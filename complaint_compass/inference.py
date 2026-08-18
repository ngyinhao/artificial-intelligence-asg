"""Artifact-only inference contract for ComplaintCompass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import (
    ARTIFACTS_DIR,
    BASE_MODEL_NAMES,
    ENSEMBLE_MODEL_NAME,
    TOP_K,
)
from .text import validate_text


class ArtifactError(RuntimeError):
    """Raised when trained artifacts are absent or incompatible."""


class ComplaintPredictor:
    """Load registered artifacts lazily and provide stable prediction results."""

    def __init__(self, artifacts_dir: Path | str = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        registry_path = self.artifacts_dir / "registry.json"
        if not registry_path.exists():
            raise ArtifactError(
                f"Model registry not found at {registry_path}. "
                "Run `python -m complaint_compass.train --all` first."
            )
        self.registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.default_model = str(self.registry.get("default_model", ""))
        self.available_models = tuple(self.registry.get("available_models", []))
        if not self.default_model or self.default_model not in self.available_models:
            raise ArtifactError("The model registry does not contain a valid default model.")
        self._loaded: dict[
            str, tuple[dict[str, Any], Any | None, Any | None]
        ] = {}

    def _load(
        self, model_name: str
    ) -> tuple[dict[str, Any], Any | None, Any | None]:
        if model_name not in self.available_models:
            raise ArtifactError(
                f"Unknown model '{model_name}'. Available: {self.available_models}"
            )
        if model_name in self._loaded:
            return self._loaded[model_name]

        model_dir = self.artifacts_dir / "models" / model_name
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise ArtifactError(f"Missing metadata for {model_name}: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        kind = metadata.get("kind")
        encoder = None
        if kind == "sklearn":
            model_path = model_dir / metadata["model_file"]
            if not model_path.exists():
                raise ArtifactError(f"Missing model artifact: {model_path}")
            model = joblib.load(model_path)
        elif kind == "minilm":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ArtifactError(
                    "sentence-transformers is required to load the MiniLM model."
                ) from exc
            classifier_path = model_dir / metadata["classifier_file"]
            encoder_path = model_dir / metadata["encoder_directory"]
            if not classifier_path.exists() or not encoder_path.exists():
                raise ArtifactError(
                    f"Incomplete MiniLM artifact directory: {model_dir}"
                )
            model = joblib.load(classifier_path)
            encoder = SentenceTransformer(str(encoder_path))
        elif kind == "ensemble":
            self._validate_ensemble_metadata(model_name, metadata)
            model = None
        else:
            raise ArtifactError(f"Unsupported artifact kind '{kind}' for {model_name}.")

        loaded = (metadata, model, encoder)
        self._loaded[model_name] = loaded
        return loaded

    def _validate_ensemble_metadata(
        self, model_name: str, metadata: dict[str, Any]
    ) -> None:
        base_models = metadata.get("base_models")
        weights = metadata.get("weights")
        labels = metadata.get("labels")
        if model_name != ENSEMBLE_MODEL_NAME:
            raise ArtifactError("Ensemble metadata uses an unsupported model name.")
        if not isinstance(base_models, list) or tuple(base_models) != BASE_MODEL_NAMES:
            raise ArtifactError(
                f"{model_name} must reference the registered base models in order."
            )
        if not isinstance(weights, dict) or set(weights) != set(BASE_MODEL_NAMES):
            raise ArtifactError(f"{model_name} has missing or unexpected weights.")
        values = np.asarray([weights[name] for name in BASE_MODEL_NAMES], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ArtifactError(f"{model_name} weights must be finite and positive.")
        if not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise ArtifactError(f"{model_name} weights must sum to one.")
        if not isinstance(labels, list) or len(labels) != len(set(labels)):
            raise ArtifactError(f"{model_name} labels are missing or duplicated.")
        missing = set(base_models).difference(self.available_models)
        if missing:
            raise ArtifactError(
                f"{model_name} is missing registered base models: {sorted(missing)}"
            )

    @staticmethod
    def _align_probabilities(
        probabilities: np.ndarray,
        classes: np.ndarray,
        target_classes: np.ndarray,
    ) -> np.ndarray:
        class_names = [str(value) for value in classes]
        target_names = [str(value) for value in target_classes]
        if len(set(class_names)) != len(class_names) or set(class_names) != set(
            target_names
        ):
            raise ArtifactError("Base-model classes do not match ensemble labels.")
        return probabilities[
            :, [class_names.index(label) for label in target_names]
        ]

    @staticmethod
    def _validate_probabilities(
        probabilities: np.ndarray, classes: np.ndarray, row_count: int
    ) -> None:
        if probabilities.shape != (row_count, len(classes)):
            raise ArtifactError("Model probabilities do not match the registered classes.")
        if not np.isfinite(probabilities).all():
            raise ArtifactError("Model returned non-finite probabilities.")
        if np.any(probabilities < -1e-12):
            raise ArtifactError("Model returned negative probabilities.")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ArtifactError("Model probabilities do not sum to one.")

    def _predict_normalized(
        self,
        normalized: list[str],
        model_name: str,
        *,
        ancestry: tuple[str, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if model_name in ancestry:
            raise ArtifactError("Recursive ensemble model references are not supported.")
        metadata, model, encoder = self._load(model_name)
        kind = metadata["kind"]
        if kind == "minilm":
            features = encoder.encode(
                normalized,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            classes = np.asarray(model.classes_, dtype=str)
        elif kind == "sklearn":
            probabilities = np.asarray(model.predict_proba(normalized), dtype=float)
            classes = np.asarray(model.classes_, dtype=str)
        else:
            target_classes = np.asarray(metadata["labels"], dtype=str)
            combined = np.zeros((len(normalized), len(target_classes)), dtype=float)
            expected_checksum = metadata.get("dataset_sha256")
            for base_name in metadata["base_models"]:
                base_metadata, _, _ = self._load(base_name)
                if base_metadata.get("kind") == "ensemble":
                    raise ArtifactError("Recursive ensemble components are not supported.")
                if base_metadata.get("dataset_sha256") != expected_checksum:
                    raise ArtifactError(
                        f"Dataset checksum mismatch for ensemble component {base_name}."
                    )
                _, base_probabilities, base_classes = self._predict_normalized(
                    normalized,
                    base_name,
                    ancestry=(*ancestry, model_name),
                )
                combined += float(metadata["weights"][base_name]) * (
                    self._align_probabilities(
                        base_probabilities, base_classes, target_classes
                    )
                )
            row_sums = combined.sum(axis=1, keepdims=True)
            if not np.isfinite(row_sums).all() or np.any(row_sums <= 0):
                raise ArtifactError("Ensemble produced invalid probability totals.")
            probabilities = combined / row_sums
            classes = target_classes

        self._validate_probabilities(probabilities, classes, len(normalized))
        labels = classes[np.argmax(probabilities, axis=1)]
        return labels, probabilities, classes

    def predict_batch(
        self, texts: list[str], *, model_name: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return labels, probabilities, and model class order for validated text."""

        if not texts:
            raise ValueError("At least one complaint is required.")
        normalized = [validate_text(text) for text in texts]
        selected = model_name or self.default_model
        return self._predict_normalized(normalized, selected)

    def artifact_size_bytes(self, model_name: str) -> int:
        """Return the unique on-disk footprint required by a registered model."""

        visited_models: set[str] = set()
        files: set[Path] = set()

        def visit(name: str, ancestry: tuple[str, ...] = ()) -> None:
            if name in ancestry:
                raise ArtifactError("Recursive ensemble model references are not supported.")
            if name in visited_models:
                return
            visited_models.add(name)
            metadata, _, _ = self._load(name)
            model_dir = self.artifacts_dir / "models" / name
            files.update(item.resolve() for item in model_dir.rglob("*") if item.is_file())
            if metadata["kind"] == "ensemble":
                for base_name in metadata["base_models"]:
                    visit(base_name, (*ancestry, name))

        visit(model_name)
        return sum(path.stat().st_size for path in files)

    def predict(self, text: str, model_name: str | None = None) -> dict[str, Any]:
        """Return the public prediction contract for one complaint narrative."""

        labels, probabilities, classes = self.predict_batch(
            [text], model_name=model_name
        )
        order = np.argsort(probabilities[0])[::-1][:TOP_K]
        return {
            "label": str(labels[0]),
            "confidence": float(probabilities[0][order[0]]),
            "top_categories": [
                {
                    "label": str(classes[index]),
                    "probability": float(probabilities[0][index]),
                }
                for index in order
            ],
        }


_DEFAULT_PREDICTOR: ComplaintPredictor | None = None


def predict(text: str, model_name: str | None = None) -> dict[str, Any]:
    """Module-level convenience API backed by the default artifact registry."""

    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = ComplaintPredictor()
    return _DEFAULT_PREDICTOR.predict(text, model_name=model_name)
