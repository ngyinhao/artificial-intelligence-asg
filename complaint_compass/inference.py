"""Artifact-only inference contract for ComplaintCompass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import ARTIFACTS_DIR, BASE_MODEL_NAMES, ENSEMBLE_MODEL_NAME, TOP_K
from .fusion import FusionError, fuse_probabilities
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
        self._loaded: dict[str, tuple[dict[str, Any], Any, Any | None]] = {}

    def _load(self, model_name: str) -> tuple[dict[str, Any], Any, Any | None]:
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
            base_models = tuple(metadata.get("base_models", []))
            weights = metadata.get("weights", {})
            if model_name != ENSEMBLE_MODEL_NAME or base_models != BASE_MODEL_NAMES:
                raise ArtifactError("Weighted ensemble metadata is incompatible.")
            if set(weights) != set(BASE_MODEL_NAMES):
                raise ArtifactError("Weighted ensemble has missing or unexpected weights.")
            values = np.asarray([weights[name] for name in BASE_MODEL_NAMES], dtype=float)
            if (
                not np.isfinite(values).all()
                or np.any(values <= 0)
                or not np.isclose(values.sum(), 1.0, atol=1e-8)
            ):
                raise ArtifactError("Weighted ensemble weights must be positive and sum to one.")
            if not set(base_models).issubset(self.available_models):
                raise ArtifactError("Weighted ensemble references unavailable base models.")
            model = None
        elif kind == "adaptive_fusion":
            configuration_path = model_dir / metadata["configuration_file"]
            if not configuration_path.exists():
                raise ArtifactError(
                    f"Missing fusion configuration: {configuration_path}"
                )
            model = json.loads(configuration_path.read_text(encoding="utf-8"))
            members = tuple(model.get("member_names", []))
            if not members or model_name in members:
                raise ArtifactError("Adaptive fusion contains invalid member models.")
            if not set(members).issubset(self.available_models):
                raise ArtifactError(
                    "Adaptive fusion references models absent from the registry."
                )
        else:
            raise ArtifactError(f"Unsupported artifact kind '{kind}' for {model_name}.")

        loaded = (metadata, model, encoder)
        self._loaded[model_name] = loaded
        return loaded

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
            raise ArtifactError("Base-model classes do not match combination labels.")
        return probabilities[:, [class_names.index(label) for label in target_names]]

    def predict_batch(
        self, texts: list[str], *, model_name: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return labels, probabilities, and model class order for validated text."""

        if not texts:
            raise ValueError("At least one complaint is required.")
        normalized = [validate_text(text) for text in texts]
        selected = model_name or self.default_model
        metadata, model, encoder = self._load(selected)
        if metadata["kind"] == "minilm":
            features = encoder.encode(
                normalized,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            labels = np.asarray(model.predict(features), dtype=str)
            classes = np.asarray(model.classes_, dtype=str)
        elif metadata["kind"] == "ensemble":
            classes = np.asarray(metadata["labels"], dtype=str)
            probabilities = np.zeros((len(normalized), len(classes)), dtype=float)
            expected_checksum = metadata.get("dataset_sha256")
            for member_name in metadata["base_models"]:
                member_metadata, _, _ = self._load(member_name)
                if member_metadata.get("dataset_sha256") != expected_checksum:
                    raise ArtifactError(
                        f"Dataset checksum mismatch for ensemble member {member_name}."
                    )
                _, member_probabilities, member_classes = self.predict_batch(
                    normalized, model_name=member_name
                )
                probabilities += float(metadata["weights"][member_name]) * (
                    self._align_probabilities(
                        member_probabilities, member_classes, classes
                    )
                )
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            labels = classes[np.argmax(probabilities, axis=1)]
        elif metadata["kind"] == "adaptive_fusion":
            classes = np.asarray(model["labels"], dtype=str)
            member_probabilities = []
            for member_name in model["member_names"]:
                _, probabilities_for_member, member_classes = self.predict_batch(
                    normalized, model_name=member_name
                )
                member_probabilities.append(
                    self._align_probabilities(
                        probabilities_for_member, member_classes, classes
                    )
                )
            try:
                probabilities = fuse_probabilities(
                    np.stack(member_probabilities, axis=0),
                    np.asarray(model["class_reliability"], dtype=float),
                    alpha=float(model["alpha"]),
                    beta=float(model["beta"]),
                    gamma=float(model["gamma"]),
                )
            except (FusionError, KeyError, TypeError, ValueError) as exc:
                raise ArtifactError("Adaptive fusion configuration is invalid.") from exc
            labels = classes[np.argmax(probabilities, axis=1)]
        else:
            probabilities = np.asarray(model.predict_proba(normalized), dtype=float)
            labels = np.asarray(model.predict(normalized), dtype=str)
            classes = np.asarray(model.classes_, dtype=str)

        if probabilities.shape != (len(normalized), len(classes)):
            raise ArtifactError("Model probabilities do not match the registered classes.")
        if not np.isfinite(probabilities).all():
            raise ArtifactError("Model returned non-finite probabilities.")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ArtifactError("Model probabilities do not sum to one.")
        return labels, probabilities, classes

    def artifact_size_bytes(self, model_name: str) -> int:
        """Return the unique on-disk footprint required by a registered model."""

        visited_models: set[str] = set()
        files: set[Path] = set()

        def visit(name: str, ancestry: tuple[str, ...] = ()) -> None:
            if name in ancestry:
                raise ArtifactError("Recursive combination references are unsupported.")
            if name in visited_models:
                return
            visited_models.add(name)
            metadata, model, _ = self._load(name)
            model_dir = self.artifacts_dir / "models" / name
            files.update(item.resolve() for item in model_dir.rglob("*") if item.is_file())
            if metadata["kind"] == "ensemble":
                for member_name in metadata["base_models"]:
                    visit(member_name, (*ancestry, name))
            elif metadata["kind"] == "adaptive_fusion":
                for member_name in model["member_names"]:
                    visit(member_name, (*ancestry, name))

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
