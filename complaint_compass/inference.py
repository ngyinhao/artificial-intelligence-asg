"""Artifact-only inference contract for ComplaintCompass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import ARTIFACTS_DIR, TOP_K
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
        else:
            raise ArtifactError(f"Unsupported artifact kind '{kind}' for {model_name}.")

        loaded = (metadata, model, encoder)
        self._loaded[model_name] = loaded
        return loaded

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
