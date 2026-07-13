from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from complaint_compass.config import PRODUCT_LABELS
from complaint_compass.inference import ArtifactError, ComplaintPredictor


class StaticProbabilityModel:
    def __init__(self) -> None:
        self.classes_ = np.asarray(PRODUCT_LABELS, dtype=str)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        probabilities = np.asarray([0.05, 0.10, 0.15, 0.40, 0.20, 0.10])
        return np.tile(probabilities, (len(texts), 1))

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.asarray([PRODUCT_LABELS[3]] * len(texts), dtype=str)


def build_artifacts(root: Path) -> None:
    model_dir = root / "models" / "static"
    model_dir.mkdir(parents=True)
    joblib.dump(StaticProbabilityModel(), model_dir / "model.joblib")
    metadata = {
        "model_name": "static",
        "model_version": "test",
        "kind": "sklearn",
        "labels": list(PRODUCT_LABELS),
        "model_file": "model.joblib",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    registry = {
        "default_model": "static",
        "available_models": ["static"],
        "models": {"static": metadata},
    }
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


def test_predict_returns_stable_public_contract(tmp_path: Path) -> None:
    build_artifacts(tmp_path)
    predictor = ComplaintPredictor(tmp_path)
    result = predictor.predict(
        "My mortgage payment was applied incorrectly and support did not fix it."
    )

    assert result["label"] == PRODUCT_LABELS[3]
    assert result["confidence"] == pytest.approx(0.40)
    assert len(result["top_categories"]) == 3
    assert result["top_categories"][0]["label"] == PRODUCT_LABELS[3]


def test_predict_batch_probabilities_are_valid(tmp_path: Path) -> None:
    build_artifacts(tmp_path)
    predictor = ComplaintPredictor(tmp_path)
    labels, probabilities, classes = predictor.predict_batch(
        [
            "The bank has not corrected this sufficiently detailed complaint.",
            "The company keeps contacting me about a debt I do not recognize.",
        ]
    )
    assert labels.shape == (2,)
    assert probabilities.shape == (2, 6)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert classes.tolist() == list(PRODUCT_LABELS)


def test_missing_registry_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="train --all"):
        ComplaintPredictor(tmp_path)
