from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from complaint_compass.config import BASE_MODEL_NAMES, PRODUCT_LABELS
from complaint_compass.inference import ArtifactError, ComplaintPredictor


class StaticProbabilityModel:
    def __init__(
        self,
        probabilities: list[float] | None = None,
        classes: list[str] | tuple[str, ...] = PRODUCT_LABELS,
    ) -> None:
        self.classes_ = np.asarray(classes, dtype=str)
        self.probabilities = np.asarray(
            probabilities or [0.05, 0.10, 0.15, 0.40, 0.20, 0.10],
            dtype=float,
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.tile(self.probabilities, (len(texts), 1))

    def predict(self, texts: list[str]) -> np.ndarray:
        label = self.classes_[int(np.argmax(self.probabilities))]
        return np.asarray([label] * len(texts), dtype=str)


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


def add_fusion_artifact(root: Path) -> None:
    registry_path = root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    model_dir = root / "models" / "adaptive_fusion"
    model_dir.mkdir(parents=True)
    configuration = {
        "member_names": ["static"],
        "labels": list(PRODUCT_LABELS),
        "class_reliability": [[0.8] * len(PRODUCT_LABELS)],
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.1,
    }
    metadata = {
        "model_name": "adaptive_fusion",
        "model_version": "test",
        "kind": "adaptive_fusion",
        "labels": list(PRODUCT_LABELS),
        "configuration_file": "configuration.json",
    }
    (model_dir / "configuration.json").write_text(
        json.dumps(configuration), encoding="utf-8"
    )
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    registry["available_models"].append("adaptive_fusion")
    registry["models"]["adaptive_fusion"] = metadata
    registry_path.write_text(json.dumps(registry), encoding="utf-8")


def build_weighted_artifacts(root: Path) -> None:
    rows = {
        "naive_bayes": [0.70, 0.06, 0.06, 0.06, 0.06, 0.06],
        "linear_svm": [0.06, 0.70, 0.06, 0.06, 0.06, 0.06],
        "minilm_logreg": [0.06, 0.06, 0.70, 0.06, 0.06, 0.06],
    }
    registry_models = {}
    for name in BASE_MODEL_NAMES:
        model_dir = root / "models" / name
        model_dir.mkdir(parents=True)
        classes = list(reversed(PRODUCT_LABELS)) if name == "naive_bayes" else PRODUCT_LABELS
        probabilities = list(reversed(rows[name])) if name == "naive_bayes" else rows[name]
        joblib.dump(StaticProbabilityModel(probabilities, classes), model_dir / "model.joblib")
        metadata = {
            "model_name": name,
            "kind": "sklearn",
            "labels": list(classes),
            "dataset_sha256": "test-dataset",
            "model_file": "model.joblib",
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        registry_models[name] = metadata

    model_dir = root / "models" / "weighted_ensemble"
    model_dir.mkdir(parents=True)
    metadata = {
        "model_name": "weighted_ensemble",
        "kind": "ensemble",
        "labels": list(PRODUCT_LABELS),
        "dataset_sha256": "test-dataset",
        "base_models": list(BASE_MODEL_NAMES),
        "weights": {"naive_bayes": 0.20, "linear_svm": 0.50, "minilm_logreg": 0.30},
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    registry_models["weighted_ensemble"] = metadata
    registry = {
        "default_model": "weighted_ensemble",
        "available_models": [*BASE_MODEL_NAMES, "weighted_ensemble"],
        "models": registry_models,
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


def test_adaptive_fusion_uses_registered_member_probabilities(tmp_path: Path) -> None:
    build_artifacts(tmp_path)
    add_fusion_artifact(tmp_path)
    predictor = ComplaintPredictor(tmp_path)
    result = predictor.predict(
        "My mortgage payment was applied incorrectly and support did not fix it.",
        model_name="adaptive_fusion",
    )
    assert result["label"] == PRODUCT_LABELS[3]
    assert result["confidence"] == pytest.approx(0.40)


def test_weighted_ensemble_combines_aligned_probabilities(tmp_path: Path) -> None:
    build_weighted_artifacts(tmp_path)
    predictor = ComplaintPredictor(tmp_path)
    labels, probabilities, classes = predictor.predict_batch(
        ["My detailed complaint remains unresolved after repeated support requests."],
        model_name="weighted_ensemble",
    )

    assert probabilities[0] == pytest.approx([0.188, 0.380, 0.252, 0.06, 0.06, 0.06])
    assert labels.tolist() == [PRODUCT_LABELS[1]]
    assert classes.tolist() == list(PRODUCT_LABELS)
    assert predictor.default_model == "weighted_ensemble"
