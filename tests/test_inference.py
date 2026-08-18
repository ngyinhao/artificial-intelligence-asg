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


def build_ensemble_artifacts(root: Path) -> None:
    probability_rows = {
        "naive_bayes": [0.70, 0.06, 0.06, 0.06, 0.06, 0.06],
        "linear_svm": [0.06, 0.70, 0.06, 0.06, 0.06, 0.06],
        "minilm_logreg": [0.06, 0.06, 0.70, 0.06, 0.06, 0.06],
    }
    registry_models = {}
    for name in BASE_MODEL_NAMES:
        model_dir = root / "models" / name
        model_dir.mkdir(parents=True)
        classes = list(reversed(PRODUCT_LABELS)) if name == "naive_bayes" else PRODUCT_LABELS
        probabilities = probability_rows[name]
        if name == "naive_bayes":
            probabilities = list(reversed(probabilities))
        joblib.dump(
            StaticProbabilityModel(probabilities, classes), model_dir / "model.joblib"
        )
        metadata = {
            "model_name": name,
            "model_version": "test",
            "kind": "sklearn",
            "labels": list(classes),
            "dataset_sha256": "dataset-test",
            "model_file": "model.joblib",
        }
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        registry_models[name] = metadata

    ensemble_dir = root / "models" / "weighted_ensemble"
    ensemble_dir.mkdir(parents=True)
    ensemble = {
        "model_name": "weighted_ensemble",
        "model_version": "test",
        "kind": "ensemble",
        "labels": list(PRODUCT_LABELS),
        "dataset_sha256": "dataset-test",
        "base_models": list(BASE_MODEL_NAMES),
        "weights": {
            "naive_bayes": 0.20,
            "linear_svm": 0.50,
            "minilm_logreg": 0.30,
        },
    }
    (ensemble_dir / "metadata.json").write_text(
        json.dumps(ensemble), encoding="utf-8"
    )
    registry_models["weighted_ensemble"] = ensemble
    registry = {
        "default_model": "linear_svm",
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


def test_ensemble_combines_aligned_probabilities_exactly(tmp_path: Path) -> None:
    build_ensemble_artifacts(tmp_path)
    predictor = ComplaintPredictor(tmp_path)

    labels, probabilities, classes = predictor.predict_batch(
        ["My detailed complaint remains unresolved after several support requests."],
        model_name="weighted_ensemble",
    )

    expected = np.asarray([0.188, 0.380, 0.252, 0.06, 0.06, 0.06])
    assert probabilities[0] == pytest.approx(expected)
    assert probabilities.sum(axis=1) == pytest.approx([1.0])
    assert classes.tolist() == list(PRODUCT_LABELS)
    assert labels.tolist() == [PRODUCT_LABELS[1]]


def test_ensemble_reports_effective_unique_artifact_size(tmp_path: Path) -> None:
    build_ensemble_artifacts(tmp_path)
    predictor = ComplaintPredictor(tmp_path)
    expected = sum(
        path.stat().st_size
        for path in (tmp_path / "models").rglob("*")
        if path.is_file()
    )
    assert predictor.artifact_size_bytes("weighted_ensemble") == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata["weights"].update({"naive_bayes": 0.0}), "positive"),
        (lambda metadata: metadata["weights"].update({"naive_bayes": 0.30}), "sum to one"),
    ],
)
def test_ensemble_rejects_malformed_weights(
    tmp_path: Path, mutation, message: str
) -> None:
    build_ensemble_artifacts(tmp_path)
    path = tmp_path / "models" / "weighted_ensemble" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mutation(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")

    predictor = ComplaintPredictor(tmp_path)
    with pytest.raises(ArtifactError, match=message):
        predictor.predict(
            "My detailed complaint remains unresolved after several support requests.",
            model_name="weighted_ensemble",
        )


def test_ensemble_rejects_component_checksum_mismatch(tmp_path: Path) -> None:
    build_ensemble_artifacts(tmp_path)
    path = tmp_path / "models" / "naive_bayes" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["dataset_sha256"] = "different-dataset"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    predictor = ComplaintPredictor(tmp_path)
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        predictor.predict(
            "My detailed complaint remains unresolved after several support requests.",
            model_name="weighted_ensemble",
        )


def test_ensemble_rejects_missing_component_artifact(tmp_path: Path) -> None:
    build_ensemble_artifacts(tmp_path)
    (tmp_path / "models" / "naive_bayes" / "metadata.json").unlink()

    predictor = ComplaintPredictor(tmp_path)
    with pytest.raises(ArtifactError, match="Missing metadata"):
        predictor.predict(
            "My detailed complaint remains unresolved after several support requests.",
            model_name="weighted_ensemble",
        )


def test_ensemble_rejects_component_label_mismatch(tmp_path: Path) -> None:
    build_ensemble_artifacts(tmp_path)
    model_path = tmp_path / "models" / "naive_bayes" / "model.joblib"
    incompatible = [*PRODUCT_LABELS[:-1], "Other"]
    joblib.dump(
        StaticProbabilityModel(
            [0.70, 0.06, 0.06, 0.06, 0.06, 0.06], incompatible
        ),
        model_path,
    )

    predictor = ComplaintPredictor(tmp_path)
    with pytest.raises(ArtifactError, match="classes do not match"):
        predictor.predict(
            "My detailed complaint remains unresolved after several support requests.",
            model_name="weighted_ensemble",
        )


def test_ensemble_rejects_recursive_component_metadata(tmp_path: Path) -> None:
    build_ensemble_artifacts(tmp_path)
    path = tmp_path / "models" / "naive_bayes" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "kind": "ensemble",
            "base_models": list(BASE_MODEL_NAMES),
            "weights": {
                "naive_bayes": 0.20,
                "linear_svm": 0.50,
                "minilm_logreg": 0.30,
            },
        }
    )
    path.write_text(json.dumps(metadata), encoding="utf-8")

    predictor = ComplaintPredictor(tmp_path)
    with pytest.raises(ArtifactError, match="unsupported model name"):
        predictor.predict(
            "My detailed complaint remains unresolved after several support requests.",
            model_name="weighted_ensemble",
        )
