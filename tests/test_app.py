from __future__ import annotations

from pathlib import Path

import json
import joblib
import numpy as np
from streamlit.testing.v1 import AppTest

from app import _metric_frame
from complaint_compass.config import PRODUCT_LABELS


class StaticAppModel:
    def __init__(self) -> None:
        self.classes_ = np.asarray(PRODUCT_LABELS, dtype=str)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        probabilities = np.asarray([0.05, 0.10, 0.15, 0.10, 0.10, 0.50])
        return np.tile(probabilities, (len(texts), 1))

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.asarray([PRODUCT_LABELS[5]] * len(texts), dtype=str)


def build_app_artifacts(root: Path) -> None:
    model_dir = root / "models" / "static"
    model_dir.mkdir(parents=True)
    joblib.dump(StaticAppModel(), model_dir / "model.joblib")
    metadata = {
        "model_name": "static",
        "model_version": "test",
        "kind": "sklearn",
        "labels": list(PRODUCT_LABELS),
        "model_file": "model.joblib",
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    registry = {
        "default_model": "static",
        "available_models": ["static"],
        "models": {"static": metadata},
    }
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


def test_app_reports_missing_artifacts_without_crashing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("COMPLAINT_COMPASS_ARTIFACTS_DIR", str(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert app.error
    assert "Model registry not found" in app.error[0].value


def test_metric_frame_omits_weighted_f1() -> None:
    frame = _metric_frame(
        {
            "example": {
                "accuracy": 0.8,
                "macro_precision": 0.8,
                "macro_recall": 0.8,
                "macro_f1": 0.8,
                "weighted_f1": 0.9,
            }
        }
    )

    assert "Weighted F1" not in frame.columns


def test_app_classifies_with_a_registered_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    build_app_artifacts(tmp_path)
    monkeypatch.setenv("COMPLAINT_COMPASS_ARTIFACTS_DIR", str(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    app.text_area[0].input(
        "My mortgage payment was applied incorrectly and remains unresolved."
    )
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.success[0].value == PRODUCT_LABELS[5]
    assert app.metric[0].value == "50.0000%"
