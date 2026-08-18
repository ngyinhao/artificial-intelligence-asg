from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_reports_missing_artifacts_without_crashing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("COMPLAINT_COMPASS_ARTIFACTS_DIR", str(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert app.error
    assert "Model registry not found" in app.error[0].value


def test_app_lists_ensemble_and_keeps_linear_svm_default(
    tmp_path: Path, monkeypatch
) -> None:
    registry = {
        "default_model": "linear_svm",
        "available_models": [
            "naive_bayes",
            "linear_svm",
            "minilm_logreg",
            "weighted_ensemble",
        ],
        "models": {
            "weighted_ensemble": {
                "kind": "ensemble",
                "weights": {
                    "naive_bayes": 0.20,
                    "linear_svm": 0.50,
                    "minilm_logreg": 0.30,
                },
            }
        },
    }
    (tmp_path / "registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    monkeypatch.setenv("COMPLAINT_COMPASS_ARTIFACTS_DIR", str(tmp_path))
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.selectbox[0].value == "linear_svm"
    assert any(
        option.startswith("weighted_ensemble")
        for option in app.selectbox[0].options
    )

    app.selectbox[0].set_value("weighted_ensemble")
    app.run(timeout=30)
    assert not app.exception
    assert any("validation data only" in item.value for item in app.info)
    assert app.expander[0].label == "How the ensemble combines models"
