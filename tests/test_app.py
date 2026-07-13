from __future__ import annotations

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
