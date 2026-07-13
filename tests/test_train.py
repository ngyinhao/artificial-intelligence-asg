from __future__ import annotations

from complaint_compass.train import select_default_model


def test_default_selection_uses_size_for_near_tie() -> None:
    selected = select_default_model(
        {
            "slightly_better": {
                "macro_f1": 0.900,
                "artifact_size_bytes": 1_000,
                "mean_inference_latency_ms": 1.0,
            },
            "smaller_near_tie": {
                "macro_f1": 0.895,
                "artifact_size_bytes": 100,
                "mean_inference_latency_ms": 2.0,
            },
            "outside_tie": {
                "macro_f1": 0.880,
                "artifact_size_bytes": 1,
                "mean_inference_latency_ms": 0.1,
            },
        }
    )
    assert selected == "smaller_near_tie"


def test_default_selection_uses_best_score_when_not_tied() -> None:
    selected = select_default_model(
        {
            "best": {"macro_f1": 0.90, "artifact_size_bytes": 1_000},
            "small": {"macro_f1": 0.80, "artifact_size_bytes": 1},
        }
    )
    assert selected == "best"
