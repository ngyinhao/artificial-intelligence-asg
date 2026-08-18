from __future__ import annotations

import numpy as np
import pytest

from complaint_compass.config import BASE_MODEL_NAMES, PRODUCT_LABELS
from complaint_compass.train import (
    TrainingError,
    align_probabilities,
    select_default_model,
    select_ensemble_weights,
)


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


def test_align_probabilities_reorders_model_classes() -> None:
    reversed_labels = tuple(reversed(PRODUCT_LABELS))
    source = np.arange(12, dtype=float).reshape(2, 6)
    aligned = align_probabilities(source, reversed_labels)
    assert aligned[:, 0].tolist() == source[:, -1].tolist()
    assert aligned[:, -1].tolist() == source[:, 0].tolist()


def test_align_probabilities_rejects_incompatible_classes() -> None:
    with pytest.raises(TrainingError, match="canonical product labels"):
        align_probabilities(np.ones((1, 6)) / 6, [*PRODUCT_LABELS[:-1], "Other"])


def test_weight_selection_is_positive_reproducible_and_prefers_best_evidence() -> None:
    truth = list(PRODUCT_LABELS)
    perfect = np.full((6, 6), 0.02)
    np.fill_diagonal(perfect, 0.90)
    wrong = np.roll(perfect, shift=1, axis=1)
    probabilities = {
        "naive_bayes": wrong,
        "linear_svm": perfect,
        "minilm_logreg": wrong,
    }
    classes = {name: PRODUCT_LABELS for name in BASE_MODEL_NAMES}

    first = select_ensemble_weights(truth, probabilities, classes)
    second = select_ensemble_weights(truth, probabilities, classes)

    assert first["weights"] == second["weights"]
    assert first["weights"] == {
        "naive_bayes": pytest.approx(0.05),
        "linear_svm": pytest.approx(0.90),
        "minilm_logreg": pytest.approx(0.05),
    }
    assert sum(first["weights"].values()) == pytest.approx(1.0)


def test_weight_selection_uses_equal_distance_then_lexicographic_tie_break() -> None:
    truth = list(PRODUCT_LABELS)
    common = np.full((6, 6), 0.02)
    np.fill_diagonal(common, 0.90)
    probabilities = {name: common for name in BASE_MODEL_NAMES}
    classes = {name: PRODUCT_LABELS for name in BASE_MODEL_NAMES}

    selected = select_ensemble_weights(truth, probabilities, classes)

    assert tuple(selected["weights"].values()) == pytest.approx((0.30, 0.35, 0.35))
