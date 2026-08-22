from __future__ import annotations

import numpy as np
import pytest

from complaint_compass.fusion import (
    FusionError,
    fit_adaptive_fusion,
    fuse_probabilities,
    normalized_entropy,
)


def test_normalized_entropy_distinguishes_certainty_from_uniformity() -> None:
    probabilities = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.9, 0.1], [0.6, 0.4]],
        ]
    )
    entropy = normalized_entropy(probabilities)
    assert entropy[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert entropy[0, 1] == pytest.approx(1.0)
    assert entropy[1, 0] < entropy[1, 1]


def test_fusion_uses_class_reliability_and_returns_probabilities() -> None:
    probabilities = np.asarray(
        [
            [[0.80, 0.20]],
            [[0.30, 0.70]],
        ]
    )
    reliability = np.asarray(
        [
            [0.95, 0.20],
            [0.20, 0.95],
        ]
    )
    fused = fuse_probabilities(
        probabilities, reliability, alpha=2.0, beta=1.0, gamma=0.0
    )
    assert fused.shape == (1, 2)
    assert fused.sum() == pytest.approx(1.0)
    assert fused[0, 0] > fused[0, 1]


def test_agreement_bonus_increases_the_agreed_category() -> None:
    probabilities = np.asarray(
        [
            [[0.55, 0.45]],
            [[0.55, 0.45]],
            [[0.10, 0.90]],
        ]
    )
    reliability = np.ones((3, 2))
    without_bonus = fuse_probabilities(
        probabilities, reliability, alpha=1.0, beta=0.0, gamma=0.0
    )
    with_bonus = fuse_probabilities(
        probabilities, reliability, alpha=1.0, beta=0.0, gamma=0.2
    )
    assert with_bonus[0, 0] > without_bonus[0, 0]


def test_fit_adaptive_fusion_is_deterministic() -> None:
    labels = ("a", "b")
    probabilities = np.asarray(
        [
            [[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]],
            [[0.6, 0.4], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]],
        ]
    )
    first, first_probabilities = fit_adaptive_fusion(
        ["a", "b", "a", "b"],
        probabilities,
        labels=labels,
        member_names=("lexical", "semantic"),
    )
    second, second_probabilities = fit_adaptive_fusion(
        ["a", "b", "a", "b"],
        probabilities,
        labels=labels,
        member_names=("lexical", "semantic"),
    )
    assert first == second
    assert np.allclose(first_probabilities, second_probabilities)


def test_fusion_rejects_mismatched_reliability_shape() -> None:
    with pytest.raises(FusionError, match="Class reliability"):
        fuse_probabilities(
            np.ones((2, 1, 3)),
            np.ones((2, 2)),
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
        )
