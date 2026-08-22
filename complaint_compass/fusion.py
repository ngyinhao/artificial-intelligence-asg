"""Adaptive reliability-uncertainty fusion for aligned class probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

_EPSILON = 1e-12


class FusionError(ValueError):
    """Raised when fusion inputs or configuration are inconsistent."""


def _probability_cube(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 3:
        raise FusionError(
            "Member probabilities must have shape (models, samples, classes)."
        )
    if min(probabilities.shape) < 1:
        raise FusionError("Member probabilities cannot contain an empty dimension.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise FusionError("Member probabilities must be finite and non-negative.")
    row_sums = probabilities.sum(axis=2, keepdims=True)
    if (row_sums <= 0).any():
        raise FusionError("Every member probability row must have positive mass.")
    return probabilities / row_sums


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return entropy in [0, 1] for each model/sample probability row."""

    normalized = _probability_cube(probabilities)
    class_count = normalized.shape[2]
    if class_count == 1:
        return np.zeros(normalized.shape[:2], dtype=float)
    safe = np.clip(normalized, _EPSILON, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=2) / np.log(class_count)
    return np.clip(entropy, 0.0, 1.0)


def fuse_probabilities(
    member_probabilities: np.ndarray,
    class_reliability: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Fuse aligned member probabilities with adaptive reliability weights."""

    probabilities = _probability_cube(member_probabilities)
    reliability = np.asarray(class_reliability, dtype=float)
    expected = (probabilities.shape[0], probabilities.shape[2])
    if reliability.shape != expected:
        raise FusionError(
            f"Class reliability must have shape {expected}, got {reliability.shape}."
        )
    if not np.isfinite(reliability).all() or not (
        (reliability >= 0) & (reliability <= 1)
    ).all():
        raise FusionError("Class reliability values must be finite and in [0, 1].")
    if alpha < 0 or beta < 0 or gamma < 0:
        raise FusionError("Fusion parameters alpha, beta, and gamma must be non-negative.")

    confidence = np.clip(1.0 - normalized_entropy(probabilities), _EPSILON, 1.0)
    weights = (
        np.clip(reliability, _EPSILON, 1.0)[:, None, :] ** alpha
        * confidence[:, :, None] ** beta
    )
    scores = (weights * probabilities).sum(axis=0)

    votes = np.argmax(probabilities, axis=2)
    vote_counts = np.zeros_like(scores)
    for class_index in range(scores.shape[1]):
        vote_counts[:, class_index] = (votes == class_index).sum(axis=0)
    scores *= 1.0 + gamma * np.maximum(vote_counts - 1.0, 0.0)

    totals = scores.sum(axis=1, keepdims=True)
    if (totals <= 0).any() or not np.isfinite(totals).all():
        raise FusionError("Adaptive fusion produced invalid probability mass.")
    return scores / totals


def fit_adaptive_fusion(
    y_true: Sequence[str],
    member_probabilities: np.ndarray,
    *,
    labels: Sequence[str],
    member_names: Sequence[str],
    alpha_grid: Sequence[float] = (0.5, 1.0, 2.0),
    beta_grid: Sequence[float] = (0.5, 1.0, 2.0),
    gamma_grid: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit deterministic ARUF parameters using validation probabilities only."""

    probabilities = _probability_cube(member_probabilities)
    label_order = tuple(str(label) for label in labels)
    names = tuple(str(name) for name in member_names)
    targets = np.asarray([str(label) for label in y_true], dtype=str)
    if probabilities.shape[0] != len(names):
        raise FusionError("Member names do not match the probability cube.")
    if probabilities.shape[1] != len(targets):
        raise FusionError("Targets do not match the probability cube sample count.")
    if probabilities.shape[2] != len(label_order):
        raise FusionError("Labels do not match the probability cube class count.")
    if not targets.size or not set(targets).issubset(label_order):
        raise FusionError("Targets must be non-empty and present in the label order.")

    reliability = np.vstack(
        [
            f1_score(
                targets,
                np.asarray(label_order)[np.argmax(model_probabilities, axis=1)],
                labels=list(label_order),
                average=None,
                zero_division=0,
            )
            for model_probabilities in probabilities
        ]
    )
    target_indices = np.asarray(
        [label_order.index(target) for target in targets], dtype=int
    )

    candidates: list[tuple[tuple[float, ...], dict[str, float], np.ndarray]] = []
    for alpha in alpha_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                fused = fuse_probabilities(
                    probabilities,
                    reliability,
                    alpha=float(alpha),
                    beta=float(beta),
                    gamma=float(gamma),
                )
                predicted = np.asarray(label_order)[np.argmax(fused, axis=1)]
                macro_f1 = float(
                    f1_score(
                        targets,
                        predicted,
                        labels=list(label_order),
                        average="macro",
                        zero_division=0,
                    )
                )
                loss = float(
                    -np.log(
                        np.clip(
                            fused[np.arange(len(targets)), target_indices],
                            _EPSILON,
                            1.0,
                        )
                    ).mean()
                )
                neutral_distance = abs(float(alpha) - 1.0) + abs(float(beta) - 1.0)
                rank = (-macro_f1, loss, neutral_distance, float(gamma), float(alpha), float(beta))
                candidates.append(
                    (
                        rank,
                        {
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "gamma": float(gamma),
                            "validation_macro_f1": macro_f1,
                            "validation_log_loss": loss,
                        },
                        fused,
                    )
                )
    if not candidates:
        raise FusionError("At least one fusion parameter candidate is required.")

    _, selected, fused = min(candidates, key=lambda item: item[0])
    configuration: dict[str, Any] = {
        "algorithm": "Adaptive Reliability-Uncertainty Fusion",
        "member_names": list(names),
        "labels": list(label_order),
        "class_reliability": reliability.tolist(),
        **selected,
        "selection": {
            "primary": "validation_macro_f1",
            "tie_breakers": [
                "validation_log_loss",
                "distance_from_neutral_parameters",
                "gamma",
                "alpha",
                "beta",
            ],
        },
    }
    return configuration, fused
