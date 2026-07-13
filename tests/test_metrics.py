from __future__ import annotations

import pytest

from complaint_compass.metrics import classification_metrics


def test_classification_metrics_use_fixed_label_order() -> None:
    labels = ["a", "b", "c"]
    metrics = classification_metrics(
        ["a", "a", "b", "c"],
        ["a", "b", "b", "c"],
        labels=labels,
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert len(metrics["confusion_matrix"]) == 3
    assert list(metrics["per_class"])[0:3] == labels
