"""Evaluate registered models on the exploratory test benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_PATH,
    PRODUCT_LABELS,
    REPORTS_DIR,
    TEST_METRICS_PATH,
    ensure_runtime_directories,
)
from .inference import ComplaintPredictor
from .metrics import classification_metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _plot_confusion_matrix(model_name: str, matrix: list[list[int]]) -> Path:
    output = REPORTS_DIR / f"confusion_matrix_{model_name}.png"
    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=PRODUCT_LABELS,
        yticklabels=PRODUCT_LABELS,
        ax=axis,
    )
    axis.set_title(f"Confusion matrix: {model_name}")
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("Actual category")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def evaluate_models(
    *,
    data_path: Path = PROCESSED_DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, dict[str, Any]]:
    ensure_runtime_directories()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_path}. Run data prepare first."
        )
    frame = pd.read_csv(data_path, dtype={"complaint_id": str})
    test = frame.loc[frame["split"] == "test"].copy()
    if test.empty:
        raise ValueError("The processed dataset has no test benchmark records.")
    predictor = ComplaintPredictor(artifacts_dir)
    results: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for model_name in predictor.available_models:
        started = time.perf_counter()
        labels, probabilities, _ = predictor.predict_batch(
            test["text"].tolist(), model_name=model_name
        )
        elapsed = time.perf_counter() - started
        metrics = classification_metrics(
            test["label"], labels, labels=PRODUCT_LABELS
        )
        metrics["mean_inference_latency_ms"] = float(
            elapsed * 1000 / len(test)
        )
        metrics["artifact_size_bytes"] = predictor.artifact_size_bytes(model_name)
        metrics["confusion_matrix_path"] = str(
            _plot_confusion_matrix(model_name, metrics["confusion_matrix"])
        )
        results[model_name] = metrics

        confidence = probabilities.max(axis=1)
        error_mask = labels != test["label"].to_numpy()
        errors = test.loc[
            error_mask, ["complaint_id", "text_sha256", "label"]
        ].copy()
        errors["predicted_label"] = labels[error_mask]
        errors["confidence"] = confidence[error_mask]
        errors = (
            errors.sort_values(["label", "confidence"], ascending=[True, False])
            .groupby("label", as_index=False, group_keys=False)
            .head(5)
        )
        errors.to_csv(
            REPORTS_DIR / f"error_samples_{model_name}.csv", index=False
        )
        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "artifact_size_bytes": metrics["artifact_size_bytes"],
                "mean_inference_latency_ms": metrics[
                    "mean_inference_latency_ms"
                ],
            }
        )

    _write_json(TEST_METRICS_PATH, results)
    pd.DataFrame(comparison_rows).sort_values(
        "macro_f1", ascending=False
    ).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = evaluate_models(data_path=args.data, artifacts_dir=args.artifacts)
    for model_name, metrics in results.items():
        print(f"{model_name}: macro-F1={metrics['macro_f1']:.4f}")
    print(f"Test metrics: {TEST_METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
