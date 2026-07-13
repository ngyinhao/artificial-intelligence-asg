from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

import complaint_compass.evaluate as evaluation
import complaint_compass.train as training
from complaint_compass.config import PRODUCT_LABELS
from complaint_compass.inference import ComplaintPredictor

KEYWORDS = (
    "credit report",
    "debt",
    "card",
    "mortgage",
    "checking",
    "transfer",
)


class FakeSentenceTransformer:
    """Small deterministic encoder used to test MiniLM artifact plumbing offline."""

    def __init__(self, source: str) -> None:
        self.source = source

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        features = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray(
                [lowered.count(keyword) for keyword in KEYWORDS]
                + [len(lowered) / 1_000, lowered.count("problem")],
                dtype=float,
            )
            if normalize_embeddings:
                vector /= max(float(np.linalg.norm(vector)), 1.0)
            features.append(vector)
        return np.vstack(features)

    def save(self, path: str) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "fake_encoder.txt").write_text("test encoder", encoding="utf-8")


def synthetic_dataset() -> pd.DataFrame:
    rows = []
    complaint_id = 1
    for label, keyword in zip(PRODUCT_LABELS, KEYWORDS, strict=True):
        for index in range(12):
            split = "train" if index < 8 else "validation" if index < 10 else "test"
            rows.append(
                {
                    "complaint_id": str(complaint_id),
                    "date_received": "2024-01-01",
                    "text": (
                        f"My {keyword} problem number {index} remains unresolved "
                        f"after several detailed requests about {keyword}."
                    ),
                    "text_sha256": f"hash-{complaint_id}",
                    "label": label,
                    "split": split,
                }
            )
            complaint_id += 1
    return pd.DataFrame(rows)


def test_all_training_paths_create_reloadable_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "complaints.csv"
    synthetic_dataset().to_csv(data_path, index=False)
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(training, "MODELS_DIR", artifacts_dir / "models")
    monkeypatch.setattr(training, "REGISTRY_PATH", artifacts_dir / "registry.json")
    monkeypatch.setattr(
        training, "VALIDATION_METRICS_PATH", reports_dir / "validation_metrics.json"
    )
    monkeypatch.setattr(
        training, "DATASET_MANIFEST_PATH", tmp_path / "dataset_manifest.json"
    )
    monkeypatch.setattr(evaluation, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(
        evaluation, "TEST_METRICS_PATH", reports_dir / "test_metrics.json"
    )
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    registry = training.train_models(training.MODEL_NAMES, data_path=data_path)

    assert registry["available_models"] == sorted(training.MODEL_NAMES)
    assert registry["default_model"] in training.MODEL_NAMES
    predictor = ComplaintPredictor(artifacts_dir)
    for model_name in training.MODEL_NAMES:
        result = predictor.predict(
            "My mortgage problem remains unresolved after several detailed requests.",
            model_name=model_name,
        )
        assert result["label"] in PRODUCT_LABELS
        assert len(result["top_categories"]) == 3

    test_results = evaluation.evaluate_models(
        data_path=data_path, artifacts_dir=artifacts_dir
    )
    assert set(test_results) == set(training.MODEL_NAMES)
    assert (reports_dir / "test_metrics.json").exists()
    assert (reports_dir / "model_comparison.csv").exists()
    for model_name in training.MODEL_NAMES:
        assert (reports_dir / f"confusion_matrix_{model_name}.png").exists()
