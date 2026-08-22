"""Train and register the ComplaintCompass NLP approaches."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    ARTIFACTS_DIR,
    BASE_MODEL_NAMES,
    DATASET_MANIFEST_PATH,
    FUSION_MODEL_NAME,
    MINILM_MODEL_ID,
    MODEL_DESCRIPTIONS,
    MODEL_NAMES,
    MODEL_VERSION,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    PRODUCT_LABELS,
    RANDOM_SEED,
    REGISTRY_PATH,
    VALIDATION_METRICS_PATH,
    ensure_runtime_directories,
)
from .data import file_sha256
from .fusion import fit_adaptive_fusion
from .metrics import classification_metrics


class TrainingError(RuntimeError):
    """Raised when the processed data or training configuration is invalid."""


@dataclass(frozen=True)
class TrainingOutcome:
    """Validation evidence retained long enough to train the fusion algorithm."""

    metrics: dict[str, Any]
    validation_probabilities: np.ndarray
    classes: np.ndarray


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=60_000,
        sublinear_tf=True,
    )


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def _load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run data prepare first."
        )
    frame = pd.read_csv(path, dtype={"complaint_id": str})
    required = {"text", "label", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise TrainingError(f"Processed dataset is missing: {sorted(missing)}")
    training = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    if training.empty or validation.empty:
        raise TrainingError("Training and validation splits must both be non-empty.")
    return training, validation


def _artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _mean_latency_ms(operation: Callable[[], Any], sample_size: int) -> float:
    operation()  # Warm up lazy allocations and model paths.
    start = time.perf_counter()
    operation()
    return float((time.perf_counter() - start) * 1000 / max(sample_size, 1))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_sklearn_artifact(
    model_name: str,
    estimator: Any,
    *,
    dataset_sha256: str,
    best_params: dict[str, Any],
) -> dict[str, Any]:
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(estimator, model_path)
    metadata = {
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "kind": "sklearn",
        "description": MODEL_DESCRIPTIONS[model_name],
        "labels": [str(label) for label in estimator.classes_],
        "dataset_sha256": dataset_sha256,
        "best_params": best_params,
        "model_file": model_path.name,
    }
    _write_json(model_dir / "metadata.json", metadata)
    metadata["artifact_size_bytes"] = _artifact_size(model_dir)
    _write_json(model_dir / "metadata.json", metadata)
    return metadata


def _fit_naive_bayes(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dataset_sha256: str,
) -> TrainingOutcome:
    pipeline = Pipeline(
        [("tfidf", _tfidf()), ("model", MultinomialNB())]
    )
    search = GridSearchCV(
        pipeline,
        {"model__alpha": [0.1, 0.5, 1.0]},
        scoring="f1_macro",
        cv=_cv(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(train["text"], train["label"])
    validation_pred = search.best_estimator_.predict(validation["text"])
    validation_probabilities = search.best_estimator_.predict_proba(
        validation["text"]
    )
    metrics = classification_metrics(
        validation["label"], validation_pred, labels=PRODUCT_LABELS
    )
    combined = pd.concat([train, validation], ignore_index=True)
    final_model = clone(search.best_estimator_).fit(combined["text"], combined["label"])
    metadata = _save_sklearn_artifact(
        "naive_bayes",
        final_model,
        dataset_sha256=dataset_sha256,
        best_params=search.best_params_,
    )
    sample = validation["text"].iloc[: min(100, len(validation))].tolist()
    metrics.update(
        {
            "cv_macro_f1_mean": float(search.best_score_),
            "cv_macro_f1_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "best_params": search.best_params_,
            "artifact_size_bytes": metadata["artifact_size_bytes"],
            "mean_inference_latency_ms": _mean_latency_ms(
                lambda: final_model.predict_proba(sample), len(sample)
            ),
        }
    )
    return TrainingOutcome(
        metrics=metrics,
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        classes=np.asarray(search.best_estimator_.classes_, dtype=str),
    )


def _fit_linear_svm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dataset_sha256: str,
) -> TrainingOutcome:
    pipeline = Pipeline(
        [
            ("tfidf", _tfidf()),
            ("model", LinearSVC(random_state=RANDOM_SEED)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"model__C": [0.5, 1.0, 2.0]},
        scoring="f1_macro",
        cv=_cv(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(train["text"], train["label"])
    validation_model = CalibratedClassifierCV(
        clone(search.best_estimator_), method="sigmoid", cv=5, n_jobs=-1
    ).fit(train["text"], train["label"])
    validation_pred = validation_model.predict(validation["text"])
    validation_probabilities = validation_model.predict_proba(validation["text"])
    metrics = classification_metrics(
        validation["label"], validation_pred, labels=PRODUCT_LABELS
    )

    combined = pd.concat([train, validation], ignore_index=True)
    final_model = CalibratedClassifierCV(
        clone(search.best_estimator_), method="sigmoid", cv=5, n_jobs=-1
    ).fit(combined["text"], combined["label"])
    metadata = _save_sklearn_artifact(
        "linear_svm",
        final_model,
        dataset_sha256=dataset_sha256,
        best_params=search.best_params_,
    )
    sample = validation["text"].iloc[: min(100, len(validation))].tolist()
    metrics.update(
        {
            "cv_macro_f1_mean": float(search.best_score_),
            "cv_macro_f1_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "best_params": search.best_params_,
            "artifact_size_bytes": metadata["artifact_size_bytes"],
            "mean_inference_latency_ms": _mean_latency_ms(
                lambda: final_model.predict_proba(sample), len(sample)
            ),
        }
    )
    return TrainingOutcome(
        metrics=metrics,
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        classes=np.asarray(validation_model.classes_, dtype=str),
    )


def _fit_minilm_logreg(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dataset_sha256: str,
) -> TrainingOutcome:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise TrainingError(
            "sentence-transformers is required for minilm_logreg. "
            "Install requirements.txt first."
        ) from exc

    local_encoder = MODELS_DIR / "minilm_logreg" / "encoder"
    encoder_source = str(local_encoder) if local_encoder.exists() else MINILM_MODEL_ID
    encoder = SentenceTransformer(encoder_source)
    train_embeddings = encoder.encode(
        train["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    validation_embeddings = encoder.encode(
        validation["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    classifier = LogisticRegression(
        max_iter=2_000,
        random_state=RANDOM_SEED,
    )
    search = GridSearchCV(
        classifier,
        {"C": [0.5, 1.0, 2.0]},
        scoring="f1_macro",
        cv=_cv(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_embeddings, train["label"])
    validation_pred = search.best_estimator_.predict(validation_embeddings)
    validation_probabilities = search.best_estimator_.predict_proba(
        validation_embeddings
    )
    metrics = classification_metrics(
        validation["label"], validation_pred, labels=PRODUCT_LABELS
    )

    combined_embeddings = np.concatenate(
        [train_embeddings, validation_embeddings], axis=0
    )
    combined_labels = pd.concat(
        [train["label"], validation["label"]], ignore_index=True
    )
    final_classifier = clone(search.best_estimator_).fit(
        combined_embeddings, combined_labels
    )

    model_dir = MODELS_DIR / "minilm_logreg"
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = model_dir / "classifier.joblib"
    encoder_path = model_dir / "encoder"
    joblib.dump(final_classifier, classifier_path)
    encoder.save(str(encoder_path))
    metadata = {
        "model_name": "minilm_logreg",
        "model_version": MODEL_VERSION,
        "kind": "minilm",
        "description": MODEL_DESCRIPTIONS["minilm_logreg"],
        "labels": [str(label) for label in final_classifier.classes_],
        "dataset_sha256": dataset_sha256,
        "best_params": search.best_params_,
        "classifier_file": classifier_path.name,
        "encoder_directory": encoder_path.name,
        "encoder_source": MINILM_MODEL_ID,
    }
    metadata["artifact_size_bytes"] = _artifact_size(model_dir)
    _write_json(model_dir / "metadata.json", metadata)

    sample = validation["text"].iloc[: min(100, len(validation))].tolist()

    def predict_sample() -> np.ndarray:
        sample_embeddings = encoder.encode(
            sample,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return final_classifier.predict_proba(sample_embeddings)

    metrics.update(
        {
            "cv_macro_f1_mean": float(search.best_score_),
            "cv_macro_f1_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "best_params": search.best_params_,
            "artifact_size_bytes": metadata["artifact_size_bytes"],
            "mean_inference_latency_ms": _mean_latency_ms(
                predict_sample, len(sample)
            ),
        }
    )
    return TrainingOutcome(
        metrics=metrics,
        validation_probabilities=np.asarray(validation_probabilities, dtype=float),
        classes=np.asarray(search.best_estimator_.classes_, dtype=str),
    )


TRAINERS: dict[
    str, Callable[[pd.DataFrame, pd.DataFrame, str], TrainingOutcome]
] = {
    "naive_bayes": _fit_naive_bayes,
    "linear_svm": _fit_linear_svm,
    "minilm_logreg": _fit_minilm_logreg,
}


def _align_probabilities(outcome: TrainingOutcome) -> np.ndarray:
    indices = []
    classes = outcome.classes.tolist()
    for label in PRODUCT_LABELS:
        if label not in classes:
            raise TrainingError(f"Validation probabilities are missing class '{label}'.")
        indices.append(classes.index(label))
    return outcome.validation_probabilities[:, indices]


def _fit_adaptive_fusion(
    outcomes: dict[str, TrainingOutcome],
    validation: pd.DataFrame,
    dataset_sha256: str,
) -> dict[str, Any]:
    missing = set(BASE_MODEL_NAMES).difference(outcomes)
    if missing:
        raise TrainingError(
            "Adaptive fusion requires fresh validation predictions from all base "
            f"models; missing {sorted(missing)}. Run training with --all."
        )
    probability_cube = np.stack(
        [_align_probabilities(outcomes[name]) for name in BASE_MODEL_NAMES], axis=0
    )
    configuration, fused_probabilities = fit_adaptive_fusion(
        validation["label"].tolist(),
        probability_cube,
        labels=PRODUCT_LABELS,
        member_names=BASE_MODEL_NAMES,
    )
    fused_labels = np.asarray(PRODUCT_LABELS, dtype=str)[
        np.argmax(fused_probabilities, axis=1)
    ]
    metrics = classification_metrics(
        validation["label"], fused_labels, labels=PRODUCT_LABELS
    )
    model_dir = MODELS_DIR / FUSION_MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = model_dir / "configuration.json"
    _write_json(configuration_path, configuration)
    member_size = sum(
        int(outcomes[name].metrics["artifact_size_bytes"])
        for name in BASE_MODEL_NAMES
    )
    fusion_latency = _mean_latency_ms(
        lambda: fit_adaptive_fusion(
            validation["label"].iloc[:100].tolist(),
            probability_cube[:, :100, :],
            labels=PRODUCT_LABELS,
            member_names=BASE_MODEL_NAMES,
            alpha_grid=(configuration["alpha"],),
            beta_grid=(configuration["beta"],),
            gamma_grid=(configuration["gamma"],),
        ),
        min(100, len(validation)),
    )
    effective_latency = sum(
        float(outcomes[name].metrics["mean_inference_latency_ms"])
        for name in BASE_MODEL_NAMES
    ) + fusion_latency
    metadata = {
        "model_name": FUSION_MODEL_NAME,
        "model_version": MODEL_VERSION,
        "kind": "adaptive_fusion",
        "description": MODEL_DESCRIPTIONS[FUSION_MODEL_NAME],
        "labels": list(PRODUCT_LABELS),
        "dataset_sha256": dataset_sha256,
        "member_models": list(BASE_MODEL_NAMES),
        "configuration_file": configuration_path.name,
        "best_params": {
            "alpha": configuration["alpha"],
            "beta": configuration["beta"],
            "gamma": configuration["gamma"],
        },
        "evidence_split": "validation",
        "evidence_status": "exploratory; confirm on a fresh untouched holdout",
    }
    _write_json(model_dir / "metadata.json", metadata)
    metadata["artifact_size_bytes"] = member_size + _artifact_size(model_dir)
    _write_json(model_dir / "metadata.json", metadata)
    metrics.update(
        {
            "best_params": metadata["best_params"],
            "validation_log_loss": configuration["validation_log_loss"],
            "member_models": list(BASE_MODEL_NAMES),
            "artifact_size_bytes": metadata["artifact_size_bytes"],
            "mean_inference_latency_ms": effective_latency,
        }
    )
    return metrics


def select_default_model(metrics: dict[str, dict[str, Any]]) -> str:
    """Select the highest validation macro-F1, breaking exact ties by efficiency."""

    if not metrics:
        raise TrainingError("No trained models are available for selection.")
    return min(
        metrics,
        key=lambda name: (
            -float(metrics[name]["macro_f1"]),
            int(metrics[name].get("artifact_size_bytes", 2**63 - 1)),
            float(metrics[name].get("mean_inference_latency_ms", float("inf"))),
            name,
        ),
    )


def train_models(
    model_names: Sequence[str],
    *,
    data_path: Path = PROCESSED_DATA_PATH,
) -> dict[str, Any]:
    ensure_runtime_directories()
    unknown = set(model_names).difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    train, validation = _load_data(data_path)
    dataset_sha256 = file_sha256(data_path)

    existing: dict[str, dict[str, Any]] = {}
    if VALIDATION_METRICS_PATH.exists() and REGISTRY_PATH.exists():
        current_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if current_registry.get("dataset_sha256") == dataset_sha256:
            existing = json.loads(
                VALIDATION_METRICS_PATH.read_text(encoding="utf-8")
            )

    outcomes: dict[str, TrainingOutcome] = {}
    requested = tuple(model_names)
    if FUSION_MODEL_NAME in requested and not set(BASE_MODEL_NAMES).issubset(requested):
        raise TrainingError(
            "Adaptive fusion must be trained with all base models in the same run. "
            "Use `python -m complaint_compass.train --all`."
        )

    for model_name in requested:
        if model_name == FUSION_MODEL_NAME:
            continue
        print(f"Training {model_name}...")
        outcome = TRAINERS[model_name](train, validation, dataset_sha256)
        outcomes[model_name] = outcome
        existing[model_name] = outcome.metrics
        _write_json(VALIDATION_METRICS_PATH, existing)

    if FUSION_MODEL_NAME in requested:
        print(f"Training {FUSION_MODEL_NAME}...")
        existing[FUSION_MODEL_NAME] = _fit_adaptive_fusion(
            outcomes, validation, dataset_sha256
        )
        _write_json(VALIDATION_METRICS_PATH, existing)

    registry_models: dict[str, Any] = {}
    for model_name in existing:
        metadata_path = MODELS_DIR / model_name / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_sha256") == dataset_sha256:
            registry_models[model_name] = metadata
    registered_metrics = {
        model_name: existing[model_name] for model_name in registry_models
    }
    default_model = select_default_model(registered_metrics)
    registry = {
        "model_version": MODEL_VERSION,
        "dataset_sha256": dataset_sha256,
        "dataset_manifest": str(DATASET_MANIFEST_PATH),
        "default_model": default_model,
        "available_models": sorted(registry_models),
        "models": registry_models,
    }
    _write_json(REGISTRY_PATH, registry)
    return registry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Train all models")
    group.add_argument("--model", choices=MODEL_NAMES, action="append")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    names = MODEL_NAMES if args.all else tuple(args.model)
    registry = train_models(names, data_path=args.data)
    print(f"Default model: {registry['default_model']}")
    print(f"Registry: {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
