"""Shared configuration for the ComplaintCompass pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_POOL_PATH = RAW_DATA_DIR / "cfpb_complaint_pool.csv"
DOWNLOAD_MANIFEST_PATH = RAW_DATA_DIR / "download_manifest.json"
RANGE_CHECKPOINT_PATH = RAW_DATA_DIR / "cfpb_range_checkpoint.csv"
RANGE_CHECKPOINT_MANIFEST_PATH = RAW_DATA_DIR / "cfpb_range_checkpoint.json"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "complaints.csv"
DATASET_MANIFEST_PATH = PROCESSED_DATA_DIR / "dataset_manifest.json"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"
VALIDATION_METRICS_PATH = REPORTS_DIR / "validation_metrics.json"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"

CFPB_API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/api/v1/"
)
CFPB_CSV_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv"

# date_received_max is exclusive in the CFPB API. These windows distribute the
# download across the approved period rather than taking one contiguous block.
DATE_WINDOWS = (
    ("2023-08-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"),
    ("2025-07-01", "2026-01-01"),
)

PRODUCT_LABELS = (
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Credit card",
    "Mortgage",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
)

BASE_MODEL_NAMES = ("naive_bayes", "linear_svm", "minilm_logreg")
ENSEMBLE_MODEL_NAME = "weighted_ensemble"
FUSION_MODEL_NAME = "adaptive_fusion"
COMBINATION_MODEL_NAMES = (ENSEMBLE_MODEL_NAME, FUSION_MODEL_NAME)
MODEL_NAMES = (*BASE_MODEL_NAMES, *COMBINATION_MODEL_NAMES)
MODEL_DESCRIPTIONS = {
    "naive_bayes": "Multinomial Naive Bayes with word unigram/bigram TF-IDF",
    "linear_svm": "Calibrated Linear SVM with word unigram/bigram TF-IDF",
    "minilm_logreg": (
        "all-MiniLM-L6-v2 sentence embeddings with Logistic Regression"
    ),
    ENSEMBLE_MODEL_NAME: (
        "Validation-weighted soft-voting ensemble of all three base models"
    ),
    "adaptive_fusion": (
        "Adaptive Reliability-Uncertainty Fusion of all three base models"
    ),
}

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_VERSION = "1.1.0"
RANDOM_SEED = 42
TARGET_PER_CLASS = 3_000
NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.90
NEAR_DUPLICATE_MAX_FEATURES = 100_000
NEAR_DUPLICATE_MIN_DOCUMENT_FREQUENCY = 1
CSV_POOL_TARGET_PER_CLASS = TARGET_PER_CLASS + 1_250
POOL_PER_WINDOW = 850
CSV_RANGE_CHUNK_BYTES = 8 * 1024 * 1024
CSV_MAX_RANGE_CHUNKS = 180
MAX_TEXT_LENGTH = 2_000
MIN_TEXT_LENGTH = 20
TOP_K = 3


def ensure_runtime_directories() -> None:
    """Create directories used by generated data, artifacts, and reports."""

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
