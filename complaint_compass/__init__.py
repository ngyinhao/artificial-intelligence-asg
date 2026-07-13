"""ComplaintCompass: comparative NLP for complaint routing."""

from .config import MODEL_VERSION, PRODUCT_LABELS
from .inference import ComplaintPredictor, predict

__all__ = [
    "ComplaintPredictor",
    "MODEL_VERSION",
    "PRODUCT_LABELS",
    "predict",
]

__version__ = MODEL_VERSION
