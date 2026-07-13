"""Text normalization and validation shared by training and inference."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

from .config import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH

_REDACTION_RE = re.compile(r"\b[xX]{2,}\b")
_WHITESPACE_RE = re.compile(r"\s+")


class TextValidationError(ValueError):
    """Raised when a complaint cannot be classified safely."""


def normalize_text(value: object, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Normalize a CFPB narrative without linguistic stemming or stop-word removal."""

    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _REDACTION_RE.sub("<REDACTED>", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:max_length]


def validate_text(value: object) -> str:
    """Normalize user input and enforce the public inference contract."""

    if not isinstance(value, str):
        raise TextValidationError("Complaint text must be a string.")
    text = normalize_text(value)
    if len(text) < MIN_TEXT_LENGTH:
        raise TextValidationError(
            f"Complaint text must contain at least {MIN_TEXT_LENGTH} characters."
        )
    return text


def text_fingerprint(text: str) -> str:
    """Return a stable digest used for deduplication and leakage checks."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
