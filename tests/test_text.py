from __future__ import annotations

import pytest

from complaint_compass.text import (
    TextValidationError,
    normalize_text,
    text_fingerprint,
    validate_text,
)


def test_normalize_text_handles_unicode_whitespace_and_redactions() -> None:
    text = normalize_text("  Card\u00a0issue\nXXXX   was not fixed.  ")
    assert text == "Card issue <REDACTED> was not fixed."


def test_normalize_text_enforces_character_limit() -> None:
    assert normalize_text("a" * 25, max_length=10) == "a" * 10


@pytest.mark.parametrize("value", ["", "   ", "too short"])
def test_validate_text_rejects_short_input(value: str) -> None:
    with pytest.raises(TextValidationError, match="at least 20"):
        validate_text(value)


def test_validate_text_rejects_non_string() -> None:
    with pytest.raises(TextValidationError, match="must be a string"):
        validate_text(None)


def test_text_fingerprint_is_stable() -> None:
    assert text_fingerprint("same") == text_fingerprint("same")
    assert text_fingerprint("same") != text_fingerprint("different")
