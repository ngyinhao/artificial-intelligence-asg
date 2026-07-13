# PDF text extraction failed under Windows console encoding

- **Date:** 2026-07-11
- **Context and intended action:** Extract all text from the assignment PDF using the bundled Python `pypdf` library.
- **Observable symptom:** Extraction succeeded internally, but printing page text stopped on page 1 with `UnicodeEncodeError` because Windows `cp1252` could not encode a bullet character (`U+25CF`).
- **Impact:** The first extraction attempt did not expose all eight pages for review.
- **Cause:** Python inherited a non-UTF-8 Windows console output encoding.
- **Workaround:** Set `PYTHONIOENCODING=utf-8` before printing extracted PDF text.
- **Prevention:** Configure the bundled Python runner to default to UTF-8 output, or set `PYTHONUTF8=1` for document-processing commands.
