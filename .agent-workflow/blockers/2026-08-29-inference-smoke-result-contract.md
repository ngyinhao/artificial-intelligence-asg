# Inference smoke check assumed an object result

- **Context:** Final direct inference smoke test after rebuilding Streamlit model artifacts.
- **Intended action:** Print the default model, predicted label, and confidence from `ComplaintPredictor.predict`.
- **Symptom:** The one-line harness raised `AttributeError: 'dict' object has no attribute 'label'`.
- **Impact:** Only the optional smoke harness failed; the complete automated test suite had already passed.
- **Cause:** `ComplaintPredictor.predict` returns a dictionary, not an object with attribute access.
- **Troubleshooting:** The exception itself identified the returned type; existing inference and application tests confirm the supported contract.
- **Workaround:** Access `result['label']` and `result['confidence']`. An initial retry used the plausible but unsupported key `prediction` and raised `KeyError`; reading the implementation/tests confirmed the exact key.
- **Prevention:** Reuse the result access pattern from `app.py` or `tests/test_inference.py` when writing ad hoc smoke checks.
