# Streamlit AppTest exposes formatted selectbox options

- **Context / intended action:** Verify that the real hosted-style application defaults to Linear SVM and allows selecting the new `adaptive_fusion` method.
- **Symptom:** The smoke harness asserted that the raw string `adaptive_fusion` appeared in `selectbox.options`, but the assertion failed.
- **Impact:** The first ARUF UI smoke attempt stopped before classification.
- **Cause:** Streamlit AppTest exposes option display labels after the widget's `format_func` is applied, while `selectbox.value` remains the underlying raw registry value. ComplaintCompass displays each model name together with its description.
- **Troubleshooting:** The default raw value assertion succeeded; only membership in the formatted option list failed.
- **Workaround:** Assert presence against the formatted display options, but pass the raw registry value (`adaptive_fusion`) to AppTest's `select` method.
- **Prevention:** Tests for formatted Streamlit selectors should distinguish stored values from displayed options and assert both contracts deliberately.

## Follow-up evidence

Passing the formatted display label to `select` caused AppTest to apply the `format_func` a second time and raise a `ValueError` containing a duplicated label. This confirmed that selection requires the raw value even though inspection returns formatted options.
