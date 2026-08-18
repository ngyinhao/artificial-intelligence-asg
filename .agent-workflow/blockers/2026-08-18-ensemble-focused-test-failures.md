# Weighted ensemble focused-test failures

## Context and intended action

Focused unit and Streamlit tests were run after adding validation-weighted soft voting
and the fourth model selector.

## Observable symptoms

1. Weight selection returned a counterintuitive mixture and emitted repeated
   scikit-learn warnings that the supplied labels were not lexicographically ordered.
2. The Streamlit selector test could not find the raw model name in `options`, although
   the rendered option was visibly present with its description.

## Impact

The ensemble weight tie-breaking objective was being evaluated against mismatched
probability columns, and one UI assertion did not reflect Streamlit's testing API.

## Confirmed causes

- `sklearn.metrics.log_loss` internally uses lexicographically ordered class labels;
  the project intentionally uses a different canonical `PRODUCT_LABELS` order.
- `AppTest` exposes selectbox options after applying the widget's `format_func`.

## Workaround and prevention

- Reorder the combined probability columns to `sorted(PRODUCT_LABELS)` only for the
  log-loss calculation while retaining the project order everywhere else.
- Assert against the formatted selectbox option text while continuing to set the
  widget by its raw value.
- Keep explicit probability-column alignment tests whenever fixed label orders differ
  from library defaults.

## Recurrence: floating-point tie instability

After rounding generated weights for cleaner metadata, the full suite showed that two
mathematically identical combined probability matrices could differ in log loss at
machine precision because of summation order. This allowed log loss to decide a tie
before the intended distance-from-equal criterion. Round ranking-only metric values to
12 decimal places while retaining the unrounded values for reporting.
