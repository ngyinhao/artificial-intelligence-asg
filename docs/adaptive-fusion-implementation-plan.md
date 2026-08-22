# Adaptive Reliability-Uncertainty Fusion Implementation Plan

## Objective

Add **Adaptive Reliability-Uncertainty Fusion
(ARUF)**, that combines the existing Naive Bayes, calibrated Linear SVM, and
MiniLM Logistic Regression probability outputs using category reliability,
per-input uncertainty, and prediction agreement. The implementation must be
reproducible, avoid validation/test leakage, remain compatible with the current
artifact-only Streamlit application, and be deployed through the repository's
existing Streamlit hosting path.

ARUF is a project-specific algorithm assembled from established ensemble,
calibration, and uncertainty concepts. Documentation must not claim that it is
globally novel without a separate literature review.

## Algorithm

For member model `m`, category `c`, and input `x`:

```text
confidence_m(x) = 1 - normalized_entropy(P_m(x))
weight_m,c(x) = reliability_m,c^alpha * confidence_m(x)^beta
score_c(x) = agreement_c(x) * sum_m(weight_m,c(x) * P_m,c(x))
agreement_c(x) = 1 + gamma * max(votes_c(x) - 1, 0)
prediction(x) = argmax_c normalize(score(x))
```

`reliability_m,c` is the member's validation F1 for category `c`. The
hyperparameters `alpha`, `beta`, and `gamma` are selected deterministically on
validation predictions using macro-F1, with multiclass log loss and distance
from the neutral configuration as tie-breakers.

## Evidence protocol

1. Fit each validation-stage member on the training split only.
2. Generate member probabilities for the validation split.
3. Calculate per-class member reliability and select ARUF parameters using only
   these validation predictions.
4. Save only the derived ARUF configuration; keep the existing final member
   models trained on training plus validation data.
5. Evaluate all five registered methods once through the shared test pipeline,
   including the concurrently implemented fixed weighted ensemble.
6. Record that the existing test set has previously been inspected. The ARUF
   result is exploratory until confirmed on a later untouched or time-based
   holdout.

## Implementation phases

### 1. Deep fusion module

- Add a small public interface for fitting a fusion configuration and combining
  aligned member probability tensors.
- Validate shapes, finite probabilities, class alignment, reliability ranges,
  entropy behavior, and deterministic tie-breaking.
- Keep all weighting, agreement, and normalization logic inside this module.

### 2. Training and artifact registration

- Generate leakage-safe validation probabilities for the three base models.
- Tune the ARUF parameters and calculate validation metrics.
- Save member names, class order, reliability matrix, chosen parameters,
  evidence split, and effective deployment size in the ARUF artifact metadata.
- Register `adaptive_fusion` alongside the three base methods and fixed weighted
  ensemble.

### 3. Inference, evaluation, and interface

- Load the ARUF configuration without serializing duplicate base models.
- Obtain member probabilities through the existing predictor interface, align
  their classes, and fuse them deterministically.
- Include ARUF in test evaluation, error samples, comparison metrics, and the
  Streamlit model selector.
- Show a concise explanation that the method adapts weights by validation
  reliability, input uncertainty, and model agreement.

### 4. Verification

- Add unit tests for entropy, class reliability, agreement adjustment,
  normalization, deterministic tuning, and invalid inputs.
- Extend the training integration test to prove the ARUF artifact can be
  trained, reloaded, predicted, and evaluated.
- Add a successful Streamlit classification test using generated artifacts.
- Run the complete test suite and a real-artifact application smoke test.

### 5. Documentation and report evidence

- Update the README, model card, methodology documentation, artifact registry,
  and generated comparison outputs.
- Replace the report's planned/missing-ensemble language with the implemented
  ARUF method, parameters, evidence limitations, and regenerated results.
- Preserve the supplied report template and perform structural and rendered
  visual QA after editing.

### 6. Deployment

- Commit only the verified project changes required for ARUF and its
  documentation, preserving unrelated local work.
- Push the validated revision to the existing GitHub-backed Streamlit
  deployment.
- Confirm that the hosted application loads ARUF and completes a prediction.

## Completion criteria

- ARUF is a registered, selectable model with a reproducible saved
  configuration.
- Parameter selection uses no test labels or test probabilities.
- All automated tests pass.
- Generated metrics and documentation agree with the committed implementation.
- The report contains no unsupported claim of global algorithmic novelty.
- The hosted Streamlit application exposes and successfully runs ARUF.
- The highest-validation-macro-F1 method remains the Streamlit default; after the
  combined regeneration this is `weighted_ensemble`.
