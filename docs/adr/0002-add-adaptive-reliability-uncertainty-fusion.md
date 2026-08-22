# ADR 0002: Add Adaptive Reliability-Uncertainty Fusion

- **Status:** Accepted
- **Decision owner:** Project contributor
- **Date:** 22 August 2026

## Context

The three ComplaintCompass base models capture complementary evidence. Naive Bayes and
Linear SVM use sparse lexical features, while MiniLM with Logistic Regression uses a
dense semantic representation. A fixed weighted soft-voting experiment was considered,
but a single global weight cannot adapt to category-specific reliability or uncertainty
on an individual complaint.

## Decision

Add Adaptive Reliability-Uncertainty Fusion (ARUF) as a fourth registered method. For
each complaint and category, ARUF multiplies each member probability by that member's
validation F1 for the category and by an entropy-derived confidence factor. It then
applies a bounded agreement multiplier when multiple members vote for the same category,
normalizes the scores, and predicts the largest score.

Select `alpha`, `beta`, and `gamma` using probabilities generated on the validation split
by base models fitted on training data only. Use validation macro-F1 as the primary
criterion, class-order-safe multiclass log loss as the first tie-breaker, and deterministic
distance-from-neutral criteria afterward. Save the selected configuration rather than
duplicating member-model artifacts.

The highest validation macro-F1 determines the Streamlit default model. Size and latency
break exact ties only.

## Generated result

The selected configuration is `alpha = 1.0`, `beta = 0.5`, and `gamma = 0.2`.
ARUF achieved validation macro-F1 0.8263 and exploratory test macro-F1 0.8363. It
outperformed MiniLM Logistic Regression and Naive Bayes but did not outperform Linear
SVM, which achieved test macro-F1 0.8458 and remains the default.

## Consequences

- The project contains a concrete, testable adaptive fusion algorithm rather than an
  unimplemented or fixed-weight ensemble proposal.
- Inference requires all three base models, increasing effective artifact size and
  latency.
- Category reliability and input uncertainty are explicit and inspectable in the saved
  configuration.
- The existing test set had been inspected before ARUF was proposed, so the ARUF test
  result is exploratory and requires confirmation on a later untouched or time-based
  holdout.
- Documentation must describe ARUF as a project-specific synthesis of established ideas,
  not claim global research novelty without a dedicated literature review.
