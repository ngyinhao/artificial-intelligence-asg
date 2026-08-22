# ComplaintCompass NLP Implementation Plan

## Project frame

**ComplaintCompass: Comparative NLP for Automated Consumer Financial Complaint Routing**

ComplaintCompass is a single-contributor academic prototype that classifies an English
consumer complaint narrative into one of six financial-product categories. It compares
three base NLP representations plus an adaptive fusion algorithm and exposes the
trained artifacts through a Streamlit application.

The intended audience is a support-triage analyst or evaluator who needs a suggested
category and a transparent comparison of the available models. The output is routing
assistance only; it does not determine complaint merit or recommend financial, legal, or
regulatory action.

## Dataset contract

Use the official CFPB Consumer Complaint Database API and only these fields:

- Complaint ID
- Date received
- Consumer complaint narrative
- Product

The downloader queries narratives received from 1 August 2023 through 31 December 2025
and distributes requests across five bounded periods. If the CFPB API denies automated
access, the downloader uses deterministic byte ranges from the official CSV export and
records those ranges in the manifest. The target categories are:

1. Credit reporting or other personal consumer reports
2. Debt collection
3. Credit card
4. Mortgage
5. Checking or savings account
6. Money transfer, virtual currency, or money service

Normalize Unicode and whitespace, replace repeated redaction markers with
`<REDACTED>`, remove missing or short narratives, and limit each input to 2,000
characters. Preserve natural wording without stemming, lemmatization, or stop-word
removal. Remove exact normalized duplicates and all duplicate text associated with
conflicting labels.

Deterministically sample 3,000 records per category and use seed 42 for a stratified
70/15/15 training, validation, and sealed-test split. Raw and processed narratives and
trained model artifacts remain outside version control. Manifests record the query,
counts, labels, seed, and SHA-256 checksums.

## Models

Train and compare:

1. Multinomial Naive Bayes with word unigram/bigram TF-IDF.
2. Calibrated Linear SVM with word unigram/bigram TF-IDF.
3. `sentence-transformers/all-MiniLM-L6-v2` embeddings with Logistic Regression.
4. Validation-weighted soft voting, combining aligned base-model probabilities using
   positive global weights selected on a 0.05 grid.
5. Adaptive Reliability-Uncertainty Fusion (ARUF), combining the aligned probability
   outputs of all three base models using per-class validation F1, per-input normalized
   entropy, and a model-agreement multiplier.

Use five-fold stratified cross-validation on the training split. Tune Naive Bayes
`alpha` over 0.1, 0.5, and 1.0, and tune the SVM and Logistic Regression `C` over 0.5,
1.0, and 2.0. Generate validation probabilities from base models fitted on training data
only. Fit the weighted ensemble and ARUF on those probabilities. Search ARUF `alpha`
and `beta` over 0.5, 1.0,
and 2.0 and `gamma` over 0, 0.05, 0.10, and 0.20. Select configurations by validation
macro-F1, then class-order-safe log loss and deterministic neutral-parameter criteria.
Select the application default strictly by the highest validation macro-F1, using size
and latency only for an exact tie. Refit each finalized base model with training plus
validation data, then evaluate all registered methods once on the test set.

Because the original test results were inspected before the combination methods were
fully analysed, their current test results are exploratory. Confirm them on a later
untouched or time-based holdout before making a generalization claim. The weighted
ensemble currently has the highest validation macro-F1 and is the application default.

Macro-F1 is the primary metric. Also report accuracy, macro precision, macro recall,
weighted F1, per-class scores, confusion matrices, artifact size, cross-validation
variation, and inference latency.

## Application behavior

The Streamlit application provides:

- A complaint-classification view accepting 20 to 2,000 English characters.
- A model selector defaulting to the validation-selected model.
- Predicted category, calibrated confidence, and top-three candidates.
- A model-comparison view backed by generated aggregate reports.
- A data and limitations view with the intended-use disclaimer.

The application processes submitted text in memory without logging or persistence. It
loads saved preprocessing and model artifacts and never retrains at application startup.

The inference contract is:

```text
predict(text, model_name=None)
  -> {
       label: string,
       confidence: float,
       top_categories: [{label: string, probability: float}]
     }
```

## Reproducible workflow

```powershell
python -m complaint_compass.data download
python -m complaint_compass.data prepare
python -m complaint_compass.train --all
python -m complaint_compass.evaluate
streamlit run app.py
```

Every step must be repeatable without manually editing downloaded or processed data.

## Verification

- Confirm exactly six classes and 3,000 processed records per class.
- Confirm no normalized narrative crosses data splits.
- Confirm seed 42 reproduces the same complaint IDs and checksums.
- Confirm every artifact returns a registered label and six finite probabilities that
  sum to one.
- Confirm saved and reloaded artifacts produce identical predictions.
- Confirm reports include all models and all classes.
- Validate empty, short, maximum-length, and redaction-heavy input.
- Verify missing artifacts produce an actionable setup message.
- Run automated tests and the application smoke test from a clean Python 3.12
  environment.

The target is a test macro-F1 of at least 0.75 for one model. A lower result is reported
honestly with error analysis rather than changing the sealed test data.

## Boundaries and assumptions

- One contributor owns data, modeling, application, testing, and reporting.
- Version 1 is English-only and local-only.
- No crawler, user accounts, or external runtime inference API is included. The
  artifact-only Streamlit interface may be hosted without transmitting complaint text
  to a model provider.
- Network access is required only to install packages, download CFPB data, and obtain
  the MiniLM weights. Saved artifacts support offline prediction afterward.
- The official assignment document template is incorporated separately when supplied.
