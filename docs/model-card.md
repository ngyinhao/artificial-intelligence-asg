# Model Card: ComplaintCompass Routing Models

## Model family

ComplaintCompass registers three multiclass base classifiers and one adaptive fusion
algorithm over the same six-label dataset:

| Registry name | Representation | Classifier | Role |
|---|---|---|---|
| `naive_bayes` | Word unigram/bigram TF-IDF | Multinomial Naive Bayes | Interpretable baseline |
| `linear_svm` | Word unigram/bigram TF-IDF | Calibrated Linear SVM | Strong sparse-text method |
| `minilm_logreg` | Normalized MiniLM sentence embeddings | Logistic Regression | Semantic transformer representation |
| `adaptive_fusion` | Aligned probabilities, validation per-class F1, and normalized entropy | Adaptive Reliability-Uncertainty Fusion | Project-specific adaptive combination |

## Intended use

Suggest one of six CFPB financial-product categories for an English complaint narrative.
The prototype can help demonstrate triage workflows and compare NLP methods. A human
remains responsible for interpreting or acting on a suggestion.

## Out-of-scope use

- Determining whether a complaint is truthful or important.
- Predicting legal, regulatory, financial, or company-response outcomes.
- Evaluating a consumer, company, or employee.
- Routing languages or product categories outside the documented label set.
- Automated production decisions without independent validation and monitoring.

## Training and selection

All methods use the same training, validation, and test records. Hyperparameters
are selected through five-fold stratified cross-validation on training data. Validation
macro-F1 selects the application default; size and latency break exact ties only. Final
base models are refitted with training plus validation records before test evaluation.

ARUF is fitted from base-model probabilities generated on validation records by models
trained only on the training split. It combines each probability with the member's
per-class validation F1, an entropy-derived confidence factor, and a bounded agreement
multiplier. The selected parameters are `alpha=1.0`, `beta=0.5`, and `gamma=0.2`.
Because the original test results were inspected before ARUF was proposed, its current
test score is exploratory rather than a fresh confirmatory result.

## Metrics

Generated metrics are stored in:

- `reports/validation_metrics.json`
- `reports/test_metrics.json`
- `reports/model_comparison.csv`
- `reports/confusion_matrix_<model>.png`

The primary metric is macro-F1. The sealed test set contains 2,700 complaints, with 450
examples from each class. Results from `reports/test_metrics.json` are:

| Model | Macro-F1 | Accuracy | Macro precision | Macro recall | Mean latency | Artifact size |
|---|---:|---:|---:|---:|---:|---:|
| `linear_svm` | 0.8458 | 0.8456 | 0.8474 | 0.8456 | 2.25 ms | 25.46 MiB |
| `adaptive_fusion` | 0.8363 | 0.8363 | 0.8418 | 0.8363 | 24.55 ms | 120.67 MiB |
| `minilm_logreg` | 0.8147 | 0.8148 | 0.8169 | 0.8148 | 18.23 ms | 87.37 MiB |
| `naive_bayes` | 0.7524 | 0.7581 | 0.7911 | 0.7581 | 0.50 ms | 7.84 MiB |

The calibrated Linear SVM is the registered default because it achieved the highest
validation macro-F1 and also produced the strongest test result. Its best
cross-validated setting was `C=0.5`. MiniLM Logistic Regression selected `C=2.0`, and
Naive Bayes selected `alpha=0.1`. ARUF improved upon two of its three members but did
not surpass Linear SVM, showing that adaptive combination does not guarantee an accuracy
gain when the strongest member already dominates the task.

The clearest recurring confusion for the default model is between checking/savings and
credit-card complaints, while mortgage is its strongest class (F1 0.9488). These results
meet the project target but do not establish fitness for automated production routing.

## Input and output

- **Input:** One English narrative containing 20 to 2,000 normalized characters.
- **Output:** Predicted product label, calibrated confidence, and top-three label
  probabilities.
- **Persistence:** User input is not logged or saved by the application.

## Limitations and risks

- Confidence represents model behavior, not correctness or certainty.
- Keywords may dominate sparse models and may not reflect the complaint's actual focus.
- MiniLM was pretrained on broader text and can carry unmeasured representation bias.
- ARUF inherits the errors and representation limitations of all three member models.
- ARUF uncertainty weighting uses model probability dispersion, not verified epistemic
  uncertainty or real-world correctness.
- Calibration and aggregate metrics can hide class-specific or distribution-shift errors.
- CFPB narratives are opt-in, scrubbed, unverified, and US-specific.
- Performance can degrade when wording or product categories differ from training data.

## Reproducibility

Each artifact includes its model version, labels, selected parameters, dataset checksum,
and preprocessing/model files. ARUF additionally records its member names, per-class
reliability matrix, validation evidence status, and selected adaptive parameters. The
registry identifies the default and all available methods. Runtime inference loads these
local artifacts and never retrains them.
