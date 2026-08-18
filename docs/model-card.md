# Model Card: ComplaintCompass Routing Models

## Model family

ComplaintCompass registers three multiclass base classifiers trained on the same
six-label dataset and one optional ensemble:

| Registry name | Representation | Classifier | Role |
|---|---|---|---|
| `naive_bayes` | Word unigram/bigram TF-IDF | Multinomial Naive Bayes | Interpretable baseline |
| `linear_svm` | Word unigram/bigram TF-IDF | Calibrated Linear SVM | Strong sparse-text method |
| `minilm_logreg` | Normalized MiniLM sentence embeddings | Logistic Regression | Semantic transformer representation |
| `weighted_ensemble` | Aligned probabilities from all three base models | Validation-weighted soft voting | Exploratory hybrid model |

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

All base models use the same training and validation records. Hyperparameters are
selected through five-fold stratified cross-validation on training data. Validation
probabilities select positive ensemble weights on a 0.05 grid using macro-F1, log loss,
distance from equal weighting, and deterministic tie-breaking. Final base models are
refitted with training plus validation records. The Linear SVM remains the application
default and the ensemble remains optional.

The original test results were already examined before the ensemble was proposed.
Consequently, the expanded four-model comparison is an exploratory benchmark rather
than a fresh sealed evaluation. A future untouched holdout would be required for a
strong confirmatory claim about the ensemble.

## Metrics

Generated metrics are stored in:

- `reports/validation_metrics.json`
- `reports/test_metrics.json`
- `reports/model_comparison.csv`
- `reports/confusion_matrix_<model>.png`

The primary metric is macro-F1. The exploratory test benchmark contains 2,700
complaints, with 450 examples from each class. Results from
`reports/test_metrics.json` are:

| Model | Macro-F1 | Accuracy | Macro precision | Macro recall | Mean latency | Artifact size |
|---|---:|---:|---:|---:|---:|---:|
| `weighted_ensemble` | 0.8513 | 0.8511 | 0.8532 | 0.8511 | 19.87 ms | 120.67 MiB |
| `linear_svm` | 0.8458 | 0.8456 | 0.8474 | 0.8456 | 1.83 ms | 25.46 MiB |
| `minilm_logreg` | 0.8147 | 0.8148 | 0.8169 | 0.8148 | 21.61 ms | 87.37 MiB |
| `naive_bayes` | 0.7524 | 0.7581 | 0.7911 | 0.7581 | 0.39 ms | 7.84 MiB |

The ensemble selected weights of 0.05 for Naive Bayes, 0.70 for Linear SVM, and 0.25
for MiniLM Logistic Regression. It reached the highest exploratory macro-F1, but this
small post-hoc difference is not confirmatory evidence. The calibrated Linear SVM
remains the registered default because it is faster, smaller, and was selected before
the ensemble experiment. Its best cross-validated setting was `C=0.5`; MiniLM Logistic
Regression selected `C=2.0`, and Naive Bayes selected `alpha=0.1`.

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
- Calibration and aggregate metrics can hide class-specific or distribution-shift errors.
- CFPB narratives are opt-in, scrubbed, unverified, and US-specific.
- Performance can degrade when wording or product categories differ from training data.

## Reproducibility

Each artifact includes its model version, labels, selected parameters, dataset checksum,
and preprocessing/model files. The registry identifies the default and all available
models. Runtime inference loads these local artifacts and never retrains them.
