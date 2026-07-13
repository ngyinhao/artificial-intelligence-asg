# Model Card: ComplaintCompass Routing Models

## Model family

ComplaintCompass registers three multiclass classifiers trained on the same six-label
dataset:

| Registry name | Representation | Classifier | Role |
|---|---|---|---|
| `naive_bayes` | Word unigram/bigram TF-IDF | Multinomial Naive Bayes | Interpretable baseline |
| `linear_svm` | Word unigram/bigram TF-IDF | Calibrated Linear SVM | Strong sparse-text method |
| `minilm_logreg` | Normalized MiniLM sentence embeddings | Logistic Regression | Semantic transformer representation |

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

All models use the same training, validation, and sealed-test records. Hyperparameters
are selected through five-fold stratified cross-validation on training data. Validation
macro-F1 selects the application default; near-ties favor the smaller and faster model.
Final models are refitted with training plus validation records before one sealed-test
evaluation.

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
| `linear_svm` | 0.8458 | 0.8456 | 0.8474 | 0.8456 | 3.22 ms | 25.46 MiB |
| `minilm_logreg` | 0.8147 | 0.8148 | 0.8169 | 0.8148 | 23.91 ms | 87.37 MiB |
| `naive_bayes` | 0.7524 | 0.7581 | 0.7911 | 0.7581 | 0.78 ms | 7.84 MiB |

The calibrated Linear SVM is the registered default because it achieved the highest
validation macro-F1 and also produced the strongest sealed-test result. Its best
cross-validated setting was `C=0.5`. MiniLM Logistic Regression selected `C=2.0`, and
Naive Bayes selected `alpha=0.1`.

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
