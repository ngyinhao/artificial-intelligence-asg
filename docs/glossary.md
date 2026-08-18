# ComplaintCompass Glossary

| Term | Project definition |
|---|---|
| Artifact registry | JSON index identifying available saved models, their metadata, and the default application model. |
| Calibration | Adjustment that makes classifier scores more suitable for probability-like confidence reporting. |
| Complaint narrative | Public, consumer-authored CFPB description used as model input after normalization. |
| Conflicting duplicate | Identical normalized text associated with more than one target product; every copy is removed. |
| Confusion matrix | Counts showing how actual product categories map to predicted categories. |
| Cross-validation | Repeated training/validation folds used only within the training split for hyperparameter selection. |
| Data leakage | Improper transfer of duplicate text or sealed evaluation information into model fitting or selection. |
| Default model | The calibrated Linear SVM retained as the application default while the post-hoc ensemble remains optional. |
| Embedding | Dense numeric representation of a narrative produced by MiniLM. |
| Macro-F1 | Unweighted mean of the six per-class F1 scores and the project's primary metric. |
| MiniLM | Compact pretrained transformer used to generate semantic sentence embeddings. |
| Multiclass classification | Prediction of exactly one product category from the six registered labels. |
| Narrative scrubbing | CFPB process intended to remove personal information before an opted-in narrative is published. |
| Exploratory test benchmark | The original test records reused to compare the post-hoc ensemble; useful for exploration but no longer described as untouched for the new experiment. |
| Soft voting | Combination of aligned class probabilities from multiple models using weights that sum to one. |
| Weighted ensemble | The optional model that combines Naive Bayes, Linear SVM, and MiniLM Logistic Regression probabilities using validation-selected positive weights. |
| TF-IDF | Sparse text representation that weights terms by frequency within a narrative and rarity across narratives. |
| Top-three categories | Three labels with the highest predicted probability for one narrative. |
