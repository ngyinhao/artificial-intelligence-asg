# ComplaintCompass Glossary

| Term | Project definition |
|---|---|
| Artifact registry | JSON index identifying available saved models, their metadata, and the default application model. |
| Adaptive Reliability-Uncertainty Fusion (ARUF) | Project-specific algorithm that combines aligned base-model probabilities using category reliability, input uncertainty, and prediction agreement. |
| Calibration | Adjustment that makes classifier scores more suitable for probability-like confidence reporting. |
| Complaint narrative | Public, consumer-authored CFPB description used as model input after normalization. |
| Conflicting duplicate | Identical normalized text associated with more than one target product; every copy is removed. |
| Confusion matrix | Counts showing how actual product categories map to predicted categories. |
| Cross-validation | Repeated training/validation folds used only within the training split for hyperparameter selection. |
| Data leakage | Improper transfer of duplicate text or sealed evaluation information into model fitting or selection. |
| Default model | Registered method with the highest validation macro-F1; size and latency break exact ties only. |
| Embedding | Dense numeric representation of a narrative produced by MiniLM. |
| Normalized entropy | A 0-to-1 measure of how dispersed a model's class probabilities are; ARUF uses one minus this value as an input-specific confidence factor. |
| Per-class reliability | A base model's validation F1 for one product category, used by ARUF to vary influence across categories. |
| Macro-F1 | Unweighted mean of the six per-class F1 scores and the project's primary metric. |
| MiniLM | Compact pretrained transformer used to generate semantic sentence embeddings. |
| Multiclass classification | Prediction of exactly one product category from the six registered labels. |
| Narrative scrubbing | CFPB process intended to remove personal information before an opted-in narrative is published. |
| Sealed test set | Records not used for preprocessing decisions, tuning, calibration, or default-model selection. |
| TF-IDF | Sparse text representation that weights terms by frequency within a narrative and rarity across narratives. |
| Top-three categories | Three labels with the highest predicted probability for one narrative. |
| Weighted soft-voting ensemble | Combination that aligns class probabilities and applies one validation-selected positive weight to each base model; it is the current default. |
