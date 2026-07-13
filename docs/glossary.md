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
| Default model | Registered model selected by validation macro-F1, with size and latency used for near-ties. |
| Embedding | Dense numeric representation of a narrative produced by MiniLM. |
| Macro-F1 | Unweighted mean of the six per-class F1 scores and the project's primary metric. |
| MiniLM | Compact pretrained transformer used to generate semantic sentence embeddings. |
| Multiclass classification | Prediction of exactly one product category from the six registered labels. |
| Narrative scrubbing | CFPB process intended to remove personal information before an opted-in narrative is published. |
| Sealed test set | Records not used for preprocessing decisions, tuning, calibration, or default-model selection. |
| TF-IDF | Sparse text representation that weights terms by frequency within a narrative and rarity across narratives. |
| Top-three categories | Three labels with the highest predicted probability for one narrative. |
