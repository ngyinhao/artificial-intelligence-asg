# Multiclass log-loss helper assumed lexicographic class order

- **Context / intended action:** Select Adaptive Reliability-Uncertainty Fusion parameters using validation macro-F1 with multiclass log loss as a tie-breaker.
- **Symptom:** Integration tests passed but emitted repeated warnings that the supplied domain label order was not lexicographic and that probability columns must follow the helper's assumed sorted order.
- **Impact:** The log-loss tie-breaker could associate probability columns with the wrong target labels and therefore select an incorrect fusion configuration.
- **Cause:** `sklearn.metrics.log_loss` sorts the explicit label list, while ComplaintCompass uses a stable domain-specific class order shared by artifacts and probability matrices.
- **Troubleshooting:** The warning appeared only when the six real product labels were exercised; two-label tests happened to use lexicographic order and did not expose it.
- **Workaround:** Map every validation target directly to its index in the declared class order and compute negative mean log probability from those exact columns.
- **Prevention:** For probability metrics, test with a deliberately non-lexicographic label order and keep class-to-column alignment explicit throughout the pipeline.
