# Pandas rejected dict_values during CFPB checkpoint serialization

- **Date:** 2026-07-13
- **Context and intended action:** Persist the first five completed CFPB byte ranges to the new resumable checkpoint.
- **Observable symptom:** `pandas.DataFrame.from_records(records.values(), ...)` raised `TypeError: 'dict_values' object is not subscriptable`.
- **Impact:** The restarted download stopped after five ranges before writing its first checkpoint. No existing dataset or checkpoint was corrupted.
- **Cause:** Pandas 2.3 expects a subscriptable sequence for this constructor path; a live dictionary values view is iterable but not subscriptable.
- **Successful correction:** Materialize the values view with `list(records.values())` for both checkpoint and final candidate-frame construction.
- **Prevention:** Convert mapping views to concrete sequences at Pandas dataframe-construction boundaries and cover checkpoint writes in tests.
