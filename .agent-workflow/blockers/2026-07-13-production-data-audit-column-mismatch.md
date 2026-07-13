# Production data audit used the wrong fingerprint column name

- **Context and intended action:** Independently verify that no normalized narrative appears in more than one production dataset split.
- **Observable symptom:** A read-only Pandas audit raised `KeyError: 'text_fingerprint'`.
- **Impact:** The ad hoc verification command stopped before reporting split overlap; data preparation and model artifacts were unaffected.
- **Cause:** The processed CSV schema names the normalized-text digest `text_sha256`, while the audit command assumed the internal concept was exported as `text_fingerprint`.
- **Troubleshooting:** Reading only the CSV header confirmed the columns are `complaint_id,date_received,text,text_sha256,label,split`.
- **Workaround:** Group production records by `text_sha256` when checking whether any digest maps to multiple splits.
- **Prevention:** Read `dataset_manifest.json` and the processed CSV header before writing repository-external audit snippets, or add a documented processed-schema section to the dataset card if more manual audits are expected.
