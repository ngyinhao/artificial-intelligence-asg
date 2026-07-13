# CSV range parser test fixture had the wrong field count

- **Date:** 2026-07-11
- **Context and intended action:** Verify that byte-range parsing discards partial boundary rows while accepting one complete CFPB CSV record.
- **Observable symptom:** The new test expected one parsed record but received zero; the overall suite reported one failure and 19 passes.
- **Impact:** Runtime verification paused until the test or parser discrepancy was resolved.
- **Confirmed cause:** The fixture constructed an 18-field CFPB row by manually counting commas and produced the wrong number of fields. The parser correctly rejected it under its schema-integrity check.
- **Successful workaround:** Generate the fixture with Python's `csv.writer` and an explicit 18-value list.
- **Prevention:** Use schema-aware writers for CSV fixtures rather than hand-authored comma runs, especially when blank columns are significant.
