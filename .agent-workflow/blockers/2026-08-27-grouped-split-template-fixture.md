# Grouped split test fixture formed one giant similarity component

- **Context:** Adding word TF-IDF cosine near-duplicate grouping before the train/validation/test split.
- **Intended action:** Run the existing dataset preparation unit tests after enforcing whole-group split assignment.
- **Symptom:** Three tests failed with `DataQualityError` because the generated narratives all belonged to one similarity group that could not fit within any exact per-class split capacity.
- **Impact:** The grouped splitter could not be validated with the previous highly repetitive fixture.
- **Cause:** The fixture varied only a complaint number and product name inside an otherwise identical sentence. The initial `min_df=2` setting also discarded every token unique to one synthetic document, exaggerating template similarity on small samples.
- **Troubleshooting:** Inspected the group IDs and failure trace. The initial 24 fixture rows collapsed into six connected groups spanning classes. Adding a short unique suffix was insufficient because the shared template still dominated TF-IDF cosine similarity; the successful fixture makes unique lexical content dominant.
- **Workaround:** Make ordinary fixture narratives lexically distinct, retain a separate deliberate near-duplicate pair, and use `min_df=1` so rare distinguishing terms are not silently discarded. The audit utility and preparation pipeline now share this setting.
- **Prevention:** Dataset-splitting fixtures should explicitly distinguish ordinary independent records from deliberate duplicate/template cases. Very small stratified fixtures cannot accommodate large similarity groups at exact ratios.
