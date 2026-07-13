# Bounded CFPB range pass was short on mortgage records and could not resume

- **Date:** 2026-07-13
- **Context and intended action:** Collect 4,250 raw candidates for each of six CFPB product classes from deterministic 8 MB ranges of the official CSV.
- **Observable symptom:** After all 120 configured ranges, five classes met the target but Mortgage had 3,614 candidates. The command raised `DataQualityError` and exited.
- **Impact:** Approximately 960 MB of bounded source ranges had been processed, but no candidate CSV was persisted because the implementation only wrote output after full success. A continuation would have repeated completed network work.
- **Cause:** Mortgage narratives were less frequent in the sampled ranges than the conservative buffer assumed, and the original downloader had no incremental checkpoint.
- **Initial correction:** Reduced the CSV-fallback candidate target to 3,500 per class, added checkpoint CSV/JSON files containing candidates, counts, and completed offsets every five ranges and at exit, and raised the safety limit to 180 ranges.
- **Follow-up evidence:** The 3,500-class pool downloaded successfully in 116 ranges, but normalization and duplicate/conflict removal left three classes below 3,000. The target was restored to 4,250 and checkpoint compatibility was broadened to allow safe target increases.
- **Remaining limitation:** The first failed pass predated checkpointing and cannot be recovered; subsequent runs are resumable.
- **Prevention:** Persist long-running acquisition state before enforcing final sufficiency checks, and size raw buffers according to observed class frequency and cleaning loss.
