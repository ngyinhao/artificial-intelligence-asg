# Checkpoint patch context did not match current configuration layout

- **Date:** 2026-07-13
- **Context and intended action:** Add resumable CFPB CSV range checkpoints and adjust the fallback candidate buffer.
- **Observable symptom:** `apply_patch` verification failed because the expected `TARGET_PER_CLASS = 3_000` context was not located at the position implied by the combined patch.
- **Impact:** No files were modified; the downloader remained non-resumable until the patch could be reapplied.
- **Cause:** The large multi-file patch assumed a constant ordering that differed from the current `config.py` layout.
- **Workaround:** Inspect the exact surrounding lines and apply smaller, accurately anchored patches.
- **Prevention:** For multi-file patches that touch frequently edited configuration blocks, read the current snippets immediately before patching and use narrower context anchors.
