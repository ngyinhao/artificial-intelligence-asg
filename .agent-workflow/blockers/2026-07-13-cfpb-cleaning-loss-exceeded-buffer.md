# CFPB cleaning loss exceeded the reduced 500-record buffer

- **Date:** 2026-07-13
- **Context and intended action:** Clean and balance the successfully downloaded 3,500-candidate-per-class CFPB pool to 3,000 unique narratives per class.
- **Observable symptom:** Preparation stopped with cleaned counts of 2,702 for credit reporting, 2,850 for debt collection, and 2,573 for money transfer; the other three classes met the target.
- **Impact:** The frozen 18,000-record processed dataset could not yet be created.
- **Confirmed cause:** Exact normalized duplicates and conflicting-label duplicates removed more than the 500-record fallback buffer for three classes, especially money transfer.
- **Successful correction:** Restore a 4,250-candidate target per class. Allow a compatible checkpoint created under a smaller target to resume when the target increases, retaining candidates and completed byte-range offsets rather than re-downloading them.
- **Prevention:** Size acquisition buffers using observed post-normalization retention, and treat a target increase as checkpoint-compatible when source size, chunk size, and seed are unchanged.
