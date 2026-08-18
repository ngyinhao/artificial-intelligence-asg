# MiniLM retraining attempted a blocked Hugging Face request

## Context and intended action

The real `python -m complaint_compass.train --all` workflow was run to generate
validation-selected ensemble weights and refreshed deployable artifacts.

## Observable symptom

After Naive Bayes and Linear SVM completed, `SentenceTransformer` attempted a HEAD
request for MiniLM adapter metadata on Hugging Face. Windows returned socket error
10013 in the managed sandbox, followed by a closed-client runtime error.

## Impact

The full training command stopped before MiniLM validation predictions and ensemble
metadata could be generated. The first two base artifacts and partial validation report
were refreshed successfully.

## Confirmed cause

Training initialized MiniLM from its remote model identifier even though a complete,
previously downloaded encoder already existed under the repository's artifact tree.
Recent library behavior performs an online metadata check for that identifier.

## Workaround

Prefer the existing local MiniLM encoder directory during retraining and fall back to
the remote identifier only when no local encoder artifact exists. This removes the
unnecessary request while preserving the same encoder weights.

## Prevention

- Load committed/local transformer artifacts before consulting remote identifiers.
- Keep the remote model ID in metadata for provenance.
- Retain an offline integration test through the fake sentence-transformer module.
