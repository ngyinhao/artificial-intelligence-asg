"""Audit cross-split narrative similarity without exporting complaint text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complaint_compass.config import (
    NEAR_DUPLICATE_MAX_FEATURES,
    NEAR_DUPLICATE_MIN_DOCUMENT_FREQUENCY,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset, usecols=["complaint_id", "text", "split"])
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=NEAR_DUPLICATE_MIN_DOCUMENT_FREQUENCY,
        max_features=NEAR_DUPLICATE_MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(frame["text"].fillna(""))

    comparisons = []
    for reference_split, query_split in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        reference_rows = frame.index[frame["split"] == reference_split].to_numpy()
        query_rows = frame.index[frame["split"] == query_split].to_numpy()
        nearest = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
        nearest.fit(matrix[reference_rows])
        distances, indices = nearest.kneighbors(matrix[query_rows])
        similarities = 1.0 - distances[:, 0]
        reference_matches = reference_rows[indices[:, 0]]

        order = similarities.argsort()[::-1][:10]
        comparisons.append(
            {
                "reference_split": reference_split,
                "query_split": query_split,
                "query_count": int(len(query_rows)),
                "pairs_at_or_above_threshold": int((similarities >= args.threshold).sum()),
                "pairs_at_or_above_0_95": int((similarities >= 0.95).sum()),
                "maximum_similarity": float(similarities.max()),
                "top_pairs": [
                    {
                        "query_complaint_id": str(frame.at[query_rows[i], "complaint_id"]),
                        "reference_complaint_id": str(frame.at[reference_matches[i], "complaint_id"]),
                        "cosine_similarity": float(similarities[i]),
                    }
                    for i in order
                ],
            }
        )

    output = {
        "method": "word TF-IDF cosine similarity (unigrams and bigrams)",
        "threshold": args.threshold,
        "dataset_rows": int(len(frame)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "comparisons": comparisons,
        "interpretation": (
            "A screening audit for highly similar cross-split narratives; it is not a semantic "
            "duplicate detector and should be repeated if preprocessing or splitting changes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
