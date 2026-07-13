"""Download and prepare the CFPB complaint-routing dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    CFPB_API_URL,
    CFPB_CSV_URL,
    CSV_MAX_RANGE_CHUNKS,
    CSV_POOL_TARGET_PER_CLASS,
    CSV_RANGE_CHUNK_BYTES,
    DATASET_MANIFEST_PATH,
    DATE_WINDOWS,
    DOWNLOAD_MANIFEST_PATH,
    POOL_PER_WINDOW,
    PROJECT_ROOT,
    PROCESSED_DATA_PATH,
    PRODUCT_LABELS,
    RANDOM_SEED,
    RANGE_CHECKPOINT_MANIFEST_PATH,
    RANGE_CHECKPOINT_PATH,
    RAW_POOL_PATH,
    TARGET_PER_CLASS,
    ensure_runtime_directories,
)
from .text import normalize_text, text_fingerprint

LOGGER = logging.getLogger(__name__)
PAGE_SIZE = 100


class DataQualityError(RuntimeError):
    """Raised when source data cannot satisfy the frozen dataset contract."""


def _api_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_window(
    session: requests.Session,
    *,
    product: str,
    date_min: str,
    date_max: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch a stable, chronological pool for one product and date window."""

    records: list[dict[str, Any]] = []
    offset = 0
    while len(records) < limit:
        page_size = min(PAGE_SIZE, limit - len(records))
        params = {
            "date_received_min": date_min,
            "date_received_max": date_max,
            "product": product,
            "has_narrative": "yes",
            "sort": "created_date_asc",
            "size": page_size,
            "frm": offset,
            "no_aggs": "true",
            "no_highlight": "true",
        }
        response = session.get(CFPB_API_URL, params=params, timeout=(15, 90))
        response.raise_for_status()
        payload = response.json()
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            source = hit.get("_source", hit)
            records.append(
                {
                    "complaint_id": source.get("complaint_id"),
                    "date_received": source.get("date_received"),
                    "narrative": source.get("complaint_what_happened"),
                    "product": source.get("product"),
                }
            )
        if len(hits) < page_size:
            break
        offset += page_size
    return records[:limit]


def _range_offsets(
    total_size: int,
    *,
    chunk_bytes: int,
    max_chunks: int,
    seed: int,
) -> list[int]:
    """Create deterministic, non-overlapping offsets across a large remote CSV."""

    if total_size <= chunk_bytes:
        return [0]
    rng = np.random.default_rng(seed)
    offsets = [0]
    attempts = 0
    while len(offsets) < max_chunks and attempts < max_chunks * 100:
        attempts += 1
        candidate = int(rng.integers(1, total_size - chunk_bytes))
        if all(abs(candidate - existing) >= chunk_bytes for existing in offsets):
            offsets.append(candidate)
    return offsets


def _read_remote_range(
    session: requests.Session,
    *,
    start: int,
    chunk_bytes: int,
) -> bytes:
    end = start + chunk_bytes - 1
    with session.get(
        CFPB_CSV_URL,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=(15, 120),
        stream=True,
    ) as response:
        if response.status_code != 206:
            raise DataQualityError(
                "The official CFPB CSV server did not honor a bounded byte-range "
                f"request (status {response.status_code})."
            )
        return response.content


def _discover_remote_size(session: requests.Session) -> int:
    """Discover CSV size, using a one-byte GET when the edge forbids HEAD."""

    head = session.head(CFPB_CSV_URL, allow_redirects=True, timeout=(15, 60))
    if head.ok:
        total_size = int(head.headers.get("content-length", "0"))
        if total_size > 0 and "bytes" in head.headers.get(
            "accept-ranges", ""
        ).lower():
            return total_size

    with session.get(
        CFPB_CSV_URL,
        headers={"Range": "bytes=0-0"},
        timeout=(15, 60),
        stream=True,
    ) as probe:
        if probe.status_code != 206:
            raise DataQualityError(
                "The official CFPB CSV rejected both metadata discovery and a "
                f"one-byte range request (status {probe.status_code})."
            )
        content_range = probe.headers.get("content-range", "")
        match = re.search(r"/(\d+)$", content_range)
        if not match:
            raise DataQualityError(
                "The CFPB CSV range response did not report a total file size."
            )
        return int(match.group(1))


def _parse_csv_range(content: bytes, *, start: int) -> list[dict[str, str]]:
    """Parse complete CSV records from a byte range and discard boundary fragments."""

    if start > 0:
        first_newline = content.find(b"\n")
        if first_newline < 0:
            return []
        content = content[first_newline + 1 :]
    last_newline = content.rfind(b"\n")
    if last_newline < 0:
        return []
    content = content[: last_newline + 1]
    text = content.decode("utf-8", errors="replace")

    # Every range after the first lacks a header. The public CSV schema is stable;
    # malformed boundary rows are rejected by their field count.
    fieldnames = (
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Consumer complaint narrative",
        "Company public response",
        "Company",
        "State",
        "ZIP code",
        "Tags",
        "Consumer consent provided?",
        "Submitted via",
        "Date sent to company",
        "Company response to consumer",
        "Timely response?",
        "Consumer disputed?",
        "Complaint ID",
    )
    reader = csv.reader(io.StringIO(text, newline=""))
    rows: list[dict[str, str]] = []
    for values in reader:
        if values and values[0] == "Date received":
            continue
        if len(values) != len(fieldnames):
            continue
        rows.append(dict(zip(fieldnames, values, strict=True)))
    return rows


def _records_frame(records: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Materialize candidate records in the stable raw-pool column order."""

    return pd.DataFrame.from_records(
        list(records.values()),
        columns=("complaint_id", "date_received", "narrative", "product"),
    )


def _download_csv_range_pool(
    session: requests.Session,
    *,
    max_chunks: int,
    chunk_bytes: int = CSV_RANGE_CHUNK_BYTES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a bounded candidate pool from deterministic official CSV ranges."""

    total_size = _discover_remote_size(session)

    target_pool = CSV_POOL_TARGET_PER_CLASS
    records: dict[str, dict[str, Any]] = {}
    counts = {label: 0 for label in PRODUCT_LABELS}
    offsets = _range_offsets(
        total_size,
        chunk_bytes=chunk_bytes,
        max_chunks=max_chunks,
        seed=RANDOM_SEED,
    )
    used_offsets: list[int] = []

    if RANGE_CHECKPOINT_PATH.exists() and RANGE_CHECKPOINT_MANIFEST_PATH.exists():
        checkpoint_meta = json.loads(
            RANGE_CHECKPOINT_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        compatible = (
            checkpoint_meta.get("remote_size_bytes") == total_size
            and checkpoint_meta.get("range_chunk_bytes") == chunk_bytes
            and checkpoint_meta.get("seed") == RANDOM_SEED
        )
        if compatible:
            checkpoint = pd.read_csv(
                RANGE_CHECKPOINT_PATH, dtype={"complaint_id": str}
            )
            records = {
                str(row["complaint_id"]): row
                for row in checkpoint.to_dict(orient="records")
            }
            counts = {
                label: sum(
                    1 for row in records.values() if row["product"] == label
                )
                for label in PRODUCT_LABELS
            }
            used_offsets = [int(value) for value in checkpoint_meta["range_offsets"]]
            LOGGER.info(
                "Resuming from %s completed CFPB CSV ranges", len(used_offsets)
            )

    def save_checkpoint() -> None:
        checkpoint = _records_frame(records)
        RANGE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.to_csv(RANGE_CHECKPOINT_PATH, index=False, encoding="utf-8")
        RANGE_CHECKPOINT_MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "remote_size_bytes": total_size,
                    "range_chunk_bytes": chunk_bytes,
                    "seed": RANDOM_SEED,
                    "target_pool_per_class": target_pool,
                    "range_offsets": used_offsets,
                    "candidate_counts": counts,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    for index, offset in enumerate(offsets, start=1):
        if offset in used_offsets:
            continue
        LOGGER.info("Reading CFPB CSV range %s/%s", index, len(offsets))
        content = _read_remote_range(
            session, start=offset, chunk_bytes=chunk_bytes
        )
        used_offsets.append(offset)
        for source in _parse_csv_range(content, start=offset):
            product = source["Product"]
            date_received = source["Date received"]
            narrative = source["Consumer complaint narrative"]
            complaint_id = source["Complaint ID"]
            if (
                product not in counts
                or counts[product] >= target_pool
                or not narrative
                or not complaint_id
                or not ("2023-08-01" <= date_received < "2026-01-01")
                or complaint_id in records
            ):
                continue
            records[complaint_id] = {
                "complaint_id": complaint_id,
                "date_received": date_received,
                "narrative": narrative,
                "product": product,
            }
            counts[product] += 1
        if len(used_offsets) % 5 == 0:
            save_checkpoint()
        if all(count >= target_pool for count in counts.values()):
            break

    save_checkpoint()

    insufficient = {
        label: count for label, count in counts.items() if count < target_pool
    }
    if insufficient:
        raise DataQualityError(
            "The bounded official CSV sample did not collect enough candidates: "
            f"{insufficient}. Increase --max-range-chunks and rerun."
        )
    frame = _records_frame(records)
    metadata = {
        "source_mode": "official_csv_byte_ranges",
        "remote_size_bytes": total_size,
        "range_chunk_bytes": chunk_bytes,
        "range_offsets": used_offsets,
        "candidate_counts": counts,
        "target_pool_per_class": target_pool,
        "checkpoint_manifest": str(RANGE_CHECKPOINT_MANIFEST_PATH),
    }
    return frame, metadata


def download_dataset_pool(
    output_path: Path = RAW_POOL_PATH,
    *,
    pool_per_window: int = POOL_PER_WINDOW,
    session: requests.Session | None = None,
    source_mode: str = "auto",
    max_range_chunks: int = CSV_MAX_RANGE_CHUNKS,
) -> pd.DataFrame:
    """Download a temporally distributed CFPB candidate pool."""

    if pool_per_window < 1:
        raise ValueError("pool_per_window must be positive")
    ensure_runtime_directories()
    client = session or _api_session()
    if source_mode not in {"auto", "api", "csv-range"}:
        raise ValueError("source_mode must be auto, api, or csv-range")
    rows: list[dict[str, Any]] = []
    source_metadata: dict[str, Any] = {}
    if source_mode in {"auto", "api"}:
        try:
            window_counts: dict[str, int] = {}
            for product in PRODUCT_LABELS:
                for date_min, date_max in DATE_WINDOWS:
                    LOGGER.info(
                        "Fetching %s from %s to %s", product, date_min, date_max
                    )
                    fetched = _fetch_window(
                        client,
                        product=product,
                        date_min=date_min,
                        date_max=date_max,
                        limit=pool_per_window,
                    )
                    rows.extend(fetched)
                    window_counts[f"{product}|{date_min}|{date_max}"] = len(
                        fetched
                    )
            source_metadata = {
                "source_mode": "api",
                "window_counts": window_counts,
            }
        except requests.HTTPError as exc:
            if source_mode == "api" or exc.response is None or exc.response.status_code != 403:
                raise
            LOGGER.warning(
                "CFPB API access was forbidden; using official CSV byte ranges."
            )
            rows = []

    if source_mode == "csv-range" or not rows:
        frame, source_metadata = _download_csv_range_pool(
            client, max_chunks=max_range_chunks
        )
    else:
        frame = pd.DataFrame.from_records(
            rows,
            columns=("complaint_id", "date_received", "narrative", "product"),
        )
    if frame.empty:
        raise DataQualityError("The CFPB API returned no complaint narratives.")
    frame = frame.drop_duplicates(subset=["complaint_id"]).copy()
    frame["complaint_id"] = frame["complaint_id"].astype(str)
    frame = frame.sort_values(["product", "date_received", "complaint_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")

    manifest = {
        "source": CFPB_API_URL,
        "fallback_source": CFPB_CSV_URL,
        "license": "CC0 (United States Government work)",
        "date_windows": [list(window) for window in DATE_WINDOWS],
        "products": list(PRODUCT_LABELS),
        "pool_per_window": pool_per_window,
        "rows": int(len(frame)),
        "raw_sha256": file_sha256(output_path),
        **source_metadata,
    }
    DOWNLOAD_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return frame


def _split_counts(size: int) -> tuple[int, int, int]:
    if size < 3:
        raise DataQualityError("Each class needs at least three records for splitting.")
    train = max(1, int(size * 0.70))
    validation = max(1, int(size * 0.15))
    test = size - train - validation
    if test < 1:
        train -= 1 - test
        test = 1
    return train, validation, test


def prepare_dataset_frame(
    source: pd.DataFrame,
    *,
    sample_per_class: int = TARGET_PER_CLASS,
    seed: int = RANDOM_SEED,
    product_labels: Iterable[str] = PRODUCT_LABELS,
) -> pd.DataFrame:
    """Apply the frozen cleaning, balancing, and split contract."""

    required = {"complaint_id", "date_received", "narrative", "product"}
    missing = required.difference(source.columns)
    if missing:
        raise DataQualityError(f"Source data is missing columns: {sorted(missing)}")

    labels = tuple(product_labels)
    frame = source.loc[source["product"].isin(labels), list(required)].copy()
    frame["complaint_id"] = frame["complaint_id"].astype(str)
    frame["text"] = frame["narrative"].map(normalize_text)
    frame = frame.loc[frame["text"].str.len() >= 20].copy()
    frame["text_sha256"] = frame["text"].map(text_fingerprint)

    # Remove all copies of narratives carrying conflicting target labels.
    label_counts = frame.groupby("text_sha256")["product"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    if conflicting:
        frame = frame.loc[~frame["text_sha256"].isin(conflicting)].copy()
    frame = frame.drop_duplicates(subset=["text_sha256"], keep="first")

    available = frame.groupby("product").size().to_dict()
    insufficient = {
        label: int(available.get(label, 0))
        for label in labels
        if available.get(label, 0) < sample_per_class
    }
    if insufficient:
        raise DataQualityError(
            "Insufficient cleaned narratives for the frozen class sample: "
            f"{insufficient}. Increase the download pool and rerun."
        )

    sampled_parts: list[pd.DataFrame] = []
    for label in labels:
        group = frame.loc[frame["product"] == label].sort_values("complaint_id")
        sampled_parts.append(group.sample(n=sample_per_class, random_state=seed))
    sampled = pd.concat(sampled_parts, ignore_index=True)

    rng = np.random.default_rng(seed)
    split_parts: list[pd.DataFrame] = []
    for label in labels:
        group = sampled.loc[sampled["product"] == label].copy()
        group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        train_size, validation_size, _ = _split_counts(len(group))
        group["split"] = "test"
        group.loc[: train_size - 1, "split"] = "train"
        group.loc[
            train_size : train_size + validation_size - 1, "split"
        ] = "validation"
        split_parts.append(group)

    prepared = pd.concat(split_parts, ignore_index=True)
    prepared = prepared.rename(columns={"product": "label"})
    prepared = prepared[
        [
            "complaint_id",
            "date_received",
            "text",
            "text_sha256",
            "label",
            "split",
        ]
    ].sort_values(["split", "label", "complaint_id"])
    return prepared.reset_index(drop=True)


def prepare_dataset(
    input_path: Path = RAW_POOL_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    *,
    sample_per_class: int = TARGET_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    ensure_runtime_directories()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw CFPB pool not found at {input_path}. Run the download command first."
        )
    source = pd.read_csv(input_path, dtype={"complaint_id": str})
    prepared = prepare_dataset_frame(
        source, sample_per_class=sample_per_class, seed=seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, encoding="utf-8")

    manifest = {
        "source_manifest": str(DOWNLOAD_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "processed_rows": int(len(prepared)),
        "sample_per_class": sample_per_class,
        "seed": seed,
        "labels": list(PRODUCT_LABELS),
        "split_counts": {
            str(key): int(value)
            for key, value in prepared.groupby("split").size().items()
        },
        "class_counts": {
            str(key): int(value)
            for key, value in prepared.groupby("label").size().items()
        },
        "processed_sha256": file_sha256(output_path),
    }
    DATASET_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return prepared


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download a CFPB candidate pool")
    download.add_argument("--output", type=Path, default=RAW_POOL_PATH)
    download.add_argument(
        "--pool-per-window", type=int, default=POOL_PER_WINDOW
    )
    download.add_argument(
        "--source",
        choices=("auto", "api", "csv-range"),
        default="auto",
        help="Try the API first or use deterministic official CSV byte ranges",
    )
    download.add_argument(
        "--max-range-chunks", type=int, default=CSV_MAX_RANGE_CHUNKS
    )

    prepare = subparsers.add_parser("prepare", help="Prepare the balanced dataset")
    prepare.add_argument("--input", type=Path, default=RAW_POOL_PATH)
    prepare.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH)
    prepare.add_argument("--sample-per-class", type=int, default=TARGET_PER_CLASS)
    prepare.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    if args.command == "download":
        frame = download_dataset_pool(
            args.output,
            pool_per_window=args.pool_per_window,
            source_mode=args.source,
            max_range_chunks=args.max_range_chunks,
        )
        print(f"Downloaded {len(frame):,} unique complaint records to {args.output}")
    elif args.command == "prepare":
        frame = prepare_dataset(
            args.input,
            args.output,
            sample_per_class=args.sample_per_class,
            seed=args.seed,
        )
        print(f"Prepared {len(frame):,} records at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
