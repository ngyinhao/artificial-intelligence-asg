from __future__ import annotations

import csv
import io
from typing import Any

import pandas as pd
import pytest

from complaint_compass.config import PRODUCT_LABELS
from complaint_compass.data import (
    DataQualityError,
    _discover_remote_size,
    _fetch_window,
    _parse_csv_range,
    _records_frame,
    _range_offsets,
    prepare_dataset_frame,
)


def source_frame(rows_per_label: int = 7) -> pd.DataFrame:
    rows = []
    complaint_id = 1
    for label in PRODUCT_LABELS:
        for index in range(rows_per_label):
            rows.append(
                {
                    "complaint_id": str(complaint_id),
                    "date_received": "2024-01-01",
                    "narrative": (
                        f"This is complaint number {index} about {label}; "
                        "the requested resolution did not occur."
                    ),
                    "product": label,
                }
            )
            complaint_id += 1
    return pd.DataFrame(rows)


def test_prepare_dataset_is_balanced_leak_free_and_reproducible() -> None:
    source = source_frame()
    first = prepare_dataset_frame(source, sample_per_class=4, seed=42)
    second = prepare_dataset_frame(source, sample_per_class=4, seed=42)

    assert len(first) == 24
    assert first.groupby("label").size().eq(4).all()
    assert first.groupby(["label", "split"]).size().unstack().to_dict() == {
        "test": {label: 1 for label in PRODUCT_LABELS},
        "train": {label: 2 for label in PRODUCT_LABELS},
        "validation": {label: 1 for label in PRODUCT_LABELS},
    }
    assert first["text_sha256"].is_unique
    assert first["complaint_id"].tolist() == second["complaint_id"].tolist()


def test_prepare_dataset_removes_conflicting_duplicate_text() -> None:
    source = source_frame()
    conflict_text = "The exact same sufficiently long complaint narrative appears twice."
    conflict_rows = pd.DataFrame(
        [
            {
                "complaint_id": "9001",
                "date_received": "2024-01-01",
                "narrative": conflict_text,
                "product": PRODUCT_LABELS[0],
            },
            {
                "complaint_id": "9002",
                "date_received": "2024-01-01",
                "narrative": conflict_text,
                "product": PRODUCT_LABELS[1],
            },
        ]
    )
    prepared = prepare_dataset_frame(
        pd.concat([source, conflict_rows], ignore_index=True),
        sample_per_class=4,
    )
    assert conflict_text not in set(prepared["text"])


def test_prepare_dataset_reports_insufficient_class_counts() -> None:
    with pytest.raises(DataQualityError, match="Insufficient cleaned narratives"):
        prepare_dataset_frame(source_frame(rows_per_label=2), sample_per_class=3)


class FakeResponse:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"hits": {"hits": self.hits}}


class FakeSession:
    def __init__(self) -> None:
        self.params: dict[str, Any] | None = None

    def get(
        self, url: str, *, params: dict[str, Any], timeout: tuple[int, int]
    ) -> FakeResponse:
        self.params = params
        return FakeResponse(
            [
                {
                    "_source": {
                        "complaint_id": 123,
                        "date_received": "2024-01-02",
                        "complaint_what_happened": "A sufficiently long narrative.",
                        "product": PRODUCT_LABELS[0],
                    }
                }
            ]
        )


def test_fetch_window_uses_official_filter_contract() -> None:
    session = FakeSession()
    rows = _fetch_window(
        session,  # type: ignore[arg-type]
        product=PRODUCT_LABELS[0],
        date_min="2024-01-01",
        date_max="2024-07-01",
        limit=1,
    )
    assert rows[0]["complaint_id"] == 123
    assert session.params == {
        "date_received_min": "2024-01-01",
        "date_received_max": "2024-07-01",
        "product": PRODUCT_LABELS[0],
        "has_narrative": "yes",
        "sort": "created_date_asc",
        "size": 1,
        "frm": 0,
        "no_aggs": "true",
        "no_highlight": "true",
    }


class RangeProbeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        ok: bool,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.ok = ok

    def __enter__(self) -> "RangeProbeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class HeadForbiddenSession:
    def head(self, *args: object, **kwargs: object) -> RangeProbeResponse:
        return RangeProbeResponse(status_code=403, headers={}, ok=False)

    def get(self, *args: object, **kwargs: object) -> RangeProbeResponse:
        return RangeProbeResponse(
            status_code=206,
            headers={"content-range": "bytes 0-0/8905555191"},
            ok=True,
        )


def test_discover_remote_size_falls_back_when_head_is_forbidden() -> None:
    assert _discover_remote_size(HeadForbiddenSession()) == 8_905_555_191  # type: ignore[arg-type]


def test_csv_range_parser_discards_boundary_fragments() -> None:
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerow(
        [
            "2024-01-02",
            "Mortgage",
            "",
            "",
            "",
            "A complete and sufficiently long complaint",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "12345",
        ]
    )
    row = buffer.getvalue()
    parsed = _parse_csv_range(f"partial row\n{row}truncated".encode(), start=100)
    assert len(parsed) == 1
    assert parsed[0]["Date received"] == "2024-01-02"
    assert parsed[0]["Product"] == "Mortgage"
    assert parsed[0]["Complaint ID"] == "12345"


def test_range_offsets_are_deterministic_and_non_overlapping() -> None:
    first = _range_offsets(1_000_000, chunk_bytes=10_000, max_chunks=20, seed=42)
    second = _range_offsets(1_000_000, chunk_bytes=10_000, max_chunks=20, seed=42)
    assert first == second
    assert first[0] == 0
    for index, value in enumerate(first):
        assert all(
            abs(value - other) >= 10_000
            for other in first[index + 1 :]
        )


def test_candidate_record_mapping_materializes_for_checkpoint() -> None:
    frame = _records_frame(
        {
            "123": {
                "complaint_id": "123",
                "date_received": "2024-01-02",
                "narrative": "A complete and sufficiently long complaint narrative.",
                "product": "Mortgage",
            }
        }
    )
    assert frame.columns.tolist() == [
        "complaint_id",
        "date_received",
        "narrative",
        "product",
    ]
    assert frame.loc[0, "complaint_id"] == "123"
