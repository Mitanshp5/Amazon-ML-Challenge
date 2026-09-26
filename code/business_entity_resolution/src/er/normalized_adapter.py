"""Normalized record component adapter (Step 1: D1-00).

Bridges raw record rows (from TSVs or pool dicts) to the shared normalizer
and feature extraction schema, ensuring:
- Real name_core stripping legal suffixes (not just naive split()[0]).
- Full house, unit, and postal extraction.
- Preserved Unicode primary representation and Latin-folded views.
- Thread-safe caching for fast O(1) feature extraction across workers.
"""
from __future__ import annotations

from typing import Any, Mapping

from er.normalization import normalize_record


class NormalizedRecordAdapter:
    """Thread-safe cached adapter for normalizing business records."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def normalize(self, record_id: str, name: str, address: str, country: str) -> dict[str, Any]:
        if record_id in self._cache:
            return self._cache[record_id]
        rec = normalize_record(name_raw=name, address_raw=address, country_raw=country)
        rec["record_id"] = record_id
        self._cache[record_id] = rec
        return rec

    def from_row_dict(self, row: Mapping[str, Any], country: str) -> dict[str, Any]:
        rec_id = str(row.get("entity_id", row.get("target_id", "")))
        if rec_id and rec_id in self._cache:
            return self._cache[rec_id]
        name = str(row.get("business_name", ""))
        addr = str(row.get("business_address", ""))
        rec = normalize_record(name_raw=name, address_raw=addr, country_raw=country)
        rec["record_id"] = rec_id
        if rec_id:
            self._cache[rec_id] = rec
        return rec

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


_GLOBAL_ADAPTER = NormalizedRecordAdapter()


def get_normalized_record(record_id: str, name: str, address: str, country: str) -> dict[str, Any]:
    """Convenience functional interface using global shared cache."""
    return _GLOBAL_ADAPTER.normalize(record_id, name, address, country)


def record_from_dict(row: Mapping[str, Any], country: str) -> dict[str, Any]:
    """Extract and normalize a record from a dictionary row."""
    return _GLOBAL_ADAPTER.from_row_dict(row, country)
