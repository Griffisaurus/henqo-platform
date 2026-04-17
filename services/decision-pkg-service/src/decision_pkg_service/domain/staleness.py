"""
EP-09: Staleness checks for predictions, process capabilities, and benchmarks.

Thresholds:
  - Prediction:          stale if age > 365 days
  - ProcessCapability:   stale if age > 365 days (12 months)
  - BenchmarkResult:     stale if age > 90 days
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp. Assumes UTC if no tzinfo."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _reference_now(reference_date: datetime | None) -> datetime:
    if reference_date is None:
        return datetime.now(tz=timezone.utc)
    if reference_date.tzinfo is None:
        return reference_date.replace(tzinfo=timezone.utc)
    return reference_date


def is_prediction_stale(
    created_at: str,
    reference_date: datetime | None = None,
) -> bool:
    """Stale if age > 365 days."""
    age = _reference_now(reference_date) - _parse_iso(created_at)
    return age > timedelta(days=365)


def is_process_capability_stale(
    created_at: str,
    reference_date: datetime | None = None,
) -> bool:
    """Stale if age > 12 months (365 days)."""
    age = _reference_now(reference_date) - _parse_iso(created_at)
    return age > timedelta(days=365)


def is_benchmark_stale(
    created_at: str,
    reference_date: datetime | None = None,
) -> bool:
    """Stale if age > 90 days."""
    age = _reference_now(reference_date) - _parse_iso(created_at)
    return age > timedelta(days=90)


def check_staleness(
    predictions: list[dict],
    process_capabilities: list[dict],
    benchmark_results: list[dict],
    reference_date: datetime | None = None,
) -> dict[str, list[str]]:
    """
    Return dict of stale entity IDs grouped by type.

    Returns:
        {
            "stale_predictions": [...ids],
            "stale_capabilities": [...ids],
            "stale_benchmarks": [...ids],
        }
    """
    ref = _reference_now(reference_date)

    stale_predictions: list[str] = []
    for p in predictions:
        entity_id = p.get("entity_id", p.get("prediction_id", ""))
        created_at = p.get("created_at", "")
        if created_at and is_prediction_stale(created_at, ref):
            stale_predictions.append(entity_id)

    stale_capabilities: list[str] = []
    for pc in process_capabilities:
        entity_id = pc.get("entity_id", pc.get("capability_id", ""))
        created_at = pc.get("created_at", "")
        if created_at and is_process_capability_stale(created_at, ref):
            stale_capabilities.append(entity_id)

    stale_benchmarks: list[str] = []
    for br in benchmark_results:
        entity_id = br.get("entity_id", br.get("benchmark_id", ""))
        created_at = br.get("created_at", "")
        if created_at and is_benchmark_stale(created_at, ref):
            stale_benchmarks.append(entity_id)

    return {
        "stale_predictions": stale_predictions,
        "stale_capabilities": stale_capabilities,
        "stale_benchmarks": stale_benchmarks,
    }
