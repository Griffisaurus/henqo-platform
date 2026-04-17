"""
Tests for decision_pkg_service.domain.staleness (EP-09).

≥8 tests covering:
- Prediction 400 days old → stale
- Prediction 300 days old → not stale
- BenchmarkResult 100 days old → stale
- Custom reference date works
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from decision_pkg_service.domain.staleness import (
    check_staleness,
    is_benchmark_stale,
    is_prediction_stale,
    is_process_capability_stale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REF = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)


def _ts(days_ago: int, ref: datetime = _REF) -> str:
    """Return ISO timestamp for `days_ago` before reference."""
    return (ref - timedelta(days=days_ago)).isoformat()


# ---------------------------------------------------------------------------
# is_prediction_stale
# ---------------------------------------------------------------------------

class TestIsPredictionStale:
    def test_400_days_old_is_stale(self):
        assert is_prediction_stale(_ts(400), _REF) is True

    def test_300_days_old_is_not_stale(self):
        assert is_prediction_stale(_ts(300), _REF) is False

    def test_exactly_365_days_is_not_stale(self):
        # Boundary: exactly 365 days is NOT stale (> 365 required)
        assert is_prediction_stale(_ts(365), _REF) is False

    def test_366_days_is_stale(self):
        assert is_prediction_stale(_ts(366), _REF) is True

    def test_custom_reference_date(self):
        custom_ref = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ts = (custom_ref - timedelta(days=400)).isoformat()
        assert is_prediction_stale(ts, custom_ref) is True


# ---------------------------------------------------------------------------
# is_process_capability_stale
# ---------------------------------------------------------------------------

class TestIsProcessCapabilityStale:
    def test_400_days_old_is_stale(self):
        assert is_process_capability_stale(_ts(400), _REF) is True

    def test_300_days_old_is_not_stale(self):
        assert is_process_capability_stale(_ts(300), _REF) is False

    def test_custom_reference_date(self):
        custom_ref = datetime(2025, 6, 1, tzinfo=timezone.utc)
        ts = (custom_ref - timedelta(days=370)).isoformat()
        assert is_process_capability_stale(ts, custom_ref) is True


# ---------------------------------------------------------------------------
# is_benchmark_stale
# ---------------------------------------------------------------------------

class TestIsBenchmarkStale:
    def test_100_days_old_is_stale(self):
        assert is_benchmark_stale(_ts(100), _REF) is True

    def test_30_days_old_is_not_stale(self):
        assert is_benchmark_stale(_ts(30), _REF) is False

    def test_exactly_90_days_is_not_stale(self):
        assert is_benchmark_stale(_ts(90), _REF) is False

    def test_91_days_is_stale(self):
        assert is_benchmark_stale(_ts(91), _REF) is True


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------

class TestCheckStaleness:
    def test_stale_prediction_returned_in_result(self):
        predictions = [
            {"entity_id": "P-001", "created_at": _ts(400)},
            {"entity_id": "P-002", "created_at": _ts(200)},
        ]
        result = check_staleness(predictions, [], [], reference_date=_REF)
        assert "P-001" in result["stale_predictions"]
        assert "P-002" not in result["stale_predictions"]

    def test_stale_benchmark_returned_in_result(self):
        benchmarks = [
            {"entity_id": "B-001", "created_at": _ts(100)},
            {"entity_id": "B-002", "created_at": _ts(50)},
        ]
        result = check_staleness([], [], benchmarks, reference_date=_REF)
        assert "B-001" in result["stale_benchmarks"]
        assert "B-002" not in result["stale_benchmarks"]

    def test_stale_capability_returned_in_result(self):
        caps = [
            {"entity_id": "CAP-001", "created_at": _ts(400)},
        ]
        result = check_staleness([], caps, [], reference_date=_REF)
        assert "CAP-001" in result["stale_capabilities"]

    def test_empty_inputs_all_empty(self):
        result = check_staleness([], [], [], reference_date=_REF)
        assert result == {
            "stale_predictions": [],
            "stale_capabilities": [],
            "stale_benchmarks": [],
        }

    def test_custom_reference_date_respected(self):
        custom_ref = datetime(2025, 1, 1, tzinfo=timezone.utc)
        predictions = [
            {"entity_id": "P-CUSTOM", "created_at": (custom_ref - timedelta(days=400)).isoformat()},
        ]
        result = check_staleness(predictions, [], [], reference_date=custom_ref)
        assert "P-CUSTOM" in result["stale_predictions"]
