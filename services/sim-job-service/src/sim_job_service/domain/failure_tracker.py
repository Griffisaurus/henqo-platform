"""
Failure tracker for simulation jobs.

Tracks how many times a job (identified by idempotency_key) has failed,
enabling double-failure detection.
"""
from __future__ import annotations


class FailureTracker:
    """Tracks failure counts per job idempotency_key to detect double failures."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def record_failure(self, idempotency_key: str) -> int:
        """Increment failure count and return the new count."""
        current = self._counts.get(idempotency_key, 0)
        new_count = current + 1
        self._counts[idempotency_key] = new_count
        return new_count

    def get_count(self, idempotency_key: str) -> int:
        """Return current failure count for the given key (0 if never failed)."""
        return self._counts.get(idempotency_key, 0)

    def is_double_failure(self, idempotency_key: str) -> bool:
        """Return True if failure count is >= 2."""
        return self._counts.get(idempotency_key, 0) >= 2

    def reset(self, idempotency_key: str) -> None:
        """Clear the failure count for the given key."""
        self._counts.pop(idempotency_key, None)
