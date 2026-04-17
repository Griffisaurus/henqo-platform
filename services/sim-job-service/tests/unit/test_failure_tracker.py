"""
Unit tests for domain/failure_tracker.py.
"""
from __future__ import annotations

import pytest

from sim_job_service.domain.failure_tracker import FailureTracker


class TestFailureTracker:
    def test_initial_count_is_zero(self):
        tracker = FailureTracker()
        assert tracker.get_count("job-abc") == 0

    def test_first_failure_returns_count_one(self):
        tracker = FailureTracker()
        count = tracker.record_failure("job-abc")
        assert count == 1

    def test_first_failure_not_double_failure(self):
        tracker = FailureTracker()
        tracker.record_failure("job-abc")
        assert not tracker.is_double_failure("job-abc")

    def test_second_failure_returns_count_two(self):
        tracker = FailureTracker()
        tracker.record_failure("job-abc")
        count = tracker.record_failure("job-abc")
        assert count == 2

    def test_second_failure_is_double_failure(self):
        tracker = FailureTracker()
        tracker.record_failure("job-abc")
        tracker.record_failure("job-abc")
        assert tracker.is_double_failure("job-abc")

    def test_third_failure_also_is_double_failure(self):
        tracker = FailureTracker()
        tracker.record_failure("job-abc")
        tracker.record_failure("job-abc")
        tracker.record_failure("job-abc")
        assert tracker.is_double_failure("job-abc")
        assert tracker.get_count("job-abc") == 3

    def test_reset_clears_count(self):
        tracker = FailureTracker()
        tracker.record_failure("job-abc")
        tracker.record_failure("job-abc")
        tracker.reset("job-abc")
        assert tracker.get_count("job-abc") == 0
        assert not tracker.is_double_failure("job-abc")

    def test_reset_nonexistent_key_is_noop(self):
        tracker = FailureTracker()
        tracker.reset("unknown-key")   # should not raise
        assert tracker.get_count("unknown-key") == 0

    def test_multiple_keys_tracked_independently(self):
        tracker = FailureTracker()
        tracker.record_failure("job-1")
        tracker.record_failure("job-1")
        tracker.record_failure("job-2")

        assert tracker.is_double_failure("job-1")
        assert not tracker.is_double_failure("job-2")
        assert tracker.get_count("job-1") == 2
        assert tracker.get_count("job-2") == 1

    def test_reset_one_key_does_not_affect_other(self):
        tracker = FailureTracker()
        tracker.record_failure("job-1")
        tracker.record_failure("job-2")
        tracker.reset("job-1")

        assert tracker.get_count("job-1") == 0
        assert tracker.get_count("job-2") == 1
