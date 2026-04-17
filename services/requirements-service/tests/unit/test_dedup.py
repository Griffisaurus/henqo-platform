"""
Unit tests for requirements_service.domain.dedup.

Covers remaining branches in _jaccard for full coverage.
"""
from __future__ import annotations

from requirements_service.domain.dedup import _jaccard, _tokenize, is_duplicate


class TestJaccard:
    def test_both_empty_returns_one(self) -> None:
        """Both empty token sets → similarity = 1.0 (both identical empty sets)."""
        assert _jaccard(frozenset(), frozenset()) == 1.0

    def test_disjoint_sets(self) -> None:
        a = frozenset(["alpha", "beta"])
        b = frozenset(["gamma", "delta"])
        assert _jaccard(a, b) == 0.0

    def test_identical_sets(self) -> None:
        s = frozenset(["foo", "bar"])
        assert _jaccard(s, s) == 1.0

    def test_partial_overlap(self) -> None:
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        # |intersection| = 2, |union| = 4
        assert abs(_jaccard(a, b) - 0.5) < 1e-9


class TestTokenize:
    def test_stopwords_removed(self) -> None:
        tokens = _tokenize("shall not exceed the limit")
        assert "shall" not in tokens
        assert "the" not in tokens
        assert "not" not in tokens
        assert "exceed" in tokens
        assert "limit" in tokens

    def test_lowercased(self) -> None:
        tokens = _tokenize("Temperature EXCEEDS Limit")
        assert "temperature" in tokens
        assert "exceeds" in tokens


class TestIsDuplicate:
    def test_empty_candidates_returns_false(self) -> None:
        dup, idx = is_duplicate("< 2 mm", [])
        assert dup is False
        assert idx is None

    def test_exact_match(self) -> None:
        dup, idx = is_duplicate("< 2 mm", ["< 2 mm", "at least 10 N"])
        assert dup is True
        assert idx == 0

    def test_second_candidate_matches(self) -> None:
        dup, idx = is_duplicate("at least 10 N", ["< 2 mm", "at least 10 N"])
        assert dup is True
        assert idx == 1

    def test_below_threshold_not_duplicate(self) -> None:
        dup, idx = is_duplicate("< 2 mm", ["at least 1000 kN during operation"])
        assert dup is False
        assert idx is None

    def test_custom_threshold(self) -> None:
        """Lower threshold catches partial overlaps."""
        dup, idx = is_duplicate("< 2 mm", ["< 2 mm nominal"], threshold=0.5)
        assert dup is True

    def test_all_stopwords_empty_sets(self) -> None:
        """Two texts that reduce to empty token sets → _jaccard returns 1.0 → duplicate."""
        # "shall" and "must" are both stopwords, so both tokenize to empty sets
        dup, idx = is_duplicate("shall", ["must"])
        assert dup is True
        assert idx == 0
