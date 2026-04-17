"""
Unit tests for surrogate_service.domain.applicability.

Covers:
  - compute_a_density: in-distribution, out-of-distribution, empty training set
  - compute_a_range: all in bounds, some out, empty bounds
  - compute_a_ensemble: tight spread, wide spread, single member
  - compute_applicability_score: combined formula
"""
from __future__ import annotations

import math

import pytest

from surrogate_service.domain.applicability import (
    compute_a_density,
    compute_a_ensemble,
    compute_a_range,
    compute_applicability_score,
)


# ---------------------------------------------------------------------------
# A_density
# ---------------------------------------------------------------------------

class TestComputeADensity:
    def _make_grid(self) -> list[dict[str, float]]:
        """3x3 grid of training points in [0,1]x[0,1]."""
        return [
            {"x1": i / 2.0, "x2": j / 2.0}
            for i in range(3)
            for j in range(3)
        ]

    def test_in_distribution_point(self):
        """A point at the center of the training cloud should have high density."""
        training = self._make_grid()
        x = {"x1": 0.5, "x2": 0.5}
        score = compute_a_density(x, training)
        assert 0.0 <= score <= 1.0
        assert score > 0.5, f"Expected high density for center point, got {score}"

    def test_out_of_distribution_point(self):
        """A point far from all training data should have low density."""
        training = self._make_grid()
        x = {"x1": 100.0, "x2": 100.0}
        score = compute_a_density(x, training)
        assert score == 0.0, f"Expected 0.0 for far-OOD point, got {score}"

    def test_empty_training_data(self):
        """Empty training set returns 0.0."""
        score = compute_a_density({"x1": 0.5}, [])
        assert score == 0.0

    def test_single_training_point(self):
        """Single training point — pairwise distance is 0, radius=0, only exact matches count."""
        training = [{"x1": 0.5, "x2": 0.5}]
        x_match = {"x1": 0.5, "x2": 0.5}
        x_other = {"x1": 1.0, "x2": 1.0}
        assert compute_a_density(x_match, training) == 1.0
        assert compute_a_density(x_other, training) == 0.0

    def test_no_shared_keys(self):
        """No shared keys between x and training → 0.0."""
        training = [{"a": 1.0}, {"a": 2.0}]
        x = {"b": 1.0}
        assert compute_a_density(x, training) == 0.0

    def test_score_bounded(self):
        """Score is always in [0, 1]."""
        training = [{"x": float(i)} for i in range(10)]
        for xval in [-10.0, 0.5, 5.0, 100.0]:
            s = compute_a_density({"x": xval}, training)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# A_range
# ---------------------------------------------------------------------------

class TestComputeARange:
    def test_all_in_bounds(self):
        x = {"x1": 0.5, "x2": 0.5}
        bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
        assert compute_a_range(x, bounds) == 1.0

    def test_some_out_of_bounds(self):
        x = {"x1": 0.5, "x2": 2.0}
        bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
        score = compute_a_range(x, bounds)
        assert math.isclose(score, 0.5, abs_tol=1e-9)

    def test_all_out_of_bounds(self):
        x = {"x1": -1.0, "x2": 2.0}
        bounds = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
        assert compute_a_range(x, bounds) == 0.0

    def test_empty_bounds_returns_one(self):
        """No bounds defined → no features are out-of-range → 1.0."""
        assert compute_a_range({"x1": 99.0}, {}) == 1.0

    def test_boundary_values_included(self):
        """Lower and upper bounds are inclusive."""
        bounds = {"x": (0.0, 1.0)}
        assert compute_a_range({"x": 0.0}, bounds) == 1.0
        assert compute_a_range({"x": 1.0}, bounds) == 1.0
        assert compute_a_range({"x": 1.001}, bounds) == 0.0


# ---------------------------------------------------------------------------
# A_ensemble
# ---------------------------------------------------------------------------

class TestComputeAEnsemble:
    def test_tight_spread(self):
        """Nearly identical outputs → score near 1.0."""
        outputs = [1.0, 1.001, 0.999, 1.0002, 0.9998]
        score = compute_a_ensemble(outputs)
        assert score > 0.95, f"Expected high agreement, got {score}"

    def test_wide_spread(self):
        """Very wide spread relative to mean → score near 0.0."""
        outputs = [1.0, 10.0, -8.0, 5.0, -3.0]
        score = compute_a_ensemble(outputs)
        assert score == 0.0 or score < 0.3, f"Expected low agreement, got {score}"

    def test_single_member_returns_zero(self):
        assert compute_a_ensemble([1.0]) == 0.0

    def test_empty_returns_zero(self):
        assert compute_a_ensemble([]) == 0.0

    def test_two_identical_outputs(self):
        """Two identical outputs → std=0 → score=1.0."""
        score = compute_a_ensemble([5.0, 5.0])
        assert math.isclose(score, 1.0, abs_tol=1e-6)

    def test_clamped_to_zero(self):
        """Score never goes below 0.0."""
        outputs = [0.001, 1000.0, -999.0, 500.0, -500.0]
        assert compute_a_ensemble(outputs) == 0.0

    def test_clamped_to_one(self):
        """Score never exceeds 1.0."""
        outputs = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert compute_a_ensemble(outputs) <= 1.0


# ---------------------------------------------------------------------------
# Combined applicability score
# ---------------------------------------------------------------------------

class TestComputeApplicabilityScore:
    def test_weights_applied_correctly(self):
        """
        With known components the combined formula must equal
        0.30*A_d + 0.30*A_r + 0.40*A_e.
        """
        # Single training point at origin; x at origin (density=1.0)
        training = [{"x": 0.0}]
        x = {"x": 0.0}
        bounds = {"x": (0.0, 1.0)}
        # ensemble: two identical outputs → A_ensemble = 1.0
        ensemble = [0.0, 0.0]

        a_total, a_d, a_r, a_e = compute_applicability_score(x, training, bounds, ensemble)

        expected = 0.30 * a_d + 0.30 * a_r + 0.40 * a_e
        assert math.isclose(a_total, expected, abs_tol=1e-9)

    def test_returns_four_tuple(self):
        result = compute_applicability_score(
            {"x": 0.5},
            [{"x": 0.5}],
            {"x": (0.0, 1.0)},
            [1.0, 1.0],
        )
        assert len(result) == 4

    def test_total_bounded(self):
        """Total score is always in [0, 1]."""
        a_total, *_ = compute_applicability_score(
            {"x": 50.0},
            [{"x": float(i)} for i in range(5)],
            {"x": (0.0, 4.0)},
            [1.0, 1.0, 1.0],
        )
        assert 0.0 <= a_total <= 1.0

    def test_ood_point_low_score(self):
        """Far-OOD point should produce a low total score."""
        training = [{"x": float(i)} for i in range(5)]
        x = {"x": 1000.0}
        bounds = {"x": (0.0, 4.0)}
        ensemble = [1.0, 1.0]
        a_total, _, _, _ = compute_applicability_score(x, training, bounds, ensemble)
        # A_density = 0 (far OOD), A_range = 0 (out of bounds),
        # A_ensemble = 1.0 (tight).  Total = 0.30*0 + 0.30*0 + 0.40*1 = 0.40
        assert a_total < 0.5
