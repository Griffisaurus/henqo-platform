"""
Unit tests for tolerance stack methods.

Validates MR-003 golden case from eval-acceptance-spec §8:
  3 contributors: σ₁=0.05, σ₂=0.03, σ₃=0.04 (all independent)
  worst_case = 0.05 + 0.03 + 0.04 = 0.12 mm
  rss = sqrt(0.0025 + 0.0009 + 0.0016) = sqrt(0.005) ≈ 0.07071 mm

Also validates:
  - Guard band (10% of spec)
  - Monte Carlo deterministic with seed
  - Correlation matrix in RSS
"""
import math
import pytest

from mfg_service.domain.tolerance_stack import (
    ToleranceContributor,
    ToleranceStackResult,
    compute_monte_carlo,
    compute_rss,
    compute_worst_case,
)


# ---------------------------------------------------------------------------
# Golden contributors from MR-003
# ---------------------------------------------------------------------------

def _mr003_contributors() -> list[ToleranceContributor]:
    """σ₁=0.05, σ₂=0.03, σ₃=0.04 with sigma_factor=1.0 → tolerance = sigma."""
    return [
        ToleranceContributor(name="c1", nominal=1.0, tolerance=0.05, sigma_factor=1.0),
        ToleranceContributor(name="c2", nominal=1.0, tolerance=0.03, sigma_factor=1.0),
        ToleranceContributor(name="c3", nominal=1.0, tolerance=0.04, sigma_factor=1.0),
    ]


# ---------------------------------------------------------------------------
# Worst-case tests
# ---------------------------------------------------------------------------

class TestWorstCase:
    def test_mr003_worst_case_sum(self):
        """MR-003: worst_case = 0.05+0.03+0.04 = 0.12 mm (± 0.001)."""
        contributors = _mr003_contributors()
        result = compute_worst_case(contributors, spec_min_gap=0.0)
        # gap_nominal = 3.0 mm; worst_case total = 0.12; gap_wc = 3.0 - 0.12 = 2.88
        assert result.gap_worst_case == pytest.approx(3.0 - 0.12, abs=0.001)
        assert result.method == "worst_case"

    def test_worst_case_gap_nominal(self):
        """gap_nominal is sum of contributor nominals."""
        contributors = [
            ToleranceContributor("a", 5.0, 0.1),
            ToleranceContributor("b", 3.0, 0.2),
        ]
        result = compute_worst_case(contributors, spec_min_gap=0.0)
        assert result.gap_nominal == pytest.approx(8.0)

    def test_worst_case_passes_spec(self):
        """3 contributors with large nominals easily clear spec."""
        contributors = [
            ToleranceContributor("a", 10.0, 0.05),
            ToleranceContributor("b", 5.0, 0.03),
            ToleranceContributor("c", 2.0, 0.04),
        ]
        result = compute_worst_case(contributors, spec_min_gap=0.5)
        # gap_wc = 17.0 - 0.12 = 16.88; spec_min + guard = 0.5 + 0.05 = 0.55
        assert result.passes_spec is True

    def test_worst_case_fails_spec(self):
        """Stack total exceeds spec when gap is too tight."""
        contributors = [
            ToleranceContributor("a", 0.0, 0.05),
            ToleranceContributor("b", 0.0, 0.03),
            ToleranceContributor("c", 0.0, 0.04),
        ]
        result = compute_worst_case(contributors, spec_min_gap=0.20)
        # gap_wc = 0.0 - 0.12 = -0.12; well below spec
        assert result.passes_spec is False

    def test_guard_band_is_10_percent(self):
        """Guard band = 10% of spec_min_gap."""
        result = compute_worst_case([], spec_min_gap=1.0)
        assert result.guard_band == pytest.approx(0.10)

    def test_guard_band_zero_when_spec_zero(self):
        """Guard band is 0 when spec_min_gap=0."""
        result = compute_worst_case([], spec_min_gap=0.0)
        assert result.guard_band == pytest.approx(0.0)

    def test_gap_rss_is_none_for_worst_case(self):
        """gap_rss and gap_p99 are None for worst_case method."""
        result = compute_worst_case(_mr003_contributors(), spec_min_gap=0.0)
        assert result.gap_rss is None
        assert result.gap_p99 is None
        assert result.n_samples is None
        assert result.seed is None

    def test_single_contributor(self):
        """Single contributor: worst_case = its tolerance."""
        result = compute_worst_case(
            [ToleranceContributor("x", 5.0, 0.5)],
            spec_min_gap=0.0,
        )
        assert result.gap_worst_case == pytest.approx(5.0 - 0.5)


# ---------------------------------------------------------------------------
# RSS tests
# ---------------------------------------------------------------------------

class TestRSS:
    def test_mr003_rss_value(self):
        """MR-003: RSS σ_stack = sqrt(0.05²+0.03²+0.04²) = sqrt(0.005) ≈ 0.07071."""
        contributors = _mr003_contributors()
        result = compute_rss(contributors, spec_min_gap=0.0)
        expected_sigma = math.sqrt(0.0025 + 0.0009 + 0.0016)  # ≈ 0.07071
        # gap_rss = nominal (3.0) - 3*sigma_stack
        expected_gap_rss = 3.0 - 3.0 * expected_sigma
        assert result.gap_rss == pytest.approx(expected_gap_rss, abs=0.001)

    def test_rss_sigma_less_than_worst_case(self):
        """RSS 3σ gap vs worst-case gap depends on sigma_factor.
        With sigma_factor=1.0 (tolerance=sigma), RSS uses 3σ=3× tolerance
        so RSS gap is actually more conservative (smaller) than worst-case sum.
        With the typical sigma_factor=3.0 (tolerance=3σ), RSS gap > worst-case gap.
        """
        # sigma_factor=3.0 (default): tolerance = 3σ, so sigma = tolerance/3
        # RSS gap = nominal - 3*(tolerance/3) = nominal - tolerance  (per contributor)
        # WC gap  = nominal - sum(tolerances)
        # For multiple contributors with sigma_factor=3, RSS > WC because 3*σ_stack < sum(t_i)
        contributors_default = [
            ToleranceContributor(name="c1", nominal=1.0, tolerance=0.05),  # sigma_factor=3.0
            ToleranceContributor(name="c2", nominal=1.0, tolerance=0.03),
            ToleranceContributor(name="c3", nominal=1.0, tolerance=0.04),
        ]
        wc = compute_worst_case(contributors_default, spec_min_gap=0.0)
        rss = compute_rss(contributors_default, spec_min_gap=0.0)
        # With sigma_factor=3: σ_i = t_i/3; σ_stack = sqrt(Σ(t_i/3)²) = (1/3)*sqrt(Σt_i²)
        # 3*σ_stack = sqrt(Σt_i²) < Σt_i  (by RMS < sum inequality for > 1 term)
        assert rss.gap_rss > wc.gap_worst_case

    def test_rss_worst_case_always_computed(self):
        """RSS result also contains gap_worst_case."""
        result = compute_rss(_mr003_contributors(), spec_min_gap=0.0)
        assert result.gap_worst_case == pytest.approx(3.0 - 0.12, abs=0.001)

    def test_rss_with_correlation_matrix(self):
        """RSS with full positive correlation matrix inflates sigma."""
        contributors = [
            ToleranceContributor("a", 0.0, 0.05, sigma_factor=1.0),
            ToleranceContributor("b", 0.0, 0.03, sigma_factor=1.0),
        ]
        rho = 1.0  # perfect positive correlation
        corr = [[1.0, rho], [rho, 1.0]]
        result = compute_rss(contributors, spec_min_gap=0.0, correlation_matrix=corr)
        # sigma = sqrt(0.05²+0.03²+2*1*0.05*0.03) = sqrt(0.0025+0.0009+0.003) = sqrt(0.0064) = 0.08
        expected_sigma = math.sqrt(0.0025 + 0.0009 + 2 * 1.0 * 0.05 * 0.03)
        expected_gap_rss = 0.0 - 3.0 * expected_sigma
        assert result.gap_rss == pytest.approx(expected_gap_rss, abs=1e-6)

    def test_rss_no_correlation_independent_contributors(self):
        """RSS without correlation matrix assumes independence (ρ=0)."""
        contributors = [
            ToleranceContributor("a", 0.0, 0.05, sigma_factor=1.0),
            ToleranceContributor("b", 0.0, 0.03, sigma_factor=1.0),
        ]
        result_no_corr = compute_rss(contributors, spec_min_gap=0.0)
        result_zero_corr = compute_rss(
            contributors,
            spec_min_gap=0.0,
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert result_no_corr.gap_rss == pytest.approx(result_zero_corr.gap_rss, abs=1e-9)

    def test_rss_method_name(self):
        assert compute_rss(_mr003_contributors(), 0.0).method == "rss"


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_deterministic_with_fixed_seed(self):
        """Monte Carlo with same seed produces identical results."""
        contributors = _mr003_contributors()
        r1 = compute_monte_carlo(contributors, spec_min_gap=0.0, n_samples=10_000, seed=42)
        r2 = compute_monte_carlo(contributors, spec_min_gap=0.0, n_samples=10_000, seed=42)
        assert r1.gap_p99 == r2.gap_p99

    def test_different_seeds_may_differ(self):
        """Different seeds generally produce different results."""
        contributors = _mr003_contributors()
        r1 = compute_monte_carlo(contributors, spec_min_gap=0.0, n_samples=10_000, seed=42)
        r2 = compute_monte_carlo(contributors, spec_min_gap=0.0, n_samples=10_000, seed=99)
        # They should not be exactly equal (statistically extremely unlikely)
        assert r1.gap_p99 != r2.gap_p99

    def test_mc_p99_near_nominal_for_small_tolerances(self):
        """With tight tolerances and large nominals, p99 should be close to nominal."""
        contributors = [
            ToleranceContributor("a", 10.0, 0.01),
            ToleranceContributor("b", 5.0, 0.01),
        ]
        result = compute_monte_carlo(contributors, spec_min_gap=0.0, seed=42)
        # 99th percentile of gap should be near 15.0 ± a few sigma
        assert result.gap_p99 == pytest.approx(15.0, abs=0.1)

    def test_mc_n_samples_and_seed_recorded(self):
        """n_samples and seed are recorded in the result."""
        result = compute_monte_carlo(_mr003_contributors(), 0.0, n_samples=5_000, seed=7)
        assert result.n_samples == 5_000
        assert result.seed == 7

    def test_mc_method_name(self):
        assert compute_monte_carlo(_mr003_contributors(), 0.0).method == "monte_carlo"

    def test_mc_worst_case_still_computed(self):
        """Monte Carlo result still contains gap_worst_case."""
        result = compute_monte_carlo(_mr003_contributors(), spec_min_gap=0.0, seed=42)
        assert result.gap_worst_case == pytest.approx(3.0 - 0.12, abs=0.001)

    def test_mc_rss_still_computed(self):
        """Monte Carlo result also contains gap_rss."""
        result = compute_monte_carlo(_mr003_contributors(), spec_min_gap=0.0, seed=42)
        assert result.gap_rss is not None
        sigma = math.sqrt(0.0025 + 0.0009 + 0.0016)
        assert result.gap_rss == pytest.approx(3.0 - 3.0 * sigma, abs=0.001)

    def test_guard_band_applied(self):
        """Guard band is 10% of spec_min_gap."""
        result = compute_monte_carlo(_mr003_contributors(), spec_min_gap=2.0, seed=42)
        assert result.guard_band == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Cross-method consistency
# ---------------------------------------------------------------------------

class TestCrossMethodConsistency:
    def test_worst_case_ge_rss_gap(self):
        """With default sigma_factor=3.0: RSS gap > WC gap (RSS is less conservative).
        Both methods compute gap_worst_case identically."""
        contributors = [
            ToleranceContributor(name="c1", nominal=1.0, tolerance=0.05),  # default sigma_factor=3
            ToleranceContributor(name="c2", nominal=1.0, tolerance=0.03),
            ToleranceContributor(name="c3", nominal=1.0, tolerance=0.04),
        ]
        wc = compute_worst_case(contributors, spec_min_gap=0.0)
        rss = compute_rss(contributors, spec_min_gap=0.0)
        # WC gap: 3.0 - (0.05+0.03+0.04) = 2.88
        # RSS gap: 3.0 - 3*(1/3)*sqrt(0.05²+0.03²+0.04²) = 3.0 - sqrt(0.005) ≈ 2.929
        # RSS gives a larger (less conservative) gap than WC
        assert wc.gap_worst_case <= rss.gap_rss

    def test_same_spec_min_gap_recorded(self):
        """All methods record the same spec_min_gap."""
        spec = 0.50
        c = _mr003_contributors()
        for method_fn in [compute_worst_case, compute_rss, compute_monte_carlo]:
            result = method_fn(c, spec_min_gap=spec)
            assert result.spec_min_gap == pytest.approx(spec)
