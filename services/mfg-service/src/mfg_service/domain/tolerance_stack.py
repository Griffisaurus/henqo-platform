"""
Tolerance stack-up analysis — three methods (§5 of manufacturability-subsystem-spec.md).

Methods:
  worst_case   — T_wc = Σ |tolerance_i|  (sum of half-widths)
  rss          — σ_stack = sqrt(Σ σ_i² + 2 Σ_{i<j} ρ_ij σ_i σ_j)
  monte_carlo  — simulate n_samples, report p1 as minimum gap

Guard band: ISO 14253-1 — 10% of spec range (spec_max - spec_min).
For the gap problem we only have spec_min_gap, so the guard band is
10% of spec_min_gap when spec_min_gap > 0.  The pass criterion is:
  gap_result ≥ spec_min_gap + guard_band  (gap must be at least spec_min + guard)

Uses stdlib only: math, statistics, random.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field


@dataclass
class ToleranceContributor:
    name: str
    nominal: float
    tolerance: float       # half-width (±)
    sigma_factor: float = 3.0  # tolerance / sigma_factor = std_dev


@dataclass
class ToleranceStackResult:
    method: str                       # "worst_case" | "rss" | "monte_carlo"
    gap_nominal: float
    gap_worst_case: float             # always computed
    gap_rss: float | None             # None if method == worst_case
    gap_p99: float | None             # None if not monte_carlo
    n_samples: int | None
    seed: int | None
    passes_spec: bool
    spec_min_gap: float
    guard_band: float                 # ISO 14253-1: 10% of spec_min_gap (if > 0)


def _guard_band(spec_min_gap: float) -> float:
    """ISO 14253-1: guard band = 10% of the spec range.
    We treat spec_min_gap as the full range reference."""
    return 0.10 * abs(spec_min_gap) if spec_min_gap != 0.0 else 0.0


def _nominal_gap(contributors: list[ToleranceContributor]) -> float:
    return sum(c.nominal for c in contributors)


def _worst_case_total(contributors: list[ToleranceContributor]) -> float:
    """T_wc = Σ |tolerance_i|."""
    return sum(abs(c.tolerance) for c in contributors)


def compute_worst_case(
    contributors: list[ToleranceContributor],
    spec_min_gap: float,
) -> ToleranceStackResult:
    """Worst-case tolerance stack.

    The worst-case gap is: nominal_gap - T_wc
    Passes if: gap_worst_case >= spec_min_gap + guard_band
    """
    g_nom = _nominal_gap(contributors)
    t_wc = _worst_case_total(contributors)
    gap_wc = g_nom - t_wc
    gb = _guard_band(spec_min_gap)
    passes = gap_wc >= (spec_min_gap + gb)

    return ToleranceStackResult(
        method="worst_case",
        gap_nominal=g_nom,
        gap_worst_case=gap_wc,
        gap_rss=None,
        gap_p99=None,
        n_samples=None,
        seed=None,
        passes_spec=passes,
        spec_min_gap=spec_min_gap,
        guard_band=gb,
    )


def compute_rss(
    contributors: list[ToleranceContributor],
    spec_min_gap: float,
    correlation_matrix: list[list[float]] | None = None,
) -> ToleranceStackResult:
    """RSS tolerance stack with optional correlation matrix.

    σ_i = tolerance_i / sigma_factor_i
    σ_stack = sqrt(Σ σ_i² + 2 Σ_{i<j} ρ_ij σ_i σ_j)
    gap_rss = nominal_gap - 3*σ_stack  (3σ coverage ≈ 99.73%)
    Passes if: gap_rss >= spec_min_gap + guard_band
    """
    sigmas = [c.tolerance / c.sigma_factor for c in contributors]
    n = len(contributors)

    variance = sum(s ** 2 for s in sigmas)

    if correlation_matrix is not None:
        for i in range(n):
            for j in range(i + 1, n):
                rho = correlation_matrix[i][j]
                variance += 2.0 * rho * sigmas[i] * sigmas[j]

    sigma_stack = math.sqrt(max(variance, 0.0))
    g_nom = _nominal_gap(contributors)
    t_wc = _worst_case_total(contributors)
    gap_wc = g_nom - t_wc
    gap_rss_val = g_nom - 3.0 * sigma_stack
    gb = _guard_band(spec_min_gap)
    passes = gap_rss_val >= (spec_min_gap + gb)

    return ToleranceStackResult(
        method="rss",
        gap_nominal=g_nom,
        gap_worst_case=gap_wc,
        gap_rss=gap_rss_val,
        gap_p99=None,
        n_samples=None,
        seed=None,
        passes_spec=passes,
        spec_min_gap=spec_min_gap,
        guard_band=gb,
    )


def compute_monte_carlo(
    contributors: list[ToleranceContributor],
    spec_min_gap: float,
    n_samples: int = 10_000,
    seed: int = 42,
) -> ToleranceStackResult:
    """Monte Carlo tolerance stack.

    Each contributor drawn from normal(nominal, tolerance/3.0).
    Computes p1 (1st percentile) as the worst-case gap estimate.
    Passes if: p1 >= spec_min_gap + guard_band
    """
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(n_samples):
        total = 0.0
        for c in contributors:
            std_dev = c.tolerance / 3.0
            total += rng.gauss(c.nominal, std_dev)
        gaps.append(total)

    gaps.sort()
    p1_idx = max(0, int(math.ceil(0.01 * n_samples)) - 1)
    p99_idx = min(n_samples - 1, int(math.floor(0.99 * n_samples)))

    gap_p1 = gaps[p1_idx]    # 1st percentile = minimum gap estimate
    gap_p99 = gaps[p99_idx]  # 99th percentile

    g_nom = _nominal_gap(contributors)
    t_wc = _worst_case_total(contributors)
    gap_wc = g_nom - t_wc

    sigmas = [c.tolerance / c.sigma_factor for c in contributors]
    variance = sum(s ** 2 for s in sigmas)
    sigma_stack = math.sqrt(max(variance, 0.0))
    gap_rss_val = g_nom - 3.0 * sigma_stack

    gb = _guard_band(spec_min_gap)
    passes = gap_p1 >= (spec_min_gap + gb)

    return ToleranceStackResult(
        method="monte_carlo",
        gap_nominal=g_nom,
        gap_worst_case=gap_wc,
        gap_rss=gap_rss_val,
        gap_p99=gap_p99,
        n_samples=n_samples,
        seed=seed,
        passes_spec=passes,
        spec_min_gap=spec_min_gap,
        guard_band=gb,
    )
