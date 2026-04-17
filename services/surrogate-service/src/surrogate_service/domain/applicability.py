"""
Applicability / domain-validity score computation.

Formula (surrogate-trust-policy.md §2):
    A(x) = 0.30 * A_density + 0.30 * A_range + 0.40 * A_ensemble
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def compute_a_density(
    x: dict[str, float],
    training_data: list[dict[str, float]],
) -> float:
    """
    Density-based applicability: fraction of training points within a
    hypersphere of radius = 2 * mean pairwise distance in the training set.

    Returns 0.0–1.0.  If training_data is empty, returns 0.0.
    Uses L2 distance over shared feature keys.
    """
    if not training_data:
        return 0.0

    # Determine shared feature keys between x and the training set
    shared_keys = sorted(set(x.keys()) & set(training_data[0].keys()))
    if not shared_keys:
        return 0.0

    n = len(training_data)

    def l2(a: dict[str, float], b: dict[str, float]) -> float:
        return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in shared_keys))

    # Mean pairwise distance within training set (sample pairs for efficiency)
    # For small sets compute exhaustively; the spec does not size-cap this.
    pairwise_sum = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_sum += l2(training_data[i], training_data[j])
            pair_count += 1

    mean_pairwise = pairwise_sum / pair_count if pair_count > 0 else 0.0
    radius = 2.0 * mean_pairwise

    # Count training points within radius of x
    within = sum(1 for pt in training_data if l2(x, pt) <= radius)
    return within / n


def compute_a_range(
    x: dict[str, float],
    feature_bounds: dict[str, tuple[float, float]],
) -> float:
    """
    Range check: fraction of features of x that fall within [lower, upper].

    Returns 0.0–1.0.  If feature_bounds is empty, returns 1.0 (no bounds
    means no features are out-of-range).
    """
    if not feature_bounds:
        return 1.0

    in_range = sum(
        1 for feat, (lo, hi) in feature_bounds.items()
        if lo <= x.get(feat, lo) <= hi  # missing features treated as in-range
    )
    return in_range / len(feature_bounds)


def compute_a_ensemble(outputs: list[float]) -> float:
    """
    Ensemble agreement: 1 - (std / (|mean| + epsilon)).

    High spread → low score; tight agreement → high score.
    Clamped to [0.0, 1.0].  If len(outputs) < 2, returns 0.0.
    """
    if len(outputs) < 2:
        return 0.0

    epsilon = 1e-8
    n = len(outputs)
    mean_val = sum(outputs) / n
    variance = sum((v - mean_val) ** 2 for v in outputs) / (n - 1)
    std_val = math.sqrt(variance)

    score = 1.0 - std_val / (abs(mean_val) + epsilon)
    return max(0.0, min(1.0, score))


def compute_applicability_score(
    x: dict[str, float],
    training_data: list[dict[str, float]],
    feature_bounds: dict[str, tuple[float, float]],
    ensemble_outputs: list[float],
) -> tuple[float, float, float, float]:
    """
    Compute the composite applicability score.

    Returns (A_total, A_density, A_range, A_ensemble).

    Weights per surrogate-trust-policy.md §2 defaults:
        w1 = 0.30, w2 = 0.30, w3 = 0.40
    """
    a_density = compute_a_density(x, training_data)
    a_range = compute_a_range(x, feature_bounds)
    a_ensemble = compute_a_ensemble(ensemble_outputs)

    a_total = 0.30 * a_density + 0.30 * a_range + 0.40 * a_ensemble
    return a_total, a_density, a_range, a_ensemble
