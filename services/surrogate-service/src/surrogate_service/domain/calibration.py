"""
Split-conformal calibration utilities.

Implements the conformal calibration protocol from
benchmark-eval-plan.md §4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    coverage_achieved: float        # fraction of calibration set within interval
    target_coverage: float          # e.g. 0.90
    conformal_quantile: float       # q_hat
    interval_width_mean: float      # mean(upper - lower) across calibration set
    passed: bool                    # coverage_achieved >= target_coverage - 0.02


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_nonconformity_scores(
    predictions: list[float],
    actuals: list[float],
    std_devs: list[float],
) -> list[float]:
    """
    Nonconformity score = |y - y_hat| / sigma_hat.

    sigma_hat is replaced by epsilon (1e-8) when it is zero to avoid
    division-by-zero.
    """
    if not (len(predictions) == len(actuals) == len(std_devs)):
        raise ValueError(
            "predictions, actuals, and std_devs must have the same length"
        )
    epsilon = 1e-8
    return [
        abs(y - y_hat) / max(sigma, epsilon)
        for y, y_hat, sigma in zip(actuals, predictions, std_devs)
    ]


def compute_conformal_quantile(scores: list[float], coverage: float = 0.90) -> float:
    """
    Return the ceil((n+1)*(1-alpha)) / n quantile of scores.

    This is the standard split-conformal quantile from Angelopoulos &
    Bates (2021).  Clipped to the largest score when the index exceeds n.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    if not (0.0 < coverage < 1.0):
        raise ValueError("coverage must be strictly between 0 and 1")

    n = len(scores)
    alpha = 1.0 - coverage
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    # Convert to 0-based index position in sorted list
    # quantile position = ceil((n+1)*(1-alpha)) - 1 (0-indexed)
    idx = math.ceil((n + 1) * (1.0 - alpha)) - 1
    idx = min(idx, n - 1)  # clip to last element
    sorted_scores = sorted(scores)
    return sorted_scores[idx]


def evaluate_calibration(
    cal_predictions: list[float],
    cal_actuals: list[float],
    cal_std_devs: list[float],
    target_coverage: float = 0.90,
) -> CalibrationResult:
    """
    Full calibration pipeline:
      1. Compute nonconformity scores
      2. Compute conformal quantile q_hat
      3. Build prediction intervals [y_hat - q*sigma, y_hat + q*sigma]
      4. Measure empirical coverage
      5. Compute mean interval width
      6. Determine pass/fail (coverage >= target - 0.02)
    """
    scores = compute_nonconformity_scores(cal_predictions, cal_actuals, cal_std_devs)
    q_hat = compute_conformal_quantile(scores, target_coverage)

    epsilon = 1e-8
    within_interval = 0
    widths: list[float] = []
    for y_hat, y_true, sigma in zip(cal_predictions, cal_actuals, cal_std_devs):
        s = max(sigma, epsilon)
        lower = y_hat - q_hat * s
        upper = y_hat + q_hat * s
        widths.append(upper - lower)
        if lower <= y_true <= upper:
            within_interval += 1

    n = len(cal_predictions)
    coverage_achieved = within_interval / n if n > 0 else 0.0
    interval_width_mean = sum(widths) / len(widths) if widths else 0.0
    passed = coverage_achieved >= target_coverage - 0.02

    return CalibrationResult(
        coverage_achieved=coverage_achieved,
        target_coverage=target_coverage,
        conformal_quantile=q_hat,
        interval_width_mean=interval_width_mean,
        passed=passed,
    )
