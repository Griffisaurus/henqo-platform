"""
Manufacturability Readiness Score (MRS) computation.

Formula (§4 of manufacturability-subsystem-spec.md):

  MRS = 0.60·S_A + 0.30·S_B + 0.10·S_C

  S_A = 0   if any Class A violation is open (not in resolved_rule_ids)
  S_A = 1   otherwise

  S_B = 1 − (open_b / total_b)          if total_b > 0,  else 1.0
  S_C = 1 − 0.5 × (open_c / total_c)   if total_c > 0,  else 1.0

Note: S_C uses a 0.5 half-weight factor as specified in §4.
"""
from __future__ import annotations

from dataclasses import dataclass

from mfg_service.domain.dfm_rules import DFMViolation


@dataclass
class MRSResult:
    mrs_score: float
    s_a: float           # 0 if any Class A open, else 1.0
    s_b: float           # 1 - (open_b / total_b) if total_b > 0 else 1.0
    s_c: float           # 1 - 0.5*(open_c / total_c) if total_c > 0 else 1.0
    open_a_count: int
    open_b_count: int
    open_c_count: int
    total_b_count: int
    total_c_count: int


def compute_mrs(
    violations: list[DFMViolation],
    resolved_rule_ids: list[str] | None = None,
) -> MRSResult:
    """
    Compute the Manufacturability Readiness Score.

    Parameters
    ----------
    violations:
        All DFMViolation objects from rule evaluation.
    resolved_rule_ids:
        Rule IDs that have been resolved/waived. Any violation whose
        rule_id appears here is treated as closed for scoring purposes.
    """
    resolved = set(resolved_rule_ids or [])

    open_a = 0
    open_b = 0
    open_c = 0
    total_b = 0
    total_c = 0

    for v in violations:
        is_open = v.rule_id not in resolved
        if v.tier == "A":
            if is_open:
                open_a += 1
        elif v.tier == "B":
            total_b += 1
            if is_open:
                open_b += 1
        elif v.tier == "C":
            total_c += 1
            if is_open:
                open_c += 1

    s_a = 0.0 if open_a > 0 else 1.0
    s_b = (1.0 - open_b / total_b) if total_b > 0 else 1.0
    s_c = (1.0 - 0.5 * (open_c / total_c)) if total_c > 0 else 1.0

    mrs = 0.60 * s_a + 0.30 * s_b + 0.10 * s_c

    return MRSResult(
        mrs_score=round(mrs, 6),
        s_a=s_a,
        s_b=round(s_b, 6),
        s_c=round(s_c, 6),
        open_a_count=open_a,
        open_b_count=open_b,
        open_c_count=open_c,
        total_b_count=total_b,
        total_c_count=total_c,
    )
