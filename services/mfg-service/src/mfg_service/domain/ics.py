"""
Inspection Coverage Score (ICS) computation.

ICS = qualifying_count / total_characteristics

A characteristic qualifies if it has at least one InspectionResult where:
  - characteristic_id matches the characteristic's entity_id
  - component_revision_id matches the current revision being evaluated
  - decision_rule is non-empty (populated per ASME Y14.45)
  - measurement_uncertainty is non-empty (populated per GUM)

InspectionResults produced against a prior revision do NOT qualify.

Reference: §8 of manufacturability-subsystem-spec.md
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ICSResult:
    ics_score: float                  # 0.0–1.0
    qualifying_count: int
    total_characteristics: int
    non_qualifying: list[str] = field(default_factory=list)  # entity_ids


def compute_ics(
    characteristics: list[dict],
    inspection_results: list[dict],
    component_revision_id: str,
) -> ICSResult:
    """
    Compute the Inspection Coverage Score.

    Parameters
    ----------
    characteristics:
        List of Characteristic payload dicts.  Each must have ``entity_id``.
    inspection_results:
        List of InspectionResult payload dicts.  Each must have:
          ``characteristic_id``, ``component_revision_id``,
          ``decision_rule``, ``measurement_uncertainty``.
    component_revision_id:
        The current ComponentRevision ID.  Only InspectionResults
        against this revision are counted.
    """
    if not characteristics:
        return ICSResult(
            ics_score=1.0,
            qualifying_count=0,
            total_characteristics=0,
            non_qualifying=[],
        )

    # Index qualifying inspection results by characteristic_id
    qualifying_char_ids: set[str] = set()
    for ir in inspection_results:
        char_id = ir.get("characteristic_id", "")
        rev_id = ir.get("component_revision_id", "")
        decision_rule = ir.get("decision_rule", "")
        meas_unc = ir.get("measurement_uncertainty", "")

        if (
            rev_id == component_revision_id
            and decision_rule
            and meas_unc
        ):
            qualifying_char_ids.add(char_id)

    qualifying_count = 0
    non_qualifying: list[str] = []

    for char in characteristics:
        eid = char.get("entity_id", "")
        if eid in qualifying_char_ids:
            qualifying_count += 1
        else:
            non_qualifying.append(eid)

    total = len(characteristics)
    score = qualifying_count / total if total > 0 else 1.0

    return ICSResult(
        ics_score=round(score, 6),
        qualifying_count=qualifying_count,
        total_characteristics=total,
        non_qualifying=non_qualifying,
    )
