"""
EP-08: Evidence completeness checker.

Each Requirement must have at least one EvidenceItem with status in
{provisionally_supportive, definitively_supportive}.

Each Characteristic with criticality="Critical" must have at minimum
definitively_supportive evidence at CDR and above.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Gate level ordering: PDR=1, CDR=2, PRR=3, FRR=4
_GATE_LEVELS: dict[str, int] = {
    "PDR": 1,
    "CDR": 2,
    "PRR": 3,
    "FRR": 4,
}

_SUPPORTIVE_STATUSES = frozenset(["provisionally_supportive", "definitively_supportive"])


@dataclass
class EvidenceItem:
    characteristic_id: str
    evidence_type: str        # "Prediction" | "SimulationCase" | "InspectionResult"
    evidence_entity_id: str
    status: str               # "provisionally_supportive" | "definitively_supportive"
                              # | "insufficient" | "conflicting"
    decision_class: str       # class this evidence supports


@dataclass
class CompletenessResult:
    complete: bool
    covered_requirements: list[str] = field(default_factory=list)
    uncovered_requirements: list[str] = field(default_factory=list)
    insufficient_characteristics: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)


def check_completeness(
    requirements: list[dict],
    characteristics: list[dict],
    evidence_items: list[EvidenceItem],
    review_gate: str,  # "PDR" | "CDR" | "PRR" | "FRR"
) -> CompletenessResult:
    """
    Check evidence completeness for a review gate.

    Rules:
    1. Each Requirement must have at least one EvidenceItem with status in
       {provisionally_supportive, definitively_supportive}.
    2. Each Characteristic with criticality="Critical" must have at minimum
       definitively_supportive evidence at CDR and above.
    """
    gate_level = _GATE_LEVELS.get(review_gate, 1)
    cdr_or_above = gate_level >= _GATE_LEVELS["CDR"]

    # Build lookup: characteristic_id → list[EvidenceItem]
    char_to_evidence: dict[str, list[EvidenceItem]] = {}
    for item in evidence_items:
        char_to_evidence.setdefault(item.characteristic_id, []).append(item)

    # Build lookup: requirement_id → governing characteristic_ids
    # A requirement is "covered" if any of its linked characteristics has supportive evidence
    # We check by looking for evidence items whose characteristic is linked to the requirement.
    # Requirements carry a list of characteristic_ids they govern (or evidence_items link directly)
    # Strategy: map requirement→entity_ids it constrains via characteristics
    req_to_char_ids: dict[str, list[str]] = {}
    for char in characteristics:
        char_id = char.get("entity_id", char.get("characteristic_id", ""))
        req_id = char.get("governing_requirement_id", "")
        if req_id:
            req_to_char_ids.setdefault(req_id, []).append(char_id)

    covered_requirements: list[str] = []
    uncovered_requirements: list[str] = []
    detail: list[str] = []

    for req in requirements:
        req_id = req.get("entity_id", req.get("requirement_id", ""))
        # Gather all evidence items that cover this requirement's characteristics
        linked_char_ids = req_to_char_ids.get(req_id, [])
        has_supportive = False
        for char_id in linked_char_ids:
            for ev in char_to_evidence.get(char_id, []):
                if ev.status in _SUPPORTIVE_STATUSES:
                    has_supportive = True
                    break
            if has_supportive:
                break

        # Also check if any evidence_item directly references req_id as decision_class
        # (fallback: some requirements may have no characteristics, evidence covers them directly)
        if not has_supportive:
            for ev in evidence_items:
                if ev.characteristic_id == req_id and ev.status in _SUPPORTIVE_STATUSES:
                    has_supportive = True
                    break

        if has_supportive:
            covered_requirements.append(req_id)
        else:
            uncovered_requirements.append(req_id)
            detail.append(f"Requirement {req_id!r} has no supportive evidence")

    # Rule 2: Critical characteristics need definitively_supportive at CDR+
    insufficient_characteristics: list[str] = []
    if cdr_or_above:
        for char in characteristics:
            char_id = char.get("entity_id", char.get("characteristic_id", ""))
            criticality = char.get("criticality", "")
            if criticality.lower() == "critical":
                evs = char_to_evidence.get(char_id, [])
                has_definitive = any(
                    ev.status == "definitively_supportive" for ev in evs
                )
                if not has_definitive:
                    insufficient_characteristics.append(char_id)
                    detail.append(
                        f"Critical Characteristic {char_id!r} requires definitively_supportive"
                        f" evidence at {review_gate} but has none"
                    )

    complete = (
        len(uncovered_requirements) == 0
        and len(insufficient_characteristics) == 0
    )

    return CompletenessResult(
        complete=complete,
        covered_requirements=covered_requirements,
        uncovered_requirements=uncovered_requirements,
        insufficient_characteristics=insufficient_characteristics,
        detail=detail,
    )
