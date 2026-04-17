"""
EP-08: Characteristic gating checker.

Rejects any Characteristic whose status is unverified or unresolved.
At CDR and above (DesignGate+), also rejects surrogate-only evidence
where TrustBundle.evaluated_decision_class < "DesignGate".
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Gate level ordering
_GATE_LEVELS: dict[str, int] = {
    "PDR": 1,
    "CDR": 2,
    "PRR": 3,
    "FRR": 4,
}

# Decision class ordering (lower index = lower class)
_DECISION_CLASS_ORDER: list[str] = [
    "Exploratory",
    "DesignGate",
    "ReleaseCritical",
    "SafetyCritical",
]

_BLOCKED_STATUSES = frozenset(["unverified", "unresolved"])


@dataclass
class GatingResult:
    passed: bool
    blocked_characteristics: list[str] = field(default_factory=list)  # entity_ids of blocked chars
    reason: str = ""


def _gate_level(review_gate: str) -> int:
    """PDR=1, CDR=2, PRR=3, FRR=4. Used to determine if gate is DesignGate+."""
    return _GATE_LEVELS.get(review_gate, 1)


def _decision_class_level(decision_class: str) -> int:
    """Return numeric level for a decision class; -1 if unknown."""
    try:
        return _DECISION_CLASS_ORDER.index(decision_class)
    except ValueError:
        return -1


def check_characteristic_gating(
    characteristics: list[dict],  # list of Characteristic payloads with status field
    review_gate: str,
) -> GatingResult:
    """
    Check all characteristics for gate readiness.

    Rules:
    1. Reject (blocked) any Characteristic with status in {"unverified", "unresolved"}.
    2. At DesignGate (CDR) and above: also reject surrogate-only evidence where
       TrustBundle.evaluated_decision_class < "DesignGate".
    """
    gate_lvl = _gate_level(review_gate)
    design_gate_plus = gate_lvl >= _GATE_LEVELS["CDR"]

    blocked: list[str] = []
    reasons: list[str] = []

    for char in characteristics:
        char_id = char.get("entity_id", char.get("characteristic_id", ""))
        status = char.get("status", "")

        # Rule 1: blocked statuses
        if status in _BLOCKED_STATUSES:
            blocked.append(char_id)
            reasons.append(
                f"Characteristic {char_id!r} has status {status!r} which is not gate-ready"
            )
            continue

        # Rule 2: at CDR+, surrogate-only chars need DesignGate-level trust
        if design_gate_plus:
            # Check if this characteristic is surrogate-only
            # Indicated by status="surrogate_estimated" and trust_bundle present
            if status == "surrogate_estimated":
                trust_bundle = char.get("trust_bundle") or {}
                evaluated_class = trust_bundle.get("evaluated_decision_class", "")
                design_gate_level = _decision_class_level("DesignGate")
                evaluated_level = _decision_class_level(evaluated_class)

                if evaluated_level < design_gate_level:
                    blocked.append(char_id)
                    reasons.append(
                        f"Characteristic {char_id!r} is surrogate-only with"
                        f" evaluated_decision_class={evaluated_class!r}"
                        f" which is below DesignGate at {review_gate}"
                    )

    if blocked:
        return GatingResult(
            passed=False,
            blocked_characteristics=blocked,
            reason="; ".join(reasons),
        )

    return GatingResult(passed=True, blocked_characteristics=[], reason="")
