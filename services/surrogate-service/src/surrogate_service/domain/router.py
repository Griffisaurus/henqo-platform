"""
Surrogate family router and decision-class threshold lookup.

Maps (physics_domain, discipline) pairs to surrogate families S1–S4.
Decision-class thresholds from surrogate-trust-policy.md §3.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

# Keys are (physics_domain, discipline); values are surrogate family IDs.
_ROUTING_TABLE: dict[tuple[str, str], str] = {
    ("aerodynamics", "cfd"): "S1",
    ("aerodynamics", "airfoil"): "S1",
    ("structural", "fem"): "S2",
    ("structural", "structural"): "S2",
    ("thermal", "fem"): "S2",
    ("prognostics", "rul"): "S3",
    ("materials", "atomistic"): "S4",
    ("materials", "molecular"): "S4",
}

# ---------------------------------------------------------------------------
# Decision-class thresholds
# ---------------------------------------------------------------------------

# Minimum A(x) required per class (surrogate-trust-policy.md §3).
# ReleaseCritical and SafetyCritical always require simulation / formal
# evidence; surrogates always abstain for those classes.
_DECISION_CLASS_THRESHOLDS: dict[str, float] = {
    "Exploratory": 0.80,
    "DesignGate": 0.90,
    "ReleaseCritical": 1.01,    # Effectively unreachable — always abstain
    "SafetyCritical": 1.01,     # Effectively unreachable — always abstain
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def route_to_surrogate(physics_domain: str, discipline: str) -> str:
    """
    Return the surrogate family (S1–S4) for the given domain/discipline pair.

    Raises ValueError if no route is found.
    """
    key = (physics_domain.lower(), discipline.lower())
    family = _ROUTING_TABLE.get(key)
    if family is None:
        raise ValueError(
            f"No surrogate route found for physics_domain={physics_domain!r}, "
            f"discipline={discipline!r}. "
            f"Known routes: {sorted(_ROUTING_TABLE.keys())}"
        )
    return family


def get_decision_class_threshold(decision_class: str) -> float:
    """
    Return the minimum applicability score required for the given decision class.

    Raises ValueError for unknown classes.
    """
    threshold = _DECISION_CLASS_THRESHOLDS.get(decision_class)
    if threshold is None:
        raise ValueError(
            f"Unknown decision class {decision_class!r}. "
            f"Valid classes: {sorted(_DECISION_CLASS_THRESHOLDS.keys())}"
        )
    return threshold
