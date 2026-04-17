"""
Design variable domain models and in-memory store for EP-04.

DesignVariable, DesignVariableSet, and DesignVariableStore implement the
design-variable layer described in the Henqo EP-04 specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from graph_service.persistence.store import _now, new_uuid


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DesignVariable:
    name: str
    quantity_kind: str
    unit: str
    lower_bound: float
    upper_bound: float
    nominal: Optional[float] = None
    description: str = ""


@dataclass
class DesignVariableSet:
    set_id: str
    component_revision_id: str
    variables: list[DesignVariable]
    created_at: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_design_variable(var: DesignVariable) -> None:
    """
    Raise ValueError if lower_bound >= upper_bound, or if nominal is set
    but falls outside [lower_bound, upper_bound].
    """
    if var.lower_bound >= var.upper_bound:
        raise ValueError(
            f"DesignVariable '{var.name}': lower_bound ({var.lower_bound}) "
            f"must be strictly less than upper_bound ({var.upper_bound})"
        )
    if var.nominal is not None:
        if not (var.lower_bound <= var.nominal <= var.upper_bound):
            raise ValueError(
                f"DesignVariable '{var.name}': nominal ({var.nominal}) "
                f"is outside bounds [{var.lower_bound}, {var.upper_bound}]"
            )


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class DesignVariableStore:
    """In-memory store for DesignVariableSets (test/dev only)."""

    def __init__(self) -> None:
        self._sets: dict[str, DesignVariableSet] = {}

    def create(self, dvs: DesignVariableSet) -> DesignVariableSet:
        if not dvs.set_id:
            dvs.set_id = new_uuid()
        if not dvs.created_at:
            dvs.created_at = _now()
        self._sets[dvs.set_id] = dvs
        return dvs

    def get(self, set_id: str) -> Optional[DesignVariableSet]:
        return self._sets.get(set_id)

    def get_by_component(self, component_revision_id: str) -> list[DesignVariableSet]:
        return [
            dvs for dvs in self._sets.values()
            if dvs.component_revision_id == component_revision_id
        ]
