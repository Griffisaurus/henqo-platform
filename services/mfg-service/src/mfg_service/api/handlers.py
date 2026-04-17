"""
Manufacturability Service Handler (EP-07).

Implements compute_report() which:
  1. Evaluates all applicable DFM rules.
  2. Detects Class A violations and emits an escalation event to the graph store.
  3. Computes the MRS score.
  4. Optionally computes a worst-case tolerance stack.
  5. Optionally computes ICS if characteristics/inspections provided.
  6. Returns a ManufacturabilityReport.
"""
from __future__ import annotations

import sys
import os

# Allow running without installation — mirror graph-service path-injection pattern
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..", "graph-service", "src",
    ),
)

from dataclasses import dataclass, field
from typing import Any

from mfg_service.domain.dfm_rules import DFMViolation, evaluate_all_rules
from mfg_service.domain.mrs import MRSResult, compute_mrs
from mfg_service.domain.tolerance_stack import (
    ToleranceContributor,
    ToleranceStackResult,
    compute_worst_case,
)
from mfg_service.domain.ics import ICSResult, compute_ics

try:
    from graph_service.persistence.store import (
        EventRecord,
        InMemoryEntityStore,
        _now,
        new_uuid,
    )
    _GRAPH_STORE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GRAPH_STORE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Request / Response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComputeManufacturabilityRequest:
    component_revision_id: str
    component_data: dict[str, Any]
    process_families: list[str]
    tolerance_contributors: list[dict[str, Any]] = field(default_factory=list)
    spec_min_gap: float = 0.0
    resolved_rule_ids: list[str] = field(default_factory=list)
    # Optional ICS inputs
    characteristics: list[dict[str, Any]] = field(default_factory=list)
    inspection_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ManufacturabilityReport:
    component_revision_id: str
    mrs: MRSResult
    violations: list[DFMViolation]
    ics: ICSResult | None
    class_a_found: bool
    tolerance_result: ToleranceStackResult | None
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class MfgServiceHandler:
    """Manufacturability service request handler."""

    def __init__(self, store: Any | None = None) -> None:
        if store is not None:
            self._store = store
        elif _GRAPH_STORE_AVAILABLE:
            self._store = InMemoryEntityStore()
        else:
            self._store = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # compute_report
    # ------------------------------------------------------------------

    def compute_report(
        self,
        req: ComputeManufacturabilityRequest,
    ) -> ManufacturabilityReport:
        """
        Run the full manufacturability evaluation pipeline.

        Steps:
          1. evaluate_all_rules → violations list
          2. Detect Class A violations
          3. If Class A found: emit escalation event to graph store
          4. compute_mrs → MRS result
          5. If tolerance_contributors provided: compute_worst_case
          6. If characteristics provided: compute_ics
          7. Return ManufacturabilityReport
        """
        # Step 1 — DFM rule evaluation
        violations = evaluate_all_rules(
            req.component_data,
            req.process_families,
        )

        # Step 2 — Class A detection
        class_a_found = any(v.tier == "A" for v in violations)

        # Step 3 — Escalation event for Class A violations
        if class_a_found and self._store is not None:
            event_id = new_uuid()
            event = EventRecord(
                event_id=event_id,
                event_type="dfm_violation.class_a_found",
                entity_type="ComponentRevision",
                entity_id=req.component_revision_id,
                previous_state="manufacturing_review_requested",
                new_state="manufacturing_review_requested",
                triggered_by="mfg_service",
                timestamp=_now(),
            )
            self._store.emit_event(event)

        # Step 4 — MRS computation
        mrs_result = compute_mrs(violations, req.resolved_rule_ids)

        # Step 5 — Tolerance stack (worst-case only at this API level)
        tolerance_result: ToleranceStackResult | None = None
        if req.tolerance_contributors:
            contributors = [
                ToleranceContributor(
                    name=c.get("name", f"contributor_{i}"),
                    nominal=float(c.get("nominal", 0.0)),
                    tolerance=float(c.get("tolerance", 0.0)),
                    sigma_factor=float(c.get("sigma_factor", 3.0)),
                )
                for i, c in enumerate(req.tolerance_contributors)
            ]
            tolerance_result = compute_worst_case(contributors, req.spec_min_gap)

        # Step 6 — ICS computation
        ics_result: ICSResult | None = None
        if req.characteristics:
            ics_result = compute_ics(
                req.characteristics,
                req.inspection_results,
                req.component_revision_id,
            )

        return ManufacturabilityReport(
            component_revision_id=req.component_revision_id,
            mrs=mrs_result,
            violations=violations,
            ics=ics_result,
            class_a_found=class_a_found,
            tolerance_result=tolerance_result,
        )
