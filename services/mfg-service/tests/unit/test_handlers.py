"""
Unit tests for MfgServiceHandler.compute_report().

Validates:
  - Class A violation triggers escalation event (MR-001)
  - No violations → MRS = 1.0
  - Tolerance stack computed when contributors provided
  - ICS computed when characteristics provided
  - class_a_found flag reflects violation state
"""
import sys
import os

import pytest

# Ensure both services are importable
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "src",
    ),
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "graph-service", "src",
    ),
)

from mfg_service.api.handlers import (
    ComputeManufacturabilityRequest,
    ManufacturabilityReport,
    MfgServiceHandler,
)
from mfg_service.domain.dfm_rules import DFMViolation

try:
    from graph_service.persistence.store import InMemoryEntityStore
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_store():
    if GRAPH_AVAILABLE:
        return InMemoryEntityStore()
    return None


def _make_handler(store=None):
    return MfgServiceHandler(store=store or _make_store())


# ---------------------------------------------------------------------------
# MR-001: Class A violation fires and escalates
# ---------------------------------------------------------------------------

class TestClassAEscalation:
    def test_class_a_violation_sets_flag(self):
        """MR-001: CNC wall thickness violation → class_a_found=True."""
        store = _make_store()
        handler = _make_handler(store)
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-001",
            component_data={"wall_thickness_mm": 0.5, "material": "aluminum"},
            process_families=["cnc"],
        )
        report = handler.compute_report(req)
        assert report.class_a_found is True

    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="graph-service not on path")
    def test_class_a_emits_escalation_event(self):
        """MR-001: Class A violation emits dfm_violation.class_a_found event."""
        store = InMemoryEntityStore()
        handler = MfgServiceHandler(store=store)
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-001",
            component_data={"wall_thickness_mm": 0.5, "material": "aluminum"},
            process_families=["cnc"],
        )
        handler.compute_report(req)

        events = store.get_events("rev-001")
        escalation_events = [e for e in events if e.event_type == "dfm_violation.class_a_found"]
        assert len(escalation_events) == 1
        assert escalation_events[0].entity_id == "rev-001"
        assert escalation_events[0].triggered_by == "mfg_service"

    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="graph-service not on path")
    def test_no_class_a_no_escalation_event(self):
        """No Class A violations → no escalation event emitted."""
        store = InMemoryEntityStore()
        handler = MfgServiceHandler(store=store)
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-002",
            component_data={"setup_count": 6},  # only Class B violation
            process_families=["cnc"],
        )
        handler.compute_report(req)

        events = store.get_events("rev-002")
        escalation_events = [e for e in events if e.event_type == "dfm_violation.class_a_found"]
        assert len(escalation_events) == 0

    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="graph-service not on path")
    def test_multiple_class_a_one_event_emitted(self):
        """Multiple Class A violations → single escalation event (not one per violation)."""
        store = InMemoryEntityStore()
        handler = MfgServiceHandler(store=store)
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-003",
            component_data={
                "wall_thickness_mm": 0.3,       # DFM-CNC-001
                "inspection_reachability": 0.5, # DFM-CNC-007
            },
            process_families=["cnc"],
        )
        handler.compute_report(req)
        events = store.get_events("rev-003")
        escalation_events = [e for e in events if e.event_type == "dfm_violation.class_a_found"]
        assert len(escalation_events) == 1


# ---------------------------------------------------------------------------
# No violations → MRS = 1.0
# ---------------------------------------------------------------------------

class TestNoViolations:
    def test_no_violations_mrs_is_one(self):
        """No violations → MRS = 1.0."""
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-clean",
            component_data={},  # no parameters → no rules trigger
            process_families=["cnc"],
        )
        report = handler.compute_report(req)
        assert report.mrs.mrs_score == pytest.approx(1.0)
        assert report.class_a_found is False
        assert report.violations == []

    def test_no_violations_report_ok(self):
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-ok",
            component_data={},
            process_families=["cnc"],
        )
        report = handler.compute_report(req)
        assert report.ok is True
        assert report.error_code == ""


# ---------------------------------------------------------------------------
# Tolerance stack computed when contributors provided
# ---------------------------------------------------------------------------

class TestToleranceStack:
    def test_tolerance_result_none_when_no_contributors(self):
        """No tolerance_contributors → tolerance_result is None."""
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-t1",
            component_data={},
            process_families=["cnc"],
            tolerance_contributors=[],
        )
        report = handler.compute_report(req)
        assert report.tolerance_result is None

    def test_tolerance_result_computed_when_contributors_given(self):
        """tolerance_contributors provided → tolerance_result computed."""
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-t2",
            component_data={},
            process_families=["cnc"],
            tolerance_contributors=[
                {"name": "c1", "nominal": 1.0, "tolerance": 0.05},
                {"name": "c2", "nominal": 1.0, "tolerance": 0.03},
                {"name": "c3", "nominal": 1.0, "tolerance": 0.04},
            ],
            spec_min_gap=0.0,
        )
        report = handler.compute_report(req)
        assert report.tolerance_result is not None
        assert report.tolerance_result.method == "worst_case"
        # gap_worst_case = 3.0 - 0.12 = 2.88
        assert report.tolerance_result.gap_worst_case == pytest.approx(2.88, abs=0.001)

    def test_tolerance_passes_spec_when_gap_large_enough(self):
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-t3",
            component_data={},
            process_families=["cnc"],
            tolerance_contributors=[
                {"name": "c1", "nominal": 5.0, "tolerance": 0.05},
            ],
            spec_min_gap=0.5,
        )
        report = handler.compute_report(req)
        assert report.tolerance_result is not None
        assert report.tolerance_result.passes_spec is True

    def test_tolerance_fails_spec_when_gap_too_tight(self):
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-t4",
            component_data={},
            process_families=["cnc"],
            tolerance_contributors=[
                {"name": "c1", "nominal": 0.0, "tolerance": 0.10},
            ],
            spec_min_gap=0.5,
        )
        report = handler.compute_report(req)
        assert report.tolerance_result is not None
        assert report.tolerance_result.passes_spec is False


# ---------------------------------------------------------------------------
# ICS computed when characteristics provided
# ---------------------------------------------------------------------------

class TestICSComputation:
    def test_ics_none_when_no_characteristics(self):
        """No characteristics → ics is None."""
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-i1",
            component_data={},
            process_families=["cnc"],
        )
        report = handler.compute_report(req)
        assert report.ics is None

    def test_ics_computed_when_characteristics_given(self):
        """ICS computed when characteristics list is non-empty."""
        handler = _make_handler()
        chars = [
            {"entity_id": "ch-001"},
            {"entity_id": "ch-002"},
        ]
        inspections = [
            {
                "characteristic_id": "ch-001",
                "component_revision_id": "rev-i2",
                "decision_rule": "accept",
                "measurement_uncertainty": "0.01 mm",
            }
        ]
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-i2",
            component_data={},
            process_families=["cnc"],
            characteristics=chars,
            inspection_results=inspections,
        )
        report = handler.compute_report(req)
        assert report.ics is not None
        assert report.ics.ics_score == pytest.approx(0.5)
        assert report.ics.qualifying_count == 1
        assert report.ics.total_characteristics == 2

    def test_ics_full_coverage(self):
        """All characteristics have qualifying inspections → ICS = 1.0."""
        handler = _make_handler()
        chars = [{"entity_id": "ch-001"}, {"entity_id": "ch-002"}]
        inspections = [
            {
                "characteristic_id": "ch-001",
                "component_revision_id": "rev-full",
                "decision_rule": "accept",
                "measurement_uncertainty": "0.01",
            },
            {
                "characteristic_id": "ch-002",
                "component_revision_id": "rev-full",
                "decision_rule": "accept",
                "measurement_uncertainty": "0.02",
            },
        ]
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-full",
            component_data={},
            process_families=["cnc"],
            characteristics=chars,
            inspection_results=inspections,
        )
        report = handler.compute_report(req)
        assert report.ics.ics_score == pytest.approx(1.0)

    def test_ics_prior_revision_not_counted(self):
        """Inspection against prior revision → not counted in ICS (MR-005)."""
        handler = _make_handler()
        chars = [{"entity_id": "ch-001"}]
        inspections = [
            {
                "characteristic_id": "ch-001",
                "component_revision_id": "rev-old-v1",  # wrong revision
                "decision_rule": "accept",
                "measurement_uncertainty": "0.01",
            }
        ]
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-current-v2",
            component_data={},
            process_families=["cnc"],
            characteristics=chars,
            inspection_results=inspections,
        )
        report = handler.compute_report(req)
        assert report.ics.qualifying_count == 0
        assert report.ics.ics_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Resolved rules reduce violations in MRS
# ---------------------------------------------------------------------------

class TestResolvedRules:
    def test_resolved_a_violation_restores_s_a(self):
        """Class A violation with its rule_id in resolved_rule_ids → S_A = 1."""
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-r1",
            component_data={"wall_thickness_mm": 0.5, "material": "aluminum"},
            process_families=["cnc"],
            resolved_rule_ids=["DFM-CNC-001"],
        )
        report = handler.compute_report(req)
        # Violation is still detected but scored as resolved
        assert report.class_a_found is True  # detected before resolution applied
        assert report.mrs.s_a == pytest.approx(1.0)
        assert report.mrs.mrs_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multiple process families
# ---------------------------------------------------------------------------

class TestMultipleFamilies:
    def test_cnc_and_am_violations_combined(self):
        """Violations from both CNC and AM families are combined."""
        handler = _make_handler()
        req = ComputeManufacturabilityRequest(
            component_revision_id="rev-multi",
            component_data={
                "wall_thickness_mm": 0.3,     # DFM-CNC-001 A
                "am_process": "metal_pbf",
                "unsupported_span_mm": 2.0,   # DFM-AM-002 A
            },
            process_families=["cnc", "am"],
        )
        report = handler.compute_report(req)
        rule_ids = [v.rule_id for v in report.violations]
        assert "DFM-CNC-001" in rule_ids
        assert "DFM-AM-002" in rule_ids
        assert report.class_a_found is True
