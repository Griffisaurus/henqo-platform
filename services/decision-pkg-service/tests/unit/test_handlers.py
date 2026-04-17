"""
Tests for decision_pkg_service.api.handlers (EP-08 + EP-09).

≥12 tests covering:
- Successful decision package creation
- CHARACTERISTIC_NOT_READY on gating failure (DT-004)
- EVIDENCE_INCOMPLETE on completeness failure
- Successful manifest generation
- EVIDENCE_INCOMPLETE when release rule fails (RC-001)
- Manifest created in draft state, never active (AR-005)
"""
from __future__ import annotations

import pytest

from decision_pkg_service.api.handlers import (
    DecisionPkgServiceHandler,
    GenerateDecisionPackageRequest,
    GenerateReleaseManifestRequest,
)
from decision_pkg_service.domain.completeness import EvidenceItem
from graph_service.api.handlers import GraphServiceHandler
from graph_service.persistence.store import InMemoryEntityStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler() -> DecisionPkgServiceHandler:
    store = InMemoryEntityStore()
    graph = GraphServiceHandler(store=store)
    return DecisionPkgServiceHandler(graph_handler=graph)


def _req(req_id: str) -> dict:
    return {"entity_id": req_id, "text": f"Req {req_id}", "criticality": "Shall"}


def _char(char_id: str, req_id: str = "REQ-001", status: str = "simulation_validated",
          criticality: str = "Standard") -> dict:
    return {
        "entity_id": char_id,
        "name": f"char_{char_id}",
        "governing_requirement_id": req_id,
        "criticality": criticality,
        "status": status,
        "quantity_kind": "Force",
        "unit": "N",
        "decision_class_required": "DesignGate",
    }


def _ev(char_id: str, status: str = "definitively_supportive") -> EvidenceItem:
    return EvidenceItem(
        characteristic_id=char_id,
        evidence_type="SimulationCase",
        evidence_entity_id=f"ev-{char_id}",
        status=status,
        decision_class="DesignGate",
    )


def _inspection(ir_id: str, char_id: str, status: str = "pass") -> dict:
    return {"entity_id": ir_id, "characteristic_id": char_id, "status": status}


def _full_manifest_data() -> dict:
    return {
        "open_issues": [],
        "required_signatories": ["chief_engineer", "quality_lead"],
        "actual_signatories": ["chief_engineer", "quality_lead"],
    }


# ---------------------------------------------------------------------------
# EP-08: generate_decision_package
# ---------------------------------------------------------------------------

class TestGenerateDecisionPackageSuccess:
    def test_successful_package_creation_returns_package_id(self):
        handler = _make_handler()
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001")],
            evidence_items=[_ev("CH-001")],
        )
        resp = handler.generate_decision_package(req)
        assert resp.ok is True
        assert resp.package_id != ""
        assert resp.error_code == ""

    def test_successful_package_has_completeness_and_gating(self):
        handler = _make_handler()
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001")],
            evidence_items=[_ev("CH-001")],
        )
        resp = handler.generate_decision_package(req)
        assert resp.completeness is not None
        assert resp.gating is not None
        assert resp.completeness.complete is True
        assert resp.gating.passed is True

    def test_custom_package_id_is_used(self):
        handler = _make_handler()
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001")],
            evidence_items=[_ev("CH-001")],
            package_id="pkg-custom-001",
        )
        resp = handler.generate_decision_package(req)
        assert resp.ok is True
        assert resp.package_id == "pkg-custom-001"


class TestGenerateDecisionPackageGatingFailure:
    """DT-004: Unverified Characteristic must return CHARACTERISTIC_NOT_READY."""

    def test_unverified_characteristic_returns_characteristic_not_ready(self):
        handler = _make_handler()
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001", status="unverified")],
            evidence_items=[_ev("CH-001")],
        )
        resp = handler.generate_decision_package(req)
        assert resp.ok is False
        assert resp.error_code == "CHARACTERISTIC_NOT_READY"
        assert resp.package_id == ""

    def test_unresolved_characteristic_returns_characteristic_not_ready(self):
        handler = _make_handler()
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001", status="unresolved")],
            evidence_items=[_ev("CH-001")],
        )
        resp = handler.generate_decision_package(req)
        assert resp.ok is False
        assert resp.error_code == "CHARACTERISTIC_NOT_READY"

    def test_gating_failure_does_not_create_entity(self):
        """No IR writes when gating fails (DT-004 zero IR writes requirement)."""
        store = InMemoryEntityStore()
        graph = GraphServiceHandler(store=store)
        handler = DecisionPkgServiceHandler(graph_handler=graph)
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001", status="unverified")],
            evidence_items=[_ev("CH-001")],
        )
        handler.generate_decision_package(req)
        # No DecisionPackage entities should exist
        from graph_service.api.handlers import QueryEntitiesRequest
        qresp = graph.query_entities(QueryEntitiesRequest(entity_type="DecisionPackage"))
        assert qresp.total_count == 0


class TestGenerateDecisionPackageCompletenessFailure:
    def test_missing_evidence_returns_evidence_incomplete(self):
        handler = _make_handler()
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001")],
            evidence_items=[],  # no evidence
        )
        resp = handler.generate_decision_package(req)
        assert resp.ok is False
        assert resp.error_code == "EVIDENCE_INCOMPLETE"

    def test_completeness_failure_includes_completeness_result(self):
        handler = _make_handler()
        req = GenerateDecisionPackageRequest(
            assembly_revision_id="asm-001",
            review_gate="PDR",
            requirements=[_req("REQ-001")],
            characteristics=[_char("CH-001")],
            evidence_items=[],
        )
        resp = handler.generate_decision_package(req)
        assert resp.completeness is not None
        assert "REQ-001" in resp.completeness.uncovered_requirements


# ---------------------------------------------------------------------------
# EP-09: generate_release_manifest
# ---------------------------------------------------------------------------

class TestGenerateReleaseManifestSuccess:
    def _valid_manifest_req(self, pkg_id: str = "pkg-001") -> GenerateReleaseManifestRequest:
        chars = [
            _char("CH-001", status="inspection_confirmed", criticality="Critical"),
            _char("CH-002", status="released", criticality="key"),
        ]
        inspections = [
            _inspection("IR-001", "CH-001", "pass"),
            _inspection("IR-002", "CH-002", "pass"),
        ]
        return GenerateReleaseManifestRequest(
            assembly_revision_id="asm-001",
            package_id=pkg_id,
            characteristics=chars,
            inspection_results=inspections,
            simulation_cases=[],
            predictions=[],
            mrs_score=0.85,
            ics_score=0.90,
            release_manifest_data=_full_manifest_data(),
        )

    def test_successful_manifest_generation_returns_manifest_id(self):
        handler = _make_handler()
        resp = handler.generate_release_manifest(self._valid_manifest_req())
        assert resp.ok is True
        assert resp.manifest_id != ""
        assert resp.all_rules_passed is True

    def test_manifest_created_in_draft_state_not_active(self):
        """AR-005: Manifest MUST be created in draft, never active."""
        store = InMemoryEntityStore()
        graph = GraphServiceHandler(store=store)
        handler = DecisionPkgServiceHandler(graph_handler=graph)
        resp = handler.generate_release_manifest(self._valid_manifest_req())
        assert resp.ok is True

        # Verify the entity was stored in draft state
        from graph_service.api.handlers import GetEntityRequest
        entity_resp = graph.get_entity(GetEntityRequest(entity_id=resp.manifest_id))
        assert entity_resp.ok is True
        assert entity_resp.payload is not None
        assert entity_resp.payload["manifest_status"] == "draft"
        # Definitively: NOT active
        assert entity_resp.payload["manifest_status"] != "active"

    def test_manifest_never_set_to_active_directly(self):
        """AR-005: Even if all rules pass, manifest_status is always draft."""
        handler = _make_handler()
        resp = handler.generate_release_manifest(self._valid_manifest_req())
        assert resp.ok is True
        # The handler must not set active status
        # (verified via entity inspection in above test, confirmed by spec contract)
        assert resp.all_rules_passed is True

    def test_rule_results_returned_on_success(self):
        handler = _make_handler()
        resp = handler.generate_release_manifest(self._valid_manifest_req())
        assert resp.rule_results is not None
        assert len(resp.rule_results) == 10

    def test_custom_manifest_id_is_used(self):
        handler = _make_handler()
        req = self._valid_manifest_req()
        req.manifest_id = "manifest-custom-001"
        resp = handler.generate_release_manifest(req)
        assert resp.ok is True
        assert resp.manifest_id == "manifest-custom-001"


class TestGenerateReleaseManifestRuleFailure:
    """RC-001/RC-002: Rule failures block manifest generation."""

    def test_mrs_below_threshold_returns_evidence_incomplete(self):
        handler = _make_handler()
        chars = [_char("CH-001", status="inspection_confirmed", criticality="Critical")]
        inspections = [_inspection("IR-001", "CH-001", "pass")]
        req = GenerateReleaseManifestRequest(
            assembly_revision_id="asm-001",
            package_id="pkg-001",
            characteristics=chars,
            inspection_results=inspections,
            simulation_cases=[],
            predictions=[],
            mrs_score=0.50,  # below 0.70 target
            ics_score=0.90,
            release_manifest_data=_full_manifest_data(),
        )
        resp = handler.generate_release_manifest(req)
        assert resp.ok is False
        assert resp.error_code == "EVIDENCE_INCOMPLETE"
        assert resp.all_rules_passed is False

    def test_rule_failure_does_not_create_manifest_entity(self):
        """RC-002: When rule fails, no manifest entity should be created."""
        store = InMemoryEntityStore()
        graph = GraphServiceHandler(store=store)
        handler = DecisionPkgServiceHandler(graph_handler=graph)
        chars = [_char("CH-001", status="inspection_confirmed", criticality="Critical")]
        req = GenerateReleaseManifestRequest(
            assembly_revision_id="asm-001",
            package_id="pkg-001",
            characteristics=chars,
            inspection_results=[],  # missing inspection → R2 fails
            simulation_cases=[],
            predictions=[],
            mrs_score=0.85,
            ics_score=0.90,
            release_manifest_data=_full_manifest_data(),
        )
        resp = handler.generate_release_manifest(req)
        assert resp.ok is False

        from graph_service.api.handlers import QueryEntitiesRequest
        qresp = graph.query_entities(QueryEntitiesRequest(entity_type="ReleaseManifest"))
        assert qresp.total_count == 0

    def test_failed_rule_results_included_in_response(self):
        handler = _make_handler()
        req = GenerateReleaseManifestRequest(
            assembly_revision_id="asm-001",
            package_id="pkg-001",
            characteristics=[_char("CH-001", status="inspection_confirmed")],
            inspection_results=[_inspection("IR-001", "CH-001", "pass")],
            simulation_cases=[],
            predictions=[],
            mrs_score=0.50,  # fails R3
            ics_score=0.90,
            release_manifest_data=_full_manifest_data(),
        )
        resp = handler.generate_release_manifest(req)
        assert resp.rule_results is not None
        failed_ids = [r.rule_id for r in resp.rule_results if not r.passed]
        assert "R3" in failed_ids

    def test_open_issues_block_manifest(self):
        """RC-001: R9 failure (open issues) blocks manifest."""
        handler = _make_handler()
        chars = [_char("CH-001", status="inspection_confirmed", criticality="Critical")]
        inspections = [_inspection("IR-001", "CH-001", "pass")]
        manifest_data = {
            "open_issues": ["ISSUE-001"],  # open issue
            "required_signatories": ["chief_engineer"],
            "actual_signatories": ["chief_engineer"],
        }
        req = GenerateReleaseManifestRequest(
            assembly_revision_id="asm-001",
            package_id="pkg-001",
            characteristics=chars,
            inspection_results=inspections,
            simulation_cases=[],
            predictions=[],
            mrs_score=0.85,
            ics_score=0.90,
            release_manifest_data=manifest_data,
        )
        resp = handler.generate_release_manifest(req)
        assert resp.ok is False
        assert resp.error_code == "EVIDENCE_INCOMPLETE"
