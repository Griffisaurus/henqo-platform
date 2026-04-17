"""
Decision Package Service handler — EP-08 (DecisionPackage) + EP-09 (ReleaseManifest).

AR-005: This service MUST NOT set manifest_status="active" directly.
        Manifests are always created in "draft" state.
        The transition to "active" requires approver_id through graph service UpdateEntityState.
"""
from __future__ import annotations

import sys
import os

# Allow running without installation
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..", "..",
        "services", "graph-service", "src",
    ),
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..", "..",
        "packages", "schema", "src",
    ),
)

from dataclasses import dataclass, field
from typing import Any

from decision_pkg_service.domain.completeness import (
    CompletenessResult,
    EvidenceItem,
    check_completeness,
)
from decision_pkg_service.domain.gating import GatingResult, check_characteristic_gating
from decision_pkg_service.domain.release_rules import RuleResult, check_release_rules

from graph_service.api.handlers import (
    CreateEntityRequest,
    GraphServiceHandler,
)
from graph_service.persistence.store import new_uuid


# ---------------------------------------------------------------------------
# EP-08 request / response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenerateDecisionPackageRequest:
    assembly_revision_id: str
    review_gate: str
    requirements: list[dict]
    characteristics: list[dict]
    evidence_items: list[EvidenceItem]
    package_id: str = ""


@dataclass
class GenerateDecisionPackageResponse:
    package_id: str = ""
    completeness: CompletenessResult | None = None
    gating: GatingResult | None = None
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


# ---------------------------------------------------------------------------
# EP-09 request / response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenerateReleaseManifestRequest:
    assembly_revision_id: str
    package_id: str                          # existing DecisionPackage
    characteristics: list[dict]
    inspection_results: list[dict]
    simulation_cases: list[dict]
    predictions: list[dict]
    process_capabilities: list[dict] = field(default_factory=list)
    benchmark_results: list[dict] = field(default_factory=list)
    provenance_bundles: list[dict] = field(default_factory=list)
    mrs_score: float = 1.0
    ics_score: float = 1.0
    manifest_id: str = ""
    release_manifest_data: dict = field(default_factory=dict)


@dataclass
class GenerateReleaseManifestResponse:
    manifest_id: str = ""
    rule_results: list[RuleResult] | None = None
    all_rules_passed: bool = False
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class DecisionPkgServiceHandler:
    """
    Handles DecisionPackage and ReleaseManifest generation.

    Depends on a GraphServiceHandler for IR persistence.
    """

    def __init__(self, graph_handler: GraphServiceHandler | None = None) -> None:
        self._graph = graph_handler or GraphServiceHandler()

    # ------------------------------------------------------------------
    # EP-08: Generate Decision Package
    # ------------------------------------------------------------------

    def generate_decision_package(
        self,
        req: GenerateDecisionPackageRequest,
    ) -> GenerateDecisionPackageResponse:
        """
        1. check_characteristic_gating → if not passed: return CHARACTERISTIC_NOT_READY
        2. check_completeness → if not complete: return EVIDENCE_INCOMPLETE
        3. Build DecisionPackage payload and create via graph_handler
        4. Return GenerateDecisionPackageResponse with package_id
        """
        # Step 1: Gate check
        gating = check_characteristic_gating(req.characteristics, req.review_gate)
        if not gating.passed:
            return GenerateDecisionPackageResponse(
                gating=gating,
                error_code="CHARACTERISTIC_NOT_READY",
                error_message=gating.reason,
            )

        # Step 2: Completeness check
        completeness = check_completeness(
            requirements=req.requirements,
            characteristics=req.characteristics,
            evidence_items=req.evidence_items,
            review_gate=req.review_gate,
        )
        if not completeness.complete:
            return GenerateDecisionPackageResponse(
                completeness=completeness,
                gating=gating,
                error_code="EVIDENCE_INCOMPLETE",
                error_message=(
                    f"Evidence incomplete: uncovered_requirements={completeness.uncovered_requirements},"
                    f" insufficient_characteristics={completeness.insufficient_characteristics}"
                ),
            )

        # Step 3: Build and persist DecisionPackage
        package_id = req.package_id or new_uuid()
        payload: dict[str, Any] = {
            "entity_id": package_id,   # graph handler uses this as the stored entity_id
            "package_id": package_id,
            "assembly_revision_id": req.assembly_revision_id,
            "review_gate": req.review_gate,
            "covered_requirements": completeness.covered_requirements,
        }
        create_resp = self._graph.create_entity(
            CreateEntityRequest(
                entity_type="DecisionPackage",
                payload=payload,
            )
        )
        if not create_resp.ok:
            return GenerateDecisionPackageResponse(
                error_code=create_resp.error_code,
                error_message=create_resp.error_message,
            )

        # Step 4: Return success
        return GenerateDecisionPackageResponse(
            package_id=create_resp.entity_id,
            completeness=completeness,
            gating=gating,
        )

    # ------------------------------------------------------------------
    # EP-09: Generate Release Manifest
    # ------------------------------------------------------------------

    def generate_release_manifest(
        self,
        req: GenerateReleaseManifestRequest,
    ) -> GenerateReleaseManifestResponse:
        """
        1. check_release_rules → rule_results
        2. If any rule fails: return EVIDENCE_INCOMPLETE with rule_results
        3. Build ReleaseManifest payload (status="draft" — NEVER "active")
        4. create_entity("ReleaseManifest", payload) via graph_handler
        5. Return response with manifest_id and all_rules_passed=True

        AR-005: manifest_status is ALWAYS "draft" on creation.
                Transition to "active" requires approver_id via graph service.
        """
        # Step 1: Evaluate all 10 release rules
        release_manifest_data = req.release_manifest_data or {}
        rule_results = check_release_rules(
            assembly_revision_id=req.assembly_revision_id,
            characteristics=req.characteristics,
            inspection_results=req.inspection_results,
            simulation_cases=req.simulation_cases,
            predictions=req.predictions,
            release_manifest_data=release_manifest_data,
            mrs_score=req.mrs_score,
            ics_score=req.ics_score,
        )

        # Step 2: If any rule fails, return EVIDENCE_INCOMPLETE
        failed = [r for r in rule_results if not r.passed]
        if failed:
            failed_ids = [r.rule_id for r in failed]
            failed_reasons = "; ".join(r.reason for r in failed)
            return GenerateReleaseManifestResponse(
                rule_results=rule_results,
                all_rules_passed=False,
                error_code="EVIDENCE_INCOMPLETE",
                error_message=f"Release rules failed: {failed_ids}. Details: {failed_reasons}",
            )

        # Step 3 & 4: Build manifest in "draft" state (AR-005: NEVER "active")
        manifest_id = req.manifest_id or new_uuid()
        payload: dict[str, Any] = {
            "entity_id": manifest_id,   # graph handler uses this as the stored entity_id
            "manifest_id": manifest_id,
            "assembly_revision_id": req.assembly_revision_id,
            "manifest_status": "draft",   # AR-005: always draft on creation
            "package_id": req.package_id,
        }
        create_resp = self._graph.create_entity(
            CreateEntityRequest(
                entity_type="ReleaseManifest",
                payload=payload,
            )
        )
        if not create_resp.ok:
            return GenerateReleaseManifestResponse(
                rule_results=rule_results,
                error_code=create_resp.error_code,
                error_message=create_resp.error_message,
            )

        # Step 5: Return success
        return GenerateReleaseManifestResponse(
            manifest_id=create_resp.entity_id,
            rule_results=rule_results,
            all_rules_passed=True,
        )
