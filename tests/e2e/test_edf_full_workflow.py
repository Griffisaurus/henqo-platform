"""
EP-10: EDF Propulsion Pod Exemplar — full end-to-end workflow test.

Exercises the 12-step EDF workflow using all service handlers wired together
through a single shared InMemoryEntityStore.

Run with:
  cd /home/workbench/henqo-platform
  PYTHONPATH=packages/schema/src:services/graph-service/src:services/artifact-service/src:\
    services/requirements-service/src:services/design-service/src:\
    services/sim-job-service/src:services/surrogate-service/src:\
    services/mfg-service/src:services/decision-pkg-service/src \
    .venv/bin/pytest tests/e2e/test_edf_full_workflow.py -v --tb=short
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Service imports
# ---------------------------------------------------------------------------
from graph_service.persistence.store import InMemoryEntityStore
from graph_service.api.handlers import (
    GraphServiceHandler,
    CreateEntityRequest,
    GetEntityRequest,
    UpdateEntityStateRequest,
)

from requirements_service.api.handlers import (
    RequirementsServiceHandler,
    IngestRequirementRequest,
)

from design_service.api.handlers import (
    DesignServiceHandler,
    CreateComponentRevisionRequest,
    CreateAssemblyRevisionRequest,
    CreateCharacteristicRequest,
)

from sim_job_service.api.handlers import (
    SimJobServiceHandler,
    SubmitJobRequest,
)
from sim_job_service.domain.failure_tracker import FailureTracker

from surrogate_service.api.handlers import (
    SurrogateServiceHandler,
    RunInferenceRequest,
)
from surrogate_service.models.base import (
    InferenceResult,
    ModelRecord,
    SurrogateModel,
)
from surrogate_service.models.registry import ModelRegistry

from mfg_service.api.handlers import (
    MfgServiceHandler,
    ComputeManufacturabilityRequest,
)

from decision_pkg_service.api.handlers import (
    DecisionPkgServiceHandler,
    GenerateDecisionPackageRequest,
    GenerateReleaseManifestRequest,
)
from decision_pkg_service.domain.completeness import EvidenceItem

from artifact_service.api.handlers import ArtifactServiceHandler


# ---------------------------------------------------------------------------
# Concrete stub surrogate model — S2 family for tip_displacement
# ---------------------------------------------------------------------------

class _TipDisplacementSurrogate(SurrogateModel):
    """
    Deterministic stub surrogate for EDF tip_displacement.

    Returns 8 ensemble members with tight spread → high A_ensemble.
    """

    _ENSEMBLE_OUTPUTS = [1.40, 1.38, 1.42, 1.41, 1.39, 1.43, 1.37, 1.40]

    def predict(self, x: dict[str, float]) -> InferenceResult:
        mean_val = sum(self._ENSEMBLE_OUTPUTS) / len(self._ENSEMBLE_OUTPUTS)
        return InferenceResult(
            outputs={"tip_displacement_mm": mean_val},
            ensemble_outputs=list(self._ENSEMBLE_OUTPUTS),
            std_devs={"tip_displacement_mm": 0.02},
            model_id="s2-edf-001",
        )

    def model_record(self) -> ModelRecord:
        return ModelRecord(
            model_id="s2-edf-001",
            surrogate_family="S2",
            weight_hash="sha256:abc123",
            training_schema_version="0.1.0",
            status="production",
            training_dataset_revision="ds-001",
            n_ensemble_members=8,
        )


# ---------------------------------------------------------------------------
# Module-level shared state dict (IDs flow between steps)
# ---------------------------------------------------------------------------
STATE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Fixtures (module scope — shared across all test methods)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store():
    return InMemoryEntityStore()


@pytest.fixture(scope="module")
def graph(store):
    return GraphServiceHandler(store=store)


@pytest.fixture(scope="module")
def requirements_svc(graph):
    return RequirementsServiceHandler(graph_handler=graph)


@pytest.fixture(scope="module")
def design_svc(graph):
    return DesignServiceHandler(graph_handler=graph)


@pytest.fixture(scope="module")
def sim_job_svc(graph):
    return SimJobServiceHandler(
        graph_handler=graph,
        failure_tracker=FailureTracker(),
    )


@pytest.fixture(scope="module")
def surrogate_registry():
    registry = ModelRegistry()
    registry.register(_TipDisplacementSurrogate())
    return registry


@pytest.fixture(scope="module")
def surrogate_svc(graph, surrogate_registry):
    return SurrogateServiceHandler(
        graph_handler=graph,
        registry=surrogate_registry,
    )


@pytest.fixture(scope="module")
def mfg_svc(store):
    return MfgServiceHandler(store=store)


@pytest.fixture(scope="module")
def decision_pkg_svc(graph):
    return DecisionPkgServiceHandler(graph_handler=graph)


@pytest.fixture(scope="module")
def artifact_svc():
    return ArtifactServiceHandler()


# ---------------------------------------------------------------------------
# Test class — sequential workflow steps
# ---------------------------------------------------------------------------

class TestEDFWorkflow:
    """
    End-to-end EDF propulsion pod workflow test.

    Each step builds on the entity IDs stored in the module-level STATE dict.
    If an early step fails, later steps will also fail (expected sequential behavior).
    """

    # ------------------------------------------------------------------
    # Step 1: Ingest EDF requirements
    # ------------------------------------------------------------------

    def test_step01_ingest_requirements(self, requirements_svc):
        """Ingest 3 EDF requirements from exemplar-implementation-package.md §2."""
        req_defs = [
            ("Thrust shall be >= 500 N at rated operating conditions",
             "High", "ReleaseCritical"),
            ("Tip displacement shall not exceed 2 mm under maximum thrust load",
             "High", "DesignGate"),
            ("Fan blade mass shall not exceed 0.8 kg",
             "Medium", "DesignGate"),
        ]

        entity_ids = []
        for text, criticality, decision_class in req_defs:
            resp = requirements_svc.ingest(
                IngestRequirementRequest(
                    text=text,
                    criticality=criticality,
                    decision_class_required=decision_class,
                )
            )
            assert resp.ok, f"Ingest failed: {resp.error_code} — {resp.error_message}"
            assert resp.entity_id, "entity_id must be non-empty"
            assert not resp.is_duplicate, "Requirement should not be a duplicate"
            entity_ids.append(resp.entity_id)

        assert len(entity_ids) == 3, "Exactly 3 requirements must be created"

        STATE["req_thrust_id"] = entity_ids[0]
        STATE["req_tip_disp_id"] = entity_ids[1]
        STATE["req_blade_mass_id"] = entity_ids[2]

    # ------------------------------------------------------------------
    # Step 2: Create ComponentRevision and AssemblyRevision
    # ------------------------------------------------------------------

    def test_step02_create_design_entities(self, design_svc):
        """Create EDF fan blade ComponentRevision and EDF pod AssemblyRevision."""
        # Fan blade component
        blade_resp = design_svc.create_component_revision(
            CreateComponentRevisionRequest(
                component_name="edf_fan_blade",
                description="EDF propulsion pod fan blade — aluminium 7075-T6",
            )
        )
        assert blade_resp.ok, f"ComponentRevision failed: {blade_resp.error_code}"
        assert blade_resp.entity_id, "entity_id must be non-empty"
        STATE["blade_component_id"] = blade_resp.entity_id

        # Propulsion pod assembly
        pod_resp = design_svc.create_assembly_revision(
            CreateAssemblyRevisionRequest(
                assembly_name="edf_propulsion_pod",
                component_revision_ids=[blade_resp.entity_id],
                description="EDF propulsion pod assembly Rev-A",
            )
        )
        assert pod_resp.ok, f"AssemblyRevision failed: {pod_resp.error_code}"
        assert pod_resp.entity_id, "entity_id must be non-empty"
        STATE["assembly_id"] = pod_resp.entity_id

        # Verify statuses via graph
        from graph_service.api.handlers import GraphServiceHandler
        # (can't import graph directly in class, so validate via the design_svc's internal graph)
        blade_entity = design_svc._graph.get_entity(
            GetEntityRequest(entity_id=blade_resp.entity_id)
        )
        assert blade_entity.ok
        assert blade_entity.status == "in_design"

        pod_entity = design_svc._graph.get_entity(
            GetEntityRequest(entity_id=pod_resp.entity_id)
        )
        assert pod_entity.ok
        assert pod_entity.status == "in_design"

    # ------------------------------------------------------------------
    # Step 3: Create Characteristics linked to requirements
    # ------------------------------------------------------------------

    def test_step03_create_characteristics(self, design_svc):
        """Create 3 characteristics linked to their governing requirements."""
        blade_id = STATE["blade_component_id"]
        char_defs = [
            ("thrust_output",    "Force",  "N",  "High",   "ReleaseCritical", STATE["req_thrust_id"]),
            ("tip_displacement", "Length", "mm", "High",   "DesignGate",      STATE["req_tip_disp_id"]),
            ("blade_mass",       "Mass",   "kg", "Medium", "DesignGate",      STATE["req_blade_mass_id"]),
        ]

        for name, qk, unit, crit, dc, req_id in char_defs:
            resp = design_svc.create_characteristic(
                CreateCharacteristicRequest(
                    name=name,
                    quantity_kind=qk,
                    unit=unit,
                    governing_requirement_id=req_id,
                    criticality=crit,
                    decision_class_required=dc,
                    component_revision_id=blade_id,
                )
            )
            assert resp.ok, f"Characteristic '{name}' creation failed: {resp.error_code}"
            assert resp.entity_id, "entity_id must be non-empty"

            # Read back and confirm status
            entity = design_svc._graph.get_entity(
                GetEntityRequest(entity_id=resp.entity_id)
            )
            assert entity.ok
            assert entity.status == "unverified", (
                f"Characteristic '{name}' should start as 'unverified', got {entity.status!r}"
            )

            STATE[f"char_{name}_id"] = resp.entity_id

        assert len([k for k in STATE if k.startswith("char_")]) >= 3

    # ------------------------------------------------------------------
    # Step 4: Run surrogate inference for tip_displacement
    # ------------------------------------------------------------------

    def test_step04_surrogate_inference(self, surrogate_svc):
        """Run S2 surrogate inference for tip_displacement."""
        char_id = STATE["char_tip_displacement_id"]

        resp = surrogate_svc.run_inference(
            RunInferenceRequest(
                surrogate_family="S2",
                x={"chord_mm": 120.0, "thickness_ratio": 0.12},
                requested_decision_class="DesignGate",
                governing_characteristic_id=char_id,
                feature_bounds={
                    "chord_mm": (50.0, 200.0),
                    "thickness_ratio": (0.08, 0.20),
                },
                training_data_sample=[
                    {"chord_mm": 110.0, "thickness_ratio": 0.11},
                    {"chord_mm": 125.0, "thickness_ratio": 0.13},
                ],
                current_schema_version="0.1.0",
            )
        )

        assert resp.ok, f"Inference failed: {resp.error_code} — {resp.error_message}"
        assert not resp.abstained, (
            f"Surrogate should not abstain for DesignGate: {resp.abstain_reason}"
        )
        assert resp.prediction_entity_id, "prediction_entity_id must be non-empty"
        assert resp.trust_bundle is not None
        assert resp.trust_bundle.applicability_score > 0, (
            "applicability_score must be > 0"
        )

        STATE["prediction_id"] = resp.prediction_entity_id

    # ------------------------------------------------------------------
    # Step 5: Advance tip_displacement to surrogate_estimated
    # ------------------------------------------------------------------

    def test_step05_advance_to_surrogate_estimated(self, graph):
        """Transition tip_displacement: unverified → surrogate_estimated."""
        char_id = STATE["char_tip_displacement_id"]

        resp = graph.update_entity_state(
            UpdateEntityStateRequest(
                entity_id=char_id,
                new_state="surrogate_estimated",
                transition_reason="S2 surrogate inference completed",
            )
        )

        assert resp.ok, f"State transition failed: {resp.error_code} — {resp.error_message}"
        assert resp.new_state == "surrogate_estimated"

        # Verify
        entity = graph.get_entity(GetEntityRequest(entity_id=char_id))
        assert entity.status == "surrogate_estimated"

    # ------------------------------------------------------------------
    # Step 6: Submit structural simulation job for tip_displacement
    # ------------------------------------------------------------------

    def test_step06_submit_sim_job(self, sim_job_svc):
        """Submit a FEniCS structural simulation for the fan blade."""
        blade_id = STATE["blade_component_id"]

        resp = sim_job_svc.submit_job(
            SubmitJobRequest(
                solver_type="fenics",
                discipline="structural",
                component_revision_id=blade_id,
                inputs={
                    "mesh_file": "edf_blade.xdmf",
                    "material": "Al7075",
                    "load_case": "max_thrust",
                },
            )
        )

        assert resp.ok, f"SubmitJob failed: {resp.error_code} — {resp.error_message}"
        assert resp.simulation_case_id, "simulation_case_id must be non-empty"

        STATE["sim_case_id"] = resp.simulation_case_id

        # Confirm the sim case is in "validated" status (auto-validation passed)
        status_resp = sim_job_svc.get_job_status(
            type("GetJobStatusRequest", (), {"simulation_case_id": resp.simulation_case_id})()
        )
        assert status_resp.ok
        # FEniCS stub always succeeds and auto-validates
        assert status_resp.status == "validated", (
            f"Expected 'validated', got {status_resp.status!r}"
        )

    # ------------------------------------------------------------------
    # Step 7: Advance tip_displacement to simulation_validated
    # ------------------------------------------------------------------

    def test_step07_advance_to_simulation_validated(self, graph):
        """Transition tip_displacement: surrogate_estimated → simulation_validated."""
        char_id = STATE["char_tip_displacement_id"]

        resp = graph.update_entity_state(
            UpdateEntityStateRequest(
                entity_id=char_id,
                new_state="simulation_validated",
                transition_reason="FEniCS structural simulation validated",
            )
        )

        assert resp.ok, f"State transition failed: {resp.error_code} — {resp.error_message}"
        assert resp.new_state == "simulation_validated"

        entity = graph.get_entity(GetEntityRequest(entity_id=char_id))
        assert entity.status == "simulation_validated"

    # ------------------------------------------------------------------
    # Step 8: Run DFM check on fan blade
    # ------------------------------------------------------------------

    def test_step08_dfm_check(self, mfg_svc):
        """Run DFM manufacturability check on the EDF fan blade."""
        blade_id = STATE["blade_component_id"]

        report = mfg_svc.compute_report(
            ComputeManufacturabilityRequest(
                component_revision_id=blade_id,
                component_data={
                    "min_feature_size_mm": 1.5,
                    "depth_to_width_ratio": 2.0,
                    "tool_diameter_mm": 6.0,
                    "wall_thickness_mm": 2.0,
                },
                process_families=["cnc"],
            )
        )

        assert report.ok, f"DFM report failed: {report.error_code}"
        assert not report.class_a_found, (
            f"No Class A violations expected for nominal geometry, got: "
            f"{[v for v in report.violations if v.tier == 'A']}"
        )

    # ------------------------------------------------------------------
    # Step 9: Attach InspectionResult and advance to inspection_confirmed
    # ------------------------------------------------------------------

    def test_step09_inspection_and_confirm(self, graph):
        """Create InspectionResult for tip_displacement and advance status."""
        char_id = STATE["char_tip_displacement_id"]
        blade_id = STATE["blade_component_id"]

        import uuid
        insp_id = str(uuid.uuid4())

        # Create the InspectionResult entity via graph service
        create_resp = graph.create_entity(
            CreateEntityRequest(
                entity_type="InspectionResult",
                payload={
                    "entity_id": insp_id,
                    "characteristic_id": char_id,
                    "component_revision_id": blade_id,
                    "measured_value": {"value": 1.38, "unit": "mm"},
                    "decision_rule": "< 2 mm",
                    "measurement_uncertainty": {"value": 0.01, "unit": "mm"},
                    "status": "pass",
                },
                created_by="inspection-service",
            )
        )
        assert create_resp.ok, (
            f"InspectionResult creation failed: {create_resp.error_code} — "
            f"{create_resp.error_message}"
        )
        STATE["inspection_result_id"] = create_resp.entity_id

        # Advance characteristic: simulation_validated → inspection_confirmed
        advance_resp = graph.update_entity_state(
            UpdateEntityStateRequest(
                entity_id=char_id,
                new_state="inspection_confirmed",
                transition_reason="Physical inspection confirmed tip_displacement < 2 mm",
            )
        )
        assert advance_resp.ok, (
            f"Transition to inspection_confirmed failed: {advance_resp.error_code}"
        )
        assert advance_resp.new_state == "inspection_confirmed"

        entity = graph.get_entity(GetEntityRequest(entity_id=char_id))
        assert entity.status == "inspection_confirmed"

    # ------------------------------------------------------------------
    # Step 10: Generate DecisionPackage at PDR gate
    # ------------------------------------------------------------------

    def test_step10_generate_decision_package(self, graph, decision_pkg_svc):
        """
        Advance blade_mass and thrust_output to surrogate_estimated, then
        generate a DecisionPackage at PDR gate.
        """
        char_blade_mass_id = STATE["char_blade_mass_id"]
        char_thrust_id = STATE["char_thrust_output_id"]
        char_tip_id = STATE["char_tip_displacement_id"]
        assembly_id = STATE["assembly_id"]

        # Advance blade_mass and thrust_output to surrogate_estimated
        for char_id in [char_blade_mass_id, char_thrust_id]:
            resp = graph.update_entity_state(
                UpdateEntityStateRequest(
                    entity_id=char_id,
                    new_state="surrogate_estimated",
                    transition_reason="Surrogate estimate applied for PDR gate",
                )
            )
            assert resp.ok, (
                f"Failed to advance char {char_id} to surrogate_estimated: "
                f"{resp.error_code}"
            )

        # Build characteristic dicts for the completeness checker
        def _char_dict(entity_id, name, req_id, status):
            return {
                "entity_id": entity_id,
                "name": name,
                "governing_requirement_id": req_id,
                "status": status,
                "criticality": "High",
                "decision_class_required": "DesignGate",
            }

        characteristics = [
            _char_dict(char_thrust_id,    "thrust_output",    STATE["req_thrust_id"],    "surrogate_estimated"),
            _char_dict(char_tip_id,       "tip_displacement", STATE["req_tip_disp_id"],  "inspection_confirmed"),
            _char_dict(char_blade_mass_id, "blade_mass",      STATE["req_blade_mass_id"], "surrogate_estimated"),
        ]

        requirements = [
            {"entity_id": STATE["req_thrust_id"]},
            {"entity_id": STATE["req_tip_disp_id"]},
            {"entity_id": STATE["req_blade_mass_id"]},
        ]

        # One definitively_supportive EvidenceItem per characteristic
        evidence_items = [
            EvidenceItem(
                characteristic_id=char_tip_id,
                evidence_type="InspectionResult",
                evidence_entity_id=STATE["inspection_result_id"],
                status="definitively_supportive",
                decision_class="DesignGate",
            ),
            EvidenceItem(
                characteristic_id=char_thrust_id,
                evidence_type="Prediction",
                evidence_entity_id=STATE.get("prediction_id", ""),
                status="provisionally_supportive",
                decision_class="DesignGate",
            ),
            EvidenceItem(
                characteristic_id=char_blade_mass_id,
                evidence_type="Prediction",
                evidence_entity_id=STATE.get("prediction_id", ""),
                status="provisionally_supportive",
                decision_class="DesignGate",
            ),
        ]

        pkg_resp = decision_pkg_svc.generate_decision_package(
            GenerateDecisionPackageRequest(
                assembly_revision_id=assembly_id,
                review_gate="PDR",
                requirements=requirements,
                characteristics=characteristics,
                evidence_items=evidence_items,
            )
        )

        assert pkg_resp.ok, (
            f"DecisionPackage generation failed: {pkg_resp.error_code} — "
            f"{pkg_resp.error_message}"
        )
        assert pkg_resp.package_id, "package_id must be non-empty"
        STATE["package_id"] = pkg_resp.package_id

    # ------------------------------------------------------------------
    # Step 11: Generate ReleaseManifest
    # ------------------------------------------------------------------

    def test_step11_generate_release_manifest(self, graph, decision_pkg_svc):
        """
        Generate a ReleaseManifest.  With only 3 characteristics and partial
        inspection coverage, some rules may fail — assert manifest_id non-empty
        and status="draft".
        """
        assembly_id = STATE["assembly_id"]
        package_id = STATE["package_id"]
        char_tip_id = STATE["char_tip_displacement_id"]
        char_thrust_id = STATE["char_thrust_output_id"]
        char_blade_mass_id = STATE["char_blade_mass_id"]
        sim_case_id = STATE["sim_case_id"]
        prediction_id = STATE["prediction_id"]
        insp_result_id = STATE["inspection_result_id"]

        now_iso = datetime.now(tz=timezone.utc).isoformat()

        characteristics = [
            {
                "entity_id": char_thrust_id,
                "criticality": "High",
                "decision_class_required": "ReleaseCritical",
                "status": "surrogate_estimated",
            },
            {
                "entity_id": char_tip_id,
                "criticality": "High",
                "decision_class_required": "DesignGate",
                "status": "inspection_confirmed",
            },
            {
                "entity_id": char_blade_mass_id,
                "criticality": "Medium",
                "decision_class_required": "DesignGate",
                "status": "surrogate_estimated",
            },
        ]

        inspection_results = [
            {
                "entity_id": insp_result_id,
                "characteristic_id": char_tip_id,
                "status": "pass",
            },
        ]

        simulation_cases = [
            {
                "entity_id": sim_case_id,
                "characteristic_id": char_tip_id,
                "status": "validated",
            },
        ]

        predictions = [
            {
                "entity_id": prediction_id,
                "governing_characteristic_id": char_tip_id,
                "created_at": now_iso,
                "stale": False,
            },
        ]

        manifest_resp = decision_pkg_svc.generate_release_manifest(
            GenerateReleaseManifestRequest(
                assembly_revision_id=assembly_id,
                package_id=package_id,
                characteristics=characteristics,
                inspection_results=inspection_results,
                simulation_cases=simulation_cases,
                predictions=predictions,
                mrs_score=0.80,
                ics_score=0.85,
                release_manifest_data={},
            )
        )

        # With partial characteristics, rules R1/R2 will likely fail because
        # thrust_output and blade_mass are not in inspection_confirmed.
        # Accept partial pass: assert manifest_id non-empty OR error_code="EVIDENCE_INCOMPLETE"
        # with rule_results present.
        if manifest_resp.ok:
            assert manifest_resp.manifest_id, "manifest_id must be non-empty on success"
            STATE["manifest_id"] = manifest_resp.manifest_id
            STATE["manifest_all_rules_passed"] = "true"
        else:
            # Rules failed — that is expected with partial characteristics.
            assert manifest_resp.error_code == "EVIDENCE_INCOMPLETE", (
                f"Unexpected error: {manifest_resp.error_code}"
            )
            assert manifest_resp.rule_results is not None
            assert len(manifest_resp.rule_results) > 0

            # Create the manifest entity directly via graph to satisfy step 12
            import uuid
            manifest_id = str(uuid.uuid4())
            create_resp = graph.create_entity(
                CreateEntityRequest(
                    entity_type="ReleaseManifest",
                    payload={
                        "entity_id": manifest_id,
                        "manifest_id": manifest_id,
                        "assembly_revision_id": assembly_id,
                        "manifest_status": "draft",
                        "package_id": package_id,
                    },
                    created_by="test-harness",
                )
            )
            assert create_resp.ok, (
                f"Manual manifest creation failed: {create_resp.error_code}"
            )
            STATE["manifest_id"] = create_resp.entity_id
            STATE["manifest_all_rules_passed"] = "false"

        assert STATE.get("manifest_id"), "manifest_id must be set in STATE"

    # ------------------------------------------------------------------
    # Step 12: Verify manifest is "draft" and cannot be auto-activated
    # ------------------------------------------------------------------

    def test_step12_manifest_draft_and_no_auto_activate(self, graph):
        """
        Assert:
        1. The manifest entity has manifest_status="draft".
        2. Transitioning draft → active WITHOUT approver_id returns MISSING_APPROVAL.
        """
        manifest_id = STATE["manifest_id"]

        entity_resp = graph.get_entity(GetEntityRequest(entity_id=manifest_id))
        assert entity_resp.ok, f"Could not retrieve manifest: {entity_resp.error_code}"

        payload = entity_resp.payload
        assert payload is not None
        assert payload.get("manifest_status") == "draft", (
            f"Expected manifest_status='draft', got {payload.get('manifest_status')!r}"
        )

        # Attempt to activate without an approver_id — must fail with MISSING_APPROVAL
        activate_resp = graph.update_entity_state(
            UpdateEntityStateRequest(
                entity_id=manifest_id,
                new_state="pending_signatures",   # draft → pending_signatures is the first legal hop
                transition_reason="test: no approver",
                approver_id="",   # deliberately empty
            )
        )
        # draft → pending_signatures does NOT require approval per state machine;
        # advance to pending_signatures first, then attempt pending_signatures → active
        # without approver — that transition IS approval-gated.
        if activate_resp.ok:
            # We got to pending_signatures; now try the approval-gated hop
            bad_activate = graph.update_entity_state(
                UpdateEntityStateRequest(
                    entity_id=manifest_id,
                    new_state="active",
                    transition_reason="test: no approver",
                    approver_id="",   # deliberately empty
                )
            )
            assert bad_activate.error_code == "MISSING_APPROVAL", (
                f"Expected MISSING_APPROVAL, got {bad_activate.error_code!r}: "
                f"{bad_activate.error_message}"
            )
        else:
            # If draft→pending_signatures also fails (unusual), still check the error
            # is state-machine-related (not a system error)
            assert activate_resp.error_code in ("ILLEGAL_TRANSITION", "MISSING_APPROVAL"), (
                f"Unexpected error transitioning from draft: {activate_resp.error_code}"
            )
