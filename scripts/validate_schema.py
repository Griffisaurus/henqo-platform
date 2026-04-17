#!/usr/bin/env python3
"""
Schema validation script — exits 0 if all example payloads pass validate_entity.

Run from the repo root:
    python scripts/validate_schema.py
"""
from __future__ import annotations

import sys
import os

# Add packages/schema/src to the path when running outside of an installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "schema", "src"))

from henqo_schema.validation import ValidationError, validate_entity

# ---------------------------------------------------------------------------
# Example payloads — one per registered entity type
# ---------------------------------------------------------------------------

EXAMPLES: list[tuple[str, dict]] = [
    ("Requirement", {
        "entity_id": "a1b2c3d4-e5f6-4789-ab12-cd34ef567890",
        "text": "Max tip displacement < 2 mm under 500 N axial load",
        "criticality": "High",
        "decision_class_required": "DesignGate",
    }),
    ("Requirement", {
        "entity_id": "a1b2c3d4-e5f6-4789-ab12-cd34ef567891",
        "text": "Structural requirement with inline quantity",
        "criticality": "High",
        "decision_class_required": "ReleaseCritical",
        "quantity": {"value": 2.0, "unit": "mm"},
    }),
    ("Characteristic", {
        "entity_id": "b2c3d4e5-f6a7-4890-bc23-de45fa678901",
        "name": "tip_displacement",
        "quantity_kind": "Length",
        "unit": "mm",
        "governing_requirement_id": "a1b2c3d4-e5f6-4789-ab12-cd34ef567890",
        "criticality": "High",
        "decision_class_required": "DesignGate",
    }),
    ("Prediction", {
        "entity_id": "c3d4e5f6-a7b8-4901-cd34-ef56ab789012",
        "governing_characteristic_id": "b2c3d4e5-f6a7-4890-bc23-de45fa678901",
        "surrogate_family": "S2",
        "outputs": [{"name": "tip_displacement", "value": 1.42, "unit": "mm"}],
        "trust_bundle": {
            "model_revision_id": "mr-001",
            "training_dataset_revision": "ds-rev-42",
            "weight_hash": "sha256:abcdef0123456789",
            "applicability_score": 0.91,
            "evaluated_decision_class": "DesignGate",
            "policy_version": "v0.1.0",
            "uncertainty": {"n_ensemble_members": 8},
        },
    }),
    ("SimulationCase", {
        "entity_id": "d4e5f6a7-b8c9-4012-de45-fa67bc890123",
        "discipline": "structural",
        "solver": "FEniCS",
        "solver_version": "2019.1.0",
        "component_revision_id": "e5f6a7b8-c9d0-4123-ef56-ab78cd901234",
        "status": "completed",
    }),
    ("InspectionResult", {
        "entity_id": "e5f6a7b8-c9d0-4123-ef56-ab78cd901234",
        "characteristic_id": "b2c3d4e5-f6a7-4890-bc23-de45fa678901",
        "component_revision_id": "f6a7b8c9-d0e1-4234-ab67-bc89de012345",
        "measured_value": {"value": 1.38, "unit": "mm"},
        "decision_rule": "< 2 mm",
        "measurement_uncertainty": {"value": 0.01, "unit": "mm"},
    }),
    ("ComponentRevision", {
        "entity_id": "f6a7b8c9-d0e1-4234-ab67-bc89de012345",
        "component_name": "wing_spar_assembly",
        "status": "in_design",
    }),
    ("AssemblyRevision", {
        "entity_id": "a7b8c9d0-e1f2-4345-bc78-cd90ef123456",
        "assembly_name": "propulsion_pod",
        "status": "manufacturing_reviewed",
    }),
    ("ProvenanceBundle", {
        "bundle_id": "pb-001",
        "activity": "simulation_run",
        "agent_id": "sim-agent-sa-001",
    }),
    ("ReleaseManifest", {
        "manifest_id": "man-001",
        "assembly_revision_id": "a7b8c9d0-e1f2-4345-bc78-cd90ef123456",
        "manifest_status": "draft",
    }),
    ("DecisionPackage", {
        "package_id": "dpk-001",
        "assembly_revision_id": "a7b8c9d0-e1f2-4345-bc78-cd90ef123456",
        "review_gate": "CDR",
    }),
    ("SurrogateModel", {
        "model_id": "sm-s2-001",
        "surrogate_family": "S2",
        "weight_hash": "sha256:abcdef0123456789abcdef0123456789",
        "weight_format": "safetensors",
        "status": "production",
    }),
    ("TrainingJob", {
        "job_id": "tj-001",
        "surrogate_family": "S2",
        "training_dataset_revision": "ds-rev-42",
        "state": "completed",
    }),
    ("OrchestratorSession", {
        "session_id": "os-001",
        "role": "SE",
        "orchestrator_model_id": "llama-3.1-8b-instruct-awq",
        "policy_version": "v0.1.0",
    }),
    ("SimulationCampaign", {
        "campaign_id": "sc-001",
        "surrogate_family": "S1",
        "status": "planned",
    }),
    ("OodRegion", {
        "region_id": "ood-001",
        "surrogate_family": "S1",
        "model_revision_id": "mr-s1-003",
    }),
    ("RetrainingRequest", {
        "request_id": "rr-001",
        "surrogate_family": "S1",
        "trigger_type": "campaign_merged",
    }),
]


def main() -> int:
    failures: list[str] = []

    for entity_type, payload in EXAMPLES:
        try:
            validate_entity(entity_type, payload)
            print(f"  PASS  {entity_type}")
        except ValidationError as exc:
            msg = f"  FAIL  {entity_type}: [{exc.error_code}] field={exc.field!r} — {exc.message}"
            print(msg)
            failures.append(msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"  ERROR {entity_type}: {type(exc).__name__}: {exc}"
            print(msg)
            failures.append(msg)

    print()
    if failures:
        print(f"schema validation FAILED — {len(failures)} of {len(EXAMPLES)} examples failed:")
        for f in failures:
            print(f"  {f}")
        return 1

    print(f"schema validation PASSED — all {len(EXAMPLES)} examples validated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
