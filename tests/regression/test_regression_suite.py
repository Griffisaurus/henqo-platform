"""
EP-10: Regression smoke-test suite.

One smoke-test assertion per service domain function.  These tests exercise
the key domain logic independently of the full E2E workflow and serve as a
fast sanity-check that no internal regressions have been introduced.

Run with:
  cd /home/workbench/henqo-platform
  PYTHONPATH=packages/schema/src:services/graph-service/src:services/artifact-service/src:\
    services/requirements-service/src:services/design-service/src:\
    services/sim-job-service/src:services/surrogate-service/src:\
    services/mfg-service/src:services/decision-pkg-service/src \
    .venv/bin/pytest tests/regression/ -v --tb=short
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Schema validation smoke
# ---------------------------------------------------------------------------

def test_schema_validation_smoke():
    """validate_entity('Requirement', ...) must accept a well-formed payload."""
    from henqo_schema.validation import validate_entity, ValidationError

    payload = {
        "entity_id": "11111111-1111-4111-a111-111111111111",
        "text": "Thrust shall be >= 500 N at rated conditions",
        "criticality": "High",
        "decision_class_required": "ReleaseCritical",
    }
    # Should not raise
    validate_entity("Requirement", payload)

    # A missing required field must raise
    with pytest.raises(ValidationError):
        validate_entity("Requirement", {"entity_id": "abc"})


# ---------------------------------------------------------------------------
# 2. State machine smoke
# ---------------------------------------------------------------------------

def test_state_machine_smoke():
    """check_transition must allow unverified → surrogate_estimated for Characteristic."""
    from graph_service.domain.state_machine import (
        check_transition,
        IllegalTransitionError,
        MissingApprovalError,
    )

    # Legal transition — should not raise
    check_transition("Characteristic", "unverified", "surrogate_estimated")

    # Illegal transition — must raise
    with pytest.raises(IllegalTransitionError):
        check_transition("Characteristic", "unverified", "released")

    # Approval-gated transition without approver — must raise
    with pytest.raises(MissingApprovalError):
        check_transition("Characteristic", "inspection_confirmed", "released")

    # Approval-gated transition WITH approver — should not raise
    check_transition(
        "Characteristic", "inspection_confirmed", "released",
        approver_id="lead-engineer-01",
    )


# ---------------------------------------------------------------------------
# 3. Applicability score smoke
# ---------------------------------------------------------------------------

def test_applicability_score_smoke():
    """compute_applicability_score must return a score > 0 for in-range inputs."""
    from surrogate_service.domain.applicability import compute_applicability_score

    x = {"chord_mm": 120.0, "thickness_ratio": 0.12}
    training_data = [
        {"chord_mm": 110.0, "thickness_ratio": 0.11},
        {"chord_mm": 125.0, "thickness_ratio": 0.13},
    ]
    feature_bounds = {
        "chord_mm": (50.0, 200.0),
        "thickness_ratio": (0.08, 0.20),
    }
    ensemble_outputs = [1.40, 1.38, 1.42, 1.41, 1.39, 1.43, 1.37, 1.40]

    a_total, a_density, a_range, a_ensemble = compute_applicability_score(
        x=x,
        training_data=training_data,
        feature_bounds=feature_bounds,
        ensemble_outputs=ensemble_outputs,
    )

    assert 0.0 <= a_total <= 1.0, f"a_total out of range: {a_total}"
    assert a_range == 1.0, "All features in-range → a_range should be 1.0"
    assert a_ensemble > 0.0, "Tight ensemble spread → a_ensemble must be > 0"
    assert a_total > 0.0, "Overall applicability score must be > 0"


# ---------------------------------------------------------------------------
# 4. DFM rules smoke
# ---------------------------------------------------------------------------

def test_dfm_rules_smoke():
    """evaluate_all_rules must return no Class A violations for safe geometry."""
    from mfg_service.domain.dfm_rules import evaluate_all_rules

    # Geometry well within CNC tolerances
    component_data = {
        "min_feature_size_mm": 2.0,
        "depth_to_width_ratio": 1.5,
        "wall_thickness_mm": 3.0,
        "tool_diameter_mm": 6.0,
    }
    violations = evaluate_all_rules(component_data, process_families=["cnc"])

    class_a = [v for v in violations if v.tier == "A"]
    assert len(class_a) == 0, (
        f"Expected no Class A violations for safe geometry, got: {class_a}"
    )


# ---------------------------------------------------------------------------
# 5. Tolerance stack smoke
# ---------------------------------------------------------------------------

def test_tolerance_stack_smoke():
    """compute_worst_case must return a ToleranceStackResult with passes_spec correct."""
    from mfg_service.domain.tolerance_stack import (
        ToleranceContributor,
        compute_worst_case,
    )

    contributors = [
        ToleranceContributor(name="blade_root_bore", nominal=10.0, tolerance=0.05),
        ToleranceContributor(name="hub_shaft",       nominal=10.0, tolerance=0.03),
    ]
    result = compute_worst_case(contributors, spec_min_gap=0.0)

    assert result.method == "worst_case"
    assert result.gap_worst_case >= 0.0


# ---------------------------------------------------------------------------
# 6. MRS smoke
# ---------------------------------------------------------------------------

def test_mrs_smoke():
    """compute_mrs on an empty violations list must return MRS = 1.0."""
    from mfg_service.domain.mrs import compute_mrs

    result = compute_mrs(violations=[], resolved_rule_ids=[])

    assert result.mrs_score == pytest.approx(1.0), (
        f"MRS for zero violations should be 1.0, got {result.mrs_score}"
    )
    assert result.s_a == 1.0
    assert result.open_a_count == 0


# ---------------------------------------------------------------------------
# 7. Completeness smoke
# ---------------------------------------------------------------------------

def test_completeness_smoke():
    """check_completeness must return complete=False for uncovered requirements."""
    from decision_pkg_service.domain.completeness import (
        EvidenceItem,
        check_completeness,
    )

    requirements = [{"entity_id": "req-001"}, {"entity_id": "req-002"}]
    characteristics = [
        {
            "entity_id": "char-001",
            "governing_requirement_id": "req-001",
            "criticality": "Medium",
        }
    ]
    # Only req-001 is covered; req-002 has no evidence
    evidence_items = [
        EvidenceItem(
            characteristic_id="char-001",
            evidence_type="Prediction",
            evidence_entity_id="pred-001",
            status="provisionally_supportive",
            decision_class="DesignGate",
        )
    ]

    result = check_completeness(
        requirements=requirements,
        characteristics=characteristics,
        evidence_items=evidence_items,
        review_gate="PDR",
    )

    assert not result.complete, "Should be incomplete: req-002 has no evidence"
    assert "req-002" in result.uncovered_requirements
    assert "req-001" in result.covered_requirements


# ---------------------------------------------------------------------------
# 8. Release rules smoke
# ---------------------------------------------------------------------------

def test_release_rules_smoke():
    """check_release_rules must pass R3 and R4 when mrs/ics scores are above threshold."""
    from decision_pkg_service.domain.release_rules import check_release_rules

    results = check_release_rules(
        assembly_revision_id="asm-001",
        characteristics=[],
        inspection_results=[],
        simulation_cases=[],
        predictions=[],
        release_manifest_data={},
        mrs_score=0.95,
        ics_score=0.90,
    )

    rule_map = {r.rule_id: r for r in results}
    assert rule_map["R3"].passed, f"R3 should pass with mrs=0.95: {rule_map['R3'].reason}"
    assert rule_map["R4"].passed, f"R4 should pass with ics=0.90: {rule_map['R4'].reason}"
    assert rule_map["R9"].passed, "R9 should pass with no open issues"
    assert rule_map["R10"].passed, "R10 should pass with no required signatories"


# ---------------------------------------------------------------------------
# 9. Staleness smoke
# ---------------------------------------------------------------------------

def test_staleness_smoke():
    """is_prediction_stale must return True for a 2020 timestamp."""
    from decision_pkg_service.domain.staleness import is_prediction_stale

    # 2020-01-01 is more than 365 days ago relative to 2026
    assert is_prediction_stale("2020-01-01T00:00:00+00:00"), (
        "A 2020 prediction must be stale in 2026"
    )

    # A timestamp from today (or very recent) must not be stale
    from datetime import datetime, timezone
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    assert not is_prediction_stale(now_iso), (
        "A prediction created right now must not be stale"
    )


# ---------------------------------------------------------------------------
# 10. Signatory smoke
# ---------------------------------------------------------------------------

def test_signatory_smoke():
    """SignatoryWorkflow must correctly track required roles."""
    from decision_pkg_service.domain.signatory import (
        SignatoryRecord,
        SignatoryWorkflow,
    )

    workflow = SignatoryWorkflow(required_roles=["SE", "QA"])

    # Not complete yet
    assert not workflow.is_complete()
    assert set(workflow.missing_roles()) == {"SE", "QA"}

    # Add SE signature
    from datetime import datetime, timezone
    workflow.add_signature(
        SignatoryRecord(
            signatory_id="eng-001",
            role="SE",
            signed_at=datetime.now(tz=timezone.utc).isoformat(),
            signature="sig-hash-se",
        )
    )
    assert not workflow.is_complete()
    assert workflow.missing_roles() == ["QA"]

    # Add QA signature
    workflow.add_signature(
        SignatoryRecord(
            signatory_id="qa-001",
            role="QA",
            signed_at=datetime.now(tz=timezone.utc).isoformat(),
            signature="sig-hash-qa",
        )
    )
    assert workflow.is_complete(), "Workflow should be complete after both roles signed"
    assert workflow.missing_roles() == []
