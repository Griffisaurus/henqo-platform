"""
Tests for decision_pkg_service.domain.release_rules (EP-09).

≥12 tests — at least one per rule (R1–R10).
"""
from __future__ import annotations

import pytest

from decision_pkg_service.domain.release_rules import RuleResult, check_release_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASSEMBLY_ID = "asm-rev-001"


def _char(
    char_id: str,
    status: str = "inspection_confirmed",
    criticality: str = "Standard",
    decision_class_required: str = "DesignGate",
    inspect_required: bool = False,
) -> dict:
    return {
        "entity_id": char_id,
        "status": status,
        "criticality": criticality,
        "decision_class_required": decision_class_required,
        "inspect_required": inspect_required,
    }


def _inspection(ir_id: str, char_id: str, status: str = "pass") -> dict:
    return {"entity_id": ir_id, "characteristic_id": char_id, "status": status}


def _sim_case(sc_id: str, char_id: str, status: str = "validated") -> dict:
    return {"entity_id": sc_id, "characteristic_id": char_id, "status": status}


def _prediction(pred_id: str, char_id: str, dc: str = "ReleaseCritical", stale: bool = False) -> dict:
    return {
        "entity_id": pred_id,
        "governing_characteristic_id": char_id,
        "trust_bundle": {"evaluated_decision_class": dc},
        "stale": stale,
    }


def _manifest_data(
    open_issues: list | None = None,
    required_signatories: list | None = None,
    actual_signatories: list | None = None,
) -> dict:
    return {
        "open_issues": open_issues or [],
        "required_signatories": required_signatories or ["chief_engineer", "quality_lead"],
        "actual_signatories": actual_signatories or ["chief_engineer", "quality_lead"],
    }


def _all_pass_inputs():
    """Returns inputs where all 10 rules pass."""
    chars = [
        _char("CH-001", status="inspection_confirmed", criticality="Critical", inspect_required=True),
        _char("CH-002", status="released", criticality="key", inspect_required=True),
    ]
    inspections = [
        _inspection("IR-001", "CH-001", status="pass"),
        _inspection("IR-002", "CH-002", status="pass"),
    ]
    sim_cases = [
        _sim_case("SC-001", "CH-001", status="validated"),
    ]
    predictions: list[dict] = []
    manifest_data = _manifest_data()
    return dict(
        assembly_revision_id=ASSEMBLY_ID,
        characteristics=chars,
        inspection_results=inspections,
        simulation_cases=sim_cases,
        predictions=predictions,
        release_manifest_data=manifest_data,
        mrs_score=0.85,
        ics_score=0.90,
    )


def _run(**kwargs) -> list[RuleResult]:
    defaults = _all_pass_inputs()
    defaults.update(kwargs)
    return check_release_rules(**defaults)


def _get_rule(results: list[RuleResult], rule_id: str) -> RuleResult:
    for r in results:
        if r.rule_id == rule_id:
            return r
    raise KeyError(f"Rule {rule_id} not found in results")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllRulesPass:
    def test_all_10_rules_pass(self):
        results = check_release_rules(**_all_pass_inputs())
        assert len(results) == 10
        for r in results:
            assert r.passed is True, f"Rule {r.rule_id} unexpectedly failed: {r.reason}"

    def test_returns_exactly_10_results(self):
        results = check_release_rules(**_all_pass_inputs())
        rule_ids = {r.rule_id for r in results}
        assert rule_ids == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"}


class TestR1CriticalCharsStatus:
    def test_r1_fails_when_critical_char_not_confirmed(self):
        chars = [_char("CH-001", status="simulation_validated", criticality="Critical")]
        inspections = [_inspection("IR-001", "CH-001", "pass")]
        results = _run(characteristics=chars, inspection_results=inspections)
        r1 = _get_rule(results, "R1")
        assert r1.passed is False
        assert "CH-001" in r1.reason

    def test_r1_passes_when_critical_char_is_released(self):
        chars = [_char("CH-001", status="released", criticality="Critical")]
        inspections = [_inspection("IR-001", "CH-001", "pass")]
        results = _run(characteristics=chars, inspection_results=inspections)
        r1 = _get_rule(results, "R1")
        assert r1.passed is True


class TestR2InspectionEvidence:
    def test_r2_fails_when_critical_char_missing_pass_inspection(self):
        chars = [_char("CH-001", status="inspection_confirmed", criticality="Critical")]
        inspections: list[dict] = []  # no inspections
        results = _run(characteristics=chars, inspection_results=inspections)
        r2 = _get_rule(results, "R2")
        assert r2.passed is False
        assert "CH-001" in r2.reason

    def test_r2_fails_when_inspection_status_not_pass(self):
        chars = [_char("CH-001", status="inspection_confirmed", criticality="key")]
        inspections = [_inspection("IR-001", "CH-001", status="fail")]
        results = _run(characteristics=chars, inspection_results=inspections)
        r2 = _get_rule(results, "R2")
        assert r2.passed is False


class TestR3MRS:
    def test_r3_fails_when_mrs_below_target(self):
        results = _run(mrs_score=0.60)
        r3 = _get_rule(results, "R3")
        assert r3.passed is False
        assert "0.600" in r3.reason

    def test_r3_passes_when_mrs_at_threshold(self):
        results = _run(mrs_score=0.70)
        r3 = _get_rule(results, "R3")
        assert r3.passed is True

    def test_r3_passes_when_mrs_above_target(self):
        results = _run(mrs_score=1.0)
        r3 = _get_rule(results, "R3")
        assert r3.passed is True


class TestR4ICS:
    def test_r4_fails_when_ics_below_threshold(self):
        results = _run(ics_score=0.50)
        r4 = _get_rule(results, "R4")
        assert r4.passed is False
        assert "0.500" in r4.reason

    def test_r4_passes_when_ics_at_threshold(self):
        results = _run(ics_score=0.80)
        r4 = _get_rule(results, "R4")
        assert r4.passed is True


class TestR5SimulationCases:
    def test_r5_fails_when_sim_case_not_validated(self):
        # CH-001 is released, its sim case is only "completed" not "validated"
        chars = [_char("CH-001", status="released", criticality="Critical")]
        inspections = [_inspection("IR-001", "CH-001", "pass")]
        sim_cases = [_sim_case("SC-001", "CH-001", status="completed")]
        results = _run(characteristics=chars, inspection_results=inspections, simulation_cases=sim_cases)
        r5 = _get_rule(results, "R5")
        assert r5.passed is False
        assert "SC-001" in r5.reason


class TestR6StalePredictions:
    def test_r6_fails_when_stale_prediction_present(self):
        preds = [_prediction("P-001", "CH-001", stale=True)]
        results = _run(predictions=preds)
        r6 = _get_rule(results, "R6")
        assert r6.passed is False
        assert "P-001" in r6.reason

    def test_r6_passes_when_no_stale_predictions(self):
        preds = [_prediction("P-001", "CH-001", stale=False)]
        results = _run(predictions=preds)
        r6 = _get_rule(results, "R6")
        assert r6.passed is True


class TestR7InspectionRequired:
    def test_r7_fails_when_inspect_required_char_has_no_inspection(self):
        chars = [_char("CH-999", status="inspection_confirmed", inspect_required=True)]
        inspections: list[dict] = []
        results = _run(characteristics=chars, inspection_results=inspections)
        r7 = _get_rule(results, "R7")
        assert r7.passed is False
        assert "CH-999" in r7.reason


class TestR8SurrogateEvidence:
    def test_r8_fails_when_release_critical_char_has_only_design_gate_prediction(self):
        chars = [
            _char("CH-001", status="inspection_confirmed", criticality="Standard",
                  decision_class_required="ReleaseCritical"),
        ]
        preds = [_prediction("P-001", "CH-001", dc="DesignGate")]
        # No sim or inspection evidence for CH-001
        results = _run(
            characteristics=chars,
            inspection_results=[],
            simulation_cases=[],
            predictions=preds,
        )
        r8 = _get_rule(results, "R8")
        assert r8.passed is False
        assert "CH-001" in r8.reason

    def test_r8_passes_when_release_critical_char_has_inspection(self):
        chars = [
            _char("CH-001", status="inspection_confirmed", criticality="Critical",
                  decision_class_required="ReleaseCritical", inspect_required=True),
        ]
        inspections = [_inspection("IR-001", "CH-001", "pass")]
        preds = [_prediction("P-001", "CH-001", dc="DesignGate")]
        results = _run(
            characteristics=chars,
            inspection_results=inspections,
            simulation_cases=[],
            predictions=preds,
        )
        r8 = _get_rule(results, "R8")
        assert r8.passed is True


class TestR9OpenIssues:
    def test_r9_fails_when_open_issues_present(self):
        manifest_data = _manifest_data(open_issues=["ISSUE-42"])
        results = _run(release_manifest_data=manifest_data)
        r9 = _get_rule(results, "R9")
        assert r9.passed is False
        assert "ISSUE-42" in r9.reason

    def test_r9_passes_when_no_open_issues(self):
        manifest_data = _manifest_data(open_issues=[])
        results = _run(release_manifest_data=manifest_data)
        r9 = _get_rule(results, "R9")
        assert r9.passed is True


class TestR10Signatories:
    def test_r10_fails_when_signatory_missing(self):
        manifest_data = _manifest_data(
            required_signatories=["chief_engineer", "quality_lead"],
            actual_signatories=["chief_engineer"],  # quality_lead missing
        )
        results = _run(release_manifest_data=manifest_data)
        r10 = _get_rule(results, "R10")
        assert r10.passed is False
        assert "quality_lead" in r10.reason

    def test_r10_passes_when_all_signed(self):
        manifest_data = _manifest_data(
            required_signatories=["chief_engineer", "quality_lead"],
            actual_signatories=["chief_engineer", "quality_lead"],
        )
        results = _run(release_manifest_data=manifest_data)
        r10 = _get_rule(results, "R10")
        assert r10.passed is True
