"""
Unit tests for surrogate_service.domain.trust_bundle.

Maps eval-acceptance-spec.md §7 SF-001 through SF-005 to unit tests,
plus additional coverage for every hard fallback rule.
"""
from __future__ import annotations

import pytest

from surrogate_service.domain.calibration import CalibrationResult
from surrogate_service.domain.trust_bundle import (
    RULE_APPLICABILITY_BELOW_THRESHOLD,
    RULE_ENSEMBLE_TOO_SMALL,
    RULE_MODEL_FROZEN,
    RULE_RELEASE_CRITICAL_NO_SURROGATE,
    RULE_SAFETY_CRITICAL_NO_SURROGATE,
    RULE_CALIBRATION_FAILED,
    TrustBundleEvaluation,
    evaluate_trust_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _passing_cal(coverage: float = 0.90) -> CalibrationResult:
    return CalibrationResult(
        coverage_achieved=coverage,
        target_coverage=0.90,
        conformal_quantile=1.645,
        interval_width_mean=0.1,
        passed=True,
    )


def _failing_cal(coverage: float = 0.80) -> CalibrationResult:
    return CalibrationResult(
        coverage_achieved=coverage,
        target_coverage=0.90,
        conformal_quantile=1.645,
        interval_width_mean=0.1,
        passed=False,
    )


def _base_kwargs(**overrides):
    defaults = dict(
        applicability_score=0.92,
        a_density=0.95,
        a_range=0.90,
        a_ensemble=0.91,
        n_ensemble_members=8,
        model_revision_id="rev-001",
        training_dataset_revision="ds-rev-001",
        weight_hash="sha256-abc",
        calibration=_passing_cal(),
        requested_decision_class="Exploratory",
        model_frozen=False,
        policy_version="v0.1.0",
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# SF-001: High applicability → not abstained, Exploratory class
# eval-acceptance-spec.md §7 SF-001
# ---------------------------------------------------------------------------

class TestSF001HighApplicability:
    def test_not_abstained_exploratory(self):
        """A(x) = 0.92 ≥ 0.80 → Exploratory, not abstained."""
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.92,
            requested_decision_class="Exploratory",
        ))
        assert not result.abstain
        assert result.evaluated_decision_class == "Exploratory"
        assert result.abstain_reason == ""

    def test_design_gate_met(self):
        """A(x) = 0.92 ≥ 0.90 when requesting DesignGate → DesignGate."""
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.92,
            requested_decision_class="DesignGate",
        ))
        assert not result.abstain
        assert result.evaluated_decision_class == "DesignGate"

    def test_fields_populated(self):
        result = evaluate_trust_bundle(**_base_kwargs())
        assert result.model_revision_id == "rev-001"
        assert result.weight_hash == "sha256-abc"
        assert result.policy_version == "v0.1.0"


# ---------------------------------------------------------------------------
# SF-002: Applicability below DesignGate threshold → abstained
# eval-acceptance-spec.md §7 SF-002 (also covers SF-001 shadow case)
# Spec: applicability_score = 0.75 for Exploratory → Blocked
# Also: 0.85 for DesignGate (below 0.90) → Blocked
# ---------------------------------------------------------------------------

class TestSF002ApplicabilityBelowThreshold:
    def test_below_exploratory_threshold(self):
        """
        SF-001 spec case: score=0.75 < 0.80 → Blocked, abstained.
        """
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.75,
            requested_decision_class="Exploratory",
        ))
        assert result.abstain
        assert result.evaluated_decision_class == "Blocked"
        assert RULE_APPLICABILITY_BELOW_THRESHOLD in result.triggered_rules

    def test_below_design_gate_threshold(self):
        """
        SF-002 spec case: score=0.85 < 0.90 when DesignGate requested → Blocked.
        """
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.85,
            requested_decision_class="DesignGate",
        ))
        assert result.abstain
        assert result.evaluated_decision_class == "Blocked"
        assert RULE_APPLICABILITY_BELOW_THRESHOLD in result.triggered_rules

    def test_characteristic_state_unchanged_implied(self):
        """
        Abstained prediction must carry abstain=True and a non-empty reason.
        """
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.75,
            requested_decision_class="Exploratory",
        ))
        assert result.abstain_reason != ""


# ---------------------------------------------------------------------------
# SF-003: n_ensemble_members < 5 → hard block
# eval-acceptance-spec.md §7 SF-003
# ---------------------------------------------------------------------------

class TestSF003EnsembleTooSmall:
    def test_four_members_blocked(self):
        result = evaluate_trust_bundle(**_base_kwargs(n_ensemble_members=4))
        assert result.abstain
        assert result.evaluated_decision_class == "Blocked"
        assert RULE_ENSEMBLE_TOO_SMALL in result.triggered_rules

    def test_five_members_ok(self):
        result = evaluate_trust_bundle(**_base_kwargs(n_ensemble_members=5))
        assert not result.abstain

    def test_zero_members_blocked(self):
        result = evaluate_trust_bundle(**_base_kwargs(n_ensemble_members=0))
        assert result.abstain
        assert RULE_ENSEMBLE_TOO_SMALL in result.triggered_rules


# ---------------------------------------------------------------------------
# SF-004: ReleaseCritical requested → always abstained
# eval-acceptance-spec.md §7 SF-004
# ---------------------------------------------------------------------------

class TestSF004ReleaseCritical:
    def test_release_critical_always_abstains(self):
        """Surrogates always abstain for ReleaseCritical (requires simulation)."""
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.99,
            requested_decision_class="ReleaseCritical",
        ))
        assert result.abstain
        assert result.evaluated_decision_class == "Blocked"
        assert RULE_RELEASE_CRITICAL_NO_SURROGATE in result.triggered_rules

    def test_safety_critical_always_abstains(self):
        """Surrogates always abstain for SafetyCritical."""
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.99,
            requested_decision_class="SafetyCritical",
        ))
        assert result.abstain
        assert result.evaluated_decision_class == "Blocked"
        assert RULE_SAFETY_CRITICAL_NO_SURROGATE in result.triggered_rules


# ---------------------------------------------------------------------------
# SF-005: model_frozen=True → abstained
# eval-acceptance-spec.md §7 SF-005
# ---------------------------------------------------------------------------

class TestSF005ModelFrozen:
    def test_frozen_model_abstains(self):
        result = evaluate_trust_bundle(**_base_kwargs(model_frozen=True))
        assert result.abstain
        assert result.evaluated_decision_class == "Blocked"
        assert RULE_MODEL_FROZEN in result.triggered_rules

    def test_non_frozen_passes(self):
        result = evaluate_trust_bundle(**_base_kwargs(model_frozen=False))
        assert not result.abstain


# ---------------------------------------------------------------------------
# Additional: calibration failure → abstained
# ---------------------------------------------------------------------------

class TestCalibrationFailure:
    def test_calibration_failed_abstains(self):
        result = evaluate_trust_bundle(**_base_kwargs(
            calibration=_failing_cal(0.80),
        ))
        assert result.abstain
        assert RULE_CALIBRATION_FAILED in result.triggered_rules

    def test_calibration_passed_no_block(self):
        result = evaluate_trust_bundle(**_base_kwargs(
            calibration=_passing_cal(0.92),
        ))
        assert not result.abstain


# ---------------------------------------------------------------------------
# Multiple rules can trigger simultaneously
# ---------------------------------------------------------------------------

class TestMultipleRules:
    def test_ensemble_and_applicability_both_trigger(self):
        result = evaluate_trust_bundle(**_base_kwargs(
            n_ensemble_members=3,
            applicability_score=0.70,
            requested_decision_class="Exploratory",
        ))
        assert result.abstain
        assert RULE_ENSEMBLE_TOO_SMALL in result.triggered_rules
        assert RULE_APPLICABILITY_BELOW_THRESHOLD in result.triggered_rules

    def test_trust_bundle_evaluation_is_deterministic(self):
        kwargs = _base_kwargs()
        r1 = evaluate_trust_bundle(**kwargs)
        r2 = evaluate_trust_bundle(**kwargs)
        assert r1.evaluated_decision_class == r2.evaluated_decision_class
        assert r1.abstain == r2.abstain


# ---------------------------------------------------------------------------
# Clamp-to-requested-class logic: score >= DesignGate but request=Exploratory
# ---------------------------------------------------------------------------

class TestClassClamp:
    def test_high_score_clamped_to_exploratory_when_requested(self):
        """
        A(x) = 0.95 qualifies for DesignGate, but if request=Exploratory,
        evaluated_class should be capped at Exploratory.
        """
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.95,
            requested_decision_class="Exploratory",
        ))
        assert not result.abstain
        assert result.evaluated_decision_class == "Exploratory"

    def test_score_between_exploratory_and_design_gate(self):
        """A(x) = 0.85: qualifies Exploratory, not DesignGate."""
        result = evaluate_trust_bundle(**_base_kwargs(
            applicability_score=0.85,
            requested_decision_class="Exploratory",
        ))
        assert not result.abstain
        assert result.evaluated_decision_class == "Exploratory"
