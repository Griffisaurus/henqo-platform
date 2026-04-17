"""
TrustBundle evaluation and fallback rule enforcement.

Implements all 7 fallback trigger rules from surrogate-trust-policy.md §5
and the four decision-class auto-accept conditions from §3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from surrogate_service.domain.calibration import CalibrationResult
from surrogate_service.domain.router import get_decision_class_threshold


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrustBundleEvaluation:
    model_revision_id: str
    training_dataset_revision: str
    weight_hash: str
    applicability_score: float
    a_density: float
    a_range: float
    a_ensemble: float
    n_ensemble_members: int
    evaluated_decision_class: str       # "Exploratory"|"DesignGate"|"ReleaseCritical"|"SafetyCritical"|"Blocked"
    abstain: bool
    abstain_reason: str                 # empty string when not abstaining
    policy_version: str
    calibration_coverage: float
    triggered_rules: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule names (constants for auditability)
# ---------------------------------------------------------------------------

RULE_APPLICABILITY_BELOW_THRESHOLD = "RULE_1_APPLICABILITY_BELOW_THRESHOLD"
RULE_INTERVAL_WIDTH_TOO_WIDE = "RULE_2_INTERVAL_WIDTH_TOO_WIDE"
RULE_ROLLOUT_UNSTABLE = "RULE_3_ROLLOUT_UNSTABLE"
RULE_RECENT_BENCHMARK_FAILURE = "RULE_4_RECENT_BENCHMARK_FAILURE"
RULE_SPOT_CHECK_FAILURE_RATE = "RULE_5_SPOT_CHECK_FAILURE_RATE"
RULE_BENCHMARK_EXPIRED = "RULE_6_BENCHMARK_EXPIRED"
RULE_SCHEMA_VERSION_MISMATCH = "RULE_7_SCHEMA_VERSION_MISMATCH"
RULE_ENSEMBLE_TOO_SMALL = "RULE_HARD_ENSEMBLE_TOO_SMALL"
RULE_MODEL_FROZEN = "RULE_HARD_MODEL_FROZEN"
RULE_RELEASE_CRITICAL_NO_SURROGATE = "RULE_HARD_RELEASE_CRITICAL_NO_SURROGATE"
RULE_SAFETY_CRITICAL_NO_SURROGATE = "RULE_HARD_SAFETY_CRITICAL_NO_SURROGATE"
RULE_CALIBRATION_FAILED = "RULE_HARD_CALIBRATION_FAILED"
RULE_OOD_REGION = "RULE_OOD_REGION_DETECTED"


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate_trust_bundle(
    applicability_score: float,
    a_density: float,
    a_range: float,
    a_ensemble: float,
    n_ensemble_members: int,
    model_revision_id: str,
    training_dataset_revision: str,
    weight_hash: str,
    calibration: CalibrationResult,
    requested_decision_class: str,
    model_frozen: bool = False,
    policy_version: str = "v0.1.0",
) -> TrustBundleEvaluation:
    """
    Apply all fallback rules and determine the evaluated decision class.

    Sets abstain=True and evaluated_decision_class="Blocked" if any hard
    rule fires, the applicability score is below threshold, calibration
    failed, or the requested class is ReleaseCritical/SafetyCritical
    (surrogates always abstain for those classes).

    Otherwise returns the highest permissible decision class given A(x).
    """
    triggered: list[str] = []
    abstain_reasons: list[str] = []

    # ------------------------------------------------------------------
    # Hard rule: ensemble too small (surrogate-trust-policy.md §4)
    # ------------------------------------------------------------------
    if n_ensemble_members < 5:
        triggered.append(RULE_ENSEMBLE_TOO_SMALL)
        abstain_reasons.append(
            f"n_ensemble_members={n_ensemble_members} < 5 (hard minimum)"
        )

    # ------------------------------------------------------------------
    # Hard rule: model frozen (schema mismatch / §8 freeze semantics)
    # ------------------------------------------------------------------
    if model_frozen:
        triggered.append(RULE_MODEL_FROZEN)
        abstain_reasons.append("Model is frozen; re-benchmarking required")

    # ------------------------------------------------------------------
    # Hard rule: ReleaseCritical / SafetyCritical always abstain
    # ------------------------------------------------------------------
    if requested_decision_class == "ReleaseCritical":
        triggered.append(RULE_RELEASE_CRITICAL_NO_SURROGATE)
        abstain_reasons.append(
            "ReleaseCritical class requires simulation + CE approval; "
            "surrogates always abstain"
        )
    elif requested_decision_class == "SafetyCritical":
        triggered.append(RULE_SAFETY_CRITICAL_NO_SURROGATE)
        abstain_reasons.append(
            "SafetyCritical class requires formal evidence; "
            "surrogates always abstain"
        )

    # ------------------------------------------------------------------
    # Hard rule: calibration failed
    # ------------------------------------------------------------------
    if not calibration.passed:
        triggered.append(RULE_CALIBRATION_FAILED)
        abstain_reasons.append(
            f"Calibration coverage {calibration.coverage_achieved:.3f} is below "
            f"target {calibration.target_coverage:.3f} - 0.02 = "
            f"{calibration.target_coverage - 0.02:.3f}"
        )

    # ------------------------------------------------------------------
    # OOD region: a_density below 0.3 (additional diagnostic rule)
    # ------------------------------------------------------------------
    if a_density < 0.3:
        triggered.append(RULE_OOD_REGION)
        # Not by itself a hard block but contributes to abstention when
        # combined with applicability threshold failure below.

    # ------------------------------------------------------------------
    # Rule 1: applicability score below class threshold
    # ------------------------------------------------------------------
    # Evaluate for the requested class (or best achievable class)
    if requested_decision_class in ("Exploratory", "DesignGate"):
        threshold = get_decision_class_threshold(requested_decision_class)
        if applicability_score < threshold:
            triggered.append(RULE_APPLICABILITY_BELOW_THRESHOLD)
            abstain_reasons.append(
                f"A(x)={applicability_score:.4f} < threshold={threshold:.2f} "
                f"for {requested_decision_class}"
            )

    # ------------------------------------------------------------------
    # Determine final evaluated class and abstain flag
    # ------------------------------------------------------------------
    must_abstain = bool(abstain_reasons)

    if must_abstain:
        evaluated_class = "Blocked"
        abstain_flag = True
        abstain_reason = "; ".join(abstain_reasons)
    else:
        # Determine highest permissible class for the given A(x)
        if applicability_score >= 0.90:
            evaluated_class = "DesignGate"
        elif applicability_score >= 0.80:
            evaluated_class = "Exploratory"
        else:
            # Should have been caught above, but be defensive
            evaluated_class = "Blocked"
            abstain_flag = True
            abstain_reason = (
                f"A(x)={applicability_score:.4f} below Exploratory threshold 0.80"
            )

        # Clamp down to requested class (do not exceed what was requested)
        _order = ["Exploratory", "DesignGate", "ReleaseCritical", "SafetyCritical"]
        if requested_decision_class in _order and evaluated_class in _order:
            req_idx = _order.index(requested_decision_class)
            eval_idx = _order.index(evaluated_class)
            if eval_idx > req_idx:
                evaluated_class = requested_decision_class
        abstain_flag = False
        abstain_reason = ""

    return TrustBundleEvaluation(
        model_revision_id=model_revision_id,
        training_dataset_revision=training_dataset_revision,
        weight_hash=weight_hash,
        applicability_score=applicability_score,
        a_density=a_density,
        a_range=a_range,
        a_ensemble=a_ensemble,
        n_ensemble_members=n_ensemble_members,
        evaluated_decision_class=evaluated_class,
        abstain=abstain_flag,
        abstain_reason=abstain_reason,
        policy_version=policy_version,
        calibration_coverage=calibration.coverage_achieved,
        triggered_rules=triggered,
    )
