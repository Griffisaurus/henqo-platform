"""
EP-09: Release readiness rules engine.

Implements the 10 evidence completeness rules from
manufacturability-subsystem-spec.md §10.

Rules are evaluated independently; all 10 must pass for release.
"""
from __future__ import annotations

from dataclasses import dataclass

# Default thresholds (overridable in production config)
_DEFAULT_MRS_TARGET: float = 0.70
_DEFAULT_ICS_THRESHOLD: float = 0.80

# Decision class ordering for surrogate trust checks
_DECISION_CLASS_ORDER: list[str] = [
    "Exploratory",
    "DesignGate",
    "ReleaseCritical",
    "SafetyCritical",
]


def _decision_class_level(dc: str) -> int:
    try:
        return _DECISION_CLASS_ORDER.index(dc)
    except ValueError:
        return -1


@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    reason: str


def check_release_rules(
    assembly_revision_id: str,
    characteristics: list[dict],
    inspection_results: list[dict],
    simulation_cases: list[dict],
    predictions: list[dict],
    release_manifest_data: dict,
    mrs_score: float,
    ics_score: float,
    mrs_target: float = _DEFAULT_MRS_TARGET,
    ics_threshold: float = _DEFAULT_ICS_THRESHOLD,
) -> list[RuleResult]:
    """
    Evaluate all 10 evidence completeness rules.
    Each rule returns a RuleResult.

    R1:  All critical/key Characteristics have status=inspection_confirmed or released
    R2:  Every critical/key Characteristic has >= 1 InspectionResult with status=pass
         (simulation alone insufficient at MRL 7+)
    R3:  MRS >= MRS target (default 0.70)
    R4:  ICS >= ICS threshold (default 0.80)
    R5:  All SimulationCases linked to released Characteristics have status=validated
    R6:  No stale Predictions (caller marks predictions with stale=True)
    R7:  All required Characteristics have at least one InspectionResult
    R8:  No Characteristic uses surrogate-only evidence below ReleaseCritical
         for release-critical decisions
    R9:  ReleaseManifest.open_issues is empty
    R10: All required signatories have signed (proxied via manifest data)
    """
    results: list[RuleResult] = []

    # Build lookups
    critical_key_char_ids: list[str] = []
    released_char_ids: list[str] = []
    for char in characteristics:
        char_id = char.get("entity_id", char.get("characteristic_id", ""))
        criticality = char.get("criticality", "").lower()
        status = char.get("status", "")
        if criticality in ("critical", "key"):
            critical_key_char_ids.append(char_id)
        if status in ("inspection_confirmed", "released"):
            released_char_ids.append(char_id)

    # Map characteristic_id → list of InspectionResults
    char_to_inspections: dict[str, list[dict]] = {}
    for ir in inspection_results:
        char_id = ir.get("characteristic_id", "")
        char_to_inspections.setdefault(char_id, []).append(ir)

    # --- R1: Critical/key chars must be inspection_confirmed or released ---
    r1_failing: list[str] = []
    for char in characteristics:
        char_id = char.get("entity_id", char.get("characteristic_id", ""))
        criticality = char.get("criticality", "").lower()
        status = char.get("status", "")
        if criticality in ("critical", "key"):
            if status not in ("inspection_confirmed", "released"):
                r1_failing.append(char_id)

    if r1_failing:
        results.append(RuleResult(
            rule_id="R1",
            passed=False,
            reason=(
                f"Critical/key Characteristics not in inspection_confirmed or released: "
                f"{r1_failing}"
            ),
        ))
    else:
        results.append(RuleResult(rule_id="R1", passed=True, reason="All critical/key Characteristics confirmed"))

    # --- R2: Critical/key chars need >= 1 InspectionResult with status=pass ---
    r2_failing: list[str] = []
    for char_id in critical_key_char_ids:
        inspections = char_to_inspections.get(char_id, [])
        has_pass = any(ir.get("status", "") == "pass" for ir in inspections)
        if not has_pass:
            r2_failing.append(char_id)

    if r2_failing:
        results.append(RuleResult(
            rule_id="R2",
            passed=False,
            reason=f"Critical/key Characteristics missing passing InspectionResult: {r2_failing}",
        ))
    else:
        results.append(RuleResult(rule_id="R2", passed=True, reason="All critical/key Characteristics have passing inspection"))

    # --- R3: MRS >= target ---
    if mrs_score >= mrs_target:
        results.append(RuleResult(rule_id="R3", passed=True, reason=f"MRS {mrs_score:.3f} >= target {mrs_target:.3f}"))
    else:
        results.append(RuleResult(
            rule_id="R3",
            passed=False,
            reason=f"MRS {mrs_score:.3f} is below target {mrs_target:.3f}",
        ))

    # --- R4: ICS >= threshold ---
    if ics_score >= ics_threshold:
        results.append(RuleResult(rule_id="R4", passed=True, reason=f"ICS {ics_score:.3f} >= threshold {ics_threshold:.3f}"))
    else:
        results.append(RuleResult(
            rule_id="R4",
            passed=False,
            reason=f"ICS {ics_score:.3f} is below threshold {ics_threshold:.3f}",
        ))

    # --- R5: SimulationCases linked to released Characteristics must be validated ---
    r5_failing: list[str] = []
    released_char_id_set = set(released_char_ids)
    for sc in simulation_cases:
        sc_id = sc.get("entity_id", sc.get("case_id", ""))
        linked_char = sc.get("characteristic_id", "")
        status = sc.get("status", "")
        # Only check if simulation case is linked to a released characteristic
        if linked_char in released_char_id_set and status != "validated":
            r5_failing.append(sc_id)

    if r5_failing:
        results.append(RuleResult(
            rule_id="R5",
            passed=False,
            reason=f"SimulationCases linked to released Characteristics are not validated: {r5_failing}",
        ))
    else:
        results.append(RuleResult(rule_id="R5", passed=True, reason="All linked SimulationCases are validated"))

    # --- R6: No stale Predictions ---
    # Caller marks predictions with stale=True if they are stale
    r6_stale: list[str] = []
    for pred in predictions:
        pred_id = pred.get("entity_id", pred.get("prediction_id", ""))
        if pred.get("stale", False):
            r6_stale.append(pred_id)

    if r6_stale:
        results.append(RuleResult(
            rule_id="R6",
            passed=False,
            reason=f"Stale Predictions detected: {r6_stale}",
        ))
    else:
        results.append(RuleResult(rule_id="R6", passed=True, reason="No stale Predictions"))

    # --- R7: All required Characteristics have at least one InspectionResult ---
    # "Required" = those with inspect_required=True or criticality in critical/key
    r7_failing: list[str] = []
    for char in characteristics:
        char_id = char.get("entity_id", char.get("characteristic_id", ""))
        criticality = char.get("criticality", "").lower()
        inspect_required = char.get("inspect_required", criticality in ("critical", "key"))
        if inspect_required:
            if not char_to_inspections.get(char_id):
                r7_failing.append(char_id)

    if r7_failing:
        results.append(RuleResult(
            rule_id="R7",
            passed=False,
            reason=f"Required Characteristics missing InspectionResult: {r7_failing}",
        ))
    else:
        results.append(RuleResult(rule_id="R7", passed=True, reason="All required Characteristics have at least one InspectionResult"))

    # --- R8: No Characteristic uses surrogate-only evidence below ReleaseCritical
    #         for release-critical characteristics ---
    # We look at characteristics requiring ReleaseCritical decision class;
    # if they have only Prediction evidence with class < ReleaseCritical and no
    # SimulationCase or InspectionResult evidence → fail.
    release_critical_level = _decision_class_level("ReleaseCritical")
    r8_failing: list[str] = []

    # Build set of char_ids with simulation or inspection evidence
    chars_with_sim_or_insp: set[str] = set()
    for sc in simulation_cases:
        cid = sc.get("characteristic_id", "")
        if cid:
            chars_with_sim_or_insp.add(cid)
    for ir in inspection_results:
        cid = ir.get("characteristic_id", "")
        if cid:
            chars_with_sim_or_insp.add(cid)

    for char in characteristics:
        char_id = char.get("entity_id", char.get("characteristic_id", ""))
        decision_class_required = char.get("decision_class_required", "")
        if _decision_class_level(decision_class_required) >= release_critical_level:
            # This char needs ReleaseCritical+ evidence
            # Check if it has simulation or inspection backing
            if char_id not in chars_with_sim_or_insp:
                # Only predictions — check if any prediction's trust class < ReleaseCritical
                char_predictions = [
                    p for p in predictions
                    if p.get("governing_characteristic_id", p.get("characteristic_id", "")) == char_id
                ]
                if char_predictions:
                    all_below = all(
                        _decision_class_level(
                            (p.get("trust_bundle") or {}).get("evaluated_decision_class", "")
                        ) < release_critical_level
                        for p in char_predictions
                    )
                    if all_below:
                        r8_failing.append(char_id)

    if r8_failing:
        results.append(RuleResult(
            rule_id="R8",
            passed=False,
            reason=(
                f"Characteristics with ReleaseCritical requirement have only surrogate evidence"
                f" below ReleaseCritical: {r8_failing}"
            ),
        ))
    else:
        results.append(RuleResult(rule_id="R8", passed=True, reason="All release-critical Characteristics have adequate evidence tier"))

    # --- R9: ReleaseManifest.open_issues is empty ---
    open_issues = release_manifest_data.get("open_issues", [])
    if open_issues:
        results.append(RuleResult(
            rule_id="R9",
            passed=False,
            reason=f"Release manifest has {len(open_issues)} open issue(s): {open_issues}",
        ))
    else:
        results.append(RuleResult(rule_id="R9", passed=True, reason="No open issues in release manifest"))

    # --- R10: All required signatories have signed ---
    # Proxied via release_manifest_data: expected_signatories vs actual_signatories
    expected_sigs = set(release_manifest_data.get("required_signatories", []))
    actual_sigs = set(release_manifest_data.get("actual_signatories", []))
    missing_sigs = expected_sigs - actual_sigs

    if missing_sigs:
        results.append(RuleResult(
            rule_id="R10",
            passed=False,
            reason=f"Missing required signatories: {sorted(missing_sigs)}",
        ))
    else:
        results.append(RuleResult(rule_id="R10", passed=True, reason="All required signatories have signed"))

    return results
