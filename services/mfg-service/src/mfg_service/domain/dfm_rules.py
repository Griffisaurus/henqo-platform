"""
DFM Rule Engine — all 49 rules across 7 process families.

Rules read geometry/process parameters from a ``component_data`` dict.
Missing parameters → rule skipped (no violation returned).

Rule IDs match §2 of manufacturability-subsystem-spec.md.
Tiers: A = hard blocker, B = soft warning, C = advisory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DFMViolation:
    rule_id: str
    tier: str                  # "A" | "B" | "C"
    description: str
    measured_value: float | None
    threshold_value: float | None
    process_family: str        # "cnc" | "am" | "sheet_metal" | "molding" | "pcb" | "harness" | "assembly"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(data: dict, key: str):
    """Return value or None if key missing."""
    return data.get(key)


def _violation(
    rule_id: str,
    tier: str,
    description: str,
    measured: float | None,
    threshold: float | None,
    family: str,
) -> DFMViolation:
    return DFMViolation(
        rule_id=rule_id,
        tier=tier,
        description=description,
        measured_value=measured,
        threshold_value=threshold,
        process_family=family,
    )


# ---------------------------------------------------------------------------
# CNC Machining — 10 rules
# ---------------------------------------------------------------------------

def evaluate_cnc_rules(component_data: dict) -> list[DFMViolation]:
    """
    Rules DFM-CNC-001 through DFM-CNC-010.

    Expected component_data keys (all optional; missing → rule skipped):
      wall_thickness_mm         float  — minimum wall thickness
      material                  str    — "aluminum" | "steel" | other
      internal_corner_radius_mm float  — smallest internal corner radius
      tool_radius_mm            float  — largest tool radius used in setup
      hole_depth_mm             float  — deepest hole depth
      hole_diameter_mm          float  — corresponding hole diameter
      has_gun_drill             bool   — True if gun-drill available
      setup_count               int    — number of machine setups
      tool_access_score         float  — [0, 1] fraction of features reachable
      datum_surface_count       int    — non-collinear datum surfaces
      inspection_reachability   float  — [0, 1]
      surface_finish_ra_min     float  — minimum Ra value on part (µm)
      surface_finish_ra_max     float  — maximum Ra value on part (µm)
      thread_engagement_ratio   float  — engagement_length / nominal_diameter
      tolerance_stack_incapable bool   — True if any ToleranceStack contributor is incapable
    """
    violations: list[DFMViolation] = []
    f = "cnc"

    # DFM-CNC-001 (A) — minimum wall thickness
    wt = _get(component_data, "wall_thickness_mm")
    mat = _get(component_data, "material") or "aluminum"
    if wt is not None:
        limit = 1.5 if str(mat).lower() == "steel" else 0.8
        if wt < limit:
            violations.append(_violation(
                "DFM-CNC-001", "A",
                f"Minimum wall thickness {wt} mm is below {limit} mm for {mat}",
                wt, limit, f,
            ))

    # DFM-CNC-002 (A) — internal corner radius
    icr = _get(component_data, "internal_corner_radius_mm")
    tr = _get(component_data, "tool_radius_mm")
    if icr is not None:
        abs_min = 0.5
        limit_cr = max(abs_min, tr) if tr is not None else abs_min
        if icr < limit_cr:
            violations.append(_violation(
                "DFM-CNC-002", "A",
                f"Internal corner radius {icr} mm below required {limit_cr} mm",
                icr, limit_cr, f,
            ))

    # DFM-CNC-003 (A) — hole depth-to-diameter ratio
    hd = _get(component_data, "hole_depth_mm")
    hdiam = _get(component_data, "hole_diameter_mm")
    has_gun = _get(component_data, "has_gun_drill")
    if hd is not None and hdiam is not None and hdiam > 0:
        ratio = hd / hdiam
        limit_ratio = 10.0 if has_gun else 6.0
        if ratio > limit_ratio:
            violations.append(_violation(
                "DFM-CNC-003", "A",
                f"Hole depth-to-diameter ratio {ratio:.2f}:1 exceeds {limit_ratio}:1",
                ratio, limit_ratio, f,
            ))

    # DFM-CNC-004 (B) — setup count
    setups = _get(component_data, "setup_count")
    if setups is not None:
        if setups > 5:
            violations.append(_violation(
                "DFM-CNC-004", "B",
                f"Setup count {setups} exceeds 5; cost-risk flag triggered",
                float(setups), 5.0, f,
            ))

    # DFM-CNC-005 (B) — tool access score
    tas = _get(component_data, "tool_access_score")
    if tas is not None:
        if tas < 0.80:
            violations.append(_violation(
                "DFM-CNC-005", "B",
                f"Tool access score {tas:.3f} below 0.80",
                tas, 0.80, f,
            ))

    # DFM-CNC-006 (B) — workholding stability
    datums = _get(component_data, "datum_surface_count")
    if datums is not None:
        if datums < 3:
            violations.append(_violation(
                "DFM-CNC-006", "B",
                f"Only {datums} non-collinear datum surface(s); need ≥ 3",
                float(datums), 3.0, f,
            ))

    # DFM-CNC-007 (A) — inspection accessibility
    ir = _get(component_data, "inspection_reachability")
    if ir is not None:
        if ir < 0.90:
            violations.append(_violation(
                "DFM-CNC-007", "A",
                f"Inspection reachability {ir:.3f} below 0.90",
                ir, 0.90, f,
            ))

    # DFM-CNC-008 (C) — surface finish consistency
    ra_min = _get(component_data, "surface_finish_ra_min")
    ra_max = _get(component_data, "surface_finish_ra_max")
    if ra_min is not None and ra_max is not None:
        if ra_min <= 0.4 and ra_max > 3.2:
            violations.append(_violation(
                "DFM-CNC-008", "C",
                f"Mixed surface finish Ra min={ra_min} µm and Ra max={ra_max} µm without documented reason",
                ra_max, 3.2, f,
            ))

    # DFM-CNC-009 (C) — thread engagement
    te = _get(component_data, "thread_engagement_ratio")
    if te is not None:
        if te < 1.5:
            violations.append(_violation(
                "DFM-CNC-009", "C",
                f"Thread engagement ratio {te:.2f}× below 1.5× nominal diameter",
                te, 1.5, f,
            ))

    # DFM-CNC-010 (B) — tolerance stack feasibility
    ts_incapable = _get(component_data, "tolerance_stack_incapable")
    if ts_incapable:
        violations.append(_violation(
            "DFM-CNC-010", "B",
            "Tolerance stack has an incapable ProcessCapability contributor",
            None, None, f,
        ))

    return violations


# ---------------------------------------------------------------------------
# Additive Manufacturing — 9 rules
# ---------------------------------------------------------------------------

def evaluate_am_rules(component_data: dict) -> list[DFMViolation]:
    """
    Rules DFM-AM-001 through DFM-AM-009.

    Expected component_data keys:
      am_process                str    — "metal_pbf" | "polymer_pbf" | "fdm" | "sla"
      wall_thickness_mm         float
      unsupported_span_mm       float  — largest unsupported horizontal span
      support_removal_clear     bool   — True if all support removal paths clear
      evacuation_port_mm        float  — smallest evacuation port diameter
      critical_surface_angle_deg float — min angle of Ra-critical surfaces from build plate
      post_process_accessible   bool   — True if all post-processing surfaces accessible
      support_volume_fraction   float  — support_volume / part_volume
      min_feature_mm            float  — smallest feature dimension
      beam_spot_size_mm         float  — machine beam spot size (or layer thickness for z)
      pmi_additive_complete     bool   — True if ASME Y14.46 PMI fields all present
    """
    violations: list[DFMViolation] = []
    f = "am"
    process = str(_get(component_data, "am_process") or "metal_pbf").lower()

    # DFM-AM-001 (A) — minimum wall thickness
    wt = _get(component_data, "wall_thickness_mm")
    if wt is not None:
        limit = 0.4 if "metal" in process else 0.8
        if wt < limit:
            violations.append(_violation(
                "DFM-AM-001", "A",
                f"AM wall thickness {wt} mm below {limit} mm for {process}",
                wt, limit, f,
            ))

    # DFM-AM-002 (A) — unsupported horizontal span
    span = _get(component_data, "unsupported_span_mm")
    if span is not None:
        if span > 1.0:
            violations.append(_violation(
                "DFM-AM-002", "A",
                f"Unsupported horizontal span {span} mm exceeds 1.0 mm without support",
                span, 1.0, f,
            ))

    # DFM-AM-003 (B) — support structure removal access
    support_clear = _get(component_data, "support_removal_clear")
    if support_clear is not None and not support_clear:
        violations.append(_violation(
            "DFM-AM-003", "B",
            "Support structure removal path blocked (blind cavity retaining supports)",
            None, None, f,
        ))

    # DFM-AM-004 (B) — powder/resin evacuation
    evac = _get(component_data, "evacuation_port_mm")
    if evac is not None:
        if evac < 2.0:
            violations.append(_violation(
                "DFM-AM-004", "B",
                f"Evacuation port diameter {evac} mm below 2.0 mm minimum",
                evac, 2.0, f,
            ))

    # DFM-AM-005 (B) — critical surface orientation
    angle = _get(component_data, "critical_surface_angle_deg")
    if angle is not None:
        if angle < 45.0:
            violations.append(_violation(
                "DFM-AM-005", "B",
                f"Ra-critical surface at {angle}° from build plate; should be ≥ 45°",
                angle, 45.0, f,
            ))

    # DFM-AM-006 (A) — post-processing access
    pp_access = _get(component_data, "post_process_accessible")
    if pp_access is not None and not pp_access:
        violations.append(_violation(
            "DFM-AM-006", "A",
            "Post-processing surfaces (HIP/heat treatment/CNC) not geometrically accessible",
            None, None, f,
        ))

    # DFM-AM-007 (C) — support volume fraction
    svf = _get(component_data, "support_volume_fraction")
    if svf is not None:
        if svf > 0.30:
            violations.append(_violation(
                "DFM-AM-007", "C",
                f"Support volume fraction {svf:.2f} exceeds 0.30",
                svf, 0.30, f,
            ))

    # DFM-AM-008 (B) — feature resolution
    mf = _get(component_data, "min_feature_mm")
    bss = _get(component_data, "beam_spot_size_mm")
    if mf is not None and bss is not None:
        limit_res = 2.0 * bss
        if mf < limit_res:
            violations.append(_violation(
                "DFM-AM-008", "B",
                f"Feature size {mf} mm below 2× beam spot size {bss} mm = {limit_res} mm",
                mf, limit_res, f,
            ))

    # DFM-AM-009 (A) — Y14.46 PMI completeness
    pmi_ok = _get(component_data, "pmi_additive_complete")
    if pmi_ok is not None and not pmi_ok:
        violations.append(_violation(
            "DFM-AM-009", "A",
            "ASME Y14.46-2022 additive PMI fields incomplete (build direction/layer thickness/support notes/post-process sequence)",
            None, None, f,
        ))

    return violations


# ---------------------------------------------------------------------------
# Sheet Metal — 6 rules
# ---------------------------------------------------------------------------

def evaluate_sheet_metal_rules(component_data: dict) -> list[DFMViolation]:
    """
    Rules DFM-SM-001 through DFM-SM-006.

    Expected component_data keys:
      sm_material               str    — "steel" | "aluminum" | other
      material_thickness_mm     float  — sheet thickness
      bend_radius_mm            float  — smallest bend radius
      flange_width_mm           float  — smallest flange width
      hole_to_edge_distance_mm  float  — smallest hole-center to edge distance
      hole_diameter_sm_mm       float  — hole diameter (for edge-distance ratio)
      gauge_consistent          bool   — True if single gauge throughout
      bend_sequence_collision   bool   — True if bend sequence simulation found collision
      weld_approach_clearance_mm float — minimum weld torch approach clearance
    """
    violations: list[DFMViolation] = []
    f = "sheet_metal"
    mat = str(_get(component_data, "sm_material") or "steel").lower()
    t = _get(component_data, "material_thickness_mm")

    # DFM-SM-001 (A) — minimum bend radius
    br = _get(component_data, "bend_radius_mm")
    if br is not None and t is not None:
        factor = 0.5 if "aluminum" in mat else 1.0
        limit = factor * t
        if br < limit:
            violations.append(_violation(
                "DFM-SM-001", "A",
                f"Bend radius {br} mm below {factor}× thickness {t} mm = {limit} mm for {mat}",
                br, limit, f,
            ))

    # DFM-SM-002 (A) — minimum flange width
    fw = _get(component_data, "flange_width_mm")
    if fw is not None and t is not None:
        limit = 3.0 * t
        if fw < limit:
            violations.append(_violation(
                "DFM-SM-002", "A",
                f"Flange width {fw} mm below 3× thickness {t} mm = {limit} mm",
                fw, limit, f,
            ))

    # DFM-SM-003 (B) — hole-to-edge distance
    hed = _get(component_data, "hole_to_edge_distance_mm")
    if hed is not None and t is not None:
        limit = 2.0 * t
        if hed < limit:
            violations.append(_violation(
                "DFM-SM-003", "B",
                f"Hole-to-edge distance {hed} mm below 2× thickness {t} mm = {limit} mm",
                hed, limit, f,
            ))

    # DFM-SM-004 (B) — material gauge consistency
    gauge_ok = _get(component_data, "gauge_consistent")
    if gauge_ok is not None and not gauge_ok:
        violations.append(_violation(
            "DFM-SM-004", "B",
            "Multiple material gauges on single part without documented reason",
            None, None, f,
        ))

    # DFM-SM-005 (C) — bend sequence feasibility
    collision = _get(component_data, "bend_sequence_collision")
    if collision:
        violations.append(_violation(
            "DFM-SM-005", "C",
            "Bend sequence simulation found a tool collision in the planned sequence",
            None, None, f,
        ))

    # DFM-SM-006 (A) — weld access clearance
    wac = _get(component_data, "weld_approach_clearance_mm")
    if wac is not None:
        if wac < 25.0:
            violations.append(_violation(
                "DFM-SM-006", "A",
                f"Weld torch approach clearance {wac} mm below 25 mm minimum",
                wac, 25.0, f,
            ))

    return violations


# ---------------------------------------------------------------------------
# Injection Molding — 6 rules
# ---------------------------------------------------------------------------

def evaluate_molding_rules(component_data: dict) -> list[DFMViolation]:
    """
    Rules DFM-IM-001 through DFM-IM-006.

    Expected component_data keys:
      draft_angle_deg               float  — minimum draft angle on vertical surfaces
      textured_surface              bool   — True if textured finish required
      wall_thickness_min_mm         float  — minimum wall thickness in any 50 mm region
      wall_thickness_max_mm         float  — maximum wall thickness in any 50 mm region
      has_undercut                  bool   — True if undercut features present
      undercut_tooling_documented   bool   — True if side-action/core-pull documented
      gate_on_cosmetic_surface      bool   — True if gate mark would be on cosmetic surface
      gate_to_critical_feature_mm   float  — distance gate mark to nearest critical mating feature
      rib_to_wall_ratio             float  — rib thickness / nominal wall thickness
      boss_od_to_wall_ratio         float  — boss OD / adjacent wall thickness
      cycle_time_est_s              float  — estimated cycle time in seconds
    """
    violations: list[DFMViolation] = []
    f = "molding"

    # DFM-IM-001 (A) — draft angle
    da = _get(component_data, "draft_angle_deg")
    textured = _get(component_data, "textured_surface")
    if da is not None:
        limit = 3.0 if textured else 1.0
        if da < limit:
            violations.append(_violation(
                "DFM-IM-001", "A",
                f"Draft angle {da}° below {limit}° for {'textured' if textured else 'non-textured'} surface",
                da, limit, f,
            ))

    # DFM-IM-002 (A) — wall thickness uniformity
    wt_min = _get(component_data, "wall_thickness_min_mm")
    wt_max = _get(component_data, "wall_thickness_max_mm")
    if wt_min is not None and wt_max is not None and wt_min > 0:
        variation = (wt_max - wt_min) / wt_min
        if variation > 0.25:
            violations.append(_violation(
                "DFM-IM-002", "A",
                f"Wall thickness variation {variation:.1%} exceeds 25% within 50 mm region",
                variation, 0.25, f,
            ))

    # DFM-IM-003 (A) — undercut features
    has_uc = _get(component_data, "has_undercut")
    uc_doc = _get(component_data, "undercut_tooling_documented")
    if has_uc and not uc_doc:
        violations.append(_violation(
            "DFM-IM-003", "A",
            "Undercut features present without documented side-action/core-pull mechanism",
            None, None, f,
        ))

    # DFM-IM-004 (B) — gate location
    gate_cosmetic = _get(component_data, "gate_on_cosmetic_surface")
    gate_dist = _get(component_data, "gate_to_critical_feature_mm")
    if gate_cosmetic:
        violations.append(_violation(
            "DFM-IM-004", "B",
            "Gate mark on cosmetic surface",
            None, None, f,
        ))
    if gate_dist is not None and gate_dist < 5.0:
        violations.append(_violation(
            "DFM-IM-004", "B",
            f"Gate mark {gate_dist} mm from tolerance-critical mating feature; need ≥ 5 mm",
            gate_dist, 5.0, f,
        ))

    # DFM-IM-005 (B) — sink and warp risk
    rib_ratio = _get(component_data, "rib_to_wall_ratio")
    boss_ratio = _get(component_data, "boss_od_to_wall_ratio")
    if rib_ratio is not None and rib_ratio > 0.60:
        violations.append(_violation(
            "DFM-IM-005", "B",
            f"Rib-to-wall ratio {rib_ratio:.2f} exceeds 0.60 (sink/warp risk)",
            rib_ratio, 0.60, f,
        ))
    if boss_ratio is not None and boss_ratio > 0.60:
        violations.append(_violation(
            "DFM-IM-005", "B",
            f"Boss OD-to-wall ratio {boss_ratio:.2f} exceeds 0.60 (sink/warp risk)",
            boss_ratio, 0.60, f,
        ))

    # DFM-IM-006 (C) — cycle time estimate
    ct = _get(component_data, "cycle_time_est_s")
    if ct is not None and ct > 60.0:
        violations.append(_violation(
            "DFM-IM-006", "C",
            f"Estimated cycle time {ct} s exceeds 60 s for target material",
            ct, 60.0, f,
        ))

    return violations


# ---------------------------------------------------------------------------
# PCB Fabrication — 6 rules
# ---------------------------------------------------------------------------

def evaluate_pcb_rules(component_data: dict) -> list[DFMViolation]:
    """
    Rules DFM-PCB-001 through DFM-PCB-006.

    Expected component_data keys:
      fab_class                    str    — "standard" | "fine_line"
      min_trace_width_mm           float
      min_trace_space_mm           float
      pad_annular_ring_mm          float  — annular ring after worst-case drill deviation
      component_body_clearance_mm  float  — min clearance between component bodies
      test_point_coverage          float  — fraction of nets with accessible test point
      high_speed_traces_present    bool   — True if any high-speed trace > 100 MHz or λ/10
      impedance_specs_documented   bool   — True if impedance specs and stack-up constraints present
      via_aspect_ratio             float  — depth / diameter for through-hole vias
    """
    violations: list[DFMViolation] = []
    f = "pcb"
    fab = str(_get(component_data, "fab_class") or "standard").lower()

    # DFM-PCB-001 (A) — minimum trace width and space
    tw = _get(component_data, "min_trace_width_mm")
    ts = _get(component_data, "min_trace_space_mm")
    limit_tw = 0.075 if "fine" in fab else 0.10
    limit_ts = 0.075 if "fine" in fab else 0.10
    if tw is not None and tw < limit_tw:
        violations.append(_violation(
            "DFM-PCB-001", "A",
            f"Minimum trace width {tw} mm below {limit_tw} mm for {fab} fab class",
            tw, limit_tw, f,
        ))
    if ts is not None and ts < limit_ts:
        violations.append(_violation(
            "DFM-PCB-001", "A",
            f"Minimum trace space {ts} mm below {limit_ts} mm for {fab} fab class",
            ts, limit_ts, f,
        ))

    # DFM-PCB-002 (A) — pad annular ring
    par = _get(component_data, "pad_annular_ring_mm")
    if par is not None and par < 0.15:
        violations.append(_violation(
            "DFM-PCB-002", "A",
            f"Pad annular ring {par} mm below 0.15 mm minimum",
            par, 0.15, f,
        ))

    # DFM-PCB-003 (B) — component body clearance
    cbc = _get(component_data, "component_body_clearance_mm")
    if cbc is not None and cbc < 0.25:
        violations.append(_violation(
            "DFM-PCB-003", "B",
            f"Component body clearance {cbc} mm below 0.25 mm",
            cbc, 0.25, f,
        ))

    # DFM-PCB-004 (B) — test point coverage
    tpc = _get(component_data, "test_point_coverage")
    if tpc is not None and tpc < 0.90:
        violations.append(_violation(
            "DFM-PCB-004", "B",
            f"Test point coverage {tpc:.1%} below 90%",
            tpc, 0.90, f,
        ))

    # DFM-PCB-005 (A) — controlled impedance documentation
    hs = _get(component_data, "high_speed_traces_present")
    imp_doc = _get(component_data, "impedance_specs_documented")
    if hs and not imp_doc:
        violations.append(_violation(
            "DFM-PCB-005", "A",
            "High-speed traces present without impedance specs and stack-up constraints in fab notes",
            None, None, f,
        ))

    # DFM-PCB-006 (C) — via aspect ratio
    var = _get(component_data, "via_aspect_ratio")
    if var is not None and var > 10.0:
        violations.append(_violation(
            "DFM-PCB-006", "C",
            f"Via aspect ratio {var:.1f}:1 exceeds 10:1",
            var, 10.0, f,
        ))

    return violations


# ---------------------------------------------------------------------------
# Harness and Wiring Assembly — 5 rules
# ---------------------------------------------------------------------------

def evaluate_harness_rules(component_data: dict) -> list[DFMViolation]:
    """
    Rules DFM-HRN-001 through DFM-HRN-005.

    Expected component_data keys:
      wire_diameter_mm              float  — largest wire diameter in the harness
      min_bend_radius_mm            float  — smallest bend radius in harness routing
      connector_approach_clearance_mm float — minimum connector approach clearance
      strain_relief_documented      bool   — True if strain relief method specified
      thermal_separation_mm         float  — min separation of signal wires from heat sources
      heat_source_temp_c            float  — max temperature of nearby heat sources
      bundle_diameter_mm            float  — largest bundle outer diameter
    """
    violations: list[DFMViolation] = []
    f = "harness"

    # DFM-HRN-001 (A) — harness bend radius
    wd = _get(component_data, "wire_diameter_mm")
    mbr = _get(component_data, "min_bend_radius_mm")
    if wd is not None and mbr is not None:
        limit = 10.0 * wd
        if mbr < limit:
            violations.append(_violation(
                "DFM-HRN-001", "A",
                f"Harness bend radius {mbr} mm below 10× wire diameter {wd} mm = {limit} mm",
                mbr, limit, f,
            ))

    # DFM-HRN-002 (A) — connector approach clearance
    cac = _get(component_data, "connector_approach_clearance_mm")
    if cac is not None and cac < 50.0:
        violations.append(_violation(
            "DFM-HRN-002", "A",
            f"Connector approach clearance {cac} mm below 50 mm",
            cac, 50.0, f,
        ))

    # DFM-HRN-003 (B) — strain relief
    sr = _get(component_data, "strain_relief_documented")
    if sr is not None and not sr:
        violations.append(_violation(
            "DFM-HRN-003", "B",
            "Strain relief method not documented for connectors",
            None, None, f,
        ))

    # DFM-HRN-004 (B) — thermal separation
    ts = _get(component_data, "thermal_separation_mm")
    ht = _get(component_data, "heat_source_temp_c")
    if ts is not None and ht is not None and ht > 85.0 and ts < 25.0:
        violations.append(_violation(
            "DFM-HRN-004", "B",
            f"Signal wire separation {ts} mm from {ht}°C heat source; need ≥ 25 mm",
            ts, 25.0, f,
        ))

    # DFM-HRN-005 (C) — bundle diameter
    bd = _get(component_data, "bundle_diameter_mm")
    if bd is not None and bd > 25.0:
        violations.append(_violation(
            "DFM-HRN-005", "C",
            f"Harness bundle diameter {bd} mm exceeds 25 mm; consider split sub-harnesses",
            bd, 25.0, f,
        ))

    return violations


# ---------------------------------------------------------------------------
# Assembly (DFA) — 7 rules
# ---------------------------------------------------------------------------

def evaluate_assembly_rules(component_data: dict) -> list[DFMViolation]:
    """
    Rules DFA-ASM-001 through DFA-ASM-007.

    Expected component_data keys:
      primary_direction_fraction    float  — fraction of parts inserted in primary direction
      fastener_accessible           bool   — True if all fasteners accessible with standard tools
      assembly_steps                int    — count of discrete assembly operations
      part_count                    int    — total part count in assembly
      join_method_compatible        bool   — True if no incompatible join methods
      key_char_inspection_reachable float  — fraction of key Chars with inspection access
      symmetric_parts_poka_yoke     bool   — True if asymmetric parts have locating features
      fru_time_min                  float  — max FRU remove/reinstall time in minutes
    """
    violations: list[DFMViolation] = []
    f = "assembly"

    # DFA-ASM-001 (A) — dominant insertion direction
    pdf = _get(component_data, "primary_direction_fraction")
    if pdf is not None and pdf < 0.80:
        violations.append(_violation(
            "DFA-ASM-001", "A",
            f"Only {pdf:.1%} of parts in primary insertion direction; need ≥ 80%",
            pdf, 0.80, f,
        ))

    # DFA-ASM-002 (A) — fastener access
    fa = _get(component_data, "fastener_accessible")
    if fa is not None and not fa:
        violations.append(_violation(
            "DFA-ASM-002", "A",
            "Fasteners not accessible with standard hand tools without removing sub-assemblies",
            None, None, f,
        ))

    # DFA-ASM-003 (B) — assembly step count
    asteps = _get(component_data, "assembly_steps")
    pc = _get(component_data, "part_count")
    if asteps is not None and pc is not None and pc > 0:
        limit_steps = 2 * pc
        if asteps > limit_steps:
            violations.append(_violation(
                "DFA-ASM-003", "B",
                f"Assembly steps {asteps} exceeds 2× part count {pc} = {limit_steps}",
                float(asteps), float(limit_steps), f,
            ))

    # DFA-ASM-004 (B) — join method compatibility
    jmc = _get(component_data, "join_method_compatible")
    if jmc is not None and not jmc:
        violations.append(_violation(
            "DFA-ASM-004", "B",
            "Incompatible join methods specified for the same joint",
            None, None, f,
        ))

    # DFA-ASM-005 (A) — datum and inspection access for key Characteristics
    kcir = _get(component_data, "key_char_inspection_reachable")
    if kcir is not None and kcir < 0.90:
        violations.append(_violation(
            "DFA-ASM-005", "A",
            f"Key Characteristic inspection reachability {kcir:.3f} below 0.90",
            kcir, 0.90, f,
        ))

    # DFA-ASM-006 (B) — field replaceability
    fru_time = _get(component_data, "fru_time_min")
    if fru_time is not None and fru_time > 10.0:
        violations.append(_violation(
            "DFA-ASM-006", "B",
            f"FRU remove/reinstall time {fru_time} min exceeds 10 min with standard tools",
            fru_time, 10.0, f,
        ))

    # DFA-ASM-007 (A) — poka-yoke (error-proofing)
    pk = _get(component_data, "symmetric_parts_poka_yoke")
    if pk is not None and not pk:
        violations.append(_violation(
            "DFA-ASM-007", "A",
            "Geometrically symmetric parts requiring specific orientation lack poka-yoke locating feature or unambiguous marking",
            None, None, f,
        ))

    return violations


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FAMILY_MAP = {
    "cnc": evaluate_cnc_rules,
    "am": evaluate_am_rules,
    "sheet_metal": evaluate_sheet_metal_rules,
    "molding": evaluate_molding_rules,
    "pcb": evaluate_pcb_rules,
    "harness": evaluate_harness_rules,
    "assembly": evaluate_assembly_rules,
}


def evaluate_all_rules(
    component_data: dict,
    process_families: list[str],
) -> list[DFMViolation]:
    """Run only the specified process family evaluators and return combined violations."""
    violations: list[DFMViolation] = []
    for family in process_families:
        fn = _FAMILY_MAP.get(family)
        if fn is not None:
            violations.extend(fn(component_data))
    return violations
