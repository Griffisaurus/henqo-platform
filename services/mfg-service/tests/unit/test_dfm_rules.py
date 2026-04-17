"""
Unit tests for DFM rule evaluation.

For each process family:
  - At least one rule fires (positive test)
  - At least one rule passes (negative test)
Total: ≥ 14 tests (2 per family × 7 families).
"""
import pytest

from mfg_service.domain.dfm_rules import (
    DFMViolation,
    evaluate_all_rules,
    evaluate_am_rules,
    evaluate_assembly_rules,
    evaluate_cnc_rules,
    evaluate_harness_rules,
    evaluate_molding_rules,
    evaluate_pcb_rules,
    evaluate_sheet_metal_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rule_ids(violations: list[DFMViolation]) -> list[str]:
    return [v.rule_id for v in violations]


def _tiers(violations: list[DFMViolation]) -> list[str]:
    return [v.tier for v in violations]


# ---------------------------------------------------------------------------
# CNC tests
# ---------------------------------------------------------------------------

class TestCNCRules:
    def test_cnc_001_fires_wall_too_thin(self):
        """DFM-CNC-001: aluminum wall thickness 0.5 mm < 0.8 mm → violation."""
        violations = evaluate_cnc_rules({"wall_thickness_mm": 0.5, "material": "aluminum"})
        assert "DFM-CNC-001" in _rule_ids(violations)
        v = next(v for v in violations if v.rule_id == "DFM-CNC-001")
        assert v.tier == "A"
        assert v.measured_value == pytest.approx(0.5)
        assert v.threshold_value == pytest.approx(0.8)

    def test_cnc_001_passes_wall_sufficient(self):
        """DFM-CNC-001: aluminum wall thickness 1.0 mm ≥ 0.8 mm → no violation."""
        violations = evaluate_cnc_rules({"wall_thickness_mm": 1.0, "material": "aluminum"})
        assert "DFM-CNC-001" not in _rule_ids(violations)

    def test_cnc_001_steel_threshold(self):
        """DFM-CNC-001: steel uses 1.5 mm threshold."""
        # 1.0 mm is fine for aluminum but not steel
        violations = evaluate_cnc_rules({"wall_thickness_mm": 1.0, "material": "steel"})
        assert "DFM-CNC-001" in _rule_ids(violations)

    def test_cnc_002_fires_corner_radius_too_small(self):
        """DFM-CNC-002: corner radius 0.3 mm < tool radius 0.5 mm → violation."""
        violations = evaluate_cnc_rules({
            "internal_corner_radius_mm": 0.3,
            "tool_radius_mm": 0.5,
        })
        assert "DFM-CNC-002" in _rule_ids(violations)

    def test_cnc_002_passes_corner_radius_ok(self):
        """DFM-CNC-002: corner radius 1.0 mm ≥ tool radius 0.5 mm → no violation."""
        violations = evaluate_cnc_rules({
            "internal_corner_radius_mm": 1.0,
            "tool_radius_mm": 0.5,
        })
        assert "DFM-CNC-002" not in _rule_ids(violations)

    def test_cnc_003_fires_deep_hole_without_gun_drill(self):
        """DFM-CNC-003: 70 mm deep / 7 mm diameter = 10:1 > 6:1 without gun drill → violation."""
        violations = evaluate_cnc_rules({
            "hole_depth_mm": 70.0,
            "hole_diameter_mm": 7.0,
            "has_gun_drill": False,
        })
        assert "DFM-CNC-003" in _rule_ids(violations)

    def test_cnc_003_passes_with_gun_drill(self):
        """DFM-CNC-003: 70/7 = 10:1 ≤ 10:1 with gun drill → no violation."""
        violations = evaluate_cnc_rules({
            "hole_depth_mm": 70.0,
            "hole_diameter_mm": 7.0,
            "has_gun_drill": True,
        })
        assert "DFM-CNC-003" not in _rule_ids(violations)

    def test_cnc_004_fires_too_many_setups(self):
        """DFM-CNC-004: 6 setups > 5 → violation."""
        violations = evaluate_cnc_rules({"setup_count": 6})
        assert "DFM-CNC-004" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-CNC-004").tier == "B"

    def test_cnc_004_passes_acceptable_setups(self):
        """DFM-CNC-004: 3 setups ≤ 5 → no violation."""
        violations = evaluate_cnc_rules({"setup_count": 3})
        assert "DFM-CNC-004" not in _rule_ids(violations)

    def test_cnc_005_fires_low_tool_access(self):
        """DFM-CNC-005: tool_access_score 0.70 < 0.80 → violation."""
        violations = evaluate_cnc_rules({"tool_access_score": 0.70})
        assert "DFM-CNC-005" in _rule_ids(violations)

    def test_cnc_005_passes_sufficient_access(self):
        """DFM-CNC-005: tool_access_score 0.95 ≥ 0.80 → no violation."""
        violations = evaluate_cnc_rules({"tool_access_score": 0.95})
        assert "DFM-CNC-005" not in _rule_ids(violations)

    def test_cnc_007_fires_inspection_reachability_low(self):
        """DFM-CNC-007 (A): inspection_reachability 0.85 < 0.90 → violation."""
        violations = evaluate_cnc_rules({"inspection_reachability": 0.85})
        assert "DFM-CNC-007" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-CNC-007").tier == "A"

    def test_cnc_008_fires_mixed_surface_finish(self):
        """DFM-CNC-008 (C): Ra_min=0.4, Ra_max=4.0 → mixed finish violation."""
        violations = evaluate_cnc_rules({
            "surface_finish_ra_min": 0.4,
            "surface_finish_ra_max": 4.0,
        })
        assert "DFM-CNC-008" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-CNC-008").tier == "C"

    def test_cnc_008_passes_consistent_finish(self):
        """DFM-CNC-008 (C): Ra_min=0.8, Ra_max=1.6 → no mixed-finish violation."""
        violations = evaluate_cnc_rules({
            "surface_finish_ra_min": 0.8,
            "surface_finish_ra_max": 1.6,
        })
        assert "DFM-CNC-008" not in _rule_ids(violations)

    def test_cnc_009_fires_thread_engagement_short(self):
        """DFM-CNC-009 (C): engagement ratio 1.2 < 1.5 → violation."""
        violations = evaluate_cnc_rules({"thread_engagement_ratio": 1.2})
        assert "DFM-CNC-009" in _rule_ids(violations)

    def test_cnc_010_fires_incapable_stack(self):
        """DFM-CNC-010 (B): tolerance_stack_incapable=True → violation."""
        violations = evaluate_cnc_rules({"tolerance_stack_incapable": True})
        assert "DFM-CNC-010" in _rule_ids(violations)

    def test_cnc_missing_params_skip(self):
        """Missing parameters → no violations (rules are skipped)."""
        violations = evaluate_cnc_rules({})
        assert violations == []


# ---------------------------------------------------------------------------
# AM tests
# ---------------------------------------------------------------------------

class TestAMRules:
    def test_am_001_fires_thin_metal_wall(self):
        """DFM-AM-001: metal PBF wall 0.3 mm < 0.4 mm → violation."""
        violations = evaluate_am_rules({
            "am_process": "metal_pbf",
            "wall_thickness_mm": 0.3,
        })
        assert "DFM-AM-001" in _rule_ids(violations)
        v = next(v for v in violations if v.rule_id == "DFM-AM-001")
        assert v.tier == "A"

    def test_am_001_passes_polymer_wall(self):
        """DFM-AM-001: polymer PBF wall 0.9 mm ≥ 0.8 mm → no violation."""
        violations = evaluate_am_rules({
            "am_process": "polymer_pbf",
            "wall_thickness_mm": 0.9,
        })
        assert "DFM-AM-001" not in _rule_ids(violations)

    def test_am_002_fires_large_unsupported_span(self):
        """DFM-AM-002: span 1.5 mm > 1.0 mm → violation."""
        violations = evaluate_am_rules({"unsupported_span_mm": 1.5})
        assert "DFM-AM-002" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-AM-002").tier == "A"

    def test_am_002_passes_small_span(self):
        """DFM-AM-002: span 0.8 mm ≤ 1.0 mm → no violation."""
        violations = evaluate_am_rules({"unsupported_span_mm": 0.8})
        assert "DFM-AM-002" not in _rule_ids(violations)

    def test_am_007_fires_high_support_volume(self):
        """DFM-AM-007 (C): support_volume_fraction 0.35 > 0.30 → violation."""
        violations = evaluate_am_rules({"support_volume_fraction": 0.35})
        assert "DFM-AM-007" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-AM-007").tier == "C"

    def test_am_007_passes_low_support_volume(self):
        """DFM-AM-007 (C): support_volume_fraction 0.20 ≤ 0.30 → no violation."""
        violations = evaluate_am_rules({"support_volume_fraction": 0.20})
        assert "DFM-AM-007" not in _rule_ids(violations)

    def test_am_009_fires_incomplete_pmi(self):
        """DFM-AM-009 (A): pmi_additive_complete=False → violation."""
        violations = evaluate_am_rules({"pmi_additive_complete": False})
        assert "DFM-AM-009" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-AM-009").tier == "A"

    def test_am_009_passes_complete_pmi(self):
        """DFM-AM-009 (A): pmi_additive_complete=True → no violation."""
        violations = evaluate_am_rules({"pmi_additive_complete": True})
        assert "DFM-AM-009" not in _rule_ids(violations)


# ---------------------------------------------------------------------------
# Sheet Metal tests
# ---------------------------------------------------------------------------

class TestSheetMetalRules:
    def test_sm_001_fires_small_bend_radius(self):
        """DFM-SM-001: steel bend radius 0.5 mm < 1× 1.0 mm thickness → violation."""
        violations = evaluate_sheet_metal_rules({
            "sm_material": "steel",
            "material_thickness_mm": 1.0,
            "bend_radius_mm": 0.5,
        })
        assert "DFM-SM-001" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-SM-001").tier == "A"

    def test_sm_001_passes_adequate_bend_radius(self):
        """DFM-SM-001: aluminum bend radius 1.0 mm ≥ 0.5× 1.2 mm = 0.6 mm → no violation."""
        violations = evaluate_sheet_metal_rules({
            "sm_material": "aluminum",
            "material_thickness_mm": 1.2,
            "bend_radius_mm": 1.0,
        })
        assert "DFM-SM-001" not in _rule_ids(violations)

    def test_sm_002_fires_small_flange(self):
        """DFM-SM-002: flange 2.0 mm < 3× 1.0 mm = 3.0 mm → violation."""
        violations = evaluate_sheet_metal_rules({
            "material_thickness_mm": 1.0,
            "flange_width_mm": 2.0,
        })
        assert "DFM-SM-002" in _rule_ids(violations)

    def test_sm_002_passes_adequate_flange(self):
        """DFM-SM-002: flange 4.0 mm ≥ 3× 1.0 mm → no violation."""
        violations = evaluate_sheet_metal_rules({
            "material_thickness_mm": 1.0,
            "flange_width_mm": 4.0,
        })
        assert "DFM-SM-002" not in _rule_ids(violations)

    def test_sm_006_fires_insufficient_weld_clearance(self):
        """DFM-SM-006: weld approach 20 mm < 25 mm → violation."""
        violations = evaluate_sheet_metal_rules({"weld_approach_clearance_mm": 20.0})
        assert "DFM-SM-006" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-SM-006").tier == "A"

    def test_sm_006_passes_sufficient_weld_clearance(self):
        """DFM-SM-006: weld approach 30 mm ≥ 25 mm → no violation."""
        violations = evaluate_sheet_metal_rules({"weld_approach_clearance_mm": 30.0})
        assert "DFM-SM-006" not in _rule_ids(violations)


# ---------------------------------------------------------------------------
# Molding tests
# ---------------------------------------------------------------------------

class TestMoldingRules:
    def test_im_001_fires_insufficient_draft(self):
        """DFM-IM-001: non-textured surface, draft 0.5° < 1° → violation."""
        violations = evaluate_molding_rules({
            "draft_angle_deg": 0.5,
            "textured_surface": False,
        })
        assert "DFM-IM-001" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-IM-001").tier == "A"

    def test_im_001_passes_textured_draft_ok(self):
        """DFM-IM-001: textured surface, draft 5° ≥ 3° → no violation."""
        violations = evaluate_molding_rules({
            "draft_angle_deg": 5.0,
            "textured_surface": True,
        })
        assert "DFM-IM-001" not in _rule_ids(violations)

    def test_im_002_fires_nonuniform_wall(self):
        """DFM-IM-002: wall 1.0–1.5 mm = 50% variation > 25% → violation."""
        violations = evaluate_molding_rules({
            "wall_thickness_min_mm": 1.0,
            "wall_thickness_max_mm": 1.5,
        })
        assert "DFM-IM-002" in _rule_ids(violations)

    def test_im_002_passes_uniform_wall(self):
        """DFM-IM-002: wall 2.0–2.2 mm = 10% variation ≤ 25% → no violation."""
        violations = evaluate_molding_rules({
            "wall_thickness_min_mm": 2.0,
            "wall_thickness_max_mm": 2.2,
        })
        assert "DFM-IM-002" not in _rule_ids(violations)

    def test_im_003_fires_undocumented_undercut(self):
        """DFM-IM-003: undercut present but not documented → violation."""
        violations = evaluate_molding_rules({
            "has_undercut": True,
            "undercut_tooling_documented": False,
        })
        assert "DFM-IM-003" in _rule_ids(violations)

    def test_im_003_passes_documented_undercut(self):
        """DFM-IM-003: undercut present and documented → no violation."""
        violations = evaluate_molding_rules({
            "has_undercut": True,
            "undercut_tooling_documented": True,
        })
        assert "DFM-IM-003" not in _rule_ids(violations)


# ---------------------------------------------------------------------------
# PCB tests
# ---------------------------------------------------------------------------

class TestPCBRules:
    def test_pcb_001_fires_trace_too_narrow(self):
        """DFM-PCB-001: trace width 0.08 mm < 0.10 mm standard → violation."""
        violations = evaluate_pcb_rules({
            "fab_class": "standard",
            "min_trace_width_mm": 0.08,
        })
        assert "DFM-PCB-001" in _rule_ids(violations)
        v = next(v for v in violations if v.rule_id == "DFM-PCB-001")
        assert v.tier == "A"

    def test_pcb_001_passes_adequate_trace(self):
        """DFM-PCB-001: trace width 0.12 mm ≥ 0.10 mm → no violation."""
        violations = evaluate_pcb_rules({
            "fab_class": "standard",
            "min_trace_width_mm": 0.12,
            "min_trace_space_mm": 0.12,
        })
        assert "DFM-PCB-001" not in _rule_ids(violations)

    def test_pcb_002_fires_small_annular_ring(self):
        """DFM-PCB-002: annular ring 0.10 mm < 0.15 mm → violation."""
        violations = evaluate_pcb_rules({"pad_annular_ring_mm": 0.10})
        assert "DFM-PCB-002" in _rule_ids(violations)

    def test_pcb_005_fires_missing_impedance_docs(self):
        """DFM-PCB-005: high-speed traces without impedance specs → violation."""
        violations = evaluate_pcb_rules({
            "high_speed_traces_present": True,
            "impedance_specs_documented": False,
        })
        assert "DFM-PCB-005" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-PCB-005").tier == "A"

    def test_pcb_005_passes_documented_impedance(self):
        """DFM-PCB-005: high-speed traces with impedance specs → no violation."""
        violations = evaluate_pcb_rules({
            "high_speed_traces_present": True,
            "impedance_specs_documented": True,
        })
        assert "DFM-PCB-005" not in _rule_ids(violations)

    def test_pcb_006_fires_high_via_aspect_ratio(self):
        """DFM-PCB-006 (C): via aspect ratio 12.0 > 10.0 → violation."""
        violations = evaluate_pcb_rules({"via_aspect_ratio": 12.0})
        assert "DFM-PCB-006" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-PCB-006").tier == "C"

    def test_pcb_006_passes_acceptable_via_ratio(self):
        """DFM-PCB-006 (C): via aspect ratio 8.0 ≤ 10.0 → no violation."""
        violations = evaluate_pcb_rules({"via_aspect_ratio": 8.0})
        assert "DFM-PCB-006" not in _rule_ids(violations)


# ---------------------------------------------------------------------------
# Harness tests
# ---------------------------------------------------------------------------

class TestHarnessRules:
    def test_hrn_001_fires_tight_bend_radius(self):
        """DFM-HRN-001: wire 3 mm, bend radius 25 mm < 10×3=30 mm → violation."""
        violations = evaluate_harness_rules({
            "wire_diameter_mm": 3.0,
            "min_bend_radius_mm": 25.0,
        })
        assert "DFM-HRN-001" in _rule_ids(violations)
        v = next(v for v in violations if v.rule_id == "DFM-HRN-001")
        assert v.tier == "A"

    def test_hrn_001_passes_adequate_bend_radius(self):
        """DFM-HRN-001: wire 3 mm, bend radius 35 mm ≥ 30 mm → no violation."""
        violations = evaluate_harness_rules({
            "wire_diameter_mm": 3.0,
            "min_bend_radius_mm": 35.0,
        })
        assert "DFM-HRN-001" not in _rule_ids(violations)

    def test_hrn_002_fires_insufficient_connector_clearance(self):
        """DFM-HRN-002: approach 40 mm < 50 mm → violation."""
        violations = evaluate_harness_rules({"connector_approach_clearance_mm": 40.0})
        assert "DFM-HRN-002" in _rule_ids(violations)

    def test_hrn_002_passes_adequate_connector_clearance(self):
        """DFM-HRN-002: approach 60 mm ≥ 50 mm → no violation."""
        violations = evaluate_harness_rules({"connector_approach_clearance_mm": 60.0})
        assert "DFM-HRN-002" not in _rule_ids(violations)

    def test_hrn_005_fires_large_bundle(self):
        """DFM-HRN-005 (C): bundle 30 mm > 25 mm → violation."""
        violations = evaluate_harness_rules({"bundle_diameter_mm": 30.0})
        assert "DFM-HRN-005" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFM-HRN-005").tier == "C"

    def test_hrn_005_passes_small_bundle(self):
        """DFM-HRN-005 (C): bundle 20 mm ≤ 25 mm → no violation."""
        violations = evaluate_harness_rules({"bundle_diameter_mm": 20.0})
        assert "DFM-HRN-005" not in _rule_ids(violations)


# ---------------------------------------------------------------------------
# Assembly tests
# ---------------------------------------------------------------------------

class TestAssemblyRules:
    def test_asm_001_fires_bad_insertion_direction(self):
        """DFA-ASM-001: 70% < 80% in primary direction → violation."""
        violations = evaluate_assembly_rules({"primary_direction_fraction": 0.70})
        assert "DFA-ASM-001" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFA-ASM-001").tier == "A"

    def test_asm_001_passes_good_insertion(self):
        """DFA-ASM-001: 90% ≥ 80% → no violation."""
        violations = evaluate_assembly_rules({"primary_direction_fraction": 0.90})
        assert "DFA-ASM-001" not in _rule_ids(violations)

    def test_asm_002_fires_inaccessible_fasteners(self):
        """DFA-ASM-002: fastener_accessible=False → violation."""
        violations = evaluate_assembly_rules({"fastener_accessible": False})
        assert "DFA-ASM-002" in _rule_ids(violations)

    def test_asm_002_passes_accessible_fasteners(self):
        """DFA-ASM-002: fastener_accessible=True → no violation."""
        violations = evaluate_assembly_rules({"fastener_accessible": True})
        assert "DFA-ASM-002" not in _rule_ids(violations)

    def test_asm_003_fires_too_many_steps(self):
        """DFA-ASM-003: 25 steps > 2×10 parts = 20 → violation."""
        violations = evaluate_assembly_rules({
            "assembly_steps": 25,
            "part_count": 10,
        })
        assert "DFA-ASM-003" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFA-ASM-003").tier == "B"

    def test_asm_003_passes_acceptable_steps(self):
        """DFA-ASM-003: 15 steps ≤ 2×10 parts = 20 → no violation."""
        violations = evaluate_assembly_rules({
            "assembly_steps": 15,
            "part_count": 10,
        })
        assert "DFA-ASM-003" not in _rule_ids(violations)

    def test_asm_007_fires_missing_poka_yoke(self):
        """DFA-ASM-007: symmetric parts without poka-yoke → violation."""
        violations = evaluate_assembly_rules({"symmetric_parts_poka_yoke": False})
        assert "DFA-ASM-007" in _rule_ids(violations)
        assert next(v for v in violations if v.rule_id == "DFA-ASM-007").tier == "A"

    def test_asm_007_passes_with_poka_yoke(self):
        """DFA-ASM-007: symmetric parts with poka-yoke → no violation."""
        violations = evaluate_assembly_rules({"symmetric_parts_poka_yoke": True})
        assert "DFA-ASM-007" not in _rule_ids(violations)


# ---------------------------------------------------------------------------
# evaluate_all_rules dispatcher
# ---------------------------------------------------------------------------

class TestEvaluateAllRules:
    def test_dispatcher_runs_correct_families(self):
        """evaluate_all_rules only runs the requested process families."""
        data = {
            "wall_thickness_mm": 0.3,   # triggers DFM-CNC-001 and DFM-AM-001
            "am_process": "metal_pbf",
        }
        violations = evaluate_all_rules(data, ["cnc"])
        rule_ids = _rule_ids(violations)
        assert "DFM-CNC-001" in rule_ids
        assert "DFM-AM-001" not in rule_ids

    def test_dispatcher_multiple_families(self):
        """evaluate_all_rules combines violations from multiple families."""
        data = {
            "wall_thickness_mm": 0.3,   # DFM-CNC-001 + DFM-AM-001
            "am_process": "metal_pbf",
        }
        violations = evaluate_all_rules(data, ["cnc", "am"])
        rule_ids = _rule_ids(violations)
        assert "DFM-CNC-001" in rule_ids
        assert "DFM-AM-001" in rule_ids

    def test_dispatcher_unknown_family_ignored(self):
        """evaluate_all_rules silently ignores unknown family names."""
        violations = evaluate_all_rules({}, ["unknown_process"])
        assert violations == []

    def test_dispatcher_empty_families(self):
        """evaluate_all_rules with empty families returns empty list."""
        violations = evaluate_all_rules({"wall_thickness_mm": 0.3}, [])
        assert violations == []
