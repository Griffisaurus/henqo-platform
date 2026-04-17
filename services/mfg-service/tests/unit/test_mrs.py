"""
Unit tests for MRS (Manufacturability Readiness Score) computation.

Validates:
  - No violations → MRS = 1.0
  - One Class A open → S_A = 0, MRS = 0.40
  - All B violations resolved → S_B = 1.0
  - Mixed violations (MR-002 golden case from eval-acceptance-spec §8)
  - Half-weight S_C formula
  - Resolved rule IDs are respected
"""
import pytest

from mfg_service.domain.dfm_rules import DFMViolation
from mfg_service.domain.mrs import MRSResult, compute_mrs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _viol(rule_id: str, tier: str) -> DFMViolation:
    return DFMViolation(
        rule_id=rule_id,
        tier=tier,
        description="test violation",
        measured_value=None,
        threshold_value=None,
        process_family="cnc",
    )


def _make_violations(
    a_count: int = 0,
    b_count: int = 0,
    c_count: int = 0,
    prefix: str = "DFM-TST",
) -> list[DFMViolation]:
    viols = []
    for i in range(a_count):
        viols.append(_viol(f"{prefix}-A{i:03d}", "A"))
    for i in range(b_count):
        viols.append(_viol(f"{prefix}-B{i:03d}", "B"))
    for i in range(c_count):
        viols.append(_viol(f"{prefix}-C{i:03d}", "C"))
    return viols


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMRSNoViolations:
    def test_no_violations_mrs_is_one(self):
        """No violations → MRS = 1.0."""
        result = compute_mrs([])
        assert result.mrs_score == pytest.approx(1.0)
        assert result.s_a == pytest.approx(1.0)
        assert result.s_b == pytest.approx(1.0)
        assert result.s_c == pytest.approx(1.0)

    def test_no_violations_counts_are_zero(self):
        result = compute_mrs([])
        assert result.open_a_count == 0
        assert result.open_b_count == 0
        assert result.open_c_count == 0
        assert result.total_b_count == 0
        assert result.total_c_count == 0


class TestMRSClassAViolation:
    def test_one_open_class_a_makes_s_a_zero(self):
        """One open Class A → S_A = 0, MRS = 0.60×0 + 0.30×1 + 0.10×1 = 0.40."""
        viols = [_viol("DFM-CNC-001", "A")]
        result = compute_mrs(viols)
        assert result.s_a == pytest.approx(0.0)
        assert result.s_b == pytest.approx(1.0)
        assert result.s_c == pytest.approx(1.0)
        assert result.mrs_score == pytest.approx(0.40)
        assert result.open_a_count == 1

    def test_two_open_class_a_makes_s_a_zero(self):
        """Multiple open Class A violations still → S_A = 0."""
        viols = _make_violations(a_count=2)
        result = compute_mrs(viols)
        assert result.s_a == pytest.approx(0.0)
        assert result.mrs_score == pytest.approx(0.40)

    def test_class_a_resolved_s_a_is_one(self):
        """Class A violation resolved → S_A = 1, MRS = 1.0."""
        viols = [_viol("DFM-CNC-001", "A")]
        result = compute_mrs(viols, resolved_rule_ids=["DFM-CNC-001"])
        assert result.s_a == pytest.approx(1.0)
        assert result.mrs_score == pytest.approx(1.0)
        assert result.open_a_count == 0


class TestMRSClassBViolations:
    def test_all_b_open(self):
        """10 Class B violations, all open: S_B = 0.0."""
        viols = _make_violations(b_count=10)
        result = compute_mrs(viols)
        assert result.s_b == pytest.approx(0.0)
        assert result.total_b_count == 10
        assert result.open_b_count == 10

    def test_all_b_resolved(self):
        """All B violations resolved → S_B = 1.0."""
        viols = _make_violations(b_count=5)
        resolved = [v.rule_id for v in viols]
        result = compute_mrs(viols, resolved_rule_ids=resolved)
        assert result.s_b == pytest.approx(1.0)
        assert result.open_b_count == 0

    def test_partial_b_open(self):
        """3 of 10 B violations open → S_B = 1 - 3/10 = 0.70."""
        viols = _make_violations(b_count=10)
        resolved = [v.rule_id for v in viols[:7]]  # resolve 7, leave 3 open
        result = compute_mrs(viols, resolved_rule_ids=resolved)
        assert result.s_b == pytest.approx(0.70)
        assert result.open_b_count == 3
        assert result.total_b_count == 10


class TestMRSClassCViolations:
    def test_all_c_open_uses_half_weight(self):
        """5 Class C all open: S_C = 1 - 0.5*(5/5) = 1 - 0.5 = 0.50."""
        viols = _make_violations(c_count=5)
        result = compute_mrs(viols)
        assert result.s_c == pytest.approx(0.50)

    def test_partial_c_open(self):
        """2 of 5 Class C open: S_C = 1 - 0.5*(2/5) = 1 - 0.2 = 0.80."""
        viols = _make_violations(c_count=5)
        resolved = [v.rule_id for v in viols[:3]]  # resolve 3, leave 2 open
        result = compute_mrs(viols, resolved_rule_ids=resolved)
        assert result.s_c == pytest.approx(0.80)
        assert result.open_c_count == 2


class TestMRSGoldenCaseMR002:
    """MR-002 from eval-acceptance-spec §8:
    0 Class A open, 3 of 10 Class B open, 2 of 5 Class C open.
    Expected: S_A=1.0, S_B=0.70, S_C=0.80, MRS=0.89.
    """

    def test_mr002_golden_case(self):
        b_viols = _make_violations(b_count=10)
        c_viols = _make_violations(c_count=5, prefix="DFM-C")
        all_viols = b_viols + c_viols
        # resolve 7 B and 3 C → 3 B open, 2 C open
        resolved = (
            [v.rule_id for v in b_viols[:7]]
            + [v.rule_id for v in c_viols[:3]]
        )
        result = compute_mrs(all_viols, resolved_rule_ids=resolved)

        assert result.s_a == pytest.approx(1.0)
        assert result.s_b == pytest.approx(0.70, abs=1e-6)
        assert result.s_c == pytest.approx(0.80, abs=1e-6)
        # MRS = 0.60*1.0 + 0.30*0.70 + 0.10*0.80 = 0.60 + 0.21 + 0.08 = 0.89
        assert result.mrs_score == pytest.approx(0.89, abs=1e-3)

    def test_mr002_counts(self):
        b_viols = _make_violations(b_count=10)
        c_viols = _make_violations(c_count=5, prefix="DFM-C")
        all_viols = b_viols + c_viols
        resolved = (
            [v.rule_id for v in b_viols[:7]]
            + [v.rule_id for v in c_viols[:3]]
        )
        result = compute_mrs(all_viols, resolved_rule_ids=resolved)

        assert result.total_b_count == 10
        assert result.open_b_count == 3
        assert result.total_c_count == 5
        assert result.open_c_count == 2
        assert result.open_a_count == 0


class TestMRSMixedViolations:
    def test_mixed_a_b_c_open(self):
        """1 A open, 5 B all open, 5 C all open."""
        viols = _make_violations(a_count=1, b_count=5, c_count=5)
        result = compute_mrs(viols)
        assert result.s_a == pytest.approx(0.0)
        assert result.s_b == pytest.approx(0.0)
        assert result.s_c == pytest.approx(0.5)
        expected_mrs = 0.60 * 0.0 + 0.30 * 0.0 + 0.10 * 0.5
        assert result.mrs_score == pytest.approx(expected_mrs)

    def test_none_resolved_ids_treated_as_empty(self):
        """resolved_rule_ids=None is equivalent to []."""
        viols = _make_violations(b_count=3)
        result_none = compute_mrs(viols, resolved_rule_ids=None)
        result_empty = compute_mrs(viols, resolved_rule_ids=[])
        assert result_none.mrs_score == result_empty.mrs_score

    def test_weight_formula(self):
        """Verify weight formula: 0.60*S_A + 0.30*S_B + 0.10*S_C."""
        viols = _make_violations(b_count=2, c_count=4)
        # 1 B resolved → S_B = 0.5; 2 C resolved → S_C = 1 - 0.5*(2/4) = 0.75
        resolved = [viols[0].rule_id, viols[2].rule_id, viols[3].rule_id]
        result = compute_mrs(viols, resolved_rule_ids=resolved)
        expected = 0.60 * 1.0 + 0.30 * result.s_b + 0.10 * result.s_c
        assert result.mrs_score == pytest.approx(expected, abs=1e-6)
