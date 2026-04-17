"""
Tests for decision_pkg_service.domain.gating (EP-08).

≥6 tests covering:
- Unverified characteristic → blocked
- Unresolved characteristic → blocked
- All characteristics simulation_validated → passes
- Released characteristic passes all gates
"""
from __future__ import annotations

import pytest

from decision_pkg_service.domain.gating import GatingResult, check_characteristic_gating


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _char(char_id: str, status: str, trust_class: str = "") -> dict:
    char: dict = {"entity_id": char_id, "status": status}
    if trust_class:
        char["trust_bundle"] = {"evaluated_decision_class": trust_class}
    return char


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBlockedStatuses:
    def test_unverified_characteristic_is_blocked(self):
        chars = [_char("CH-001", "unverified")]
        result = check_characteristic_gating(chars, "PDR")
        assert result.passed is False
        assert "CH-001" in result.blocked_characteristics
        assert result.reason != ""

    def test_unresolved_characteristic_is_blocked(self):
        chars = [_char("CH-001", "unresolved")]
        result = check_characteristic_gating(chars, "PDR")
        assert result.passed is False
        assert "CH-001" in result.blocked_characteristics

    def test_multiple_blocked_characteristics(self):
        chars = [
            _char("CH-001", "unverified"),
            _char("CH-002", "unresolved"),
        ]
        result = check_characteristic_gating(chars, "PDR")
        assert result.passed is False
        assert set(result.blocked_characteristics) == {"CH-001", "CH-002"}


class TestPassingStatuses:
    def test_simulation_validated_passes_pdr(self):
        chars = [_char("CH-001", "simulation_validated")]
        result = check_characteristic_gating(chars, "PDR")
        assert result.passed is True
        assert result.blocked_characteristics == []

    def test_released_passes_all_gates(self):
        for gate in ["PDR", "CDR", "PRR", "FRR"]:
            chars = [_char("CH-001", "released")]
            result = check_characteristic_gating(chars, gate)
            assert result.passed is True, f"Expected pass at {gate}"

    def test_inspection_confirmed_passes_frr(self):
        chars = [_char("CH-001", "inspection_confirmed")]
        result = check_characteristic_gating(chars, "FRR")
        assert result.passed is True

    def test_empty_characteristics_passes(self):
        result = check_characteristic_gating([], "CDR")
        assert result.passed is True


class TestDesignGatePlus:
    """At CDR+, surrogate_estimated chars must have DesignGate-level trust."""

    def test_surrogate_estimated_exploratory_blocked_at_cdr(self):
        chars = [_char("CH-001", "surrogate_estimated", trust_class="Exploratory")]
        result = check_characteristic_gating(chars, "CDR")
        assert result.passed is False
        assert "CH-001" in result.blocked_characteristics

    def test_surrogate_estimated_design_gate_passes_cdr(self):
        chars = [_char("CH-001", "surrogate_estimated", trust_class="DesignGate")]
        result = check_characteristic_gating(chars, "CDR")
        assert result.passed is True

    def test_surrogate_estimated_exploratory_allowed_at_pdr(self):
        """Below CDR, surrogate_estimated with Exploratory class is fine."""
        chars = [_char("CH-001", "surrogate_estimated", trust_class="Exploratory")]
        result = check_characteristic_gating(chars, "PDR")
        assert result.passed is True

    def test_surrogate_estimated_release_critical_passes_frr(self):
        chars = [_char("CH-001", "surrogate_estimated", trust_class="ReleaseCritical")]
        result = check_characteristic_gating(chars, "FRR")
        assert result.passed is True
