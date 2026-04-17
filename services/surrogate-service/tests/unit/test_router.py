"""
Unit tests for surrogate_service.domain.router.

Covers:
  - Known routes resolve to correct family
  - Unknown domain/discipline raises ValueError
  - Decision class thresholds are correct
  - Unknown decision class raises ValueError
"""
from __future__ import annotations

import pytest

from surrogate_service.domain.router import (
    get_decision_class_threshold,
    route_to_surrogate,
)


class TestRouteToSurrogate:
    def test_aerodynamics_cfd_is_s1(self):
        assert route_to_surrogate("aerodynamics", "cfd") == "S1"

    def test_aerodynamics_airfoil_is_s1(self):
        assert route_to_surrogate("aerodynamics", "airfoil") == "S1"

    def test_structural_fem_is_s2(self):
        assert route_to_surrogate("structural", "fem") == "S2"

    def test_structural_structural_is_s2(self):
        assert route_to_surrogate("structural", "structural") == "S2"

    def test_thermal_fem_is_s2(self):
        assert route_to_surrogate("thermal", "fem") == "S2"

    def test_prognostics_rul_is_s3(self):
        assert route_to_surrogate("prognostics", "rul") == "S3"

    def test_materials_atomistic_is_s4(self):
        assert route_to_surrogate("materials", "atomistic") == "S4"

    def test_materials_molecular_is_s4(self):
        assert route_to_surrogate("materials", "molecular") == "S4"

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="No surrogate route"):
            route_to_surrogate("unknown_physics", "unknown_discipline")

    def test_unknown_discipline_raises(self):
        with pytest.raises(ValueError):
            route_to_surrogate("aerodynamics", "nonexistent")

    def test_case_insensitive(self):
        """Input should be lower-cased before lookup."""
        assert route_to_surrogate("Aerodynamics", "CFD") == "S1"


class TestGetDecisionClassThreshold:
    def test_exploratory_threshold(self):
        assert get_decision_class_threshold("Exploratory") == 0.80

    def test_design_gate_threshold(self):
        assert get_decision_class_threshold("DesignGate") == 0.90

    def test_release_critical_threshold_unreachable(self):
        """ReleaseCritical threshold is > 1.0 (always abstain)."""
        t = get_decision_class_threshold("ReleaseCritical")
        assert t > 1.0

    def test_safety_critical_threshold_unreachable(self):
        t = get_decision_class_threshold("SafetyCritical")
        assert t > 1.0

    def test_unknown_class_raises(self):
        with pytest.raises(ValueError, match="Unknown decision class"):
            get_decision_class_threshold("UnknownClass")

    def test_design_gate_strictly_above_exploratory(self):
        assert get_decision_class_threshold("DesignGate") > get_decision_class_threshold("Exploratory")
