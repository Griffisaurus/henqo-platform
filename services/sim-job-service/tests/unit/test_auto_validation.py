"""
Unit tests for domain/auto_validation.py.

One test per check rule for CFD and structural disciplines, plus the
unknown-discipline passthrough case.
"""
from __future__ import annotations

import pytest

from sim_job_service.domain.auto_validation import (
    AutoValidationResult,
    run_auto_validation,
    validate_cfd_results,
    validate_structural_results,
)


# ---------------------------------------------------------------------------
# CFD tests
# ---------------------------------------------------------------------------

class TestCFDValidation:
    def _good(self) -> dict:
        """Minimal passing CFD result set."""
        return {
            "final_residual": 1e-5,
            "convergence_threshold": 1e-4,
            "solver_warnings": [],
            "max_velocity_m_s": 50.0,
            "min_pressure_pa": 101325.0,
        }

    def test_residuals_converged_passes(self):
        result = validate_cfd_results(self._good())
        assert result.passed
        assert "residuals_converged" not in result.failures

    def test_residuals_converged_fails_when_above_threshold(self):
        d = self._good()
        d["final_residual"] = 1e-3   # > default threshold 1e-4
        result = validate_cfd_results(d)
        assert not result.passed
        assert "residuals_converged" in result.failures

    def test_residuals_converged_uses_default_threshold(self):
        # No convergence_threshold key → default 1e-4
        d = {"final_residual": 1e-5, "max_velocity_m_s": 50.0, "min_pressure_pa": 0.0}
        result = validate_cfd_results(d)
        assert "residuals_converged" not in result.failures

    def test_no_solver_warnings_passes_when_empty(self):
        result = validate_cfd_results(self._good())
        assert "no_solver_warnings" not in result.failures

    def test_no_solver_warnings_fails_when_warnings_present(self):
        d = self._good()
        d["solver_warnings"] = ["divergence in cell block 3"]
        result = validate_cfd_results(d)
        assert not result.passed
        assert "no_solver_warnings" in result.failures

    def test_velocity_plausible_passes(self):
        result = validate_cfd_results(self._good())
        assert "velocity_plausible" not in result.failures

    def test_velocity_plausible_fails_when_zero(self):
        d = self._good()
        d["max_velocity_m_s"] = 0.0
        result = validate_cfd_results(d)
        assert "velocity_plausible" in result.failures

    def test_velocity_plausible_fails_when_negative(self):
        d = self._good()
        d["max_velocity_m_s"] = -10.0
        result = validate_cfd_results(d)
        assert "velocity_plausible" in result.failures

    def test_velocity_plausible_fails_when_above_1000(self):
        d = self._good()
        d["max_velocity_m_s"] = 1500.0
        result = validate_cfd_results(d)
        assert "velocity_plausible" in result.failures

    def test_pressure_plausible_passes(self):
        result = validate_cfd_results(self._good())
        assert "pressure_plausible" not in result.failures

    def test_pressure_plausible_fails_when_below_negative_1e8(self):
        d = self._good()
        d["min_pressure_pa"] = -2e8
        result = validate_cfd_results(d)
        assert "pressure_plausible" in result.failures

    def test_all_checks_listed(self):
        result = validate_cfd_results(self._good())
        assert set(result.checks) == {
            "residuals_converged",
            "no_solver_warnings",
            "velocity_plausible",
            "pressure_plausible",
        }

    def test_multiple_failures_all_reported(self):
        d = {
            "final_residual": 1.0,     # fails residuals_converged
            "solver_warnings": ["w1"],  # fails no_solver_warnings
            "max_velocity_m_s": 50.0,
            "min_pressure_pa": 0.0,
        }
        result = validate_cfd_results(d)
        assert not result.passed
        assert "residuals_converged" in result.failures
        assert "no_solver_warnings" in result.failures


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestStructuralValidation:
    def _good(self) -> dict:
        return {
            "solver_warnings": [],
            "max_displacement_mm": 5.0,
            "displacement_limit_mm": 100.0,
            "singularity_detected": False,
            "min_element_quality": 0.8,
        }

    def test_no_solver_warnings_passes(self):
        result = validate_structural_results(self._good())
        assert "no_solver_warnings" not in result.failures

    def test_no_solver_warnings_fails_when_present(self):
        d = self._good()
        d["solver_warnings"] = ["ill-conditioned stiffness matrix"]
        result = validate_structural_results(d)
        assert not result.passed
        assert "no_solver_warnings" in result.failures

    def test_displacement_plausible_passes(self):
        result = validate_structural_results(self._good())
        assert "displacement_plausible" not in result.failures

    def test_displacement_plausible_fails_when_at_limit(self):
        d = self._good()
        d["max_displacement_mm"] = 100.0  # equal to limit — not strictly less
        result = validate_structural_results(d)
        assert "displacement_plausible" in result.failures

    def test_displacement_plausible_fails_when_exceeds_limit(self):
        d = self._good()
        d["max_displacement_mm"] = 150.0
        result = validate_structural_results(d)
        assert "displacement_plausible" in result.failures

    def test_displacement_uses_default_limit(self):
        d = {"max_displacement_mm": 5.0, "singularity_detected": False, "min_element_quality": 0.8}
        result = validate_structural_results(d)
        assert "displacement_plausible" not in result.failures

    def test_no_singularity_passes(self):
        result = validate_structural_results(self._good())
        assert "no_singularity" not in result.failures

    def test_no_singularity_fails_when_detected(self):
        d = self._good()
        d["singularity_detected"] = True
        result = validate_structural_results(d)
        assert not result.passed
        assert "no_singularity" in result.failures

    def test_element_quality_ok_passes(self):
        result = validate_structural_results(self._good())
        assert "element_quality_ok" not in result.failures

    def test_element_quality_ok_fails_when_poor(self):
        d = self._good()
        d["min_element_quality"] = 0.05  # < 0.1
        result = validate_structural_results(d)
        assert not result.passed
        assert "element_quality_ok" in result.failures

    def test_element_quality_fails_at_boundary(self):
        d = self._good()
        d["min_element_quality"] = 0.1  # not strictly > 0.1
        result = validate_structural_results(d)
        assert "element_quality_ok" in result.failures

    def test_all_checks_listed(self):
        result = validate_structural_results(self._good())
        assert set(result.checks) == {
            "no_solver_warnings",
            "displacement_plausible",
            "no_singularity",
            "element_quality_ok",
        }

    def test_full_pass(self):
        result = validate_structural_results(self._good())
        assert result.passed
        assert result.failures == []


# ---------------------------------------------------------------------------
# run_auto_validation routing
# ---------------------------------------------------------------------------

class TestRunAutoValidation:
    def test_routes_cfd(self):
        result = run_auto_validation("cfd", {"final_residual": 1e-5, "max_velocity_m_s": 50.0, "min_pressure_pa": 0.0})
        assert "residuals_converged" in result.checks

    def test_routes_aerodynamics(self):
        result = run_auto_validation("aerodynamics", {"final_residual": 1e-5, "max_velocity_m_s": 50.0, "min_pressure_pa": 0.0})
        assert "residuals_converged" in result.checks

    def test_routes_thermal_fluid(self):
        result = run_auto_validation("thermal_fluid", {"final_residual": 1e-5, "max_velocity_m_s": 50.0, "min_pressure_pa": 0.0})
        assert "residuals_converged" in result.checks

    def test_routes_structural(self):
        result = run_auto_validation("structural", {"max_displacement_mm": 5.0, "min_element_quality": 0.8})
        assert "displacement_plausible" in result.checks

    def test_routes_fem(self):
        result = run_auto_validation("fem", {"max_displacement_mm": 5.0, "min_element_quality": 0.8})
        assert "displacement_plausible" in result.checks

    def test_routes_thermal_solid(self):
        result = run_auto_validation("thermal_solid", {"max_displacement_mm": 5.0, "min_element_quality": 0.8})
        assert "displacement_plausible" in result.checks

    def test_unknown_discipline_passes_with_warning(self):
        result = run_auto_validation("magnetics", {})
        assert result.passed
        assert result.failures == []
        assert len(result.warnings) == 1
        assert "magnetics" in result.warnings[0]
        assert "no_checks_for_discipline" in result.checks
