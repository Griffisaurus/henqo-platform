"""
Auto-validation rules for simulation results.

Each discipline has a set of checks that must all pass for the
AutoValidationResult to be marked passed=True.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AutoValidationResult:
    passed: bool
    checks: list[str]     # names of checks run
    failures: list[str]   # names of failed checks
    warnings: list[str]   # non-blocking issues


def validate_cfd_results(results: dict) -> AutoValidationResult:
    """
    CFD auto-validation checks (all must pass):
    1. residuals_converged  — final_residual < convergence_threshold (default 1e-4)
    2. no_solver_warnings   — solver_warnings list is empty
    3. velocity_plausible   — 0 < max_velocity_m_s < 1000.0
    4. pressure_plausible   — min_pressure_pa > -1e8
    """
    checks = [
        "residuals_converged",
        "no_solver_warnings",
        "velocity_plausible",
        "pressure_plausible",
    ]
    failures: list[str] = []
    warnings: list[str] = []

    threshold = results.get("convergence_threshold", 1e-4)
    if not (results.get("final_residual", float("inf")) < threshold):
        failures.append("residuals_converged")

    if results.get("solver_warnings", []):
        failures.append("no_solver_warnings")

    max_vel = results.get("max_velocity_m_s", 1.0)
    if not (0 < max_vel < 1000.0):
        failures.append("velocity_plausible")

    min_p = results.get("min_pressure_pa", 0.0)
    if not (min_p > -1e8):
        failures.append("pressure_plausible")

    return AutoValidationResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        warnings=warnings,
    )


def validate_structural_results(results: dict) -> AutoValidationResult:
    """
    Structural auto-validation checks (all must pass):
    1. no_solver_warnings    — solver_warnings list is empty
    2. displacement_plausible — max_displacement_mm < displacement_limit_mm (default 100.0)
    3. no_singularity        — singularity_detected is False
    4. element_quality_ok    — min_element_quality > 0.1
    """
    checks = [
        "no_solver_warnings",
        "displacement_plausible",
        "no_singularity",
        "element_quality_ok",
    ]
    failures: list[str] = []
    warnings: list[str] = []

    if results.get("solver_warnings", []):
        failures.append("no_solver_warnings")

    disp_limit = results.get("displacement_limit_mm", 100.0)
    max_disp = results.get("max_displacement_mm", 0.0)
    if not (max_disp < disp_limit):
        failures.append("displacement_plausible")

    if results.get("singularity_detected", False):
        failures.append("no_singularity")

    min_eq = results.get("min_element_quality", 1.0)
    if not (min_eq > 0.1):
        failures.append("element_quality_ok")

    return AutoValidationResult(
        passed=len(failures) == 0,
        checks=checks,
        failures=failures,
        warnings=warnings,
    )


def run_auto_validation(discipline: str, results: dict) -> AutoValidationResult:
    """Route to the appropriate discipline validator."""
    if discipline in ("cfd", "aerodynamics", "thermal_fluid"):
        return validate_cfd_results(results)
    elif discipline in ("structural", "fem", "thermal_solid"):
        return validate_structural_results(results)
    else:
        return AutoValidationResult(
            passed=True,
            checks=["no_checks_for_discipline"],
            failures=[],
            warnings=[f"No auto-validation rules for discipline: {discipline}"],
        )
