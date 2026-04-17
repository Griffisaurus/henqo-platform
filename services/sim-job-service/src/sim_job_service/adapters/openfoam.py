"""
OpenFOAM simulation adapter (stub).

Production implementation wraps subprocess calls to the OpenFOAM CLI.
This stub returns deterministic results for testing and development.
"""
from __future__ import annotations

import time

from sim_job_service.adapters.base import AdapterResult, SimulationAdapter, _hash_inputs

# Default convergent CFD result
_DEFAULT_OUTPUTS: dict = {
    "final_residual": 1e-5,
    "max_velocity_m_s": 50.0,
    "min_pressure_pa": 101325.0,
    "solver_warnings": [],
}


class OpenFOAMAdapter(SimulationAdapter):
    """Stub adapter. Production wraps subprocess calls to OpenFOAM CLI."""

    def run(self, inputs: dict, params: dict) -> AdapterResult:
        """
        Deterministic stub.

        If params contains "stub_result", use that dict as outputs.
        Otherwise use default convergent CFD outputs.
        If params contains "stub_fail": True, return a failed result.
        """
        start = time.monotonic()

        if params.get("stub_fail", False):
            elapsed = time.monotonic() - start
            return AdapterResult(
                success=False,
                outputs={},
                solver_version=self.solver_type(),
                input_file_hash=_hash_inputs(inputs),
                wall_clock_seconds=elapsed,
                error_message=params.get("stub_fail_message", "OpenFOAM solver failed"),
                solver_warnings=[],
            )

        outputs = dict(params.get("stub_result", _DEFAULT_OUTPUTS))
        # Ensure solver_warnings is present in outputs for auto-validation
        if "solver_warnings" not in outputs:
            outputs["solver_warnings"] = []

        elapsed = time.monotonic() - start
        return AdapterResult(
            success=True,
            outputs=outputs,
            solver_version=self.solver_type(),
            input_file_hash=_hash_inputs(inputs),
            wall_clock_seconds=elapsed,
            error_message="",
            solver_warnings=outputs.get("solver_warnings", []),
        )

    def solver_type(self) -> str:
        return "openfoam"
