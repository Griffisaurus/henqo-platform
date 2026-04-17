"""
FEniCS simulation adapter (stub).

Production implementation wraps FEniCS Python API calls.
This stub returns deterministic results for testing and development.
"""
from __future__ import annotations

import time

from sim_job_service.adapters.base import AdapterResult, SimulationAdapter, _hash_inputs

# Default convergent structural result
_DEFAULT_OUTPUTS: dict = {
    "max_displacement_mm": 1.5,
    "min_element_quality": 0.8,
    "singularity_detected": False,
    "solver_warnings": [],
}


class FEniCSAdapter(SimulationAdapter):
    """Stub adapter. Production wraps FEniCS Python API calls."""

    def run(self, inputs: dict, params: dict) -> AdapterResult:
        """
        Deterministic stub.

        If params contains "stub_result", use that dict as outputs.
        Otherwise use default structural outputs.
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
                error_message=params.get("stub_fail_message", "FEniCS solver failed"),
                solver_warnings=[],
            )

        outputs = dict(params.get("stub_result", _DEFAULT_OUTPUTS))
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
        return "fenics"
