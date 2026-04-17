"""
Base adapter interface for simulation solvers.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AdapterResult:
    success: bool
    outputs: dict                        # solver-specific result fields
    solver_version: str
    input_file_hash: str                 # sha256 of inputs
    wall_clock_seconds: float
    error_message: str = ""
    solver_warnings: list[str] = field(default_factory=list)


class SimulationAdapter(ABC):
    @abstractmethod
    def run(self, inputs: dict, params: dict) -> AdapterResult:
        """Execute the simulation and return an AdapterResult."""
        ...

    @abstractmethod
    def solver_type(self) -> str:
        """Return the solver identifier string."""
        ...


def _hash_inputs(inputs: dict) -> str:
    """Compute a deterministic sha256 digest of the inputs dict."""
    serialized = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()
