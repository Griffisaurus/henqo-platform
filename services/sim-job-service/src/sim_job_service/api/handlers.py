"""
Simulation Job Service request handlers.

Implements SubmitJob and GetJobStatus RPCs as described in
service-contracts-api-spec.md §4.
"""
from __future__ import annotations

import sys
import os

# Allow running without installation — resolve graph-service and schema packages
_HERE = os.path.dirname(__file__)
sys.path.insert(
    0,
    os.path.join(_HERE, "..", "..", "..", "..", "..", "packages", "schema", "src"),
)
sys.path.insert(
    0,
    os.path.join(_HERE, "..", "..", "..", "..", "..", "services", "graph-service", "src"),
)

from dataclasses import dataclass, field
from typing import Any

from graph_service.api.handlers import (
    CreateEntityRequest,
    GetEntityRequest,
    GraphServiceHandler,
    UpdateEntityStateRequest,
)
from graph_service.persistence.store import new_uuid

from sim_job_service.adapters.base import SimulationAdapter
from sim_job_service.adapters.fenics import FEniCSAdapter
from sim_job_service.adapters.openfoam import OpenFOAMAdapter
from sim_job_service.domain.auto_validation import AutoValidationResult, run_auto_validation
from sim_job_service.domain.failure_tracker import FailureTracker


# ---------------------------------------------------------------------------
# Request / Response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SubmitJobRequest:
    solver_type: str               # "openfoam" | "fenics"
    discipline: str                # "cfd" | "structural" | etc.
    component_revision_id: str
    inputs: dict                   # solver input parameters
    solver_params: dict = field(default_factory=dict)
    idempotency_key: str = ""
    requesting_service: str = "system"


@dataclass
class SubmitJobResponse:
    simulation_case_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class GetJobStatusRequest:
    simulation_case_id: str


@dataclass
class GetJobStatusResponse:
    simulation_case_id: str = ""
    status: str = ""
    auto_validation_result: AutoValidationResult | None = None
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


# ---------------------------------------------------------------------------
# Default adapter registry
# ---------------------------------------------------------------------------

_DEFAULT_ADAPTERS: dict[str, SimulationAdapter] = {
    "openfoam": OpenFOAMAdapter(),
    "fenics": FEniCSAdapter(),
}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class SimJobServiceHandler:
    def __init__(
        self,
        graph_handler: GraphServiceHandler,
        failure_tracker: FailureTracker,
        adapters: dict[str, SimulationAdapter] | None = None,
    ) -> None:
        self._graph = graph_handler
        self._failure_tracker = failure_tracker
        self._adapters: dict[str, SimulationAdapter] = (
            adapters if adapters is not None else dict(_DEFAULT_ADAPTERS)
        )

    # ------------------------------------------------------------------
    # SubmitJob
    # ------------------------------------------------------------------

    def submit_job(self, req: SubmitJobRequest) -> SubmitJobResponse:
        # Resolve adapter — fail fast with a clear error if solver unknown
        adapter = self._adapters.get(req.solver_type)
        if adapter is None:
            return SubmitJobResponse(
                error_code="INVALID_SOLVER",
                error_message=f"Unknown solver type: {req.solver_type!r}",
            )

        # 1. Build and persist the SimulationCase entity
        entity_id = new_uuid()
        payload: dict[str, Any] = {
            "entity_id": entity_id,
            "discipline": req.discipline,
            "solver": req.solver_type,
            "solver_version": adapter.solver_type(),
            "component_revision_id": req.component_revision_id,
            "status": "queued",
        }

        # Do NOT forward idempotency_key to create_entity: each submit attempt
        # must produce a new SimulationCase entity (the key is used only by the
        # failure_tracker to link retries of the same logical job).
        create_resp = self._graph.create_entity(
            CreateEntityRequest(
                entity_type="SimulationCase",
                payload=payload,
                created_by=req.requesting_service,
            )
        )
        if not create_resp.ok:
            return SubmitJobResponse(
                error_code=create_resp.error_code,
                error_message=create_resp.error_message,
            )

        sim_case_id = create_resp.entity_id

        # 2. Transition queued → running
        run_resp = self._graph.update_entity_state(
            UpdateEntityStateRequest(
                entity_id=sim_case_id,
                new_state="running",
                transition_reason="job started",
            )
        )
        if not run_resp.ok:
            return SubmitJobResponse(
                error_code=run_resp.error_code,
                error_message=run_resp.error_message,
            )

        # 3. Execute the solver
        adapter_result = adapter.run(req.inputs, req.solver_params)

        # 4. Handle adapter failure
        if not adapter_result.success:
            self._graph.update_entity_state(
                UpdateEntityStateRequest(
                    entity_id=sim_case_id,
                    new_state="failed",
                    transition_reason=adapter_result.error_message,
                )
            )

            idem_key = req.idempotency_key or sim_case_id
            self._failure_tracker.record_failure(idem_key)

            if self._failure_tracker.is_double_failure(idem_key):
                # Emit double-failure event as a graph entity
                evt_id = new_uuid()
                self._graph.create_entity(
                    CreateEntityRequest(
                        entity_type="ProvenanceBundle",
                        payload={
                            "bundle_id": evt_id,
                            "activity": "double_failure_detected",
                            "agent_id": req.requesting_service,
                            "simulation_case_id": sim_case_id,
                            "idempotency_key": idem_key,
                        },
                        created_by=req.requesting_service,
                    )
                )
                return SubmitJobResponse(
                    simulation_case_id=sim_case_id,
                    error_code="DOUBLE_FAILURE",
                    error_message=(
                        f"Job {idem_key!r} has failed twice; manual intervention required"
                    ),
                )

            return SubmitJobResponse(
                simulation_case_id=sim_case_id,
                error_code="SOLVER_FAILED",
                error_message=adapter_result.error_message,
            )

        # 5. Run auto-validation
        validation = run_auto_validation(req.discipline, adapter_result.outputs)

        # 6. Transition running → completed
        self._graph.update_entity_state(
            UpdateEntityStateRequest(
                entity_id=sim_case_id,
                new_state="completed",
                transition_reason="solver finished",
            )
        )

        # 7. If validation passed, also transition completed → validated
        if validation.passed:
            self._graph.update_entity_state(
                UpdateEntityStateRequest(
                    entity_id=sim_case_id,
                    new_state="validated",
                    transition_reason="auto-validation passed",
                )
            )

        return SubmitJobResponse(simulation_case_id=sim_case_id)

    # ------------------------------------------------------------------
    # GetJobStatus
    # ------------------------------------------------------------------

    def get_job_status(self, req: GetJobStatusRequest) -> GetJobStatusResponse:
        get_resp = self._graph.get_entity(
            GetEntityRequest(entity_id=req.simulation_case_id)
        )
        if not get_resp.ok:
            return GetJobStatusResponse(
                error_code=get_resp.error_code,
            )

        return GetJobStatusResponse(
            simulation_case_id=req.simulation_case_id,
            status=get_resp.status,
        )
