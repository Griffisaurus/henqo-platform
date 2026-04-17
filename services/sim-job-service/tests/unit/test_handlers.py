"""
Unit tests for api/handlers.py — SimJobServiceHandler.
"""
from __future__ import annotations

import sys
import os

# Ensure graph-service and schema are importable
_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, os.path.join(_REPO, "packages", "schema", "src"))
sys.path.insert(0, os.path.join(_REPO, "services", "graph-service", "src"))

import pytest

from graph_service.api.handlers import GraphServiceHandler

from sim_job_service.adapters.base import AdapterResult, SimulationAdapter, _hash_inputs
from sim_job_service.api.handlers import (
    GetJobStatusRequest,
    SimJobServiceHandler,
    SubmitJobRequest,
)
from sim_job_service.domain.failure_tracker import FailureTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _OKCFDAdapter(SimulationAdapter):
    """Always succeeds with convergent CFD results."""
    def run(self, inputs: dict, params: dict) -> AdapterResult:
        return AdapterResult(
            success=True,
            outputs={
                "final_residual": 1e-5,
                "max_velocity_m_s": 50.0,
                "min_pressure_pa": 101325.0,
                "solver_warnings": [],
            },
            solver_version="openfoam",
            input_file_hash=_hash_inputs(inputs),
            wall_clock_seconds=0.1,
        )
    def solver_type(self) -> str:
        return "openfoam"


class _FailCFDAdapter(SimulationAdapter):
    """Always fails."""
    def run(self, inputs: dict, params: dict) -> AdapterResult:
        return AdapterResult(
            success=False,
            outputs={},
            solver_version="openfoam",
            input_file_hash=_hash_inputs(inputs),
            wall_clock_seconds=0.1,
            error_message="divergence detected",
        )
    def solver_type(self) -> str:
        return "openfoam"


class _ValidationFailStructuralAdapter(SimulationAdapter):
    """Succeeds but returns results that fail structural auto-validation (singularity)."""
    def run(self, inputs: dict, params: dict) -> AdapterResult:
        return AdapterResult(
            success=True,
            outputs={
                "max_displacement_mm": 5.0,
                "displacement_limit_mm": 100.0,
                "singularity_detected": True,  # will fail auto-validation
                "min_element_quality": 0.8,
                "solver_warnings": [],
            },
            solver_version="fenics",
            input_file_hash=_hash_inputs(inputs),
            wall_clock_seconds=0.2,
        )
    def solver_type(self) -> str:
        return "fenics"


def _make_handler(adapter=None, fail_adapter=None) -> tuple[SimJobServiceHandler, GraphServiceHandler]:
    graph = GraphServiceHandler()
    tracker = FailureTracker()
    adapters: dict = {}
    if adapter:
        adapters["openfoam"] = adapter
    if fail_adapter:
        adapters["openfoam"] = fail_adapter
    return SimJobServiceHandler(graph, tracker, adapters), graph


def _cfd_request(**kwargs) -> SubmitJobRequest:
    defaults = dict(
        solver_type="openfoam",
        discipline="cfd",
        component_revision_id="comp-rev-001",
        inputs={"mesh": "box.msh"},
        solver_params={},
        idempotency_key="cfd-job-001",
        requesting_service="test-suite",
    )
    defaults.update(kwargs)
    return SubmitJobRequest(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubmitJobSuccessfulCFD:
    def test_response_is_ok(self):
        handler, _ = _make_handler(adapter=_OKCFDAdapter())
        resp = handler.submit_job(_cfd_request())
        assert resp.ok

    def test_simulation_case_id_returned(self):
        handler, _ = _make_handler(adapter=_OKCFDAdapter())
        resp = handler.submit_job(_cfd_request())
        assert resp.simulation_case_id != ""

    def test_entity_status_is_validated(self):
        handler, graph = _make_handler(adapter=_OKCFDAdapter())
        resp = handler.submit_job(_cfd_request())
        entity = graph.get_entity(__import__("graph_service.api.handlers", fromlist=["GetEntityRequest"]).GetEntityRequest(
            entity_id=resp.simulation_case_id
        ))
        assert entity.status == "validated"

    def test_graph_entity_has_correct_fields(self):
        handler, graph = _make_handler(adapter=_OKCFDAdapter())
        resp = handler.submit_job(_cfd_request())
        from graph_service.api.handlers import GetEntityRequest
        entity = graph.get_entity(GetEntityRequest(entity_id=resp.simulation_case_id))
        assert entity.payload["discipline"] == "cfd"
        assert entity.payload["solver"] == "openfoam"
        assert entity.payload["component_revision_id"] == "comp-rev-001"


class TestSubmitJobFailedAdapter:
    def test_response_has_error(self):
        handler, _ = _make_handler(fail_adapter=_FailCFDAdapter())
        resp = handler.submit_job(_cfd_request(idempotency_key="fail-job-001"))
        assert not resp.ok
        assert resp.error_code == "SOLVER_FAILED"

    def test_entity_status_is_failed(self):
        handler, graph = _make_handler(fail_adapter=_FailCFDAdapter())
        resp = handler.submit_job(_cfd_request(idempotency_key="fail-job-002"))
        from graph_service.api.handlers import GetEntityRequest
        entity = graph.get_entity(GetEntityRequest(entity_id=resp.simulation_case_id))
        assert entity.status == "failed"

    def test_failure_count_incremented(self):
        graph = GraphServiceHandler()
        tracker = FailureTracker()
        handler = SimJobServiceHandler(graph, tracker, {"openfoam": _FailCFDAdapter()})
        handler.submit_job(_cfd_request(idempotency_key="fail-count-key"))
        assert tracker.get_count("fail-count-key") == 1


class TestDoubleFailure:
    def _handler_with_fail_adapter(self):
        graph = GraphServiceHandler()
        tracker = FailureTracker()
        handler = SimJobServiceHandler(graph, tracker, {"openfoam": _FailCFDAdapter()})
        return handler, tracker

    def test_second_failure_returns_double_failure_error(self):
        handler, tracker = self._handler_with_fail_adapter()
        key = "double-fail-key"
        handler.submit_job(_cfd_request(idempotency_key=key))
        resp2 = handler.submit_job(_cfd_request(idempotency_key=key))
        assert resp2.error_code == "DOUBLE_FAILURE"

    def test_first_failure_does_not_return_double_failure(self):
        handler, _ = self._handler_with_fail_adapter()
        resp = handler.submit_job(_cfd_request(idempotency_key="single-fail-key"))
        assert resp.error_code == "SOLVER_FAILED"
        assert resp.error_code != "DOUBLE_FAILURE"

    def test_failure_tracker_count_after_double(self):
        handler, tracker = self._handler_with_fail_adapter()
        key = "count-check-key"
        handler.submit_job(_cfd_request(idempotency_key=key))
        handler.submit_job(_cfd_request(idempotency_key=key))
        assert tracker.is_double_failure(key)


class TestStructuralJobAutoValidationFailure:
    def test_status_stays_completed_when_validation_fails(self):
        graph = GraphServiceHandler()
        tracker = FailureTracker()
        handler = SimJobServiceHandler(graph, tracker, {"fenics": _ValidationFailStructuralAdapter()})
        resp = handler.submit_job(SubmitJobRequest(
            solver_type="fenics",
            discipline="structural",
            component_revision_id="comp-rev-struct-001",
            inputs={"mesh": "bracket.msh"},
            solver_params={},
            idempotency_key="struct-val-fail-001",
            requesting_service="test-suite",
        ))
        assert resp.ok
        from graph_service.api.handlers import GetEntityRequest
        entity = graph.get_entity(GetEntityRequest(entity_id=resp.simulation_case_id))
        assert entity.status == "completed"

    def test_response_is_ok_even_when_validation_fails(self):
        """Auto-validation failure is not a job error; manual review handles it."""
        graph = GraphServiceHandler()
        tracker = FailureTracker()
        handler = SimJobServiceHandler(graph, tracker, {"fenics": _ValidationFailStructuralAdapter()})
        resp = handler.submit_job(SubmitJobRequest(
            solver_type="fenics",
            discipline="structural",
            component_revision_id="comp-rev-struct-002",
            inputs={},
            solver_params={},
        ))
        assert resp.ok


class TestUnknownSolverType:
    def test_unknown_solver_returns_invalid_solver_error(self):
        handler, _ = _make_handler()
        resp = handler.submit_job(_cfd_request(solver_type="abaqus"))
        assert not resp.ok
        assert resp.error_code == "INVALID_SOLVER"

    def test_unknown_solver_message_contains_solver_name(self):
        handler, _ = _make_handler()
        resp = handler.submit_job(_cfd_request(solver_type="nastran"))
        assert "nastran" in resp.error_message


class TestGetJobStatus:
    def test_returns_status_for_existing_job(self):
        handler, _ = _make_handler(adapter=_OKCFDAdapter())
        submit_resp = handler.submit_job(_cfd_request(idempotency_key="status-test-001"))
        status_resp = handler.get_job_status(GetJobStatusRequest(
            simulation_case_id=submit_resp.simulation_case_id
        ))
        assert status_resp.ok
        assert status_resp.status == "validated"
        assert status_resp.simulation_case_id == submit_resp.simulation_case_id

    def test_returns_error_for_nonexistent_job(self):
        handler, _ = _make_handler(adapter=_OKCFDAdapter())
        status_resp = handler.get_job_status(GetJobStatusRequest(
            simulation_case_id="00000000-0000-4000-8000-000000000000"
        ))
        assert not status_resp.ok
        assert status_resp.error_code == "NOT_FOUND"
