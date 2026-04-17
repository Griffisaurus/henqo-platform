"""
Unit tests for requirements_service.api.handlers.RequirementsServiceHandler.
"""
from __future__ import annotations

import sys
import os

# Ensure graph-service and schema packages are importable when running via
# the PYTHONPATH=... pytest invocation documented in the epic backlog.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "graph-service", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "packages", "schema", "src"))

import pytest

from graph_service.api.handlers import GraphServiceHandler  # type: ignore[import]
from requirements_service.api.handlers import (
    IngestRequirementRequest,
    IngestRequirementResponse,
    RequirementsServiceHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(existing: list[tuple[str, str]] | None = None) -> RequirementsServiceHandler:
    graph = GraphServiceHandler()
    return RequirementsServiceHandler(graph_handler=graph, existing_requirements=existing)


def _req(
    text: str = "< 2 mm",
    criticality: str = "High",
    decision_class_required: str = "DesignGate",
    idempotency_key: str = "",
    source_document_id: str = "",
) -> IngestRequirementRequest:
    return IngestRequirementRequest(
        text=text,
        criticality=criticality,
        decision_class_required=decision_class_required,
        idempotency_key=idempotency_key,
        source_document_id=source_document_id,
    )


# ---------------------------------------------------------------------------
# Successful ingest
# ---------------------------------------------------------------------------

class TestSuccessfulIngest:
    def test_basic_upper_bound_stores_entity(self) -> None:
        handler = _make_handler()
        resp = handler.ingest(_req(text="< 2 mm"))
        assert resp.ok
        assert resp.entity_id != ""
        assert resp.is_duplicate is False
        assert resp.parsed is not None
        assert resp.parsed.direction == "upper_bound"
        assert resp.parsed.value == 2.0
        assert resp.parsed.unit == "mm"

    def test_lower_bound(self) -> None:
        handler = _make_handler()
        resp = handler.ingest(_req(text="> 500 N"))
        assert resp.ok
        assert resp.parsed is not None
        assert resp.parsed.direction == "lower_bound"

    def test_range_ingest(self) -> None:
        handler = _make_handler()
        resp = handler.ingest(_req(text="between 10 and 20 MPa"))
        assert resp.ok
        assert resp.parsed is not None
        assert resp.parsed.direction == "range"
        assert resp.parsed.value == 10.0
        assert resp.parsed.value_upper == 20.0

    def test_qualitative_ingest(self) -> None:
        handler = _make_handler()
        resp = handler.ingest(_req(text="minimize weight"))
        assert resp.ok
        assert resp.parsed is not None
        assert resp.parsed.is_qualitative is True

    def test_entity_stored_for_subsequent_dedup(self) -> None:
        handler = _make_handler()
        first = handler.ingest(_req(text="< 2 mm"))
        assert first.ok
        # A clearly different requirement should not be flagged as duplicate
        second = handler.ingest(_req(text="at least 1000 kN"))
        assert second.ok
        assert second.is_duplicate is False
        # There should now be two entries in internal state
        assert len(handler._existing) == 2

    def test_quantity_attached_when_quantitative(self) -> None:
        """Quantity dict must be present in the graph entity for quantitative reqs."""
        graph = GraphServiceHandler()
        handler = RequirementsServiceHandler(graph_handler=graph)
        resp = handler.ingest(_req(text="shall not exceed 85 degC"))
        assert resp.ok
        # Retrieve entity from graph to verify quantity was written
        from graph_service.api.handlers import GetEntityRequest  # type: ignore[import]
        get_resp = graph.get_entity(GetEntityRequest(entity_id=resp.entity_id))
        assert get_resp.ok
        assert "quantity" in get_resp.payload
        assert get_resp.payload["quantity"]["value"] == 85.0
        assert get_resp.payload["quantity"]["unit"] == "degC"

    def test_qualitative_has_no_quantity_field(self) -> None:
        graph = GraphServiceHandler()
        handler = RequirementsServiceHandler(graph_handler=graph)
        resp = handler.ingest(_req(text="minimize weight"))
        assert resp.ok
        from graph_service.api.handlers import GetEntityRequest  # type: ignore[import]
        get_resp = graph.get_entity(GetEntityRequest(entity_id=resp.entity_id))
        assert "quantity" not in get_resp.payload

    def test_source_document_id_stored(self) -> None:
        graph = GraphServiceHandler()
        handler = RequirementsServiceHandler(graph_handler=graph)
        resp = handler.ingest(_req(text="< 5 mm", source_document_id="DOC-001"))
        assert resp.ok
        from graph_service.api.handlers import GetEntityRequest  # type: ignore[import]
        get_resp = graph.get_entity(GetEntityRequest(entity_id=resp.entity_id))
        assert get_resp.payload.get("source_document_id") == "DOC-001"


# ---------------------------------------------------------------------------
# ParseError → REQUIREMENT_NOT_PARSEABLE
# ---------------------------------------------------------------------------

class TestParseError:
    def test_unparseable_returns_error(self) -> None:
        handler = _make_handler()
        resp = handler.ingest(_req(text="the component shall be acceptable"))
        assert not resp.ok
        assert resp.error_code == "REQUIREMENT_NOT_PARSEABLE"
        assert resp.entity_id == ""

    def test_empty_text_returns_error(self) -> None:
        handler = _make_handler()
        resp = handler.ingest(_req(text=""))
        assert not resp.ok
        assert resp.error_code == "REQUIREMENT_NOT_PARSEABLE"

    def test_ambiguous_quantity_returns_error(self) -> None:
        handler = _make_handler()
        resp = handler.ingest(_req(text="the gap is 3 mm"))
        assert not resp.ok
        assert resp.error_code == "REQUIREMENT_NOT_PARSEABLE"


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_exact_duplicate(self) -> None:
        handler = _make_handler()
        first = handler.ingest(_req(text="< 2 mm"))
        assert first.ok
        second = handler.ingest(_req(text="< 2 mm"))
        assert second.is_duplicate is True
        assert second.duplicate_of == first.entity_id
        assert second.entity_id == ""

    def test_near_duplicate(self) -> None:
        handler = _make_handler()
        first = handler.ingest(_req(text="shall not exceed 85 degC"))
        assert first.ok
        # "must" is a stopword, so token set is identical → Jaccard = 1.0
        second = handler.ingest(_req(text="must not exceed 85 degC"))
        assert second.is_duplicate is True
        assert second.duplicate_of == first.entity_id

    def test_different_requirements_not_flagged(self) -> None:
        handler = _make_handler()
        handler.ingest(_req(text="< 2 mm"))
        second = handler.ingest(_req(text="at least 1000 kN at rated speed"))
        assert second.is_duplicate is False

    def test_preloaded_existing_requirements(self) -> None:
        """Seed handler with an existing requirement; new ingest should detect dup."""
        existing_id = "11111111-1111-4111-8111-111111111111"
        handler = _make_handler(existing=[(existing_id, "< 2 mm")])
        resp = handler.ingest(_req(text="< 2 mm"))
        assert resp.is_duplicate is True
        assert resp.duplicate_of == existing_id


# ---------------------------------------------------------------------------
# Idempotency key
# ---------------------------------------------------------------------------

class TestIdempotencyKey:
    def test_idempotency_key_returns_same_entity(self) -> None:
        graph = GraphServiceHandler()
        handler = RequirementsServiceHandler(graph_handler=graph)
        key = "idem-key-abc-123"
        first = handler.ingest(_req(text="< 2 mm", idempotency_key=key))
        assert first.ok
        first_id = first.entity_id

        # Second call with same key but different text — graph should return same entity
        # (dedup fires before graph, so we use a fresh handler sharing the same graph)
        handler2 = RequirementsServiceHandler(graph_handler=graph)
        second = handler2.ingest(_req(text="< 2 mm", idempotency_key=key))
        assert second.ok
        assert second.entity_id == first_id

    def test_empty_idempotency_key_creates_new_entity(self) -> None:
        handler = _make_handler()
        # Two different requirements with no idempotency key → both succeed
        r1 = handler.ingest(_req(text="< 2 mm"))
        r2 = handler.ingest(_req(text="> 500 N"))
        assert r1.entity_id != r2.entity_id


# ---------------------------------------------------------------------------
# Graph failure propagation
# ---------------------------------------------------------------------------

class TestGraphFailure:
    def test_graph_error_propagates(self) -> None:
        """If graph service returns an error, handler surfaces it."""

        class FailingGraph:
            def create_entity(self, req):  # type: ignore[no-untyped-def]
                from graph_service.api.handlers import CreateEntityResponse  # type: ignore[import]
                return CreateEntityResponse(
                    error_code="SCHEMA_VALIDATION_ERROR",
                    error_message="injected failure",
                )

        handler = RequirementsServiceHandler(graph_handler=FailingGraph())
        resp = handler.ingest(_req(text="< 2 mm"))
        assert not resp.ok
        assert resp.error_code == "SCHEMA_VALIDATION_ERROR"
        assert "injected failure" in resp.error_message
