"""
Unit tests for graph service handlers (CreateEntity, GetEntity, UpdateEntityState,
QueryEntities, CreateRelation, GetProvenanceChain).
"""
from __future__ import annotations

import sys
import os

# Make schema package importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "packages", "schema", "src"),
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "src"),
)

import pytest

from graph_service.api.handlers import (
    CreateEntityRequest,
    CreateRelationRequest,
    GetEntityRequest,
    GetProvenanceChainRequest,
    GraphServiceHandler,
    QueryEntitiesRequest,
    UpdateEntityStateRequest,
)
from graph_service.persistence.store import InMemoryEntityStore


def _handler() -> GraphServiceHandler:
    return GraphServiceHandler(store=InMemoryEntityStore())


def _req_payload(**overrides):
    base = {
        "entity_id": "a1b2c3d4-e5f6-4789-ab12-cd34ef567890",
        "text": "Max tip displacement < 2 mm",
        "criticality": "High",
        "decision_class_required": "DesignGate",
    }
    base.update(overrides)
    return base


def _char_payload(**overrides):
    base = {
        "entity_id": "b2c3d4e5-f6a7-4890-bc23-de45fa678901",
        "name": "tip_displacement",
        "quantity_kind": "Length",
        "unit": "mm",
        "governing_requirement_id": "a1b2c3d4-e5f6-4789-ab12-cd34ef567890",
        "criticality": "High",
        "decision_class_required": "DesignGate",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# CreateEntity
# ---------------------------------------------------------------------------

class TestCreateEntity:
    def test_create_valid_requirement(self):
        h = _handler()
        resp = h.create_entity(CreateEntityRequest("Requirement", _req_payload()))
        assert resp.ok
        assert resp.entity_id == "a1b2c3d4-e5f6-4789-ab12-cd34ef567890"

    def test_create_emits_created_event(self):
        h = _handler()
        h.create_entity(CreateEntityRequest("Requirement", _req_payload()))
        events = h._store.get_events("a1b2c3d4-e5f6-4789-ab12-cd34ef567890")
        assert len(events) == 1
        assert events[0].event_type == "requirement.created"

    def test_create_invalid_payload_returns_error(self):
        h = _handler()
        resp = h.create_entity(CreateEntityRequest("Requirement", {"text": "missing fields"}))
        assert not resp.ok
        assert resp.error_code == "SCHEMA_VALIDATION_ERROR"

    def test_idempotency_returns_same_entity(self):
        h = _handler()
        req = CreateEntityRequest("Requirement", _req_payload(), idempotency_key="idem-001")
        resp1 = h.create_entity(req)
        resp2 = h.create_entity(req)
        assert resp1.entity_id == resp2.entity_id
        # Only one event should be in the store
        events = h._store.get_events(resp1.entity_id)
        assert len(events) == 1

    def test_create_characteristic_valid(self):
        h = _handler()
        resp = h.create_entity(CreateEntityRequest("Characteristic", _char_payload()))
        assert resp.ok

    def test_create_characteristic_invalid_unit(self):
        h = _handler()
        payload = _char_payload(unit="furlongs")
        resp = h.create_entity(CreateEntityRequest("Characteristic", payload))
        assert not resp.ok
        assert resp.error_code == "UNIT_PARSE_ERROR"


# ---------------------------------------------------------------------------
# GetEntity
# ---------------------------------------------------------------------------

class TestGetEntity:
    def test_get_existing_entity(self):
        h = _handler()
        h.create_entity(CreateEntityRequest("Requirement", _req_payload()))
        resp = h.get_entity(GetEntityRequest("a1b2c3d4-e5f6-4789-ab12-cd34ef567890"))
        assert resp.ok
        assert resp.entity_type == "Requirement"
        assert resp.payload is not None

    def test_get_nonexistent_entity(self):
        h = _handler()
        resp = h.get_entity(GetEntityRequest("00000000-0000-4000-8000-000000000000"))
        assert not resp.ok
        assert resp.error_code == "NOT_FOUND"


# ---------------------------------------------------------------------------
# UpdateEntityState
# ---------------------------------------------------------------------------

class TestUpdateEntityState:
    def _create_char(self, h: GraphServiceHandler, status: str = "") -> str:
        payload = _char_payload()
        if status:
            payload["status"] = status
        h.create_entity(CreateEntityRequest("Characteristic", payload))
        return payload["entity_id"]

    def test_legal_transition(self):
        h = _handler()
        eid = self._create_char(h, status="unverified")
        resp = h.update_entity_state(
            UpdateEntityStateRequest(entity_id=eid, new_state="surrogate_estimated")
        )
        assert resp.ok
        assert resp.previous_state == "unverified"
        assert resp.new_state == "surrogate_estimated"

    def test_legal_transition_updates_status(self):
        h = _handler()
        eid = self._create_char(h, status="unverified")
        h.update_entity_state(UpdateEntityStateRequest(entity_id=eid, new_state="surrogate_estimated"))
        record = h._store.get(eid)
        assert record is not None
        assert record.status == "surrogate_estimated"

    def test_legal_transition_creates_provenance_bundle(self):
        h = _handler()
        eid = self._create_char(h, status="unverified")
        resp = h.update_entity_state(
            UpdateEntityStateRequest(entity_id=eid, new_state="surrogate_estimated")
        )
        assert resp.provenance_bundle_id != ""
        pb = h._store.get(resp.provenance_bundle_id)
        assert pb is not None
        assert pb.entity_type == "ProvenanceBundle"

    def test_legal_transition_emits_status_changed_event(self):
        h = _handler()
        eid = self._create_char(h, status="unverified")
        h.update_entity_state(UpdateEntityStateRequest(entity_id=eid, new_state="surrogate_estimated"))
        events = [e for e in h._store.get_events(eid) if e.event_type.endswith("status_changed")]
        assert len(events) == 1
        assert events[0].previous_state == "unverified"
        assert events[0].new_state == "surrogate_estimated"

    def test_illegal_transition_returns_error(self):
        h = _handler()
        eid = self._create_char(h, status="unverified")
        resp = h.update_entity_state(
            UpdateEntityStateRequest(entity_id=eid, new_state="released")
        )
        assert not resp.ok
        assert resp.error_code == "ILLEGAL_TRANSITION"

    def test_illegal_transition_does_not_change_state(self):
        h = _handler()
        eid = self._create_char(h, status="unverified")
        h.update_entity_state(UpdateEntityStateRequest(entity_id=eid, new_state="released"))
        record = h._store.get(eid)
        assert record is not None
        assert record.status == "unverified"

    def test_approval_gated_without_approver_returns_missing_approval(self):
        h = _handler()
        eid = self._create_char(h, status="inspection_confirmed")
        resp = h.update_entity_state(
            UpdateEntityStateRequest(entity_id=eid, new_state="released")
        )
        assert not resp.ok
        assert resp.error_code == "MISSING_APPROVAL"

    def test_approval_gated_with_approver_succeeds(self):
        h = _handler()
        eid = self._create_char(h, status="inspection_confirmed")
        resp = h.update_entity_state(
            UpdateEntityStateRequest(entity_id=eid, new_state="released", approver_id="lse-001")
        )
        assert resp.ok

    def test_update_nonexistent_entity_returns_not_found(self):
        h = _handler()
        resp = h.update_entity_state(
            UpdateEntityStateRequest(entity_id="00000000-0000-4000-8000-000000000000", new_state="released")
        )
        assert not resp.ok
        assert resp.error_code == "NOT_FOUND"


# ---------------------------------------------------------------------------
# QueryEntities
# ---------------------------------------------------------------------------

class TestQueryEntities:
    def test_query_by_entity_type(self):
        h = _handler()
        h.create_entity(CreateEntityRequest("Requirement", _req_payload()))
        resp = h.query_entities(QueryEntitiesRequest(entity_type="Requirement"))
        assert resp.ok
        assert resp.total_count == 1

    def test_query_empty_result(self):
        h = _handler()
        resp = h.query_entities(QueryEntitiesRequest(entity_type="Prediction"))
        assert resp.ok
        assert resp.total_count == 0

    def test_query_with_filter(self):
        h = _handler()
        h.create_entity(CreateEntityRequest("Requirement", _req_payload(criticality="High")))
        h.create_entity(CreateEntityRequest(
            "Requirement",
            _req_payload(entity_id="req-002", criticality="Low"),
        ))
        resp = h.query_entities(
            QueryEntitiesRequest(entity_type="Requirement", filters={"criticality": "High"})
        )
        assert resp.total_count == 1

    def test_query_respects_limit(self):
        h = _handler()
        for i in range(5):
            h.create_entity(CreateEntityRequest(
                "Requirement",
                _req_payload(entity_id=f"req-{i:03d}"),
            ))
        resp = h.query_entities(QueryEntitiesRequest(entity_type="Requirement", limit=3))
        assert len(resp.entities) == 3


# ---------------------------------------------------------------------------
# CreateRelation
# ---------------------------------------------------------------------------

class TestCreateRelation:
    def _setup(self) -> tuple[GraphServiceHandler, str, str]:
        h = _handler()
        req_id = h.create_entity(CreateEntityRequest("Requirement", _req_payload())).entity_id
        char_id = h.create_entity(CreateEntityRequest("Characteristic", _char_payload())).entity_id
        return h, req_id, char_id

    def test_valid_relation(self):
        h, req_id, char_id = self._setup()
        resp = h.create_relation(CreateRelationRequest(
            from_entity_id=req_id, relation_type="CONSTRAINS", to_entity_id=char_id
        ))
        assert resp.ok
        assert resp.relation_id != ""

    def test_unknown_relation_type(self):
        h, req_id, char_id = self._setup()
        resp = h.create_relation(CreateRelationRequest(
            from_entity_id=req_id, relation_type="INVENTED_TYPE", to_entity_id=char_id
        ))
        assert not resp.ok
        assert resp.error_code == "UNKNOWN_RELATION_TYPE"

    def test_missing_from_entity(self):
        h, _, char_id = self._setup()
        resp = h.create_relation(CreateRelationRequest(
            from_entity_id="00000000-0000-4000-8000-000000000000",
            relation_type="CONSTRAINS",
            to_entity_id=char_id,
        ))
        assert not resp.ok
        assert resp.error_code == "LINK_RESOLUTION_ERROR"

    def test_missing_to_entity(self):
        h, req_id, _ = self._setup()
        resp = h.create_relation(CreateRelationRequest(
            from_entity_id=req_id,
            relation_type="CONSTRAINS",
            to_entity_id="00000000-0000-4000-8000-000000000000",
        ))
        assert not resp.ok
        assert resp.error_code == "LINK_RESOLUTION_ERROR"


# ---------------------------------------------------------------------------
# GetProvenanceChain
# ---------------------------------------------------------------------------

class TestGetProvenanceChain:
    def test_single_node_chain(self):
        h = _handler()
        req_id = h.create_entity(CreateEntityRequest("Requirement", _req_payload())).entity_id
        resp = h.get_provenance_chain(GetProvenanceChainRequest(req_id))
        assert resp.ok
        assert resp.chain == [req_id]

    def test_two_node_chain(self):
        h = _handler()
        req_id = h.create_entity(CreateEntityRequest("Requirement", _req_payload())).entity_id
        char_id = h.create_entity(CreateEntityRequest("Characteristic", _char_payload())).entity_id
        h.create_relation(CreateRelationRequest(
            from_entity_id=char_id, relation_type="DERIVED_FROM", to_entity_id=req_id
        ))
        resp = h.get_provenance_chain(GetProvenanceChainRequest(char_id))
        assert resp.ok
        assert resp.chain == [char_id, req_id]

    def test_nonexistent_entity_returns_single_node_chain(self):
        h = _handler()
        resp = h.get_provenance_chain(GetProvenanceChainRequest("unknown-id"))
        assert resp.ok
        assert resp.chain == ["unknown-id"]

    def test_circular_chain_detected(self):
        h = _handler()
        a_id = h.create_entity(CreateEntityRequest("Requirement", _req_payload())).entity_id
        b_id = h.create_entity(CreateEntityRequest("Characteristic", _char_payload())).entity_id
        h.create_relation(CreateRelationRequest(
            from_entity_id=a_id, relation_type="DERIVED_FROM", to_entity_id=b_id
        ))
        h.create_relation(CreateRelationRequest(
            from_entity_id=b_id, relation_type="DERIVED_FROM", to_entity_id=a_id
        ))
        resp = h.get_provenance_chain(GetProvenanceChainRequest(a_id))
        assert not resp.ok
        assert resp.error_code == "CIRCULAR_CHAIN"
