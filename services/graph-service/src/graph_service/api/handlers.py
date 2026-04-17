"""
Graph service request handlers.

Implements the 5 RPCs from service-contracts-api-spec.md §2:
  CreateEntity, GetEntity, UpdateEntityState, QueryEntities, GetProvenanceChain.
Also implements CreateRelation.
"""
from __future__ import annotations

import sys
import os

# Allow running without installation
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "packages", "schema", "src"),
)

from dataclasses import dataclass
from typing import Any

from henqo_schema.validation import ValidationError, validate_entity

from graph_service.domain.state_machine import (
    IllegalTransitionError,
    MissingApprovalError,
    check_transition,
)
from graph_service.persistence.store import (
    EntityRecord,
    EntityStore,
    EventRecord,
    InMemoryEntityStore,
    RelationRecord,
    _now,
    new_uuid,
)


# ---------------------------------------------------------------------------
# Request / Response dataclasses (stand-in for proto messages)
# ---------------------------------------------------------------------------

@dataclass
class CreateEntityRequest:
    entity_type: str
    payload: dict[str, Any]
    idempotency_key: str = ""
    created_by: str = "system"


@dataclass
class CreateEntityResponse:
    entity_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class GetEntityRequest:
    entity_id: str
    include_related: bool = False


@dataclass
class GetEntityResponse:
    payload: dict[str, Any] | None = None
    entity_type: str = ""
    status: str = ""
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class UpdateEntityStateRequest:
    entity_id: str
    new_state: str
    transition_reason: str = ""
    approver_id: str = ""
    idempotency_key: str = ""


@dataclass
class UpdateEntityStateResponse:
    previous_state: str = ""
    new_state: str = ""
    provenance_bundle_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class QueryEntitiesRequest:
    entity_type: str
    filters: dict[str, str] | None = None
    limit: int = 50


@dataclass
class QueryEntitiesResponse:
    entities: list[dict[str, Any]] | None = None
    total_count: int = 0
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class CreateRelationRequest:
    from_entity_id: str
    relation_type: str
    to_entity_id: str
    properties: dict[str, str] | None = None
    idempotency_key: str = ""


@dataclass
class CreateRelationResponse:
    relation_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class GetProvenanceChainRequest:
    entity_id: str


@dataclass
class GetProvenanceChainResponse:
    chain: list[str] | None = None   # ordered list of entity IDs
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

_VALID_RELATION_TYPES = frozenset([
    "EVALUATES",        # Prediction → Characteristic
    "CONSTRAINS",       # Requirement → Characteristic
    "DERIVED_FROM",     # entity provenance chain
    "PART_OF",          # ComponentRevision → AssemblyRevision
    "GOVERNED_BY",      # entity → ProvenanceBundle
    "EVIDENCE_FOR",     # SimulationCase/InspectionResult → Characteristic
])


class GraphServiceHandler:
    def __init__(self, store: EntityStore | None = None) -> None:
        self._store = store or InMemoryEntityStore()

    # ------------------------------------------------------------------
    # CreateEntity
    # ------------------------------------------------------------------

    def create_entity(self, req: CreateEntityRequest) -> CreateEntityResponse:
        # Idempotency check
        if req.idempotency_key:
            existing = self._store.get_by_idempotency_key(req.idempotency_key)
            if existing:
                return CreateEntityResponse(entity_id=existing.entity_id)

        # Step 1–5 validation via schema package
        try:
            validate_entity(req.entity_type, req.payload)
        except ValidationError as exc:
            return CreateEntityResponse(
                error_code=exc.error_code,
                error_message=exc.message,
            )

        entity_id = req.payload.get("entity_id") or new_uuid()
        status = req.payload.get("status", "")

        now = _now()
        record = EntityRecord(
            entity_id=entity_id,
            entity_type=req.entity_type,
            payload=dict(req.payload),
            status=status,
            created_at=now,
            updated_at=now,
            idempotency_key=req.idempotency_key or None,
        )
        self._store.create(record)

        self._store.emit_event(EventRecord(
            event_id=new_uuid(),
            event_type=f"{req.entity_type.lower()}.created",
            entity_type=req.entity_type,
            entity_id=entity_id,
            previous_state="",
            new_state=status,
            triggered_by=req.created_by,
            timestamp=now,
        ))

        return CreateEntityResponse(entity_id=entity_id)

    # ------------------------------------------------------------------
    # GetEntity
    # ------------------------------------------------------------------

    def get_entity(self, req: GetEntityRequest) -> GetEntityResponse:
        record = self._store.get(req.entity_id)
        if record is None:
            return GetEntityResponse(error_code="NOT_FOUND")
        return GetEntityResponse(
            payload=record.payload,
            entity_type=record.entity_type,
            status=record.status,
        )

    # ------------------------------------------------------------------
    # UpdateEntityState
    # ------------------------------------------------------------------

    def update_entity_state(self, req: UpdateEntityStateRequest) -> UpdateEntityStateResponse:
        record = self._store.get(req.entity_id)
        if record is None:
            return UpdateEntityStateResponse(error_code="NOT_FOUND", error_message="Entity not found")

        try:
            check_transition(
                entity_type=record.entity_type,
                current_state=record.status,
                new_state=req.new_state,
                approver_id=req.approver_id or None,
            )
        except IllegalTransitionError as exc:
            return UpdateEntityStateResponse(
                error_code="ILLEGAL_TRANSITION",
                error_message=str(exc),
            )
        except MissingApprovalError as exc:
            return UpdateEntityStateResponse(
                error_code="MISSING_APPROVAL",
                error_message=str(exc),
            )

        previous_state = record.status
        self._store.update_status(req.entity_id, req.new_state)

        # Auto-create a minimal ProvenanceBundle for the transition
        pb_id = new_uuid()
        pb_record = EntityRecord(
            entity_id=pb_id,
            entity_type="ProvenanceBundle",
            payload={
                "bundle_id": pb_id,
                "activity": f"state_transition:{previous_state}→{req.new_state}",
                "agent_id": req.approver_id or "system",
                "transition_reason": req.transition_reason,
            },
            status="",
            created_at=_now(),
            updated_at=_now(),
        )
        self._store.create(pb_record)

        self._store.emit_event(EventRecord(
            event_id=new_uuid(),
            event_type=f"{record.entity_type.lower()}.status_changed",
            entity_type=record.entity_type,
            entity_id=req.entity_id,
            previous_state=previous_state,
            new_state=req.new_state,
            triggered_by=req.approver_id or "system",
            timestamp=_now(),
        ))

        return UpdateEntityStateResponse(
            previous_state=previous_state,
            new_state=req.new_state,
            provenance_bundle_id=pb_id,
        )

    # ------------------------------------------------------------------
    # QueryEntities
    # ------------------------------------------------------------------

    def query_entities(self, req: QueryEntitiesRequest) -> QueryEntitiesResponse:
        records = self._store.query(
            entity_type=req.entity_type,
            filters=req.filters,
            limit=req.limit,
        )
        return QueryEntitiesResponse(
            entities=[r.payload for r in records],
            total_count=len(records),
        )

    # ------------------------------------------------------------------
    # CreateRelation
    # ------------------------------------------------------------------

    def create_relation(self, req: CreateRelationRequest) -> CreateRelationResponse:
        if req.relation_type not in _VALID_RELATION_TYPES:
            return CreateRelationResponse(
                error_code="UNKNOWN_RELATION_TYPE",
                error_message=f"Unknown relation type: {req.relation_type!r}",
            )

        if not self._store.get(req.from_entity_id):
            return CreateRelationResponse(
                error_code="LINK_RESOLUTION_ERROR",
                error_message=f"from_entity_id {req.from_entity_id!r} not found",
            )
        if not self._store.get(req.to_entity_id):
            return CreateRelationResponse(
                error_code="LINK_RESOLUTION_ERROR",
                error_message=f"to_entity_id {req.to_entity_id!r} not found",
            )

        relation_id = new_uuid()
        self._store.create_relation(RelationRecord(
            relation_id=relation_id,
            from_entity_id=req.from_entity_id,
            relation_type=req.relation_type,
            to_entity_id=req.to_entity_id,
            properties=req.properties or {},
        ))
        return CreateRelationResponse(relation_id=relation_id)

    # ------------------------------------------------------------------
    # GetProvenanceChain
    # ------------------------------------------------------------------

    def get_provenance_chain(self, req: GetProvenanceChainRequest) -> GetProvenanceChainResponse:
        """
        Walk DERIVED_FROM edges to build the provenance chain.
        Cycle detection: abort if we visit a node twice.
        """
        chain: list[str] = []
        visited: set[str] = set()
        current_id = req.entity_id

        while current_id:
            if current_id in visited:
                return GetProvenanceChainResponse(
                    error_code="CIRCULAR_CHAIN",
                    error_message=f"Circular provenance chain detected at {current_id!r}",
                )
            visited.add(current_id)
            chain.append(current_id)

            derived_from_relations = self._store.get_relations(
                current_id, relation_type="DERIVED_FROM"
            )
            # Only follow edges where current_id is the source
            outgoing = [r for r in derived_from_relations if r.from_entity_id == current_id]
            if not outgoing:
                break
            current_id = outgoing[0].to_entity_id

        return GetProvenanceChainResponse(chain=chain)
