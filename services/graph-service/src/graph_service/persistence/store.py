"""
Abstract entity store interface + in-memory implementation for testing.

Production will swap in an Apache AGE / PostgreSQL implementation.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EntityRecord:
    entity_id: str
    entity_type: str
    payload: dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    idempotency_key: str | None = None


@dataclass
class RelationRecord:
    relation_id: str
    from_entity_id: str
    relation_type: str
    to_entity_id: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class EventRecord:
    event_id: str
    event_type: str
    entity_type: str
    entity_id: str
    previous_state: str
    new_state: str
    triggered_by: str
    timestamp: str


class EntityStore(ABC):
    @abstractmethod
    def create(self, record: EntityRecord) -> EntityRecord: ...

    @abstractmethod
    def get(self, entity_id: str) -> EntityRecord | None: ...

    @abstractmethod
    def update_status(self, entity_id: str, new_status: str) -> EntityRecord: ...

    @abstractmethod
    def query(
        self,
        entity_type: str,
        filters: dict[str, str] | None = None,
        limit: int = 50,
    ) -> list[EntityRecord]: ...

    @abstractmethod
    def get_by_idempotency_key(self, key: str) -> EntityRecord | None: ...

    @abstractmethod
    def create_relation(self, record: RelationRecord) -> RelationRecord: ...

    @abstractmethod
    def get_relations(
        self, entity_id: str, relation_type: str | None = None
    ) -> list[RelationRecord]: ...

    @abstractmethod
    def emit_event(self, record: EventRecord) -> None: ...

    @abstractmethod
    def get_events(self, entity_id: str) -> list[EventRecord]: ...


class InMemoryEntityStore(EntityStore):
    """Thread-unsafe in-memory store for unit testing."""

    def __init__(self) -> None:
        self._entities: dict[str, EntityRecord] = {}
        self._by_idempotency: dict[str, EntityRecord] = {}
        self._relations: list[RelationRecord] = []
        self._events: list[EventRecord] = []

    def create(self, record: EntityRecord) -> EntityRecord:
        self._entities[record.entity_id] = record
        if record.idempotency_key:
            self._by_idempotency[record.idempotency_key] = record
        return record

    def get(self, entity_id: str) -> EntityRecord | None:
        return self._entities.get(entity_id)

    def update_status(self, entity_id: str, new_status: str) -> EntityRecord:
        record = self._entities[entity_id]
        record.status = new_status
        record.updated_at = _now()
        return record

    def query(
        self,
        entity_type: str,
        filters: dict[str, str] | None = None,
        limit: int = 50,
    ) -> list[EntityRecord]:
        results = [r for r in self._entities.values() if r.entity_type == entity_type]
        if filters:
            results = [
                r for r in results
                if all(str(r.payload.get(k)) == v for k, v in filters.items())
            ]
        return results[:limit]

    def get_by_idempotency_key(self, key: str) -> EntityRecord | None:
        return self._by_idempotency.get(key)

    def create_relation(self, record: RelationRecord) -> RelationRecord:
        self._relations.append(record)
        return record

    def get_relations(
        self, entity_id: str, relation_type: str | None = None
    ) -> list[RelationRecord]:
        out = [
            r for r in self._relations
            if r.from_entity_id == entity_id or r.to_entity_id == entity_id
        ]
        if relation_type:
            out = [r for r in out if r.relation_type == relation_type]
        return out

    def emit_event(self, record: EventRecord) -> None:
        self._events.append(record)

    def get_events(self, entity_id: str) -> list[EventRecord]:
        return [e for e in self._events if e.entity_id == entity_id]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_uuid() -> str:
    return str(uuid.uuid4())
