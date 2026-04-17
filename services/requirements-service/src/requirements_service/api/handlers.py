"""
Requirements Ingestion Service handler.

Accepts natural-language requirement text, parses it into a structured
ParsedRequirement, performs deduplication, and writes a Requirement entity
to the graph service.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from requirements_service.domain.parser import ParseError, ParsedRequirement, parse
from requirements_service.domain.dedup import is_duplicate


# ---------------------------------------------------------------------------
# Request / Response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IngestRequirementRequest:
    text: str
    criticality: str
    decision_class_required: str
    idempotency_key: str = ""
    source_document_id: str = ""


@dataclass
class IngestRequirementResponse:
    entity_id: str = ""
    is_duplicate: bool = False
    duplicate_of: str = ""
    error_code: str = ""
    error_message: str = ""
    parsed: Optional[ParsedRequirement] = None

    @property
    def ok(self) -> bool:
        return not self.error_code


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class RequirementsServiceHandler:
    """
    Ingests requirement text, parses it, deduplicates, and writes to the
    graph service.

    Parameters
    ----------
    graph_handler:
        An instance of GraphServiceHandler (duck-typed; no hard import to
        avoid circular dependencies in tests).
    existing_requirements:
        Optional seed list of (entity_id, text) tuples representing
        requirements already persisted.  Used to initialise the dedup state.
    """

    def __init__(
        self,
        graph_handler: object,
        existing_requirements: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        self._graph = graph_handler
        # Internal dedup state: list of (entity_id, text) pairs
        self._existing: list[tuple[str, str]] = list(existing_requirements or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, req: IngestRequirementRequest) -> IngestRequirementResponse:
        """
        Ingest a single requirement.

        Steps:
        1. Parse text — return REQUIREMENT_NOT_PARSEABLE on ParseError.
        2. Duplicate check — return is_duplicate=True if near-match found.
        3. Build Requirement entity payload and call graph_handler.create_entity.
        4. Store entity for future dedup lookups.
        5. Return IngestRequirementResponse.
        """
        # Step 1: parse
        try:
            parsed = parse(req.text)
        except ParseError as exc:
            return IngestRequirementResponse(
                error_code="REQUIREMENT_NOT_PARSEABLE",
                error_message=str(exc),
            )

        # Step 2: dedup
        existing_texts = [text for _, text in self._existing]
        dup, dup_idx = is_duplicate(req.text, existing_texts)
        if dup and dup_idx is not None:
            dup_entity_id = self._existing[dup_idx][0]
            return IngestRequirementResponse(
                is_duplicate=True,
                duplicate_of=dup_entity_id,
            )

        # Step 3: build payload
        entity_id = str(uuid.uuid4())
        payload: dict = {
            "entity_id": entity_id,
            "text": req.text,
            "criticality": req.criticality,
            "decision_class_required": req.decision_class_required,
        }
        if req.source_document_id:
            payload["source_document_id"] = req.source_document_id

        # Attach parsed quantity if present
        if not parsed.is_qualitative and parsed.value is not None and parsed.unit is not None:
            payload["quantity"] = {
                "value": parsed.value,
                "unit": parsed.unit,
            }

        # Step 4: write to graph service
        from graph_service.api.handlers import CreateEntityRequest  # type: ignore[import]
        create_req = CreateEntityRequest(
            entity_type="Requirement",
            payload=payload,
            idempotency_key=req.idempotency_key,
            created_by="requirements-service",
        )
        create_resp = self._graph.create_entity(create_req)

        if not create_resp.ok:
            return IngestRequirementResponse(
                error_code=create_resp.error_code,
                error_message=create_resp.error_message,
            )

        # Use the entity_id the graph service assigned (handles idempotency)
        persisted_id = create_resp.entity_id

        # Step 5: store for future dedup
        self._existing.append((persisted_id, req.text))

        return IngestRequirementResponse(
            entity_id=persisted_id,
            parsed=parsed,
        )
