"""
Design service request handlers for EP-04.

Wraps GraphServiceHandler to create and link:
  - ComponentRevision
  - AssemblyRevision (with PART_OF relations to component revisions)
  - Characteristic (with optional EVIDENCE_FOR relation)
  - DesignVariableSet (in-memory, associated to a ComponentRevision)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from graph_service.api.handlers import (
    CreateEntityRequest,
    CreateRelationRequest,
    GetEntityRequest,
    GraphServiceHandler,
)
from graph_service.persistence.store import _now, new_uuid

from design_service.domain.design_variables import (
    DesignVariable,
    DesignVariableSet,
    DesignVariableStore,
    validate_design_variable,
)


# ---------------------------------------------------------------------------
# Request / Response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CreateComponentRevisionRequest:
    component_name: str
    description: str = ""
    idempotency_key: str = ""


@dataclass
class CreateComponentRevisionResponse:
    entity_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class CreateAssemblyRevisionRequest:
    assembly_name: str
    component_revision_ids: list[str]
    description: str = ""
    idempotency_key: str = ""


@dataclass
class CreateAssemblyRevisionResponse:
    entity_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class CreateCharacteristicRequest:
    name: str
    quantity_kind: str
    unit: str
    governing_requirement_id: str
    criticality: str
    decision_class_required: str
    component_revision_id: str = ""   # if set, create EVIDENCE_FOR relation


@dataclass
class CreateCharacteristicResponse:
    entity_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class AddDesignVariableSetRequest:
    component_revision_id: str
    variables: list[DesignVariable]


@dataclass
class AddDesignVariableSetResponse:
    set_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class GetDesignVariableSetResponse:
    design_variable_sets: list[DesignVariableSet] = field(default_factory=list)
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class DesignServiceHandler:
    """
    Orchestrates design-candidate creation by wrapping GraphServiceHandler
    and an in-memory DesignVariableStore.
    """

    def __init__(self, graph_handler: Optional[GraphServiceHandler] = None) -> None:
        self._graph = graph_handler or GraphServiceHandler()
        self._dvs_store = DesignVariableStore()

    # ------------------------------------------------------------------
    # ComponentRevision
    # ------------------------------------------------------------------

    def create_component_revision(
        self, req: CreateComponentRevisionRequest
    ) -> CreateComponentRevisionResponse:
        entity_id = new_uuid()
        payload: dict = {
            "entity_id": entity_id,
            "component_name": req.component_name,
            "status": "in_design",
        }
        if req.description:
            payload["description"] = req.description

        graph_req = CreateEntityRequest(
            entity_type="ComponentRevision",
            payload=payload,
            idempotency_key=req.idempotency_key,
        )
        resp = self._graph.create_entity(graph_req)
        if not resp.ok:
            return CreateComponentRevisionResponse(
                error_code=resp.error_code,
                error_message=resp.error_message,
            )
        return CreateComponentRevisionResponse(entity_id=resp.entity_id)

    # ------------------------------------------------------------------
    # AssemblyRevision
    # ------------------------------------------------------------------

    def create_assembly_revision(
        self, req: CreateAssemblyRevisionRequest
    ) -> CreateAssemblyRevisionResponse:
        entity_id = new_uuid()
        payload: dict = {
            "entity_id": entity_id,
            "assembly_name": req.assembly_name,
            "status": "in_design",
        }
        if req.description:
            payload["description"] = req.description

        graph_req = CreateEntityRequest(
            entity_type="AssemblyRevision",
            payload=payload,
            idempotency_key=req.idempotency_key,
        )
        resp = self._graph.create_entity(graph_req)
        if not resp.ok:
            return CreateAssemblyRevisionResponse(
                error_code=resp.error_code,
                error_message=resp.error_message,
            )

        assembly_id = resp.entity_id

        # Create PART_OF relations from each component revision to the assembly
        for component_id in req.component_revision_ids:
            rel_req = CreateRelationRequest(
                from_entity_id=component_id,
                relation_type="PART_OF",
                to_entity_id=assembly_id,
            )
            rel_resp = self._graph.create_relation(rel_req)
            if not rel_resp.ok:
                return CreateAssemblyRevisionResponse(
                    error_code=rel_resp.error_code,
                    error_message=rel_resp.error_message,
                )

        return CreateAssemblyRevisionResponse(entity_id=assembly_id)

    # ------------------------------------------------------------------
    # Characteristic
    # ------------------------------------------------------------------

    def create_characteristic(
        self, req: CreateCharacteristicRequest
    ) -> CreateCharacteristicResponse:
        entity_id = new_uuid()
        payload: dict = {
            "entity_id": entity_id,
            "name": req.name,
            "quantity_kind": req.quantity_kind,
            "unit": req.unit,
            "governing_requirement_id": req.governing_requirement_id,
            "criticality": req.criticality,
            "decision_class_required": req.decision_class_required,
            "status": "unverified",
        }

        graph_req = CreateEntityRequest(
            entity_type="Characteristic",
            payload=payload,
        )
        resp = self._graph.create_entity(graph_req)
        if not resp.ok:
            return CreateCharacteristicResponse(
                error_code=resp.error_code,
                error_message=resp.error_message,
            )

        char_id = resp.entity_id

        # Optionally link the characteristic as EVIDENCE_FOR a component revision
        if req.component_revision_id:
            rel_req = CreateRelationRequest(
                from_entity_id=char_id,
                relation_type="EVIDENCE_FOR",
                to_entity_id=req.component_revision_id,
            )
            rel_resp = self._graph.create_relation(rel_req)
            if not rel_resp.ok:
                return CreateCharacteristicResponse(
                    error_code=rel_resp.error_code,
                    error_message=rel_resp.error_message,
                )

        return CreateCharacteristicResponse(entity_id=char_id)

    # ------------------------------------------------------------------
    # DesignVariableSet — add
    # ------------------------------------------------------------------

    def add_design_variable_set(
        self, req: AddDesignVariableSetRequest
    ) -> AddDesignVariableSetResponse:
        # Verify that the component revision exists in the graph
        get_resp = self._graph.get_entity(
            GetEntityRequest(entity_id=req.component_revision_id)
        )
        if not get_resp.ok:
            return AddDesignVariableSetResponse(
                error_code="NOT_FOUND",
                error_message=(
                    f"ComponentRevision '{req.component_revision_id}' not found"
                ),
            )

        # Validate each design variable
        try:
            for var in req.variables:
                validate_design_variable(var)
        except ValueError as exc:
            return AddDesignVariableSetResponse(
                error_code="VALIDATION_ERROR",
                error_message=str(exc),
            )

        dvs = DesignVariableSet(
            set_id=new_uuid(),
            component_revision_id=req.component_revision_id,
            variables=list(req.variables),
            created_at=_now(),
        )
        self._dvs_store.create(dvs)

        return AddDesignVariableSetResponse(set_id=dvs.set_id)

    # ------------------------------------------------------------------
    # DesignVariableSet — get by set_id
    # ------------------------------------------------------------------

    def get_design_variable_set(self, set_id: str) -> GetDesignVariableSetResponse:
        dvs = self._dvs_store.get(set_id)
        if dvs is None:
            return GetDesignVariableSetResponse(
                error_code="NOT_FOUND",
                error_message=f"DesignVariableSet '{set_id}' not found",
            )
        return GetDesignVariableSetResponse(design_variable_sets=[dvs])

    # ------------------------------------------------------------------
    # DesignVariableSet — get by component revision id
    # ------------------------------------------------------------------

    def get_design_variable_sets_by_component(
        self, component_revision_id: str
    ) -> GetDesignVariableSetResponse:
        sets = self._dvs_store.get_by_component(component_revision_id)
        return GetDesignVariableSetResponse(design_variable_sets=sets)
