"""
Unit tests for EP-04: Candidate Design Representation.

Covers DT-001 through DT-006 acceptance criteria.
"""
from __future__ import annotations

import pytest

from graph_service.api.handlers import GraphServiceHandler
from graph_service.persistence.store import InMemoryEntityStore

from design_service.api.handlers import (
    AddDesignVariableSetRequest,
    CreateAssemblyRevisionRequest,
    CreateCharacteristicRequest,
    CreateComponentRevisionRequest,
    DesignServiceHandler,
)
from design_service.domain.design_variables import (
    DesignVariable,
    validate_design_variable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_handler() -> DesignServiceHandler:
    """Return a DesignServiceHandler backed by a fresh in-memory store."""
    store = InMemoryEntityStore()
    graph = GraphServiceHandler(store=store)
    return DesignServiceHandler(graph_handler=graph)


def _create_component(handler: DesignServiceHandler, name: str = "Widget") -> str:
    """Create a component revision and return its entity_id."""
    resp = handler.create_component_revision(
        CreateComponentRevisionRequest(component_name=name)
    )
    assert resp.ok, f"create_component_revision failed: {resp.error_code} — {resp.error_message}"
    return resp.entity_id


# ---------------------------------------------------------------------------
# ComponentRevision tests
# ---------------------------------------------------------------------------

class TestCreateComponentRevision:
    def test_success_returns_entity_id(self) -> None:
        handler = make_handler()
        resp = handler.create_component_revision(
            CreateComponentRevisionRequest(component_name="BracketA")
        )
        assert resp.ok
        assert resp.entity_id != ""
        assert resp.error_code == ""

    def test_initial_status_is_in_design(self) -> None:
        handler = make_handler()
        resp = handler.create_component_revision(
            CreateComponentRevisionRequest(component_name="BracketB")
        )
        assert resp.ok
        # Verify the entity stored in the graph has status="in_design"
        from graph_service.api.handlers import GetEntityRequest
        get_resp = handler._graph.get_entity(GetEntityRequest(entity_id=resp.entity_id))
        assert get_resp.ok
        assert get_resp.status == "in_design"

    def test_idempotency_key_returns_same_entity(self) -> None:
        handler = make_handler()
        req = CreateComponentRevisionRequest(
            component_name="BracketC", idempotency_key="idem-key-001"
        )
        resp1 = handler.create_component_revision(req)
        resp2 = handler.create_component_revision(req)
        assert resp1.ok and resp2.ok
        assert resp1.entity_id == resp2.entity_id

    def test_two_components_have_different_ids(self) -> None:
        handler = make_handler()
        id1 = _create_component(handler, "Alpha")
        id2 = _create_component(handler, "Beta")
        assert id1 != id2


# ---------------------------------------------------------------------------
# AssemblyRevision tests
# ---------------------------------------------------------------------------

class TestCreateAssemblyRevision:
    def test_success_with_component_linking(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Wing")

        resp = handler.create_assembly_revision(
            CreateAssemblyRevisionRequest(
                assembly_name="WingAssembly",
                component_revision_ids=[comp_id],
            )
        )
        assert resp.ok
        assert resp.entity_id != ""

    def test_status_is_in_design(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Spar")
        resp = handler.create_assembly_revision(
            CreateAssemblyRevisionRequest(
                assembly_name="SparAssembly",
                component_revision_ids=[comp_id],
            )
        )
        assert resp.ok
        from graph_service.api.handlers import GetEntityRequest
        get_resp = handler._graph.get_entity(GetEntityRequest(entity_id=resp.entity_id))
        assert get_resp.status == "in_design"

    def test_multiple_components_linked(self) -> None:
        handler = make_handler()
        id1 = _create_component(handler, "Rib1")
        id2 = _create_component(handler, "Rib2")
        resp = handler.create_assembly_revision(
            CreateAssemblyRevisionRequest(
                assembly_name="RibAssembly",
                component_revision_ids=[id1, id2],
            )
        )
        assert resp.ok

    def test_link_resolution_error_when_component_missing(self) -> None:
        handler = make_handler()
        resp = handler.create_assembly_revision(
            CreateAssemblyRevisionRequest(
                assembly_name="BadAssembly",
                component_revision_ids=["nonexistent-component-id"],
            )
        )
        assert not resp.ok
        assert resp.error_code == "LINK_RESOLUTION_ERROR"

    def test_empty_component_list_succeeds(self) -> None:
        handler = make_handler()
        resp = handler.create_assembly_revision(
            CreateAssemblyRevisionRequest(
                assembly_name="EmptyAssembly",
                component_revision_ids=[],
            )
        )
        assert resp.ok


# ---------------------------------------------------------------------------
# Characteristic tests
# ---------------------------------------------------------------------------

class TestCreateCharacteristic:
    def _char_req(self, **kwargs) -> CreateCharacteristicRequest:
        defaults = dict(
            name="MaxStress",
            quantity_kind="Stress",
            unit="MPa",
            governing_requirement_id="req-001",
            criticality="high",
            decision_class_required="ReleaseCritical",
        )
        defaults.update(kwargs)
        return CreateCharacteristicRequest(**defaults)

    def test_success(self) -> None:
        handler = make_handler()
        resp = handler.create_characteristic(self._char_req())
        assert resp.ok
        assert resp.entity_id != ""

    def test_status_is_unverified(self) -> None:
        handler = make_handler()
        resp = handler.create_characteristic(self._char_req())
        assert resp.ok
        from graph_service.api.handlers import GetEntityRequest
        get_resp = handler._graph.get_entity(GetEntityRequest(entity_id=resp.entity_id))
        assert get_resp.status == "unverified"

    def test_creates_evidence_for_relation(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "GearHousing")
        resp = handler.create_characteristic(
            self._char_req(component_revision_id=comp_id)
        )
        assert resp.ok
        # Verify the EVIDENCE_FOR relation was stored
        relations = handler._graph._store.get_relations(
            resp.entity_id, relation_type="EVIDENCE_FOR"
        )
        assert len(relations) == 1
        assert relations[0].to_entity_id == comp_id

    def test_evidence_for_fails_when_component_missing(self) -> None:
        handler = make_handler()
        resp = handler.create_characteristic(
            self._char_req(component_revision_id="ghost-component-id")
        )
        assert not resp.ok
        assert resp.error_code == "LINK_RESOLUTION_ERROR"

    def test_validation_error_on_bad_unit(self) -> None:
        handler = make_handler()
        resp = handler.create_characteristic(
            self._char_req(unit="psi")   # not in _PARSEABLE_UNITS
        )
        assert not resp.ok
        assert resp.error_code in ("UNIT_PARSE_ERROR", "SCHEMA_VALIDATION_ERROR")

    def test_no_evidence_for_when_component_revision_id_empty(self) -> None:
        handler = make_handler()
        resp = handler.create_characteristic(self._char_req())
        assert resp.ok
        relations = handler._graph._store.get_relations(
            resp.entity_id, relation_type="EVIDENCE_FOR"
        )
        assert len(relations) == 0


# ---------------------------------------------------------------------------
# DesignVariableSet tests
# ---------------------------------------------------------------------------

class TestAddDesignVariableSet:
    def _var(self, name: str = "thickness", lo: float = 1.0, hi: float = 10.0,
             nominal: float | None = None) -> DesignVariable:
        return DesignVariable(
            name=name,
            quantity_kind="Length",
            unit="mm",
            lower_bound=lo,
            upper_bound=hi,
            nominal=nominal,
        )

    def test_success(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Plate")
        resp = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[self._var()],
            )
        )
        assert resp.ok
        assert resp.set_id != ""

    def test_set_id_is_unique_per_call(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Plate2")
        resp1 = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[self._var("t1")],
            )
        )
        resp2 = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[self._var("t2")],
            )
        )
        assert resp1.ok and resp2.ok
        assert resp1.set_id != resp2.set_id

    def test_not_found_when_component_missing(self) -> None:
        handler = make_handler()
        resp = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id="no-such-component",
                variables=[self._var()],
            )
        )
        assert not resp.ok
        assert resp.error_code == "NOT_FOUND"

    def test_validation_error_on_lower_ge_upper(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Plate3")
        bad_var = self._var(lo=5.0, hi=5.0)   # equal bounds → invalid
        resp = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[bad_var],
            )
        )
        assert not resp.ok
        assert resp.error_code == "VALIDATION_ERROR"

    def test_validation_error_on_lower_gt_upper(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Plate4")
        bad_var = self._var(lo=10.0, hi=1.0)  # inverted bounds
        resp = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[bad_var],
            )
        )
        assert not resp.ok
        assert resp.error_code == "VALIDATION_ERROR"

    def test_validation_error_on_nominal_outside_bounds(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Plate5")
        bad_var = self._var(lo=1.0, hi=10.0, nominal=15.0)  # nominal above upper
        resp = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[bad_var],
            )
        )
        assert not resp.ok
        assert resp.error_code == "VALIDATION_ERROR"

    def test_nominal_at_boundary_is_valid(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Plate6")
        var = self._var(lo=1.0, hi=10.0, nominal=1.0)  # at lower bound → valid
        resp = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[var],
            )
        )
        assert resp.ok


# ---------------------------------------------------------------------------
# GetDesignVariableSet tests
# ---------------------------------------------------------------------------

class TestGetDesignVariableSet:
    def _make_set(self, handler: DesignServiceHandler, comp_id: str) -> str:
        var = DesignVariable(
            name="radius",
            quantity_kind="Length",
            unit="mm",
            lower_bound=0.5,
            upper_bound=5.0,
        )
        resp = handler.add_design_variable_set(
            AddDesignVariableSetRequest(
                component_revision_id=comp_id,
                variables=[var],
            )
        )
        assert resp.ok
        return resp.set_id

    def test_get_by_id_returns_set(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Rod")
        set_id = self._make_set(handler, comp_id)

        resp = handler.get_design_variable_set(set_id)
        assert resp.ok
        assert len(resp.design_variable_sets) == 1
        assert resp.design_variable_sets[0].set_id == set_id

    def test_get_by_id_not_found(self) -> None:
        handler = make_handler()
        resp = handler.get_design_variable_set("nonexistent-set-id")
        assert not resp.ok
        assert resp.error_code == "NOT_FOUND"

    def test_get_by_component_returns_all_sets(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Beam")
        self._make_set(handler, comp_id)
        self._make_set(handler, comp_id)

        resp = handler.get_design_variable_sets_by_component(comp_id)
        assert resp.ok
        assert len(resp.design_variable_sets) == 2
        for dvs in resp.design_variable_sets:
            assert dvs.component_revision_id == comp_id

    def test_get_by_component_returns_empty_for_no_sets(self) -> None:
        handler = make_handler()
        comp_id = _create_component(handler, "Shaft")
        resp = handler.get_design_variable_sets_by_component(comp_id)
        assert resp.ok
        assert resp.design_variable_sets == []

    def test_get_by_component_isolates_sets(self) -> None:
        """Sets for one component don't appear under another component."""
        handler = make_handler()
        comp_a = _create_component(handler, "CompA")
        comp_b = _create_component(handler, "CompB")
        self._make_set(handler, comp_a)

        resp = handler.get_design_variable_sets_by_component(comp_b)
        assert resp.ok
        assert resp.design_variable_sets == []


# ---------------------------------------------------------------------------
# DesignVariable validation unit tests
# ---------------------------------------------------------------------------

class TestDesignVariableValidation:
    def test_valid_variable_passes(self) -> None:
        var = DesignVariable(
            name="length", quantity_kind="Length", unit="m",
            lower_bound=0.0, upper_bound=1.0, nominal=0.5,
        )
        validate_design_variable(var)  # must not raise

    def test_lower_equals_upper_raises(self) -> None:
        var = DesignVariable(
            name="x", quantity_kind="Length", unit="m",
            lower_bound=1.0, upper_bound=1.0,
        )
        with pytest.raises(ValueError, match="lower_bound"):
            validate_design_variable(var)

    def test_lower_greater_than_upper_raises(self) -> None:
        var = DesignVariable(
            name="x", quantity_kind="Length", unit="m",
            lower_bound=5.0, upper_bound=1.0,
        )
        with pytest.raises(ValueError):
            validate_design_variable(var)

    def test_nominal_below_lower_bound_raises(self) -> None:
        var = DesignVariable(
            name="x", quantity_kind="Length", unit="m",
            lower_bound=2.0, upper_bound=10.0, nominal=1.0,
        )
        with pytest.raises(ValueError, match="nominal"):
            validate_design_variable(var)

    def test_nominal_above_upper_bound_raises(self) -> None:
        var = DesignVariable(
            name="x", quantity_kind="Length", unit="m",
            lower_bound=2.0, upper_bound=10.0, nominal=11.0,
        )
        with pytest.raises(ValueError, match="nominal"):
            validate_design_variable(var)

    def test_nominal_none_is_always_valid(self) -> None:
        var = DesignVariable(
            name="x", quantity_kind="Length", unit="m",
            lower_bound=0.0, upper_bound=100.0, nominal=None,
        )
        validate_design_variable(var)  # must not raise

    def test_nominal_at_lower_bound_is_valid(self) -> None:
        var = DesignVariable(
            name="x", quantity_kind="Length", unit="m",
            lower_bound=1.0, upper_bound=10.0, nominal=1.0,
        )
        validate_design_variable(var)  # must not raise

    def test_nominal_at_upper_bound_is_valid(self) -> None:
        var = DesignVariable(
            name="x", quantity_kind="Length", unit="m",
            lower_bound=1.0, upper_bound=10.0, nominal=10.0,
        )
        validate_design_variable(var)  # must not raise
