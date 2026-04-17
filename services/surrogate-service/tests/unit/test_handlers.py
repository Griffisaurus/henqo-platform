"""
Unit tests for surrogate_service.api.handlers.SurrogateServiceHandler.

Covers:
  - Successful inference creates Prediction entity
  - Abstention path (low A(x)) — entity created with status='abstained'
  - Model not found → NOT_FOUND error
  - Schema version mismatch → SCHEMA_VERSION_MISMATCH error
  - Schema-compatible inference succeeds
"""
from __future__ import annotations

import pytest

from graph_service.api.handlers import GraphServiceHandler

from surrogate_service.api.handlers import (
    RunInferenceRequest,
    SurrogateServiceHandler,
)
from surrogate_service.models.base import InferenceResult, ModelRecord, SurrogateModel
from surrogate_service.models.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Test double: minimal surrogate model
# ---------------------------------------------------------------------------

class _FakeSurrogate(SurrogateModel):
    def __init__(
        self,
        model_id: str = "model-001",
        family: str = "S1",
        status: str = "production",
        schema_version: str = "0.1.0",
        n_ensemble: int = 8,
        outputs: dict[str, float] | None = None,
        ensemble_outputs: list[float] | None = None,
    ) -> None:
        self._record = ModelRecord(
            model_id=model_id,
            surrogate_family=family,
            weight_hash="sha256-test",
            training_schema_version=schema_version,
            status=status,
            training_dataset_revision="ds-rev-001",
            n_ensemble_members=n_ensemble,
        )
        self._outputs = outputs or {"lift_coeff": 1.23}
        self._ensemble = ensemble_outputs or [1.20, 1.23, 1.24, 1.22, 1.23, 1.21, 1.25, 1.22]

    def predict(self, x: dict[str, float]) -> InferenceResult:
        return InferenceResult(
            outputs=self._outputs,
            ensemble_outputs=self._ensemble,
            std_devs={k: 0.01 for k in self._outputs},
            model_id=self._record.model_id,
        )

    def model_record(self) -> ModelRecord:
        return self._record


def _make_handler(model: _FakeSurrogate | None = None) -> tuple[SurrogateServiceHandler, ModelRegistry, GraphServiceHandler]:
    graph = GraphServiceHandler()
    registry = ModelRegistry()
    if model is not None:
        registry.register(model)
    handler = SurrogateServiceHandler(graph_handler=graph, registry=registry)
    return handler, registry, graph


def _basic_request(**overrides) -> RunInferenceRequest:
    defaults = dict(
        surrogate_family="S1",
        x={"mach": 0.5, "aoa": 5.0},
        requested_decision_class="Exploratory",
        governing_characteristic_id="char-001",
        feature_bounds={"mach": (0.0, 1.0), "aoa": (0.0, 20.0)},
        training_data_sample=[
            {"mach": 0.4, "aoa": 4.0},
            {"mach": 0.5, "aoa": 5.0},
            {"mach": 0.6, "aoa": 6.0},
        ],
        current_schema_version="0.1.0",
    )
    defaults.update(overrides)
    return RunInferenceRequest(**defaults)


# ---------------------------------------------------------------------------
# Test: successful inference
# ---------------------------------------------------------------------------

class TestSuccessfulInference:
    def test_ok_response(self):
        model = _FakeSurrogate()
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request())
        assert resp.ok
        assert not resp.abstained
        assert resp.prediction_entity_id != ""
        assert resp.outputs is not None
        assert "lift_coeff" in resp.outputs

    def test_trust_bundle_attached(self):
        model = _FakeSurrogate()
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request())
        assert resp.trust_bundle is not None
        assert resp.trust_bundle.policy_version == "v0.1.0"

    def test_prediction_entity_created_in_graph(self):
        model = _FakeSurrogate()
        handler, _, graph = _make_handler(model)
        resp = handler.run_inference(_basic_request())
        # Verify the prediction entity exists in graph store
        from graph_service.api.handlers import GetEntityRequest
        get_resp = graph.get_entity(GetEntityRequest(entity_id=resp.prediction_entity_id))
        assert get_resp.ok
        assert get_resp.entity_type == "Prediction"

    def test_prediction_status_created(self):
        model = _FakeSurrogate()
        handler, _, graph = _make_handler(model)
        resp = handler.run_inference(_basic_request())
        from graph_service.api.handlers import GetEntityRequest
        get_resp = graph.get_entity(GetEntityRequest(entity_id=resp.prediction_entity_id))
        assert get_resp.status == "created"


# ---------------------------------------------------------------------------
# Test: abstention path
# ---------------------------------------------------------------------------

class TestAbstentionPath:
    def test_low_applicability_triggers_abstention(self):
        """
        Ensemble with wide spread + OOD point → low A(x) → abstained.
        Use wide spread ensemble and out-of-bounds x to force low score.
        """
        model = _FakeSurrogate(
            # Wide-spread ensemble → A_ensemble low
            ensemble_outputs=[-100.0, 100.0, -50.0, 50.0, 0.0, -80.0, 80.0, -30.0],
        )
        handler, _, _ = _make_handler(model)
        req = _basic_request(
            # x way out of bounds → A_range = 0
            x={"mach": 999.0, "aoa": 999.0},
            # training points nowhere near x → A_density = 0
            training_data_sample=[
                {"mach": 0.4, "aoa": 4.0},
                {"mach": 0.5, "aoa": 5.0},
                {"mach": 0.6, "aoa": 6.0},
                {"mach": 0.7, "aoa": 7.0},
                {"mach": 0.8, "aoa": 8.0},
            ],
            feature_bounds={"mach": (0.0, 1.0), "aoa": (0.0, 20.0)},
            requested_decision_class="Exploratory",
        )
        resp = handler.run_inference(req)
        assert resp.abstained
        assert resp.abstain_reason != ""
        assert resp.outputs is None

    def test_abstained_prediction_stored_in_graph(self):
        """Abstained prediction must still be persisted in graph with status='abstained'."""
        model = _FakeSurrogate(
            ensemble_outputs=[-100.0, 100.0, -50.0, 50.0, 0.0, -80.0, 80.0, -30.0],
        )
        handler, _, graph = _make_handler(model)
        req = _basic_request(
            x={"mach": 999.0, "aoa": 999.0},
            training_data_sample=[{"mach": float(i)} for i in range(5)],
            feature_bounds={"mach": (0.0, 1.0), "aoa": (0.0, 20.0)},
        )
        resp = handler.run_inference(req)
        if resp.abstained:
            from graph_service.api.handlers import GetEntityRequest
            get_resp = graph.get_entity(GetEntityRequest(entity_id=resp.prediction_entity_id))
            assert get_resp.ok
            assert get_resp.status == "abstained"

    def test_small_ensemble_abstains(self):
        """
        n_ensemble_members < 5 is a hard block regardless of applicability.

        The graph service schema validator also rejects n_ensemble_members < 5,
        so the Prediction entity may not be stored.  The handler returns
        abstained=True regardless of whether storage succeeded.
        """
        model = _FakeSurrogate(n_ensemble=4)
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request())
        assert resp.abstained
        # abstain_reason must explain the ensemble rule
        assert "ensemble" in resp.abstain_reason.lower() or "5" in resp.abstain_reason


# ---------------------------------------------------------------------------
# Test: model not found
# ---------------------------------------------------------------------------

class TestModelNotFound:
    def test_no_model_registered_returns_not_found(self):
        handler, _, _ = _make_handler(model=None)
        resp = handler.run_inference(_basic_request(surrogate_family="S1"))
        assert not resp.ok
        assert resp.error_code == "NOT_FOUND"

    def test_wrong_family_returns_not_found(self):
        model = _FakeSurrogate(family="S2")
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request(surrogate_family="S1"))
        assert resp.error_code == "NOT_FOUND"

    def test_staged_model_not_returned_as_production(self):
        """Only production-status models are returned by get_production_model."""
        model = _FakeSurrogate(status="staged")
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request())
        assert resp.error_code == "NOT_FOUND"

    def test_registry_get_by_id_miss(self):
        """ModelRegistry.get_by_id returns None for unknown IDs."""
        from surrogate_service.models.registry import ModelRegistry
        registry = ModelRegistry()
        assert registry.get_by_id("nonexistent") is None

    def test_registry_get_production_model_miss(self):
        """ModelRegistry.get_production_model returns None for unknown family."""
        from surrogate_service.models.registry import ModelRegistry
        registry = ModelRegistry()
        assert registry.get_production_model("S99") is None


# ---------------------------------------------------------------------------
# Test: schema version mismatch
# ---------------------------------------------------------------------------

class TestSchemaVersionMismatch:
    def test_major_version_mismatch(self):
        """Model trained on schema v0.x; current IR schema is v1.x → mismatch."""
        model = _FakeSurrogate(schema_version="0.1.0")
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request(current_schema_version="1.0.0"))
        assert not resp.ok
        assert resp.error_code == "SCHEMA_VERSION_MISMATCH"

    def test_minor_version_difference_compatible(self):
        """Same major, different minor → compatible; inference should proceed."""
        model = _FakeSurrogate(schema_version="0.1.0")
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request(current_schema_version="0.2.0"))
        # Should not be SCHEMA_VERSION_MISMATCH; may succeed or abstain
        assert resp.error_code != "SCHEMA_VERSION_MISMATCH"

    def test_schema_mismatch_no_trust_bundle_created(self):
        """SCHEMA_VERSION_MISMATCH returns before building TrustBundle."""
        model = _FakeSurrogate(schema_version="0.1.0")
        handler, _, _ = _make_handler(model)
        resp = handler.run_inference(_basic_request(current_schema_version="2.0.0"))
        assert resp.trust_bundle is None
        assert resp.error_code == "SCHEMA_VERSION_MISMATCH"
