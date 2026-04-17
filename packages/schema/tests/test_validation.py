"""
Unit tests for packages/schema/src/validation.py.

Golden negative test set: one invalid payload per validation rule.
"""
from __future__ import annotations

import pytest

from henqo_schema.validation import ValidationError, validate_entity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_requirement(**overrides) -> dict:
    base = {
        "entity_id": "a1b2c3d4-e5f6-4789-ab12-cd34ef567890",
        "text": "Max tip displacement < 2 mm under 500 N axial load",
        "criticality": "High",
        "decision_class_required": "DesignGate",
    }
    base.update(overrides)
    return base


def _base_characteristic(**overrides) -> dict:
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


def _base_prediction(**overrides) -> dict:
    base = {
        "entity_id": "c3d4e5f6-a7b8-4901-cd34-ef56ab789012",
        "governing_characteristic_id": "b2c3d4e5-f6a7-4890-bc23-de45fa678901",
        "surrogate_family": "S2",
        "outputs": [{"name": "tip_displacement", "value": 1.42, "unit": "mm"}],
        "trust_bundle": {
            "model_revision_id": "mr-001",
            "training_dataset_revision": "ds-rev-42",
            "weight_hash": "sha256:abc123",
            "applicability_score": 0.91,
            "evaluated_decision_class": "DesignGate",
            "policy_version": "v0.1.0",
            "uncertainty": {"n_ensemble_members": 8},
        },
    }
    base.update(overrides)
    return base


def _base_simulation_case(**overrides) -> dict:
    base = {
        "entity_id": "d4e5f6a7-b8c9-4012-de45-fa67bc890123",
        "discipline": "structural",
        "solver": "FEniCS",
        "solver_version": "2019.1.0",
        "component_revision_id": "e5f6a7b8-c9d0-4123-ef56-ab78cd901234",
        "status": "queued",
    }
    base.update(overrides)
    return base


def _base_inspection_result(**overrides) -> dict:
    base = {
        "entity_id": "e5f6a7b8-c9d0-4123-ef56-ab78cd901234",
        "characteristic_id": "b2c3d4e5-f6a7-4890-bc23-de45fa678901",
        "component_revision_id": "f6a7b8c9-d0e1-4234-ab67-bc89de012345",
        "measured_value": {"value": 1.38, "unit": "mm"},
        "decision_rule": "< 2 mm",
        "measurement_uncertainty": {"value": 0.01, "unit": "mm"},
    }
    base.update(overrides)
    return base


def _base_surrogate_model(**overrides) -> dict:
    base = {
        "model_id": "sm-001",
        "surrogate_family": "S2",
        "weight_hash": "sha256:abc123",
        "weight_format": "safetensors",
        "status": "production",
    }
    base.update(overrides)
    return base


def _base_training_job(**overrides) -> dict:
    base = {
        "job_id": "tj-001",
        "surrogate_family": "S2",
        "training_dataset_revision": "ds-rev-42",
        "state": "queued",
    }
    base.update(overrides)
    return base


def _base_orchestrator_session(**overrides) -> dict:
    base = {
        "session_id": "os-001",
        "role": "SE",
        "orchestrator_model_id": "llama-3.1-8b-instruct",
        "policy_version": "v0.1.0",
    }
    base.update(overrides)
    return base


def _base_simulation_campaign(**overrides) -> dict:
    base = {
        "campaign_id": "sc-001",
        "surrogate_family": "S1",
        "status": "planned",
    }
    base.update(overrides)
    return base


def _base_ood_region(**overrides) -> dict:
    base = {
        "region_id": "ood-001",
        "surrogate_family": "S1",
        "model_revision_id": "mr-001",
    }
    base.update(overrides)
    return base


def _base_retraining_request(**overrides) -> dict:
    base = {
        "request_id": "rr-001",
        "surrogate_family": "S1",
        "trigger_type": "campaign_merged",
    }
    base.update(overrides)
    return base


def _base_trust_bundle(**overrides) -> dict:
    base = {
        "model_revision_id": "mr-001",
        "training_dataset_revision": "ds-rev-42",
        "weight_hash": "sha256:abc123",
        "applicability_score": 0.91,
        "evaluated_decision_class": "DesignGate",
        "policy_version": "v0.1.0",
        "uncertainty": {"n_ensemble_members": 8},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestValidEntityPasses:
    def test_requirement_valid(self):
        validate_entity("Requirement", _base_requirement())

    def test_requirement_with_quantity(self):
        payload = _base_requirement()
        payload["quantity"] = {"value": 2.0, "unit": "mm"}
        validate_entity("Requirement", payload)

    def test_characteristic_valid(self):
        validate_entity("Characteristic", _base_characteristic())

    def test_characteristic_force_unit(self):
        payload = _base_characteristic(quantity_kind="Force", unit="kN")
        validate_entity("Characteristic", payload)

    def test_prediction_valid(self):
        validate_entity("Prediction", _base_prediction())

    def test_simulation_case_valid(self):
        validate_entity("SimulationCase", _base_simulation_case())

    def test_inspection_result_valid(self):
        validate_entity("InspectionResult", _base_inspection_result())

    def test_surrogate_model_valid(self):
        validate_entity("SurrogateModel", _base_surrogate_model())

    def test_training_job_valid(self):
        validate_entity("TrainingJob", _base_training_job())

    def test_orchestrator_session_valid(self):
        validate_entity("OrchestratorSession", _base_orchestrator_session())

    def test_simulation_campaign_valid(self):
        validate_entity("SimulationCampaign", _base_simulation_campaign())

    def test_ood_region_valid(self):
        validate_entity("OodRegion", _base_ood_region())

    def test_retraining_request_valid(self):
        validate_entity("RetrainingRequest", _base_retraining_request())

    def test_unknown_entity_type_passes_no_required_fields(self):
        # Entity types without registered required fields just pass
        validate_entity("UnknownFutureEntity", {"some_field": "value"})


# ---------------------------------------------------------------------------
# Step 1: Required field validation
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_requirement_missing_entity_id(self):
        payload = _base_requirement()
        del payload["entity_id"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Requirement", payload)
        err = exc.value
        assert err.error_code == "SCHEMA_VALIDATION_ERROR"
        assert err.field == "entity_id"
        assert err.entity_type == "Requirement"

    def test_requirement_missing_text(self):
        with pytest.raises(ValidationError) as exc:
            validate_entity("Requirement", _base_requirement(text=""))
        assert exc.value.field == "text"

    def test_requirement_missing_criticality(self):
        payload = _base_requirement()
        del payload["criticality"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Requirement", payload)
        assert exc.value.field == "criticality"

    def test_requirement_missing_decision_class(self):
        payload = _base_requirement()
        del payload["decision_class_required"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Requirement", payload)
        assert exc.value.field == "decision_class_required"

    def test_characteristic_missing_name(self):
        payload = _base_characteristic()
        del payload["name"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Characteristic", payload)
        assert exc.value.field == "name"

    def test_characteristic_missing_governing_req(self):
        payload = _base_characteristic()
        del payload["governing_requirement_id"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Characteristic", payload)
        assert exc.value.field == "governing_requirement_id"

    def test_prediction_missing_trust_bundle_field(self):
        payload = _base_prediction()
        del payload["trust_bundle"]
        with pytest.raises(ValidationError):
            validate_entity("Prediction", payload)

    def test_simulation_case_missing_solver(self):
        payload = _base_simulation_case()
        del payload["solver"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("SimulationCase", payload)
        assert exc.value.field == "solver"

    def test_inspection_result_missing_measured_value(self):
        payload = _base_inspection_result()
        del payload["measured_value"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("InspectionResult", payload)
        assert exc.value.field == "measured_value"

    def test_field_none_treated_as_missing(self):
        payload = _base_requirement(text=None)
        with pytest.raises(ValidationError) as exc:
            validate_entity("Requirement", payload)
        assert exc.value.field == "text"

    def test_surrogate_model_missing_weight_hash(self):
        payload = _base_surrogate_model()
        del payload["weight_hash"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("SurrogateModel", payload)
        assert exc.value.field == "weight_hash"

    def test_training_job_missing_state(self):
        payload = _base_training_job()
        del payload["state"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("TrainingJob", payload)
        assert exc.value.field == "state"

    def test_orchestrator_session_missing_policy_version(self):
        payload = _base_orchestrator_session()
        del payload["policy_version"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("OrchestratorSession", payload)
        assert exc.value.field == "policy_version"

    def test_retraining_request_missing_trigger_type(self):
        payload = _base_retraining_request()
        del payload["trigger_type"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("RetrainingRequest", payload)
        assert exc.value.field == "trigger_type"


# ---------------------------------------------------------------------------
# Step 2: Unit validation
# ---------------------------------------------------------------------------

class TestUnitValidation:
    # --- Requirement.quantity ---

    def test_requirement_quantity_missing_unit(self):
        payload = _base_requirement()
        payload["quantity"] = {"value": 2.0, "unit": ""}
        with pytest.raises(ValidationError) as exc:
            validate_entity("Requirement", payload)
        assert exc.value.error_code == "UNIT_MISSING_ERROR"
        assert "quantity.unit" in exc.value.field

    def test_requirement_quantity_unparseable_unit(self):
        payload = _base_requirement()
        payload["quantity"] = {"value": 2.0, "unit": "furlongs"}
        with pytest.raises(ValidationError) as exc:
            validate_entity("Requirement", payload)
        assert exc.value.error_code == "UNIT_PARSE_ERROR"

    def test_requirement_no_quantity_passes(self):
        validate_entity("Requirement", _base_requirement())

    def test_requirement_empty_quantity_passes(self):
        payload = _base_requirement()
        payload["quantity"] = None
        validate_entity("Requirement", payload)

    # --- Characteristic.unit ---

    def test_characteristic_empty_unit(self):
        # `unit` is required — required-field check fires before unit-check
        payload = _base_characteristic(unit="")
        with pytest.raises(ValidationError) as exc:
            validate_entity("Characteristic", payload)
        assert exc.value.error_code == "SCHEMA_VALIDATION_ERROR"
        assert exc.value.field == "unit"

    def test_characteristic_unparseable_unit(self):
        payload = _base_characteristic(unit="cubits")
        with pytest.raises(ValidationError) as exc:
            validate_entity("Characteristic", payload)
        assert exc.value.error_code == "UNIT_PARSE_ERROR"
        assert exc.value.field == "unit"

    def test_characteristic_all_valid_units(self):
        valid_units = ["N", "kN", "MN", "kg", "m", "mm", "K", "Pa", "MPa", "m/s", "Hz", "rad", "J", "W", "s"]
        for unit in valid_units:
            validate_entity("Characteristic", _base_characteristic(unit=unit))

    # --- InspectionResult quantity fields ---

    def test_inspection_measured_value_missing_unit(self):
        payload = _base_inspection_result()
        payload["measured_value"] = {"value": 1.38, "unit": ""}
        with pytest.raises(ValidationError) as exc:
            validate_entity("InspectionResult", payload)
        assert exc.value.error_code == "UNIT_MISSING_ERROR"
        assert "measured_value" in exc.value.field

    def test_inspection_measured_value_unparseable_unit(self):
        payload = _base_inspection_result()
        payload["measured_value"] = {"value": 1.38, "unit": "bananas"}
        with pytest.raises(ValidationError) as exc:
            validate_entity("InspectionResult", payload)
        assert exc.value.error_code == "UNIT_PARSE_ERROR"

    def test_inspection_uncertainty_missing_unit(self):
        payload = _base_inspection_result()
        payload["measurement_uncertainty"] = {"value": 0.01, "unit": ""}
        with pytest.raises(ValidationError) as exc:
            validate_entity("InspectionResult", payload)
        assert exc.value.error_code == "UNIT_MISSING_ERROR"
        assert "measurement_uncertainty" in exc.value.field

    def test_inspection_uncertainty_unparseable_unit(self):
        payload = _base_inspection_result()
        payload["measurement_uncertainty"] = {"value": 0.01, "unit": "smoot"}
        with pytest.raises(ValidationError) as exc:
            validate_entity("InspectionResult", payload)
        assert exc.value.error_code == "UNIT_PARSE_ERROR"

    def test_inspection_missing_quantity_fields_pass(self):
        payload = _base_inspection_result()
        del payload["measured_value"]
        del payload["measurement_uncertainty"]
        # Still fails on required-field check for measured_value
        with pytest.raises(ValidationError) as exc:
            validate_entity("InspectionResult", payload)
        assert exc.value.error_code == "SCHEMA_VALIDATION_ERROR"

    def test_inspection_null_quantity_fields_skip_unit_check(self):
        # If both quantity fields are null/empty, required-field check fires first
        payload = _base_inspection_result()
        payload["measured_value"] = None
        with pytest.raises(ValidationError) as exc:
            validate_entity("InspectionResult", payload)
        assert exc.value.field == "measured_value"


# ---------------------------------------------------------------------------
# Step 4: Status validation
# ---------------------------------------------------------------------------

class TestStatusValidation:
    def test_simulation_case_invalid_status(self):
        payload = _base_simulation_case(status="pending")
        with pytest.raises(ValidationError) as exc:
            validate_entity("SimulationCase", payload)
        assert exc.value.error_code == "INVALID_STATUS"
        assert exc.value.field == "status"

    def test_simulation_case_valid_statuses(self):
        for status in ["queued", "running", "completed", "failed", "validated", "invalidated"]:
            validate_entity("SimulationCase", _base_simulation_case(status=status))

    def test_characteristic_invalid_status(self):
        payload = _base_characteristic()
        payload["status"] = "approved"
        with pytest.raises(ValidationError) as exc:
            validate_entity("Characteristic", payload)
        assert exc.value.error_code == "INVALID_STATUS"

    def test_characteristic_valid_statuses(self):
        for status in ["unverified", "surrogate_estimated", "simulation_validated",
                       "inspection_confirmed", "released", "unresolved", "superseded"]:
            payload = _base_characteristic()
            payload["status"] = status
            validate_entity("Characteristic", payload)

    def test_status_empty_string_passes_state_check(self):
        # Characteristic.status is not required, so empty string skips state check
        payload = _base_characteristic()
        payload["status"] = ""
        validate_entity("Characteristic", payload)

    def test_entity_without_status_registry_ignores_status(self):
        # Requirement has no status registry — any status value passes
        payload = _base_requirement()
        payload["status"] = "some_random_status"
        validate_entity("Requirement", payload)

    def test_component_revision_invalid_status(self):
        payload = {
            "entity_id": "f6a7b8c9-d0e1-4234-ab67-bc89de012345",
            "component_name": "wing_spar",
            "status": "approved",
        }
        with pytest.raises(ValidationError) as exc:
            validate_entity("ComponentRevision", payload)
        assert exc.value.error_code == "INVALID_STATUS"

    def test_release_manifest_invalid_status(self):
        payload = {
            "manifest_id": "m-001",
            "assembly_revision_id": "ar-001",
            "manifest_status": "approved",
            "status": "approved",
        }
        with pytest.raises(ValidationError) as exc:
            validate_entity("ReleaseManifest", payload)
        assert exc.value.error_code == "INVALID_STATUS"

    def test_prediction_invalid_status(self):
        payload = _base_prediction()
        payload["status"] = "archived"
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.error_code == "INVALID_STATUS"

    def test_prediction_valid_statuses(self):
        for status in ["created", "blocked", "used", "superseded", "abstained"]:
            payload = _base_prediction()
            payload["status"] = status
            validate_entity("Prediction", payload)


# ---------------------------------------------------------------------------
# Step 5: TrustBundle validation
# ---------------------------------------------------------------------------

class TestTrustBundleValidation:
    def test_prediction_missing_trust_bundle(self):
        payload = _base_prediction()
        payload["trust_bundle"] = None
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.error_code == "TRUST_BUNDLE_MISSING"
        assert exc.value.field == "trust_bundle"

    def test_trust_bundle_missing_model_revision_id(self):
        payload = _base_prediction()
        del payload["trust_bundle"]["model_revision_id"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.model_revision_id"

    def test_trust_bundle_missing_training_dataset_revision(self):
        payload = _base_prediction()
        del payload["trust_bundle"]["training_dataset_revision"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.training_dataset_revision"

    def test_trust_bundle_missing_weight_hash(self):
        payload = _base_prediction()
        del payload["trust_bundle"]["weight_hash"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.weight_hash"

    def test_trust_bundle_missing_applicability_score(self):
        payload = _base_prediction()
        del payload["trust_bundle"]["applicability_score"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.applicability_score"

    def test_trust_bundle_missing_evaluated_decision_class(self):
        payload = _base_prediction()
        del payload["trust_bundle"]["evaluated_decision_class"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.evaluated_decision_class"

    def test_trust_bundle_missing_policy_version(self):
        payload = _base_prediction()
        del payload["trust_bundle"]["policy_version"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.policy_version"

    def test_trust_bundle_n_ensemble_members_below_minimum(self):
        payload = _base_prediction()
        payload["trust_bundle"]["uncertainty"]["n_ensemble_members"] = 4
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.uncertainty.n_ensemble_members"
        assert "below minimum of 5" in exc.value.message

    def test_trust_bundle_n_ensemble_members_exactly_5(self):
        payload = _base_prediction()
        payload["trust_bundle"]["uncertainty"]["n_ensemble_members"] = 5
        validate_entity("Prediction", payload)

    def test_trust_bundle_n_ensemble_members_zero_when_missing(self):
        payload = _base_prediction()
        del payload["trust_bundle"]["uncertainty"]
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.uncertainty.n_ensemble_members"

    def test_trust_bundle_invalid_decision_class(self):
        payload = _base_prediction()
        payload["trust_bundle"]["evaluated_decision_class"] = "Marketing"
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert "evaluated_decision_class" in exc.value.field

    def test_trust_bundle_blocked_decision_class_valid(self):
        payload = _base_prediction()
        payload["trust_bundle"]["evaluated_decision_class"] = "Blocked"
        validate_entity("Prediction", payload)

    def test_trust_bundle_all_valid_decision_classes(self):
        for dc in ["Exploratory", "DesignGate", "ReleaseCritical", "SafetyCritical", "Blocked"]:
            payload = _base_prediction()
            payload["trust_bundle"]["evaluated_decision_class"] = dc
            validate_entity("Prediction", payload)

    def test_trust_bundle_field_empty_string_fails(self):
        payload = _base_prediction()
        payload["trust_bundle"]["weight_hash"] = ""
        with pytest.raises(ValidationError) as exc:
            validate_entity("Prediction", payload)
        assert exc.value.field == "trust_bundle.weight_hash"


# ---------------------------------------------------------------------------
# ValidationError.to_dict()
# ---------------------------------------------------------------------------

class TestValidationErrorToDict:
    def test_to_dict_contains_all_keys(self):
        err = ValidationError(
            error_code="SCHEMA_VALIDATION_ERROR",
            field="entity_id",
            message="Required field 'entity_id' is missing or empty",
            entity_type="Requirement",
        )
        d = err.to_dict()
        assert set(d.keys()) == {"error_code", "field", "message", "entity_type", "schema_version"}
        assert d["schema_version"] == "0.1.0"
        assert d["error_code"] == "SCHEMA_VALIDATION_ERROR"

    def test_validation_error_is_exception(self):
        err = ValidationError(
            error_code="UNIT_MISSING_ERROR",
            field="unit",
            message="msg",
            entity_type="Characteristic",
        )
        assert isinstance(err, Exception)
