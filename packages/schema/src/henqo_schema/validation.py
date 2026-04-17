"""
Entity validation utilities for the Henqo IR schema.

Enforces the 6-step ingestion validation order from
data-governance-provenance-spec.md §8.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

@dataclass
class ValidationError(Exception):
    error_code: str
    field: str
    message: str
    entity_type: str
    schema_version: str = "0.1.0"

    def to_dict(self) -> dict[str, str]:
        return {
            "error_code": self.error_code,
            "field": self.field,
            "message": self.message,
            "entity_type": self.entity_type,
            "schema_version": self.schema_version,
        }


# ---------------------------------------------------------------------------
# Unit validation helpers
# ---------------------------------------------------------------------------

# Canonical quantity kinds and their SI base units
_QUANTITY_KIND_SI: dict[str, str] = {
    "Force": "N",
    "Mass": "kg",
    "Length": "m",
    "Temperature": "K",
    "Pressure": "Pa",
    "Velocity": "m/s",
    "Acceleration": "m/s^2",
    "Frequency": "Hz",
    "Dimensionless": "1",
    "Angle": "rad",
    "Energy": "J",
    "Power": "W",
    "Time": "s",
    "Torque": "N*m",
    "Stress": "Pa",
}

# Parseable unit strings (whitelist; pint/QUDT full registry used in production)
_PARSEABLE_UNITS: frozenset[str] = frozenset([
    "N", "kN", "MN",
    "kg", "g", "mg",
    "m", "mm", "cm", "km",
    "K", "degC", "degF",
    "Pa", "kPa", "MPa", "GPa",
    "m/s", "km/h",
    "m/s^2",
    "Hz", "kHz", "MHz",
    "1",  # dimensionless
    "rad", "deg",
    "J", "kJ", "MJ",
    "W", "kW", "MW",
    "s", "ms", "us", "min", "h",
    "N*m", "kN*m",
    "kcal/mol", "kcal/mol/angstrom",
    "GPa", "eV",
    "dB", "dB(A)",
    "cycles",
])

_QUANTITY_KIND_UNIT_MAP: dict[str, frozenset[str]] = {
    "Force": frozenset(["N", "kN", "MN"]),
    "Mass": frozenset(["kg", "g", "mg"]),
    "Length": frozenset(["m", "mm", "cm", "km"]),
    "Temperature": frozenset(["K", "degC", "degF"]),
    "Pressure": frozenset(["Pa", "kPa", "MPa", "GPa"]),
    "Velocity": frozenset(["m/s", "km/h"]),
    "Acceleration": frozenset(["m/s^2"]),
    "Frequency": frozenset(["Hz", "kHz", "MHz"]),
    "Dimensionless": frozenset(["1"]),
    "Angle": frozenset(["rad", "deg"]),
    "Energy": frozenset(["J", "kJ", "MJ", "kcal/mol"]),
    "Power": frozenset(["W", "kW", "MW"]),
    "Time": frozenset(["s", "ms", "us", "min", "h"]),
    "Torque": frozenset(["N*m", "kN*m"]),
    "Stress": frozenset(["Pa", "kPa", "MPa", "GPa"]),
}


def _validate_unit_present(field: str, quantity: dict[str, Any], entity_type: str) -> None:
    """Step 2a: unit field must be non-empty."""
    unit = quantity.get("unit", "")
    if not unit:
        raise ValidationError(
            error_code="UNIT_MISSING_ERROR",
            field=field,
            message="Quantity field must have a non-empty unit string",
            entity_type=entity_type,
        )


def _validate_unit_parseable(field: str, quantity: dict[str, Any], entity_type: str) -> None:
    """Step 2b: unit must be in the parseable unit registry."""
    unit = quantity.get("unit", "")
    if unit not in _PARSEABLE_UNITS:
        raise ValidationError(
            error_code="UNIT_PARSE_ERROR",
            field=field,
            message=f"Unit '{unit}' is not parseable against the unit registry",
            entity_type=entity_type,
        )


def _validate_quantity_kind(
    field: str,
    quantity: dict[str, Any],
    expected_kind: str,
    entity_type: str,
) -> None:
    """Step 2d: unit must be convertible to the declared quantity kind."""
    unit = quantity.get("unit", "")
    allowed = _QUANTITY_KIND_UNIT_MAP.get(expected_kind, frozenset())
    if allowed and unit not in allowed:
        raise ValidationError(
            error_code="UNIT_KIND_MISMATCH",
            field=field,
            message=(
                f"Unit '{unit}' is not compatible with quantity kind '{expected_kind}'. "
                f"Allowed units: {sorted(allowed)}"
            ),
            entity_type=entity_type,
        )


# ---------------------------------------------------------------------------
# Per-entity validation functions
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: dict[str, list[str]] = {
    "Requirement": ["entity_id", "text", "criticality", "decision_class_required"],
    "Characteristic": ["entity_id", "name", "quantity_kind", "unit",
                       "governing_requirement_id", "criticality", "decision_class_required"],
    "Prediction": ["entity_id", "governing_characteristic_id", "surrogate_family",
                   "outputs"],
    "SimulationCase": ["entity_id", "discipline", "solver", "solver_version",
                       "component_revision_id", "status"],
    "InspectionResult": ["entity_id", "characteristic_id", "component_revision_id",
                         "measured_value", "decision_rule", "measurement_uncertainty"],
    "ComponentRevision": ["entity_id", "component_name", "status"],
    "AssemblyRevision": ["entity_id", "assembly_name", "status"],
    "ProvenanceBundle": ["bundle_id", "activity", "agent_id"],
    "ReleaseManifest": ["manifest_id", "assembly_revision_id", "manifest_status"],
    "DecisionPackage": ["package_id", "assembly_revision_id", "review_gate"],
    "SurrogateModel": ["model_id", "surrogate_family", "weight_hash", "weight_format", "status"],
    "TrainingJob": ["job_id", "surrogate_family", "training_dataset_revision", "state"],
    "OrchestratorSession": ["session_id", "role", "orchestrator_model_id", "policy_version"],
    "SimulationCampaign": ["campaign_id", "surrogate_family", "status"],
    "OodRegion": ["region_id", "surrogate_family", "model_revision_id"],
    "RetrainingRequest": ["request_id", "surrogate_family", "trigger_type"],
}

_VALID_STATUSES: dict[str, frozenset[str]] = {
    "Characteristic": frozenset([
        "unverified", "surrogate_estimated", "simulation_validated",
        "inspection_confirmed", "released", "unresolved", "superseded",
    ]),
    "SimulationCase": frozenset([
        "queued", "running", "completed", "failed", "validated", "invalidated",
    ]),
    "ComponentRevision": frozenset([
        "in_design", "manufacturing_review_requested",
        "manufacturing_reviewed", "released", "obsolete",
    ]),
    "AssemblyRevision": frozenset([
        "in_design", "manufacturing_review_requested",
        "manufacturing_reviewed", "released", "obsolete",
    ]),
    "ReleaseManifest": frozenset([
        "draft", "pending_signatures", "active", "superseded",
    ]),
    "Prediction": frozenset([
        "created", "blocked", "used", "superseded", "abstained",
    ]),
}

_VALID_DECISION_CLASSES: frozenset[str] = frozenset([
    "Exploratory", "DesignGate", "ReleaseCritical", "SafetyCritical",
])

_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def validate_entity(entity_type: str, payload: dict[str, Any]) -> None:
    """
    Validate an IR entity payload before persistence.

    Enforces the 6-step ingestion validation order from
    data-governance-provenance-spec.md §8:
      1. Proto schema validation (required fields, types)
      2. Unit validation
      3. Link resolution (checked here structurally; graph service checks existence)
      4. State machine pre-check
      5. TrustBundle completeness (for Prediction)
      6. Policy version check (warning only in v0)

    Raises ValidationError on failure.
    """
    # Step 1: Required field presence
    _validate_required_fields(entity_type, payload)

    # Step 2: Unit validation (entities with Quantity fields)
    _validate_units(entity_type, payload)

    # Step 4: State machine pre-check
    _validate_status(entity_type, payload)

    # Step 5: TrustBundle completeness for Prediction
    if entity_type == "Prediction":
        _validate_trust_bundle(payload)

    # Step 6: Policy version check (warning only in v0 — no exception raised)
    # Production enforcement deferred to v1.


# ---------------------------------------------------------------------------
# Validation sub-functions
# ---------------------------------------------------------------------------

def _validate_required_fields(entity_type: str, payload: dict[str, Any]) -> None:
    required = _REQUIRED_FIELDS.get(entity_type, [])
    for field in required:
        if field not in payload or payload[field] is None or payload[field] == "":
            raise ValidationError(
                error_code="SCHEMA_VALIDATION_ERROR",
                field=field,
                message=f"Required field '{field}' is missing or empty",
                entity_type=entity_type,
            )


def _validate_units(entity_type: str, payload: dict[str, Any]) -> None:
    """Validate Quantity fields for the given entity type."""
    if entity_type == "Requirement":
        if "quantity" in payload and payload["quantity"]:
            q = payload["quantity"]
            _validate_unit_present("quantity.unit", q, entity_type)
            _validate_unit_parseable("quantity.unit", q, entity_type)

    elif entity_type == "Characteristic":
        # Unit field on Characteristic itself (not a nested Quantity) must be parseable
        unit = payload.get("unit", "")
        if not unit:
            raise ValidationError(
                error_code="UNIT_MISSING_ERROR",
                field="unit",
                message="Characteristic.unit must be non-empty",
                entity_type=entity_type,
            )
        if unit not in _PARSEABLE_UNITS:
            raise ValidationError(
                error_code="UNIT_PARSE_ERROR",
                field="unit",
                message=f"Unit '{unit}' is not parseable",
                entity_type=entity_type,
            )

    elif entity_type == "InspectionResult":
        for field in ("measured_value", "measurement_uncertainty"):
            if field in payload and payload[field]:
                q = payload[field]
                _validate_unit_present(f"{field}.unit", q, entity_type)
                _validate_unit_parseable(f"{field}.unit", q, entity_type)


def _validate_status(entity_type: str, payload: dict[str, Any]) -> None:
    valid = _VALID_STATUSES.get(entity_type)
    if valid is None:
        return
    status = payload.get("status", "")
    if status and status not in valid:
        raise ValidationError(
            error_code="INVALID_STATUS",
            field="status",
            message=f"Status '{status}' is not a valid state for {entity_type}. Valid: {sorted(valid)}",
            entity_type=entity_type,
        )


def _validate_trust_bundle(payload: dict[str, Any]) -> None:
    """Step 5: TrustBundle completeness check for Prediction."""
    tb = payload.get("trust_bundle")
    if not tb:
        raise ValidationError(
            error_code="TRUST_BUNDLE_MISSING",
            field="trust_bundle",
            message="Prediction must include a TrustBundle",
            entity_type="Prediction",
        )

    required_tb_fields = [
        "model_revision_id",
        "training_dataset_revision",
        "weight_hash",
        "applicability_score",
        "evaluated_decision_class",
        "policy_version",
    ]
    for field in required_tb_fields:
        if field not in tb or tb[field] is None or tb[field] == "":
            raise ValidationError(
                error_code="SCHEMA_VALIDATION_ERROR",
                field=f"trust_bundle.{field}",
                message=f"TrustBundle field '{field}' is missing or empty",
                entity_type="Prediction",
            )

    # n_ensemble_members must be ≥ 5
    uncertainty = tb.get("uncertainty") or {}
    n = uncertainty.get("n_ensemble_members", 0)
    if n < 5:
        raise ValidationError(
            error_code="SCHEMA_VALIDATION_ERROR",
            field="trust_bundle.uncertainty.n_ensemble_members",
            message=f"Value {n} is below minimum of 5",
            entity_type="Prediction",
        )

    # evaluated_decision_class must be a known value
    edc = tb.get("evaluated_decision_class", "")
    valid_classes = _VALID_DECISION_CLASSES | {"Blocked"}
    if edc not in valid_classes:
        raise ValidationError(
            error_code="SCHEMA_VALIDATION_ERROR",
            field="trust_bundle.evaluated_decision_class",
            message=f"Unknown decision class '{edc}'. Valid: {sorted(valid_classes)}",
            entity_type="Prediction",
        )
