"""
State machine unit tests.

One test per legal transition (verify succeeds).
One test per illegal transition (verify IllegalTransitionError).
One test per approval-gated transition (verify MissingApprovalError without approver).
"""
from __future__ import annotations

import pytest

from graph_service.domain.state_machine import (
    IllegalTransitionError,
    MissingApprovalError,
    check_transition,
    is_approval_gated,
    legal_next_states,
)


# ---------------------------------------------------------------------------
# Characteristic — legal transitions
# ---------------------------------------------------------------------------

class TestCharacteristicLegalTransitions:
    def test_unverified_to_surrogate_estimated(self):
        check_transition("Characteristic", "unverified", "surrogate_estimated")

    def test_unverified_to_simulation_validated(self):
        check_transition("Characteristic", "unverified", "simulation_validated")

    def test_unverified_to_unresolved(self):
        check_transition("Characteristic", "unverified", "unresolved")

    def test_surrogate_estimated_to_simulation_validated(self):
        check_transition("Characteristic", "surrogate_estimated", "simulation_validated")

    def test_surrogate_estimated_to_unresolved(self):
        check_transition("Characteristic", "surrogate_estimated", "unresolved")

    def test_simulation_validated_to_inspection_confirmed(self):
        check_transition("Characteristic", "simulation_validated", "inspection_confirmed")

    def test_simulation_validated_to_surrogate_estimated(self):
        check_transition("Characteristic", "simulation_validated", "surrogate_estimated")

    def test_inspection_confirmed_to_released_with_approver(self):
        check_transition("Characteristic", "inspection_confirmed", "released", approver_id="lse-001")

    def test_inspection_confirmed_to_simulation_validated(self):
        check_transition("Characteristic", "inspection_confirmed", "simulation_validated")

    def test_released_to_superseded(self):
        check_transition("Characteristic", "released", "superseded")

    def test_unresolved_to_surrogate_estimated(self):
        check_transition("Characteristic", "unresolved", "surrogate_estimated")

    def test_unresolved_to_simulation_validated(self):
        check_transition("Characteristic", "unresolved", "simulation_validated")


class TestCharacteristicIllegalTransitions:
    def test_unverified_to_released(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Characteristic", "unverified", "released")

    def test_released_to_unverified(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Characteristic", "released", "unverified")

    def test_surrogate_estimated_to_released(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Characteristic", "surrogate_estimated", "released")

    def test_inspection_confirmed_to_unverified(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Characteristic", "inspection_confirmed", "unverified")

    def test_superseded_to_anything(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Characteristic", "superseded", "unverified")


class TestCharacteristicApprovalGates:
    def test_inspection_confirmed_to_released_missing_approver(self):
        with pytest.raises(MissingApprovalError):
            check_transition("Characteristic", "inspection_confirmed", "released")

    def test_inspection_confirmed_to_released_empty_approver(self):
        with pytest.raises(MissingApprovalError):
            check_transition("Characteristic", "inspection_confirmed", "released", approver_id="")

    def test_is_approval_gated_inspection_confirmed_to_released(self):
        assert is_approval_gated("Characteristic", "inspection_confirmed", "released") is True

    def test_is_not_gated_unverified_to_surrogate_estimated(self):
        assert is_approval_gated("Characteristic", "unverified", "surrogate_estimated") is False


# ---------------------------------------------------------------------------
# Prediction — legal transitions
# ---------------------------------------------------------------------------

class TestPredictionLegalTransitions:
    def test_created_to_used(self):
        check_transition("Prediction", "created", "used")

    def test_created_to_blocked(self):
        check_transition("Prediction", "created", "blocked")

    def test_created_to_abstained(self):
        check_transition("Prediction", "created", "abstained")

    def test_created_to_superseded(self):
        check_transition("Prediction", "created", "superseded")

    def test_blocked_to_created(self):
        check_transition("Prediction", "blocked", "created")

    def test_used_to_superseded(self):
        check_transition("Prediction", "used", "superseded")

    def test_abstained_to_superseded(self):
        check_transition("Prediction", "abstained", "superseded")


class TestPredictionIllegalTransitions:
    def test_used_to_created(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Prediction", "used", "created")

    def test_abstained_to_used(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Prediction", "abstained", "used")

    def test_superseded_to_used(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("Prediction", "superseded", "used")


# ---------------------------------------------------------------------------
# SimulationCase — legal transitions
# ---------------------------------------------------------------------------

class TestSimulationCaseLegalTransitions:
    def test_queued_to_running(self):
        check_transition("SimulationCase", "queued", "running")

    def test_queued_to_failed(self):
        check_transition("SimulationCase", "queued", "failed")

    def test_running_to_completed(self):
        check_transition("SimulationCase", "running", "completed")

    def test_running_to_failed(self):
        check_transition("SimulationCase", "running", "failed")

    def test_completed_to_validated(self):
        check_transition("SimulationCase", "completed", "validated")

    def test_completed_to_invalidated(self):
        check_transition("SimulationCase", "completed", "invalidated")

    def test_validated_to_invalidated(self):
        check_transition("SimulationCase", "validated", "invalidated")

    def test_failed_to_queued(self):
        check_transition("SimulationCase", "failed", "queued")


class TestSimulationCaseIllegalTransitions:
    def test_queued_to_validated(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("SimulationCase", "queued", "validated")

    def test_validated_to_running(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("SimulationCase", "validated", "running")

    def test_invalidated_to_validated(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("SimulationCase", "invalidated", "validated")


# ---------------------------------------------------------------------------
# ComponentRevision — legal transitions
# ---------------------------------------------------------------------------

class TestComponentRevisionLegalTransitions:
    def test_in_design_to_review_requested(self):
        check_transition("ComponentRevision", "in_design", "manufacturing_review_requested")

    def test_review_requested_to_reviewed(self):
        check_transition("ComponentRevision", "manufacturing_review_requested", "manufacturing_reviewed")

    def test_review_requested_to_in_design(self):
        check_transition("ComponentRevision", "manufacturing_review_requested", "in_design")

    def test_reviewed_to_released_with_approver(self):
        check_transition("ComponentRevision", "manufacturing_reviewed", "released", approver_id="mfg-lead-001")

    def test_reviewed_to_in_design(self):
        check_transition("ComponentRevision", "manufacturing_reviewed", "in_design")

    def test_released_to_obsolete(self):
        check_transition("ComponentRevision", "released", "obsolete")


class TestComponentRevisionApprovalGates:
    def test_reviewed_to_released_missing_approver(self):
        with pytest.raises(MissingApprovalError):
            check_transition("ComponentRevision", "manufacturing_reviewed", "released")

    def test_is_approval_gated(self):
        assert is_approval_gated("ComponentRevision", "manufacturing_reviewed", "released") is True


# ---------------------------------------------------------------------------
# ReleaseManifest — legal transitions
# ---------------------------------------------------------------------------

class TestReleaseManifestLegalTransitions:
    def test_draft_to_pending_signatures(self):
        check_transition("ReleaseManifest", "draft", "pending_signatures")

    def test_pending_signatures_to_active_with_approver(self):
        check_transition("ReleaseManifest", "pending_signatures", "active", approver_id="release-eng-001")

    def test_pending_signatures_to_draft(self):
        check_transition("ReleaseManifest", "pending_signatures", "draft")

    def test_active_to_superseded(self):
        check_transition("ReleaseManifest", "active", "superseded")


class TestReleaseManifestApprovalGates:
    def test_pending_signatures_to_active_missing_approver(self):
        with pytest.raises(MissingApprovalError):
            check_transition("ReleaseManifest", "pending_signatures", "active")

    def test_is_approval_gated(self):
        assert is_approval_gated("ReleaseManifest", "pending_signatures", "active") is True


# ---------------------------------------------------------------------------
# SurrogateModel — legal transitions
# ---------------------------------------------------------------------------

class TestSurrogateModelLegalTransitions:
    def test_training_to_benchmarking(self):
        check_transition("SurrogateModel", "training", "benchmarking")

    def test_training_to_failed(self):
        check_transition("SurrogateModel", "training", "failed")

    def test_benchmarking_to_staged(self):
        check_transition("SurrogateModel", "benchmarking", "staged")

    def test_benchmarking_to_failed(self):
        check_transition("SurrogateModel", "benchmarking", "failed")

    def test_staged_to_production_with_approver(self):
        check_transition("SurrogateModel", "staged", "production", approver_id="ml-lead-001")

    def test_staged_to_deprecated(self):
        check_transition("SurrogateModel", "staged", "deprecated")

    def test_production_to_deprecated(self):
        check_transition("SurrogateModel", "production", "deprecated")

    def test_deprecated_to_retired(self):
        check_transition("SurrogateModel", "deprecated", "retired")


class TestSurrogateModelApprovalGates:
    def test_staged_to_production_missing_approver(self):
        with pytest.raises(MissingApprovalError):
            check_transition("SurrogateModel", "staged", "production")


# ---------------------------------------------------------------------------
# Unknown entity type
# ---------------------------------------------------------------------------

class TestUnknownEntityType:
    def test_unknown_entity_raises_illegal_transition(self):
        with pytest.raises(IllegalTransitionError):
            check_transition("UnknownEntity", "state_a", "state_b")


# ---------------------------------------------------------------------------
# legal_next_states helper
# ---------------------------------------------------------------------------

class TestLegalNextStates:
    def test_characteristic_unverified(self):
        states = legal_next_states("Characteristic", "unverified")
        assert set(states) == {"surrogate_estimated", "simulation_validated", "unresolved"}

    def test_simulation_case_queued(self):
        states = legal_next_states("SimulationCase", "queued")
        assert set(states) == {"running", "failed"}

    def test_surrogate_model_retired_is_terminal(self):
        assert legal_next_states("SurrogateModel", "retired") == []

    def test_unknown_entity_returns_empty(self):
        assert legal_next_states("UnknownEntity", "foo") == []
