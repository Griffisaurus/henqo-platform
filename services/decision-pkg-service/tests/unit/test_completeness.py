"""
Tests for decision_pkg_service.domain.completeness (EP-08).

≥6 tests covering:
- All requirements covered → complete
- Missing evidence → incomplete with uncovered listed
- Critical characteristic needs definitively_supportive at CDR and above
"""
from __future__ import annotations

import pytest

from decision_pkg_service.domain.completeness import (
    CompletenessResult,
    EvidenceItem,
    check_completeness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(req_id: str) -> dict:
    return {"entity_id": req_id, "text": f"Req {req_id}", "criticality": "Shall"}


def _char(char_id: str, req_id: str, criticality: str = "Standard") -> dict:
    return {
        "entity_id": char_id,
        "name": f"char_{char_id}",
        "governing_requirement_id": req_id,
        "criticality": criticality,
    }


def _ev(char_id: str, status: str, dc: str = "Exploratory") -> EvidenceItem:
    return EvidenceItem(
        characteristic_id=char_id,
        evidence_type="SimulationCase",
        evidence_entity_id=f"ev-{char_id}",
        status=status,
        decision_class=dc,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompletenessAllCovered:
    """All requirements have supportive evidence → complete=True."""

    def test_single_requirement_provisionally_supportive(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001")]
        evs = [_ev("CH-001", "provisionally_supportive")]
        result = check_completeness(reqs, chars, evs, "PDR")
        assert result.complete is True
        assert "REQ-001" in result.covered_requirements
        assert result.uncovered_requirements == []

    def test_multiple_requirements_all_covered(self):
        reqs = [_req("REQ-001"), _req("REQ-002")]
        chars = [_char("CH-001", "REQ-001"), _char("CH-002", "REQ-002")]
        evs = [
            _ev("CH-001", "definitively_supportive"),
            _ev("CH-002", "provisionally_supportive"),
        ]
        result = check_completeness(reqs, chars, evs, "PDR")
        assert result.complete is True
        assert set(result.covered_requirements) == {"REQ-001", "REQ-002"}
        assert result.uncovered_requirements == []

    def test_definitively_supportive_counts_as_covered(self):
        reqs = [_req("REQ-A")]
        chars = [_char("CH-A", "REQ-A")]
        evs = [_ev("CH-A", "definitively_supportive")]
        result = check_completeness(reqs, chars, evs, "CDR")
        assert result.complete is True


class TestCompletnessMissingEvidence:
    """Missing evidence → incomplete with uncovered requirements listed."""

    def test_no_evidence_for_requirement(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001")]
        evs: list[EvidenceItem] = []
        result = check_completeness(reqs, chars, evs, "PDR")
        assert result.complete is False
        assert "REQ-001" in result.uncovered_requirements
        assert "REQ-001" not in result.covered_requirements

    def test_insufficient_status_does_not_cover(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001")]
        evs = [_ev("CH-001", "insufficient")]
        result = check_completeness(reqs, chars, evs, "PDR")
        assert result.complete is False
        assert "REQ-001" in result.uncovered_requirements

    def test_conflicting_status_does_not_cover(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001")]
        evs = [_ev("CH-001", "conflicting")]
        result = check_completeness(reqs, chars, evs, "PDR")
        assert result.complete is False

    def test_partial_coverage_lists_only_uncovered(self):
        reqs = [_req("REQ-001"), _req("REQ-002")]
        chars = [_char("CH-001", "REQ-001"), _char("CH-002", "REQ-002")]
        evs = [_ev("CH-001", "provisionally_supportive")]  # REQ-002 not covered
        result = check_completeness(reqs, chars, evs, "PDR")
        assert result.complete is False
        assert "REQ-001" in result.covered_requirements
        assert "REQ-002" in result.uncovered_requirements
        assert len(result.detail) >= 1


class TestCriticalCharCDRRule:
    """Critical characteristics need definitively_supportive at CDR+."""

    def test_critical_char_provisionally_at_cdr_fails(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001", criticality="Critical")]
        evs = [_ev("CH-001", "provisionally_supportive")]
        result = check_completeness(reqs, chars, evs, "CDR")
        # Requirement is covered, but critical char rule fails
        assert result.complete is False
        assert "CH-001" in result.insufficient_characteristics

    def test_critical_char_definitively_at_cdr_passes(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001", criticality="Critical")]
        evs = [_ev("CH-001", "definitively_supportive")]
        result = check_completeness(reqs, chars, evs, "CDR")
        assert result.complete is True
        assert result.insufficient_characteristics == []

    def test_critical_char_provisionally_at_pdr_passes(self):
        """PDR is below CDR — provisionally_supportive is OK for critical chars at PDR."""
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001", criticality="Critical")]
        evs = [_ev("CH-001", "provisionally_supportive")]
        result = check_completeness(reqs, chars, evs, "PDR")
        # At PDR the CDR rule does not apply
        assert result.complete is True
        assert result.insufficient_characteristics == []

    def test_non_critical_char_provisionally_at_cdr_passes(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001", criticality="Standard")]
        evs = [_ev("CH-001", "provisionally_supportive")]
        result = check_completeness(reqs, chars, evs, "CDR")
        assert result.complete is True

    def test_critical_char_at_frr_needs_definitive(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001", criticality="Critical")]
        evs = [_ev("CH-001", "provisionally_supportive")]
        result = check_completeness(reqs, chars, evs, "FRR")
        assert result.complete is False
        assert "CH-001" in result.insufficient_characteristics

    def test_detail_messages_populated_on_failure(self):
        reqs = [_req("REQ-001")]
        chars = [_char("CH-001", "REQ-001")]
        evs: list[EvidenceItem] = []
        result = check_completeness(reqs, chars, evs, "PDR")
        assert len(result.detail) > 0
