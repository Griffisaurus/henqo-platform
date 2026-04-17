"""
Tests for decision_pkg_service.domain.signatory (EP-09).

≥6 tests covering:
- Empty signatures → not complete
- One of two roles signed → not complete
- All roles signed → complete
- missing_roles returns correct list
"""
from __future__ import annotations

import pytest

from decision_pkg_service.domain.signatory import SignatoryRecord, SignatoryWorkflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(role: str, signatory_id: str = "user-001") -> SignatoryRecord:
    return SignatoryRecord(
        signatory_id=signatory_id,
        role=role,
        signed_at="2026-04-17T12:00:00Z",
        signature=f"sig-{role}",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignatoryWorkflowCompleteness:
    def test_empty_signatures_not_complete(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer", "quality_lead"])
        assert wf.is_complete() is False

    def test_one_of_two_roles_signed_not_complete(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer", "quality_lead"])
        wf.add_signature(_sig("chief_engineer"))
        assert wf.is_complete() is False

    def test_all_roles_signed_is_complete(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer", "quality_lead"])
        wf.add_signature(_sig("chief_engineer"))
        wf.add_signature(_sig("quality_lead"))
        assert wf.is_complete() is True

    def test_no_required_roles_is_immediately_complete(self):
        wf = SignatoryWorkflow(required_roles=[])
        assert wf.is_complete() is True

    def test_single_required_role_complete_after_one_sig(self):
        wf = SignatoryWorkflow(required_roles=["safety_lead"])
        wf.add_signature(_sig("safety_lead"))
        assert wf.is_complete() is True


class TestMissingRoles:
    def test_missing_roles_returns_unsigned_roles(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer", "quality_lead", "safety_lead"])
        wf.add_signature(_sig("chief_engineer"))
        missing = wf.missing_roles()
        assert "quality_lead" in missing
        assert "safety_lead" in missing
        assert "chief_engineer" not in missing

    def test_missing_roles_empty_when_all_signed(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer"])
        wf.add_signature(_sig("chief_engineer"))
        assert wf.missing_roles() == []

    def test_missing_roles_all_when_no_signatures(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer", "quality_lead"])
        assert set(wf.missing_roles()) == {"chief_engineer", "quality_lead"}


class TestSignedRoles:
    def test_signed_roles_empty_initially(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer"])
        assert wf.signed_roles() == []

    def test_signed_roles_contains_added_role(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer", "quality_lead"])
        wf.add_signature(_sig("quality_lead"))
        assert "quality_lead" in wf.signed_roles()
        assert "chief_engineer" not in wf.signed_roles()

    def test_duplicate_signatures_for_same_role_deduped(self):
        wf = SignatoryWorkflow(required_roles=["chief_engineer"])
        wf.add_signature(_sig("chief_engineer", "user-001"))
        wf.add_signature(_sig("chief_engineer", "user-002"))
        # Still counts as complete — role is present
        assert wf.is_complete() is True
        # signed_roles dedupes
        assert wf.signed_roles().count("chief_engineer") == 1
