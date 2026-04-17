"""
EP-09: Signatory workflow for release manifest sign-off.

IMPORTANT: This service NEVER directly sets manifest_status=active.
When all required signatures are collected, it signals readiness to the
graph service to perform the transition (requiring approver_id).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignatoryRecord:
    signatory_id: str
    role: str
    signed_at: str
    signature: str = ""


class SignatoryWorkflow:
    """
    Tracks required signatories for a release manifest.

    IMPORTANT: This service NEVER directly sets manifest_status=active.
    When all required signatures are collected, it signals readiness to the
    graph service to perform the transition (requiring approver_id).
    """

    def __init__(self, required_roles: list[str]) -> None:
        self._required_roles: list[str] = list(required_roles)
        self._signatures: list[SignatoryRecord] = []

    def add_signature(self, signatory: SignatoryRecord) -> None:
        """Record a signature. Duplicate signatures for the same role are allowed (last wins)."""
        self._signatures.append(signatory)

    def is_complete(self) -> bool:
        """True if all required_roles have at least one signature."""
        signed = self.signed_roles()
        return all(role in signed for role in self._required_roles)

    def missing_roles(self) -> list[str]:
        """Return required roles that have not yet been signed."""
        signed = set(self.signed_roles())
        return [role for role in self._required_roles if role not in signed]

    def signed_roles(self) -> list[str]:
        """Return the distinct set of roles that have signed (preserving first-seen order)."""
        seen: dict[str, bool] = {}
        for sig in self._signatures:
            seen[sig.role] = True
        return list(seen.keys())

    def all_signatures(self) -> list[SignatoryRecord]:
        """Return all recorded signatures."""
        return list(self._signatures)
