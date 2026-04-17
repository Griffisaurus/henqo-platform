"""
Unit tests for surrogate_service.domain.schema_check.

Covers:
  - Same major version → compatible
  - Different major version → incompatible
  - Minor/patch difference only → compatible
  - Invalid version strings
"""
from __future__ import annotations

import pytest

from surrogate_service.domain.schema_check import check_schema_version


class TestCheckSchemaVersion:
    def test_same_version_compatible(self):
        ok, reason = check_schema_version("0.1.0", "0.1.0")
        assert ok
        assert "compatible" in reason.lower()

    def test_same_major_different_minor_compatible(self):
        ok, reason = check_schema_version("0.1.0", "0.2.0")
        assert ok

    def test_same_major_different_patch_compatible(self):
        ok, reason = check_schema_version("1.2.3", "1.2.9")
        assert ok

    def test_different_major_incompatible(self):
        """Major version mismatch → model_frozen trigger (fallback rule 7)."""
        ok, reason = check_schema_version("0.1.0", "1.0.0")
        assert not ok
        assert "mismatch" in reason.lower() or "major" in reason.lower()

    def test_different_major_reverse_incompatible(self):
        ok, reason = check_schema_version("2.0.0", "1.0.0")
        assert not ok

    def test_major_v_prefix_compatible(self):
        """Version strings with leading 'v' should parse correctly."""
        ok, _ = check_schema_version("v1.2.3", "v1.5.0")
        assert ok

    def test_major_v_prefix_incompatible(self):
        ok, _ = check_schema_version("v0.1.0", "v1.0.0")
        assert not ok

    def test_version_1_to_2_incompatible(self):
        ok, reason = check_schema_version("1.0.0", "2.0.0")
        assert not ok
        assert "1" in reason and "2" in reason

    def test_returns_tuple_of_bool_and_str(self):
        result = check_schema_version("0.1.0", "0.2.0")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_unparseable_training_version_returns_false(self):
        """Non-semver training version string → incompatible with error reason."""
        ok, reason = check_schema_version("not_a_version", "0.1.0")
        assert not ok
        assert "parse" in reason.lower() or "Cannot" in reason

    def test_unparseable_current_version_returns_false(self):
        """Non-semver current version string → incompatible with error reason."""
        ok, reason = check_schema_version("0.1.0", "not_a_version")
        assert not ok
        assert len(reason) > 0
