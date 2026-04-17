"""Unit tests for artifact service handlers and storage domain."""
from __future__ import annotations

import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from artifact_service.api.handlers import (
    ArtifactServiceHandler,
    DeleteArtifactRequest,
    GetArtifactRequest,
    UploadArtifactRequest,
)
from artifact_service.domain.storage import (
    ArtifactImmutableError,
    ArtifactNotFoundError,
    ArtifactStore,
    DigestMismatchError,
)


def _handler() -> ArtifactServiceHandler:
    return ArtifactServiceHandler()


SAMPLE_CONTENT = b"model weights data for surrogate S2"
SAMPLE_MEDIA_TYPE = "application/octet-stream"


# ---------------------------------------------------------------------------
# UploadArtifact
# ---------------------------------------------------------------------------

class TestUploadArtifact:
    def test_upload_returns_digest(self):
        h = _handler()
        resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE))
        assert resp.ok
        assert resp.artifact_ref is not None
        assert resp.artifact_ref.digest.startswith("sha256:")

    def test_upload_digest_is_correct_sha256(self):
        h = _handler()
        resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE))
        expected = "sha256:" + hashlib.sha256(SAMPLE_CONTENT).hexdigest()
        assert resp.artifact_ref.digest == expected

    def test_upload_same_content_same_digest(self):
        h = _handler()
        resp1 = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE))
        resp2 = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE))
        assert resp1.artifact_ref.digest == resp2.artifact_ref.digest

    def test_upload_different_content_different_digest(self):
        h = _handler()
        resp1 = h.upload_artifact(UploadArtifactRequest(b"content_a", SAMPLE_MEDIA_TYPE))
        resp2 = h.upload_artifact(UploadArtifactRequest(b"content_b", SAMPLE_MEDIA_TYPE))
        assert resp1.artifact_ref.digest != resp2.artifact_ref.digest

    def test_upload_empty_content_returns_error(self):
        h = _handler()
        resp = h.upload_artifact(UploadArtifactRequest(b"", SAMPLE_MEDIA_TYPE))
        assert not resp.ok
        assert resp.error_code == "INVALID_REQUEST"

    def test_upload_invalid_retention_class_returns_error(self):
        h = _handler()
        resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, "permanent"))
        assert not resp.ok
        assert resp.error_code == "INVALID_RETENTION_CLASS"

    def test_upload_all_valid_retention_classes(self):
        for cls in ["draft", "retained", "locked", "released"]:
            h = _handler()
            resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, cls))
            assert resp.ok


# ---------------------------------------------------------------------------
# GetArtifact
# ---------------------------------------------------------------------------

class TestGetArtifact:
    def test_get_existing_artifact(self):
        h = _handler()
        upload_resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE))
        get_resp = h.get_artifact(GetArtifactRequest(upload_resp.artifact_ref.digest))
        assert get_resp.ok
        assert get_resp.content == SAMPLE_CONTENT

    def test_get_nonexistent_artifact(self):
        h = _handler()
        resp = h.get_artifact(GetArtifactRequest("sha256:" + "a" * 64))
        assert not resp.ok
        assert resp.error_code == "NOT_FOUND"

    def test_get_returns_correct_ref(self):
        h = _handler()
        upload_resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, "retained"))
        get_resp = h.get_artifact(GetArtifactRequest(upload_resp.artifact_ref.digest))
        assert get_resp.artifact_ref.retention_class == "retained"
        assert get_resp.artifact_ref.size_bytes == len(SAMPLE_CONTENT)


# ---------------------------------------------------------------------------
# DeleteArtifact
# ---------------------------------------------------------------------------

class TestDeleteArtifact:
    def test_delete_draft_artifact(self):
        h = _handler()
        upload_resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, "draft"))
        del_resp = h.delete_artifact(DeleteArtifactRequest(upload_resp.artifact_ref.digest))
        assert del_resp.ok
        # Confirm it's gone
        get_resp = h.get_artifact(GetArtifactRequest(upload_resp.artifact_ref.digest))
        assert get_resp.error_code == "NOT_FOUND"

    def test_delete_retained_artifact(self):
        h = _handler()
        upload_resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, "retained"))
        del_resp = h.delete_artifact(DeleteArtifactRequest(upload_resp.artifact_ref.digest))
        assert del_resp.ok

    def test_delete_locked_artifact_returns_immutable_error(self):
        h = _handler()
        upload_resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, "locked"))
        del_resp = h.delete_artifact(DeleteArtifactRequest(upload_resp.artifact_ref.digest))
        assert not del_resp.ok
        assert del_resp.error_code == "ENTITY_NOT_MODIFIABLE"

    def test_delete_released_artifact_returns_immutable_error(self):
        h = _handler()
        upload_resp = h.upload_artifact(UploadArtifactRequest(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, "released"))
        del_resp = h.delete_artifact(DeleteArtifactRequest(upload_resp.artifact_ref.digest))
        assert not del_resp.ok
        assert del_resp.error_code == "ENTITY_NOT_MODIFIABLE"

    def test_delete_nonexistent_artifact(self):
        h = _handler()
        del_resp = h.delete_artifact(DeleteArtifactRequest("sha256:" + "b" * 64))
        assert not del_resp.ok
        assert del_resp.error_code == "NOT_FOUND"


# ---------------------------------------------------------------------------
# ArtifactStore domain — digest verification
# ---------------------------------------------------------------------------

class TestArtifactStoreDomain:
    def test_digest_verified_on_get(self):
        store = ArtifactStore()
        ref = store.store(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE)
        record = store.get(ref.digest)
        assert record.content == SAMPLE_CONTENT

    def test_exists(self):
        store = ArtifactStore()
        ref = store.store(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE)
        assert store.exists(ref.digest)
        assert not store.exists("sha256:" + "0" * 64)

    def test_get_nonexistent_raises(self):
        store = ArtifactStore()
        with pytest.raises(ArtifactNotFoundError):
            store.get("sha256:" + "0" * 64)

    def test_delete_immutable_raises(self):
        store = ArtifactStore()
        ref = store.store(SAMPLE_CONTENT, SAMPLE_MEDIA_TYPE, "locked")
        with pytest.raises(ArtifactImmutableError):
            store.delete(ref.digest)
