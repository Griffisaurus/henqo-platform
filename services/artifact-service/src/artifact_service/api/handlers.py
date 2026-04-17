"""
Artifact service request handlers.

Implements UploadArtifact and GetArtifact from service-contracts-api-spec.md §3.
"""
from __future__ import annotations

from dataclasses import dataclass

from artifact_service.domain.storage import (
    ArtifactImmutableError,
    ArtifactNotFoundError,
    ArtifactRef,
    ArtifactStore,
)


@dataclass
class UploadArtifactRequest:
    content: bytes
    media_type: str
    retention_class: str = "draft"


@dataclass
class UploadArtifactResponse:
    artifact_ref: ArtifactRef | None = None
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class GetArtifactRequest:
    digest: str


@dataclass
class GetArtifactResponse:
    content: bytes | None = None
    artifact_ref: ArtifactRef | None = None
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


@dataclass
class DeleteArtifactRequest:
    digest: str


@dataclass
class DeleteArtifactResponse:
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error_code


_VALID_RETENTION_CLASSES = frozenset(["draft", "retained", "locked", "released"])


class ArtifactServiceHandler:
    def __init__(self, store: ArtifactStore | None = None) -> None:
        self._store = store or ArtifactStore()

    def upload_artifact(self, req: UploadArtifactRequest) -> UploadArtifactResponse:
        if not req.content:
            return UploadArtifactResponse(
                error_code="INVALID_REQUEST",
                error_message="content must be non-empty",
            )
        if req.retention_class not in _VALID_RETENTION_CLASSES:
            return UploadArtifactResponse(
                error_code="INVALID_RETENTION_CLASS",
                error_message=f"Unknown retention class: {req.retention_class!r}",
            )
        ref = self._store.store(req.content, req.media_type, req.retention_class)
        return UploadArtifactResponse(artifact_ref=ref)

    def get_artifact(self, req: GetArtifactRequest) -> GetArtifactResponse:
        try:
            record = self._store.get(req.digest)
        except ArtifactNotFoundError:
            return GetArtifactResponse(error_code="NOT_FOUND", error_message=f"Artifact {req.digest!r} not found")
        ref = ArtifactRef(
            digest=record.digest,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
            retention_class=record.retention_class,
        )
        return GetArtifactResponse(content=record.content, artifact_ref=ref)

    def delete_artifact(self, req: DeleteArtifactRequest) -> DeleteArtifactResponse:
        try:
            self._store.delete(req.digest)
        except ArtifactNotFoundError:
            return DeleteArtifactResponse(error_code="NOT_FOUND", error_message=f"Artifact {req.digest!r} not found")
        except ArtifactImmutableError as exc:
            return DeleteArtifactResponse(error_code="ENTITY_NOT_MODIFIABLE", error_message=str(exc))
        return DeleteArtifactResponse()
