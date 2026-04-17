"""
Content-addressed artifact storage.

Artifacts are stored by SHA-256 digest. Retrieval verifies the digest.
Retention classes: draft | retained | locked | released
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


_IMMUTABLE_CLASSES = frozenset(["locked", "released"])


class ArtifactNotFoundError(Exception):
    pass


class ArtifactImmutableError(Exception):
    """Raised on delete attempt for locked/released artifacts."""
    pass


class DigestMismatchError(Exception):
    pass


@dataclass
class ArtifactRecord:
    digest: str                   # sha256:<hex>
    media_type: str
    content: bytes
    retention_class: str
    size_bytes: int
    stored_at: str


@dataclass
class ArtifactRef:
    digest: str
    media_type: str
    size_bytes: int
    retention_class: str


class ArtifactStore:
    """In-memory content-addressed store. Production replaces with S3/MinIO."""

    def __init__(self) -> None:
        self._artifacts: dict[str, ArtifactRecord] = {}

    def store(self, content: bytes, media_type: str, retention_class: str = "draft") -> ArtifactRef:
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if digest not in self._artifacts:
            self._artifacts[digest] = ArtifactRecord(
                digest=digest,
                media_type=media_type,
                content=content,
                retention_class=retention_class,
                size_bytes=len(content),
                stored_at=datetime.now(timezone.utc).isoformat(),
            )
        return ArtifactRef(
            digest=digest,
            media_type=media_type,
            size_bytes=len(content),
            retention_class=retention_class,
        )

    def get(self, digest: str) -> ArtifactRecord:
        record = self._artifacts.get(digest)
        if record is None:
            raise ArtifactNotFoundError(f"Artifact not found: {digest!r}")

        # Verify digest on retrieval
        computed = "sha256:" + hashlib.sha256(record.content).hexdigest()
        if computed != record.digest:
            raise DigestMismatchError(
                f"Digest mismatch for {digest!r}: stored={record.digest!r}, computed={computed!r}"
            )
        return record

    def delete(self, digest: str) -> None:
        record = self._artifacts.get(digest)
        if record is None:
            raise ArtifactNotFoundError(f"Artifact not found: {digest!r}")
        if record.retention_class in _IMMUTABLE_CLASSES:
            raise ArtifactImmutableError(
                f"Cannot delete artifact with retention_class={record.retention_class!r}"
            )
        del self._artifacts[digest]

    def exists(self, digest: str) -> bool:
        return digest in self._artifacts
