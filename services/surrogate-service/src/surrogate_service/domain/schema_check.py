"""
Schema version compatibility check.

Implements fallback rule 7 from surrogate-trust-policy.md §5:
  Major version mismatch → incompatible → model_frozen trigger.
"""
from __future__ import annotations


def _parse_major(version: str) -> int:
    """Extract the major version integer from a semver string (e.g. '1.2.3' → 1)."""
    try:
        major_str = version.strip().lstrip("v").split(".")[0]
        return int(major_str)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse semver major from {version!r}") from exc


def check_schema_version(
    model_training_schema_version: str,
    current_schema_version: str,
) -> tuple[bool, str]:
    """
    Compare major versions of semver strings.

    Returns (compatible: bool, reason: str).

    A major version mismatch is incompatible and triggers the model_frozen
    fallback rule (surrogate-trust-policy.md §5 condition 7).

    Same major, any minor/patch → compatible.
    """
    try:
        train_major = _parse_major(model_training_schema_version)
        current_major = _parse_major(current_schema_version)
    except ValueError as exc:
        return False, str(exc)

    if train_major != current_major:
        return (
            False,
            (
                f"Schema major version mismatch: model trained on v{train_major} "
                f"but current IR schema is v{current_major}. "
                "Surrogate must be re-benchmarked against the new schema before use."
            ),
        )

    return True, f"Schema versions compatible (major={train_major})."
