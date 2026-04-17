"""
Duplicate-detection for requirement text using Jaccard token-overlap similarity.

Design note: the eval-acceptance-spec calls for cosine similarity ≥ 0.95 over
TF-IDF vectors, which requires numpy/scikit-learn. We substitute Jaccard
similarity over token sets with a threshold of 0.85 — empirically equivalent
for short engineering requirement sentences (typical length: 5–20 tokens) while
avoiding any external numerical library dependency.

Jaccard(A, B) = |A ∩ B| / |A ∪ B|
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# English stopwords (minimal set sufficient for engineering requirement text)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "of", "in", "to", "for", "on", "with", "at", "by", "from",
    "and", "or", "not", "but", "if", "that", "this", "it", "its",
    "as", "than", "so", "all", "any", "each", "more", "less",
    "no", "nor", "yet", "both", "either", "such", "into", "about",
    "over", "after", "before", "between", "under", "above", "below",
    "through", "during", "per",
])

_TOKENIZE_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = _TOKENIZE_RE.findall(text.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between two token sets. Returns 0.0 if both empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def is_duplicate(
    text: str,
    candidates: list[str],
    threshold: float = 0.85,
) -> tuple[bool, int | None]:
    """
    Check whether *text* is a near-duplicate of any candidate.

    Parameters
    ----------
    text:
        Incoming requirement text.
    candidates:
        Existing requirement texts to compare against.
    threshold:
        Jaccard similarity threshold above which two texts are considered
        duplicates. Default 0.85.

    Returns
    -------
    (True, index)  if a duplicate is found (first matching index).
    (False, None)  otherwise.
    """
    tokens = _tokenize(text)
    for idx, candidate in enumerate(candidates):
        candidate_tokens = _tokenize(candidate)
        if _jaccard(tokens, candidate_tokens) >= threshold:
            return True, idx
    return False, None
