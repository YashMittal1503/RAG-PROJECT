"""
Retrieval metrics computed on text, not chunk ids.

A gold evidence quote counts as retrieved by a context when the quote fuzzily
appears inside it. Because this compares text, it works the same for chunks,
graph passages, tree nodes or whole chapters, so different architectures can be
compared fairly.
"""

from __future__ import annotations

import math
import re

from rapidfuzz import fuzz

DEFAULT_THRESHOLD = 90.0
DEFAULT_KS = (1, 3, 5, 10)


def _norm(text: str) -> str:
    # Collapse whitespace and markdown emphasis so PDF/markdown formatting doesn't break matches.
    text = re.sub(r"[*_`#>|]", " ", text)
    return " ".join(text.lower().split())


def quote_in_context(quote: str, context: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True if `quote` appears (approximately) inside `context`."""
    q, c = _norm(quote), _norm(context)
    if not q or not c:
        return False
    if q in c:
        return True
    # partial_ratio aligns the shorter string against the best window of the longer one.
    # A quote longer than the context can only be partially covered, so require containment direction.
    if len(q) > len(c) * 1.2:
        return False
    return fuzz.partial_ratio(q, c) >= threshold


def evidence_hits(
    quotes: list[str],
    contexts: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[set[int]]:
    """For each context (in rank order), the set of evidence indices it contains."""
    return [
        {i for i, q in enumerate(quotes) if quote_in_context(q, ctx, threshold)}
        for ctx in contexts
    ]


def recall_at_k(hits: list[set[int]], n_evidence: int, k: int) -> float:
    """Fraction of evidence spans found in the top-k contexts."""
    if n_evidence == 0:
        return 0.0
    found: set[int] = set()
    for h in hits[:k]:
        found |= h
    return len(found) / n_evidence


def mrr(hits: list[set[int]]) -> float:
    """Reciprocal rank of the first context containing any evidence."""
    for rank, h in enumerate(hits, start=1):
        if h:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: list[set[int]], n_evidence: int, k: int = 10) -> float:
    """nDCG with binary gain: a context is relevant if it contains evidence not seen at a higher rank."""
    if n_evidence == 0:
        return 0.0
    seen: set[int] = set()
    dcg = 0.0
    for rank, h in enumerate(hits[:k], start=1):
        new = h - seen
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_evidence, k) + 1))
    return dcg / ideal if ideal else 0.0


def score_retrieval(
    quotes: list[str],
    contexts: list[str],
    ks: tuple[int, ...] = DEFAULT_KS,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    hits = evidence_hits(quotes, contexts, threshold)
    n = len(quotes)
    return {
        "recall_at_k": {k: recall_at_k(hits, n, k) for k in ks},
        "mrr": mrr(hits),
        "ndcg_at_10": ndcg_at_k(hits, n, 10),
    }
