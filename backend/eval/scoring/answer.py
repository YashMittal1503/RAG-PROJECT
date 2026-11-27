"""SQuAD-style lexical answer metrics: normalized exact match and token F1."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation + "“”‘’—–")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, articles and extra whitespace (SQuAD normalization)."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold answer."""
    p = normalize(prediction)
    return float(any(p == normalize(g) for g in golds if g))


def contains_answer(prediction: str, golds: list[str]) -> float:
    """1.0 if any normalized gold answer appears as a whole phrase in the prediction.

    Long-form RAG answers rarely match exactly, so this is the more useful lexical
    signal for chat-style output.
    """
    p = f" {normalize(prediction)} "
    return float(any(f" {normalize(g)} " in p for g in golds if normalize(g)))


def token_f1(prediction: str, golds: list[str]) -> float:
    """Max token-overlap F1 between the prediction and any gold answer."""
    best = 0.0
    p_tokens = normalize(prediction).split()
    for g in golds:
        g_tokens = normalize(g).split()
        if not p_tokens or not g_tokens:
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(p_tokens)
        recall = overlap / len(g_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
