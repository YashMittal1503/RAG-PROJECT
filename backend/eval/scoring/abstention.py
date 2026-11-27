"""Detects when a system declined to answer, and scores refusals against answerability."""

from __future__ import annotations

import re

# Phrases our own pipeline and common LLMs use when declining. Kept explicit so
# a change to the app's refusal wording shows up as a failing test, not silent drift.
_REFUSAL_PATTERNS = [
    r"\bnot (?:available|mentioned|stated|provided|specified|found|covered) in (?:the )?(?:provided |uploaded )?(?:document|context|excerpt|text|source)",
    r"\b(?:does|do) not (?:contain|mention|say|specify|provide|state)\b",
    r"\bi (?:don't|do not|cannot|can't|couldn't|could not) (?:find|answer|determine|locate|see)\b",
    r"\bi don't have enough information\b",
    r"\bno (?:relevant )?information (?:about|on|regarding)\b",
    r"\bcouldn't find any relevant sections\b",
    r"\bthere is no (?:mention|indication|information|evidence)\b",
    r"\bnot enough (?:information|context)\b",
    r"\bunable to (?:find|answer|determine)\b",
    r"\bthe (?:document|context|text) (?:is silent|doesn't say)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    """True if the answer declines to answer.

    Only the opening of the answer is checked: a long, substantive answer that ends
    with "the document does not mention X" is a partial answer, not a refusal.
    """
    if not answer or not answer.strip():
        return True
    head = answer.strip()[:300]
    return bool(_REFUSAL_RE.search(head))


def score_abstention(answer: str, answerable: bool) -> dict:
    refused = is_refusal(answer)
    if answerable:
        return {"refused": refused, "false_refusal": float(refused), "correct_refusal": None}
    return {"refused": refused, "false_refusal": None, "correct_refusal": float(refused)}
