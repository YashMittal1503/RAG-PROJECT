"""
Citation precision/recall against gold evidence pages.

precision: of the (doc, page) pairs the system cited, how many hold gold evidence.
recall:    of the gold evidence (doc, page) pairs, how many the system cited.

Only items whose evidence has page numbers are scored; others return None.
"""

from __future__ import annotations

from eval.schema import Citation, Evidence


def _key(doc: str | None, page: int | None) -> tuple[str | None, int | None]:
    return (doc, page)


def score_citations(
    citations: list[Citation],
    evidence: list[Evidence],
    page_tolerance: int = 0,
) -> dict:
    gold = {_key(e.doc, e.page) for e in evidence if e.page is not None}
    if not gold:
        return {"citation_precision": None, "citation_recall": None}

    cited = {_key(c.doc, c.page) for c in citations if c.page is not None}
    if not cited:
        return {"citation_precision": 0.0, "citation_recall": 0.0}

    def matches(a, b) -> bool:
        # A citation with no doc id (single-doc answers) matches any doc.
        doc_ok = a[0] is None or b[0] is None or a[0] == b[0]
        return doc_ok and abs(a[1] - b[1]) <= page_tolerance

    correct_cites = sum(1 for c in cited if any(matches(c, g) for g in gold))
    covered_gold = sum(1 for g in gold if any(matches(c, g) for c in cited))
    return {
        "citation_precision": correct_cites / len(cited),
        "citation_recall": covered_gold / len(gold),
    }
