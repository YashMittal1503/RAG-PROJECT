"""
Short "what is in this document" profiles for the document router.

The router (query.analyze_document_scope) only sees filenames. "Where is Lily Bloom's hometown?" names no file,
so with two novels in the library it asks which document the user means, although only one of them mentions
Lily. A profile lists each text document's distinctive names and terms (characters, companies, products,
places): capitalized words that are frequent in that document and rare in the others.

Built from a sample of the chunks already in Qdrant: no extra LLM call, no database column. Cached per
(document, chunk count), so a document is profiled once per process and again only after it is re-indexed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

SAMPLE_CHUNKS = 250        # chunks read per document (Qdrant returns them in id order, i.e. an arbitrary sample)
PROFILE_TERMS = 12         # terms shown per document
MIN_COUNT = 3              # a term must appear this often in the sample
MIN_CAPITALIZED = 0.8      # share of its occurrences that are capitalized (a name, not a common word)

_cache: dict[tuple[str, int | None], list[str]] = {}

# Capitalized words that are not sentence-initial. A word right after . ! ? : " or a line break is skipped,
# since any word is capitalized there.
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")   # digits allowed after the first letter: "Seed2Shelf", "GPT4"
_STOP = {
    "the", "and", "but", "for", "she", "her", "his", "him", "they", "them", "this", "that", "what", "when",
    "where", "why", "how", "who", "you", "your", "our", "its", "it's", "i'm", "i'll", "i've", "i'd", "chapter",
    "page", "table", "figure", "section", "mr", "mrs", "ms", "dr", "mom", "dad", "oh", "okay", "yes", "no",
    "god", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
    "december", "english", "american", "note", "notes", "part", "item", "inc", "llc", "ltd",
}


def _count_terms(texts: list[str]) -> tuple[Counter, Counter]:
    """(capitalized occurrences, all occurrences) per lowercased word.

    Names are counted wherever they appear, sentence starts included (a first-person novel opens many
    sentences and lines of dialogue with "Lily" or "Atlas"). Ordinary words are filtered out because they
    also appear in lowercase: "Then" starts sentences, but "then" is far more common.
    """
    caps, total = Counter(), Counter()
    for text in texts:
        for m in _WORD.finditer(text):
            word = m.group(0).strip("'-")
            if word.endswith("'s"):
                word = word[:-2]
            low = word.lower()
            if len(low) < 3 or low in _STOP:
                continue
            total[low] += 1
            if word[0].isupper() and not word.isupper():
                caps[low] += 1
    return caps, total


def distinctive_terms(samples: dict[str, list[str]], top: int = PROFILE_TERMS) -> dict[str, list[str]]:
    """{doc key: [terms...]} for every document in `samples` ({doc key: chunk texts}).

    Score = frequency in this document x log(1 + documents / documents that use the term), so a name only
    one book uses (Lily) beats a word every document capitalizes (Figure, Company).
    """
    counted = {k: _count_terms(v) for k, v in samples.items()}
    n_docs = len(counted)
    df = Counter()
    for caps, total in counted.values():
        df.update({t for t, c in total.items() if c >= 2})
    out: dict[str, list[str]] = {}
    for key, (caps, total) in counted.items():
        scored = []
        for term, c in caps.items():
            if c < MIN_COUNT or c / total[term] < MIN_CAPITALIZED:
                continue
            scored.append((c * math.log(1 + n_docs / max(df[term], 1)), term))
        scored.sort(reverse=True)
        out[key] = [t.capitalize() if t.islower() else t for _, t in scored[:top]]
    return out


async def _sample_chunks(user_id: str, doc_id: str) -> list[str]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.services.vector_store import _collection_name, get_client

    client = await get_client()
    name = _collection_name(user_id)
    if not await client.collection_exists(name):
        return []
    points, _ = await client.scroll(
        collection_name=name,
        scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id)))]),
        limit=SAMPLE_CHUNKS,
        with_payload=["content"],
        with_vectors=False,
    )
    return [p.payload.get("content") or "" for p in points]


async def attach_document_profiles(user_id: str, docs: list[dict]) -> list[dict]:
    """Add 'key_terms' (a list of distinctive names/terms) to each text document in `docs`, in place.

    Never raises: the router still works from filenames alone.
    """
    text_docs = [d for d in docs if not d.get("is_tabular")]
    if len(text_docs) < 2:   # one text document: nothing to tell apart
        return docs
    try:
        keys = {str(d["id"]): (str(d["id"]), d.get("chunk_count")) for d in text_docs}
        missing = [doc_id for doc_id, ck in keys.items() if ck not in _cache]
        if missing:
            # Scores are relative to the other documents, so all of them are (re)scored together.
            ids = list(keys)
            texts = await asyncio.gather(*(_sample_chunks(user_id, i) for i in ids))
            terms = distinctive_terms(dict(zip(ids, texts)))
            for doc_id in ids:
                _cache[keys[doc_id]] = terms.get(doc_id, [])
        for d in text_docs:
            terms = _cache.get(keys[str(d["id"])]) or []
            if terms:
                d["key_terms"] = terms
    except Exception as e:
        logger.warning(f"Could not build document profiles for routing: {e}")
    return docs


def clear_cache() -> None:
    _cache.clear()
