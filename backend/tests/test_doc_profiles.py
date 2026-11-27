"""Distinctive-term profiles that let the document router tell text documents apart (no network)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import doc_profiles
from app.services.doc_profiles import attach_document_profiles, distinctive_terms
from app.services.query import analyze_document_scope

NOVEL_A = ["Lily walked to the flower shop. Atlas was there.", "\"Lily,\" Ryle said. Then Lily left.",
           "Atlas smiled. Ryle frowned at Lily and Atlas.", "The lily in the vase was white, Ryle said, and then he left."] * 3
NOVEL_B = ["Lowen read the manuscript. Jeremy watched.", "\"Lowen,\" Jeremy said. Then Crew ran.",
           "Then the house was quiet. Jeremy and Crew slept.", "Verity was upstairs. Lowen knew it, and then the lights went out."] * 3
REPORT = ["Then Microsoft reported revenue. Azure grew.", "Azure and Xbox revenue rose. Then it fell.",
          "Microsoft said Azure demand was strong and then stable."] * 3


def test_names_win_over_common_words_and_sentence_starts():
    terms = distinctive_terms({"a": NOVEL_A, "b": NOVEL_B, "r": REPORT})
    assert {"Lily", "Atlas", "Ryle"} <= set(terms["a"])        # names count even when they open a sentence
    assert {"Lowen", "Jeremy", "Crew"} <= set(terms["b"])
    assert {"Microsoft", "Azure"} <= set(terms["r"])
    assert not any(t in ("Then", "The") for ts in terms.values() for t in ts)   # also written in lowercase
    assert "Lily" not in terms["b"]


@pytest.mark.anyio
async def test_attach_profiles_skips_spreadsheets_and_caches():
    doc_profiles.clear_cache()
    docs = [{"id": "a", "filename": "a.pdf", "is_tabular": False, "chunk_count": 5},
            {"id": "b", "filename": "b.pdf", "is_tabular": False, "chunk_count": 5},
            {"id": "s", "filename": "s.csv", "is_tabular": True}]
    sample = AsyncMock(side_effect=lambda uid, doc_id: {"a": NOVEL_A, "b": NOVEL_B}[doc_id])
    with patch.object(doc_profiles, "_sample_chunks", sample):
        await attach_document_profiles("u", docs)
        await attach_document_profiles("u", [dict(d) for d in docs])   # second call: served from the cache
    assert "Lily" in docs[0]["key_terms"] and "Lowen" in docs[1]["key_terms"]
    assert "key_terms" not in docs[2]
    assert sample.await_count == 2
    doc_profiles.clear_cache()


@pytest.mark.anyio
async def test_profile_failure_leaves_routing_working():
    doc_profiles.clear_cache()
    docs = [{"id": "a", "filename": "a.pdf", "is_tabular": False}, {"id": "b", "filename": "b.pdf", "is_tabular": False}]
    with patch.object(doc_profiles, "_sample_chunks", AsyncMock(side_effect=RuntimeError("qdrant down"))):
        assert await attach_document_profiles("u", docs) == docs
    assert all("key_terms" not in d for d in docs)


@pytest.mark.anyio
async def test_router_catalog_lists_key_terms():
    docs = [{"id": "d1", "filename": "It-Ends-with-Us.pdf", "is_tabular": False, "key_terms": ["Lily", "Ryle", "Atlas"]},
            {"id": "d2", "filename": "Verity.pdf", "is_tabular": False, "key_terms": ["Lowen", "Jeremy"]}]
    reply = MagicMock(choices=[MagicMock(message=MagicMock(content="CATEGORY: SPECIFIC\nDOCUMENT_INDEX: 1", reasoning=None))])
    llm = AsyncMock(return_value=reply)
    with patch("app.services.query.call_llm_with_fallback", llm):
        scope = await analyze_document_scope("Where is Lily Bloom's hometown?", None, docs)
    prompt = llm.call_args.kwargs["messages"][0]["content"]
    assert "It-Ends-with-Us.pdf (Text Document) — key names/terms: Lily, Ryle, Atlas" in prompt
    assert scope.status == "resolved" and scope.target_doc_id == "d1"
