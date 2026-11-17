"""Tests the vanilla adapter's orchestration with the app's query functions stubbed out (offline)."""

import sys
import types
from pathlib import Path

import pytest

import app.services as app_services
from eval.corpus import load_manifest
from eval.schema import GoldItem, Turn
from eval.systems.vanilla import NO_CHUNKS_MESSAGE, VanillaRAG

DATA = Path(__file__).resolve().parents[1] / "eval" / "data"


class _Scope:
    def __init__(self, status, doc=None, text=None):
        self.status = status
        self.target_doc_id = doc.uuid if doc else None
        self.target_filename = doc.filename if doc else None
        self.clarification_text = text
        self.cleaned_question = ""
        self.is_tabular = False


def _install_fake_query(monkeypatch, *, intent="retrieve", scope=None, chunks=None, answer="Elias [Page 1]"):
    calls = {}
    q = types.ModuleType("app.services.query")

    async def classify_intent(question, history):
        calls["history"] = history
        return intent

    async def analyze_document_scope(question, history, library):
        calls["library"] = library
        return scope

    async def rewrite_query(question, history):
        return question + " (rewritten)"

    async def retrieve_chunks(user_id, question, is_aggregation, doc_id_filter=None, filename_filter=None):
        calls["retrieve"] = dict(user_id=user_id, question=question, doc_id_filter=doc_id_filter)
        return chunks or []

    async def _gen(*a, **k):
        for tok in answer.split(" "):
            yield tok + " "

    q.classify_intent = classify_intent
    q.analyze_document_scope = analyze_document_scope
    q.rewrite_query = rewrite_query
    q.retrieve_chunks = retrieve_chunks
    q.detect_aggregation = lambda s: False
    q.generate_answer_stream = lambda question, chunks: _gen()
    q.generate_direct_response = lambda question, history: _gen()
    q.SYSTEM_PROMPT = "system"
    q._build_context = lambda chunks, query="": "ctx"
    q.validate_citations = lambda ans, chs: [{"filename": c["filename"], "page_number": c["page_number"]} for c in chs[:1]]

    chunking = types.ModuleType("app.services.chunking")
    chunking.count_tokens = lambda s: len(s.split())

    monkeypatch.setitem(sys.modules, "app.services.query", q)
    monkeypatch.setitem(sys.modules, "app.services.chunking", chunking)
    monkeypatch.setattr(app_services, "query", q, raising=False)
    monkeypatch.setattr(app_services, "chunking", chunking, raising=False)
    return calls


@pytest.fixture
def docs():
    return load_manifest(DATA / "smoke_corpus.yaml")


@pytest.mark.anyio
async def test_rag_path_maps_docs_contexts_and_citations(monkeypatch, docs):
    lighthouse = next(d for d in docs if d.id == "lighthouse")
    chunks = [{"content": "“Morning, Elias!”", "filename": lighthouse.filename, "page_number": 1, "score": 0.9}]
    calls = _install_fake_query(monkeypatch, scope=_Scope("resolved", lighthouse), chunks=chunks)

    item = GoldItem(id="t", question="Narrator's name?", answer="Elias",
                    history=[Turn(role="user", content="hi")],
                    evidence=[{"doc": "lighthouse", "page": 1, "quote": "Morning, Elias!"}])
    pred = await VanillaRAG(docs, user_id="eval-tenant").answer(item)

    assert pred.error is None
    assert pred.route == "rag" and pred.chosen_doc == "lighthouse"
    assert pred.answer.strip() == "Elias [Page 1]"
    assert pred.contexts[0].doc == "lighthouse" and pred.contexts[0].page == 1
    assert pred.citations[0].doc == "lighthouse"
    assert calls["retrieve"]["user_id"] == "eval-tenant"
    assert calls["retrieve"]["doc_id_filter"] == lighthouse.uuid
    assert calls["history"] == [{"role": "user", "content": "hi", "citations": None}]
    assert {d["filename"] for d in calls["library"]} == {d.filename for d in docs}
    assert {"intent", "scope", "rewrite", "retrieve", "generate"} <= set(pred.timings_ms)
    assert pred.completion_tokens > 0


@pytest.mark.anyio
async def test_chitchat_clarify_and_empty_retrieval(monkeypatch, docs):
    _install_fake_query(monkeypatch, intent="chitchat", answer="Hello there")
    pred = await VanillaRAG(docs).answer(GoldItem(id="c", question="hi", expected_route="chitchat"))
    assert pred.route == "chitchat" and "Hello" in pred.answer

    _install_fake_query(monkeypatch, scope=_Scope("needs_clarification", text="Which document?"))
    pred = await VanillaRAG(docs).answer(GoldItem(id="a", question="summarize", answerable=False, category="unanswerable"))
    assert pred.route == "clarify" and pred.answer == "Which document?"

    _install_fake_query(monkeypatch, scope=_Scope("all_docs"), chunks=[])
    pred = await VanillaRAG(docs).answer(GoldItem(id="e", question="q", answerable=False, category="unanswerable"))
    assert pred.route == "rag" and pred.answer == NO_CHUNKS_MESSAGE and pred.chosen_doc is None


@pytest.mark.anyio
async def test_errors_are_captured_not_raised(monkeypatch, docs):
    calls = _install_fake_query(monkeypatch, scope=_Scope("all_docs"))

    async def boom(*a, **k):
        raise RuntimeError("qdrant down")

    sys.modules["app.services.query"].retrieve_chunks = boom
    pred = await VanillaRAG(docs).answer(GoldItem(id="x", question="q", answerable=False, category="unanswerable"))
    assert pred.error == "RuntimeError: qdrant down"
    assert "retrieve" not in pred.timings_ms and "intent" in pred.timings_ms
