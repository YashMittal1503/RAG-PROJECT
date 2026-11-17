"""
Adapter for the current production pipeline (app/services/query.py), run against
the isolated eval tenant. It calls the same functions the chat endpoint
(app/routers/chat.py) calls, in the same order, without the database or SSE layer:

  classify_intent → analyze_document_scope → rewrite_query
    → detect_aggregation → retrieve_chunks → generate_answer_stream → validate_citations

Known gaps vs. production:
- The lead-chunk lookup inside retrieve_chunks reads Postgres. Without a DB it
  logs a warning and continues, so overview questions get slightly less context
  than in production.
- Spreadsheet / Text-to-SQL routing is not exercised yet (the eval corpus is PDF/TXT only).
"""

from __future__ import annotations

import time
from typing import Any

from eval.corpus import CorpusDoc, eval_user_id, index_corpus
from eval.schema import Citation, GoldItem, Prediction, RetrievedContext
from eval.systems.base import RAGSystem

NO_CHUNKS_MESSAGE = (
    "I searched through your uploaded documents, but couldn't find any relevant sections "
    "answering your question."
)


class VanillaRAG(RAGSystem):
    name = "vanilla"

    def __init__(self, docs: list[CorpusDoc], **options: Any):
        super().__init__(docs, **options)
        self.user_id = options.get("user_id") or eval_user_id()
        self._by_uuid = {d.uuid: d for d in docs}
        self._by_filename = {d.filename.lower(): d for d in docs}

    def config(self) -> dict[str, Any]:
        from app.config import settings

        return {
            **super().config(),
            "user_id": self.user_id,
            "groq_model": settings.groq_model,
            "embedding_model": settings.embedding_model,
            "reranker_enabled": settings.enable_reranker,
        }

    async def index(self, force: bool = False) -> dict:
        return await index_corpus(self.docs, self.user_id, force=force)

    def _library(self) -> list[dict]:
        return [
            {"id": d.uuid, "filename": d.filename, "file_type": d.file_type, "is_tabular": False, "chunk_count": None}
            for d in self.docs
        ]

    def _doc_id_for_filename(self, filename: str | None) -> str | None:
        if not filename:
            return None
        d = self._by_filename.get(filename.lower())
        return d.id if d else None

    async def answer(self, item: GoldItem) -> Prediction:
        from app.services import query as q
        from app.services.chunking import count_tokens

        history = [{"role": t.role, "content": t.content, "citations": None} for t in item.history]
        timings: dict[str, float] = {}
        pred = Prediction(item_id=item.id, answer="")

        def _tick(stage: str, t0: float) -> None:
            timings[stage] = round((time.perf_counter() - t0) * 1000, 1)

        try:
            t0 = time.perf_counter()
            intent = await q.classify_intent(item.question, history)
            _tick("intent", t0)
            if intent == "chitchat":
                t0 = time.perf_counter()
                text = "".join([tok async for tok in q.generate_direct_response(item.question, history)])
                _tick("generate", t0)
                pred.answer, pred.route = text, "chitchat"
                return pred

            t0 = time.perf_counter()
            scope = await q.analyze_document_scope(item.question, history, self._library())
            _tick("scope", t0)
            if scope.status in ("no_documents", "needs_clarification"):
                pred.route = "no_documents" if scope.status == "no_documents" else "clarify"
                pred.answer = scope.clarification_text or ""
                return pred
            resolved = scope.status == "resolved"
            doc_filter = scope.target_doc_id if resolved else None
            filename_filter = scope.target_filename if resolved else None
            if doc_filter and doc_filter in self._by_uuid:
                pred.chosen_doc = self._by_uuid[doc_filter].id

            t0 = time.perf_counter()
            effective_q = scope.cleaned_question or item.question
            rewritten = await q.rewrite_query(effective_q, history)
            _tick("rewrite", t0)

            pred.route = "rag"
            t0 = time.perf_counter()
            chunks = await q.retrieve_chunks(
                user_id=self.user_id,
                question=rewritten,
                is_aggregation=q.detect_aggregation(rewritten),
                doc_id_filter=doc_filter,
                filename_filter=filename_filter,
            )
            _tick("retrieve", t0)
            pred.contexts = [
                RetrievedContext(
                    text=c.get("content", ""),
                    doc=self._doc_id_for_filename(c.get("filename")),
                    page=c.get("page_number"),
                    score=c.get("score"),
                )
                for c in chunks
            ]
            if not chunks:
                pred.answer = NO_CHUNKS_MESSAGE
                return pred

            t0 = time.perf_counter()
            pred.answer = "".join([tok async for tok in q.generate_answer_stream(rewritten, chunks)])
            _tick("generate", t0)

            # Approximate generation tokens (the prompt the generator saw + its output).
            pred.prompt_tokens = count_tokens(q.SYSTEM_PROMPT) + count_tokens(q._build_context(chunks, query=rewritten))
            pred.completion_tokens = count_tokens(pred.answer)

            cites = q.validate_citations(pred.answer, chunks)
            pred.citations = [
                Citation(doc=self._doc_id_for_filename(c.get("filename")), page=c.get("page_number"))
                for c in cites
            ]
        except Exception as e:  # one failing item must not abort the run
            pred.error = f"{type(e).__name__}: {e}"
        finally:
            pred.timings_ms = timings
        return pred
