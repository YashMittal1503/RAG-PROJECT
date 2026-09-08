"""
RAGAS Evaluation Script for the DocuChat RAG Project.

Runs the full RAG pipeline end-to-end on a curated test dataset,
collects retrieved contexts and generated answers, then evaluates
using RAGAS metrics:
  - Faithfulness (anti-hallucination)
  - Answer Relevancy (on-topic)
  - Context Precision (retrieval signal-to-noise)
  - Context Recall (retrieval completeness)
  - Answer Correctness (vs ground truth)

Usage:
    cd backend
    .\\venv\\Scripts\\python.exe ragas_eval.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, List, Optional

from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

# Ensure app modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ragas_eval")

# ── Your project's user_id (the one with uploaded documents) ──────────────
USER_ID = "ef052553-0d23-40e8-a677-508a417259a5"


# ── Dynamic Evaluation Dataset ─────────────────────────────────────────────
DATASET_FILE = os.path.join(os.path.dirname(__file__), "evaluation_dataset.json")


async def discover_user_documents(user_id: str) -> dict[str, list[dict]]:
    """
    Scroll Qdrant collection for the given user and group chunks by filename.
    Returns: { filename: [ { "content": str, "page_number": int, "chunk_type": str }, ... ] }
    """
    from app.services.vector_store import get_client, _collection_name
    client = await get_client()
    col = _collection_name(user_id)
    offset = None
    chunks_by_file: dict[str, list[dict]] = {}

    while True:
        res, offset = await client.scroll(
            collection_name=col,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in res:
            fn = pt.payload.get("filename")
            content = pt.payload.get("content", "")
            if not fn or not content:
                continue
            chunks_by_file.setdefault(fn, []).append({
                "content": content,
                "page_number": pt.payload.get("page_number"),
                "chunk_type": pt.payload.get("chunk_type", "text"),
            })
        if offset is None:
            break

    return chunks_by_file


async def synthesize_qa_for_chunk(
    client: Any,
    model: str,
    filename: str,
    chunk_text: str,
) -> dict | None:
    """
    Generate a grounded question and comprehensive ground_truth answer
    from a specific document excerpt using ministral-3b-2512.
    """
    system_prompt = (
        "You are an AI benchmark creator for RAG evaluation. "
        "Given a document excerpt, generate 1 natural user question and a comprehensive, "
        "factually accurate ground-truth answer based ONLY on the provided excerpt.\n"
        "Requirements:\n"
        "1. The question must be specific and answerable purely from the excerpt.\n"
        "2. The ground_truth must be a comprehensive paragraph string directly grounded in the excerpt.\n"
        "3. Output STRICT JSON with keys: \"question\" (string) and \"ground_truth\" (string)."
    )
    user_prompt = f"Document: {filename}\n\nExcerpt:\n{chunk_text[:1500]}"
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(resp.choices[0].message.content)
        q = data.get("question", "").strip()
        gt = data.get("ground_truth", "")
        if isinstance(gt, dict):
            gt = " ".join(f"{k}: {v}" for k, v in gt.items())
        elif isinstance(gt, list):
            gt = " ".join(str(item) for item in gt)
        gt = str(gt).strip()

        if q and gt:
            return {"question": q, "ground_truth": gt}
    except Exception as e:
        logger.warning(f"Failed to synthesize QA for {filename}: {e}")
    return None


async def sync_and_load_evaluation_dataset(
    user_id: str,
    questions_per_new_doc: int = 2,
) -> list[dict]:
    """
    Discovers all uploaded documents in the user's Qdrant collection,
    loads existing questions from evaluation_dataset.json,
    and dynamically synthesizes grounded questions for any newly discovered documents.
    Returns: flattened list of [{ "question": ..., "ground_truth": ..., "document": ... }]
    """
    from openai import AsyncOpenAI
    from app.config import settings

    logger.info("Scanning vector database for user documents...")
    chunks_by_file = await discover_user_documents(user_id)
    logger.info(f"Discovered {len(chunks_by_file)} documents in vector store: {list(chunks_by_file.keys())}")

    # Load persistent dataset if it exists
    dataset_store: dict[str, list[dict]] = {}
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                dataset_store = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {DATASET_FILE}: {e}")

    # Prepare Mistral client for synthesis if new documents need questions
    mistral_keys = settings.get_mistral_keys()
    judge_model = settings.mistral_model or "ministral-3b-2512"
    if judge_model == "mistral-small-latest":
        judge_model = "ministral-3b-2512"

    mistral_clients = [
        AsyncOpenAI(api_key=k, base_url="https://api.mistral.ai/v1")
        for k in mistral_keys
    ] if mistral_keys else []

    updated = False
    for fn, chunks in chunks_by_file.items():
        existing_qs = dataset_store.get(fn, [])
        if existing_qs:
            logger.info(f"  [Existing Document] '{fn}': {len(existing_qs)} questions loaded from cache.")
            continue

        logger.info(f"  [New Document Detected] '{fn}' ({len(chunks)} chunks). Synthesizing {questions_per_new_doc} grounded questions...")
        if not mistral_clients:
            logger.error("Cannot synthesize questions for new document: No Mistral API keys configured.")
            continue

        # Filter out very short chunks (e.g. copyright, headers)
        substantive = [c for c in chunks if len(c["content"]) > 150]
        if not substantive:
            substantive = chunks

        # Pick diverse chunks across document span
        stride = max(1, len(substantive) // (questions_per_new_doc + 1))
        chosen_indices = [min((i + 1) * stride, len(substantive) - 1) for i in range(questions_per_new_doc)]
        chosen_indices = list(dict.fromkeys(chosen_indices))

        new_qa_list = []
        for idx_i, chunk_idx in enumerate(chosen_indices):
            chunk = substantive[chunk_idx]
            client = mistral_clients[idx_i % len(mistral_clients)]
            qa = await synthesize_qa_for_chunk(
                client=client,
                model=judge_model,
                filename=fn,
                chunk_text=chunk["content"],
            )
            if qa:
                new_qa_list.append(qa)
                logger.info(f"    Generated Q{idx_i+1}: '{qa['question'][:65]}...'")

        if new_qa_list:
            dataset_store[fn] = new_qa_list
            updated = True

    # Persist updated dataset to file
    if updated:
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset_store, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated evaluation dataset successfully saved to: {DATASET_FILE}")

    # Flatten into list of test questions with document tags
    flattened = []
    for fn, items in dataset_store.items():
        for item in items:
            flattened.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "document": fn,
            })

    logger.info(f"Total benchmark questions across all documents: {len(flattened)}")
    return flattened


async def run_rag_pipeline(question: str) -> dict:
    """
    Run the full RAG pipeline for a single question.
    Returns: {answer: str, contexts: list[str], retrieval_time_ms: float, generation_time_ms: float}
    """
    from app.services.query import (
        classify_intent,
        rewrite_query,
        detect_aggregation,
        retrieve_chunks,
        generate_answer_stream,
    )
    from app.services import embedding

    # Ensure embedding model is loaded
    if not embedding.is_model_loaded():
        embedding.init_model()

    result = {
        "answer": "",
        "contexts": [],
        "retrieval_time_ms": 0.0,
        "generation_time_ms": 0.0,
        "chunk_scores": [],
        "chunk_metadata": [],
    }

    # Step 1: Rewrite query directly (eval questions are known document queries, skipping intent classification)
    rewritten = await rewrite_query(question, [])

    # Step 3: Detect aggregation
    is_agg = detect_aggregation(rewritten)

    # Step 4: Retrieve chunks
    t0 = time.time()
    chunks = await retrieve_chunks(USER_ID, rewritten, is_agg)
    t1 = time.time()
    result["retrieval_time_ms"] = (t1 - t0) * 1000

    if not chunks:
        result["answer"] = "No relevant documents found."
        return result

    # Store contexts
    result["contexts"] = [c.get("content", "") for c in chunks]
    result["chunk_scores"] = [c.get("score", 0.0) for c in chunks]
    result["chunk_metadata"] = [
        {
            "filename": c.get("filename", ""),
            "page_number": c.get("page_number"),
            "chunk_type": c.get("chunk_type", "text"),
            "score": c.get("score", 0.0),
        }
        for c in chunks
    ]

    # Step 5: Generate answer
    t2 = time.time()
    full_answer = ""
    async for token in generate_answer_stream(rewritten, chunks):
        full_answer += token
    t3 = time.time()

    # Step 6: Validate citations
    from app.services.query import validate_citations
    citations = validate_citations(full_answer, chunks)

    result["answer"] = full_answer
    result["citations"] = citations
    result["generation_time_ms"] = (t3 - t2) * 1000

    return result


class RotatingJudgeLLM(BaseChatModel):
    """
    Round-robin LLM wrapper that distributes evaluation prompts across
    a pool of API keys to maximize concurrency and avoid per-key rate limits.
    """
    models: list[Any] = Field(default_factory=list)
    _current_idx: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.models:
            raise RuntimeError("No judge models available in pool.")
        idx = self._current_idx % len(self.models)
        self._current_idx += 1
        kwargs.pop("n", None)  # Ensure n=1 compatibility for non-OpenAI endpoints
        return self.models[idx]._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.models:
            raise RuntimeError("No judge models available in pool.")
        idx = self._current_idx % len(self.models)
        self._current_idx += 1
        kwargs.pop("n", None)  # Ensure n=1 compatibility for non-OpenAI endpoints
        return await self.models[idx]._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "rotating-judge-llm"


async def collect_evaluation_data(dataset: list[dict], concurrency: int = 1) -> list[dict]:
    """
    Run the RAG pipeline on all test questions freshly using sequential pacing.
    Processes 1 question at a time with a 2.0s pause to stay strictly within Groq's 30,000 TPM limit.
    """
    logger.info(f"Running fresh RAG pipeline on {len(dataset)} test questions with concurrency={concurrency}...")
    sem = asyncio.Semaphore(concurrency)

    async def _process_item(index: int, item: dict) -> dict:
        q = item["question"]
        gt = item["ground_truth"]
        doc = item.get("document", "Unknown")
        async with sem:
            logger.info(f"[{index+1}/{len(dataset)}] [{doc}] Querying: '{q[:60]}...'")
            t_start = time.time()
            try:
                result = await run_rag_pipeline(q)
                elapsed_ms = (time.time() - t_start) * 1000
                logger.info(
                    f"[{index+1}/{len(dataset)}] Finished in {elapsed_ms:.0f}ms | "
                    f"Contexts: {len(result['contexts'])} | Answer: {len(result['answer'])} chars"
                )
                # Light 0.3-second pacing buffer between queries (multi-key pool handles 429s automatically)
                await asyncio.sleep(0.3)
                return {
                    "question": q,
                    "ground_truth": gt,
                    "document": doc,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "citations": result.get("citations", []),
                    "retrieval_time_ms": result["retrieval_time_ms"],
                    "generation_time_ms": result["generation_time_ms"],
                    "chunk_scores": result["chunk_scores"],
                    "chunk_metadata": result["chunk_metadata"],
                }
            except Exception as e:
                logger.error(f"[{index+1}/{len(dataset)}] ERROR on '{q[:60]}...': {e}")
                return {
                    "question": q,
                    "ground_truth": gt,
                    "document": doc,
                    "answer": f"ERROR: {str(e)}",
                    "contexts": [],
                    "retrieval_time_ms": 0,
                    "generation_time_ms": 0,
                    "chunk_scores": [],
                    "chunk_metadata": [],
                }

    tasks = [_process_item(i, item) for i, item in enumerate(dataset)]
    all_results = await asyncio.gather(*tasks)
    return list(all_results)



def run_ragas_evaluation(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation on the collected results.
    Offloads to Google Gemini Flash with multi-key rotation and multi-worker parallelism.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            FactualCorrectness,
        )
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_openai import ChatOpenAI
        from langchain_core.embeddings import Embeddings
        from app.services.embedding import embed_texts, embed_query
        from app.config import settings

        class FastEmbedLangchain(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return embed_texts(texts)
            def embed_query(self, text: str) -> list[float]:
                return embed_query(text)

        logger.info("\n" + "=" * 60)
        logger.info("Configuring High-Performance RAGAS Evaluation...")
        logger.info("=" * 60)

        # Select and configure Judge LLM:
        # 1. Primary: Mistral AI (500k TPM, 128k context, 60 RPM — ideal for RAGAS eval)
        # 2. Secondary: Google Gemini Flash (3 distinct projects, 60 RPM aggregate)
        # 3. Tertiary: Groq (ultra-fast sub-second LPUs)
        mistral_keys = settings.get_mistral_keys()
        gemini_keys = settings.get_gemini_keys()
        groq_keys = settings.get_groq_keys()

        if mistral_keys:
            judge_provider = "Mistral AI"
            base_url = "https://api.mistral.ai/v1"
            judge_model = settings.mistral_model or "ministral-3b-2512"
            if judge_model == "mistral-small-latest":
                judge_model = "ministral-3b-2512"
            active_keys = mistral_keys
            # ministral-3b-2512 supports 12.5 RPS and 500,000 TPM
            max_workers = min(3, len(active_keys))
        elif gemini_keys:
            judge_provider = "Google Gemini Flash"
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            judge_model = settings.gemini_model
            if not judge_model or judge_model in ("gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"):
                judge_model = "gemini-flash-latest"
            active_keys = gemini_keys
            # 3 distinct projects = 3 workers (1 worker per project quota)
            max_workers = min(3, len(active_keys))
        else:
            judge_provider = "Groq (fallback)"
            base_url = "https://api.groq.com/openai/v1"
            judge_model = "qwen/qwen3.8-27b"
            active_keys = groq_keys
            max_workers = min(4, max(2, len(active_keys) * 2))

        logger.info(
            f"Judge LLM Provider: {judge_provider} ({judge_model}) | "
            f"Active keys in rotation: {len(active_keys)} | Parallel Workers: {max_workers}"
        )

        if len(active_keys) > 1:
            models = [
                ChatOpenAI(
                    base_url=base_url,
                    api_key=k,
                    model=judge_model,
                    temperature=0.0,
                    max_tokens=2000,
                    timeout=60,
                    max_retries=2,
                )
                for k in active_keys
            ]
            evaluator_llm = LangchainLLMWrapper(RotatingJudgeLLM(models=models))
        else:
            single_model = ChatOpenAI(
                base_url=base_url,
                api_key=active_keys[0],
                model=judge_model,
                temperature=0.0,
                max_tokens=2000,
                timeout=60,
                max_retries=2,
            )
            evaluator_llm = LangchainLLMWrapper(single_model)

        evaluator_embeddings = LangchainEmbeddingsWrapper(FastEmbedLangchain())

        # Build RAGAS dataset
        samples = []
        for r in results:
            if not r["contexts"] or r["answer"].startswith("ERROR"):
                continue

            sample = SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            samples.append(sample)

        if not samples:
            logger.error("No valid samples for RAGAS evaluation!")
            return {"error": "No valid samples"}

        eval_dataset = EvaluationDataset(samples=samples)

        # Define metrics
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
            FactualCorrectness(llm=evaluator_llm),
        ]

        # Run evaluation with multi-worker parallelism
        eval_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            run_config=RunConfig(max_workers=max_workers, timeout=60, max_retries=2, max_wait=10),
        )

        # Convert to dict and enrich with document metadata
        raw_scores = eval_result.to_pandas().to_dict(orient="records")
        scores = []
        for i, row in enumerate(raw_scores):
            cleaned = {}
            for k, v in row.items():
                if isinstance(v, float) and (v != v):  # NaN check
                    continue
                cleaned[k] = round(v, 4) if isinstance(v, float) else v
            if i < len(results):
                cleaned["document"] = results[i].get("document", "Unknown")
            scores.append(cleaned)

        # Compute overall aggregates
        aggregate = {}
        metric_names = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference", "context_recall", "factual_correctness(mode=f1)"]
        for metric in metric_names:
            values = [s.get(metric) for s in scores if s.get(metric) is not None and not (isinstance(s.get(metric), float) and s.get(metric) != s.get(metric))]
            if values:
                aggregate[metric] = round(sum(values) / len(values), 4)

        # Compute per-document aggregates
        by_document = {}
        for s in scores:
            doc = s.get("document", "Unknown")
            by_document.setdefault(doc, []).append(s)

        doc_aggregates = {}
        for doc, doc_scores in by_document.items():
            doc_aggs = {}
            for metric in metric_names:
                m_vals = [s.get(metric) for s in doc_scores if s.get(metric) is not None and not (isinstance(s.get(metric), float) and s.get(metric) != s.get(metric))]
                if m_vals:
                    doc_aggs[metric] = round(sum(m_vals) / len(m_vals), 4)
            doc_aggregates[doc] = doc_aggs

        return {
            "per_question": scores,
            "aggregate": aggregate,
            "by_document": doc_aggregates,
            "num_samples": len(samples),
            "evaluation_method": f"ragas ({judge_provider} {judge_model})",
        }

    except ImportError as e:
        logger.warning(f"RAGAS or langchain dependencies not installed: {e}")
        logger.info("Falling back to manual evaluation metrics...")
        return run_manual_evaluation(results)
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Falling back to manual evaluation metrics...")
        return run_manual_evaluation(results)


def run_manual_evaluation(results: list[dict]) -> dict:
    """
    Fallback manual evaluation when RAGAS is not available.
    Uses heuristic scoring based on text overlap and retrieval quality.
    """
    logger.info("\nRunning manual (heuristic) evaluation...")

    scores = []
    for r in results:
        q = r["question"]
        gt = r["ground_truth"].lower()
        ans = r["answer"].lower()
        contexts = [c.lower() for c in r["contexts"]]

        # 1. Answer-Ground Truth Overlap (proxy for correctness)
        gt_words = set(gt.split())
        ans_words = set(ans.split())
        if gt_words:
            overlap = len(gt_words & ans_words) / len(gt_words)
        else:
            overlap = 0.0

        # 2. Context Relevance (do contexts contain ground truth info?)
        combined_ctx = " ".join(contexts)
        gt_key_phrases = [w for w in gt.split() if len(w) > 3]
        if gt_key_phrases:
            ctx_recall = sum(1 for w in gt_key_phrases if w in combined_ctx) / len(gt_key_phrases)
        else:
            ctx_recall = 0.0

        # 3. Faithfulness proxy (does the answer only use info from contexts?)
        stop_words = {"which", "about", "their", "these", "those", "where", "there", "would", "could", "should", "based", "using"}
        ans_key_words = [w for w in ans.split() if len(w) > 4 and w not in stop_words]
        if ans_key_words:
            grounded = sum(1 for w in ans_key_words if w in combined_ctx) / len(ans_key_words)
        else:
            grounded = 0.0

        # 4. Retrieval quality (average cosine similarity scores)
        avg_score = sum(r["chunk_scores"]) / len(r["chunk_scores"]) if r["chunk_scores"] else 0.0

        # 5. Answer contains "don't have enough information" check
        no_info = "don't have enough" in ans or "no information" in ans or "no relevant" in ans

        scores.append({
            "question": q,
            "answer_correctness": round(overlap, 4),
            "context_recall": round(ctx_recall, 4),
            "faithfulness_proxy": round(grounded, 4),
            "avg_retrieval_score": round(avg_score, 4),
            "retrieval_time_ms": round(r["retrieval_time_ms"], 1),
            "generation_time_ms": round(r["generation_time_ms"], 1),
            "num_chunks_retrieved": len(r["contexts"]),
            "no_info_response": no_info,
        })

    # Aggregate
    aggregate = {}
    for metric in ["answer_correctness", "context_recall", "faithfulness_proxy", "avg_retrieval_score"]:
        values = [s[metric] for s in scores]
        aggregate[metric] = round(sum(values) / len(values), 4) if values else 0.0

    aggregate["avg_retrieval_time_ms"] = round(
        sum(s["retrieval_time_ms"] for s in scores) / len(scores), 1
    ) if scores else 0.0
    aggregate["avg_generation_time_ms"] = round(
        sum(s["generation_time_ms"] for s in scores) / len(scores), 1
    ) if scores else 0.0

    return {
        "per_question": scores,
        "aggregate": aggregate,
        "num_samples": len(scores),
        "evaluation_method": "manual_heuristic",
    }


def print_results(eval_results: dict, raw_results: list[dict]):
    """Pretty-print the evaluation results."""
    print("\n" + "=" * 70)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 70)

    per_q = eval_results.get("per_question", [])
    for i, raw in enumerate(raw_results):
        score = per_q[i] if i < len(per_q) else {}
        doc = raw.get("document", "Unknown")
        print(f"\n{'_' * 60}")
        print(f"Q{i+1} [{doc}]:")
        print(f"  Question:     {raw['question']}")
        gt_preview = raw['ground_truth'][:100].encode('ascii', 'replace').decode()
        ans_preview = raw['answer'][:100].encode('ascii', 'replace').decode()
        print(f"  Ground Truth: {gt_preview}...")
        print(f"  Answer:       {ans_preview}...")
        print(f"  Chunks:       {len(raw['contexts'])} retrieved")

        for key, val in score.items():
            if key not in ("question", "user_input", "response", "retrieved_contexts", "reference", "document"):
                if isinstance(val, float):
                    emoji = "[OK]" if val >= 0.7 else ("[WARN]" if val >= 0.5 else "[LOW]")
                    print(f"  {emoji} {key}: {val:.4f}")
                elif isinstance(val, (int, bool)):
                    print(f"     {key}: {val}")

    # Overall Aggregate
    print(f"\n{'=' * 70}")
    print("  AGGREGATE SCORES (ALL DOCUMENTS)")
    print(f"{'=' * 70}")
    for metric, val in eval_results.get("aggregate", {}).items():
        if isinstance(val, float) and val <= 1.0:
            emoji = "[OK]" if val >= 0.7 else ("[WARN]" if val >= 0.5 else "[LOW]")
            print(f"  {emoji} {metric}: {val:.4f}")
        else:
            print(f"       {metric}: {val}")

    # Per-Document Aggregates
    if eval_results.get("by_document"):
        print(f"\n{'=' * 70}")
        print("  PER-DOCUMENT AGGREGATES")
        print(f"{'=' * 70}")
        for doc, metrics in eval_results["by_document"].items():
            print(f"\n  Document: {doc}")
            for metric, val in metrics.items():
                if isinstance(val, float):
                    emoji = "[OK]" if val >= 0.7 else ("[WARN]" if val >= 0.5 else "[LOW]")
                    print(f"    {emoji} {metric}: {val:.4f}")

    print(f"\n  Total samples evaluated: {eval_results.get('num_samples', 0)}")
    method = eval_results.get("evaluation_method", "ragas")
    print(f"  Evaluation method: {method}")
    print("=" * 70)


async def main():
    """Main entry point with timing benchmarks."""
    t_global_start = time.time()
    logger.info("Starting High-Performance RAGAS Evaluation of DocuChat RAG Pipeline")
    logger.info(f"   User ID: {USER_ID}")
    logger.info(f"   Timestamp: {datetime.now().isoformat()}")

    # Step 0: Sync and load dynamic evaluation dataset across all uploaded documents
    dataset = await sync_and_load_evaluation_dataset(USER_ID, questions_per_new_doc=2)
    logger.info(f"   Total test questions: {len(dataset)}")

    # Step 1: Run RAG pipeline on all test questions freshly
    # Scale concurrency to match the available Groq key pool (multi-key pool handles load)
    from app.config import settings
    groq_keys = settings.get_groq_keys()
    eval_concurrency = max(2, len(groq_keys))
    logger.info(f"   Phase 1 Concurrency: {eval_concurrency} (scaled across {len(groq_keys)} Groq keys in pool)")

    t1_start = time.time()
    raw_results = await collect_evaluation_data(dataset=dataset, concurrency=eval_concurrency)
    t1_elapsed = time.time() - t1_start
    logger.info(f"\nPhase 1 (Concurrent RAG Generation) completed in {t1_elapsed:.1f}s")

    # Save raw results
    raw_path = os.path.join(os.path.dirname(__file__), "ragas_raw_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Raw results saved to: {raw_path}")

    # Step 2: Run RAGAS evaluation
    t2_start = time.time()
    eval_results = run_ragas_evaluation(raw_results)
    t2_elapsed = time.time() - t2_start
    logger.info(f"\nPhase 2 (Parallel Judge LLM Evaluation) completed in {t2_elapsed:.1f}s")

    # Save evaluation results
    eval_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Evaluation results saved to: {eval_path}")

    # Step 3: Pretty-print results with performance benchmark
    total_elapsed = time.time() - t_global_start
    eval_results["benchmark"] = {
        "phase1_generation_seconds": round(t1_elapsed, 2),
        "phase2_evaluation_seconds": round(t2_elapsed, 2),
        "total_seconds": round(total_elapsed, 2),
    }
    print_results(eval_results, raw_results)
    print(f"\n  BENCHMARK SUMMARY:")
    print(f"  Phase 1 (Concurrent Generation): {t1_elapsed:.1f}s")
    print(f"  Phase 2 (Parallel Evaluation):   {t2_elapsed:.1f}s")
    print(f"  Total Script Runtime:            {total_elapsed:.1f}s")
    print("=" * 70)

    return eval_results


if __name__ == "__main__":
    asyncio.run(main())
