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

# Ensure app modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ragas_eval")

# ── Your project's user_id (the one with uploaded documents) ──────────────
USER_ID = "ef052553-0d23-40e8-a677-508a417259a5"


# ── Evaluation Dataset ────────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "question": "What is the purpose of the Ultimate Python Handbook?",
        "ground_truth": "The purpose of the Ultimate Python Handbook is to make programming accessible and enjoyable for everyone. It is a comprehensive guide designed for beginners and anyone looking to strengthen their foundational knowledge of Python, a versatile and user-friendly programming language.",
    },
    {
        "question": "What are the different types of comments in Python?",
        "ground_truth": "There are two types of comments in Python: Single-line comments use a '#' at the start of the line, and multi-line comments can use '#' at each line or use triple quotes (triple double-quotes).",
    },
    {
        "question": "What is a function in Python according to the handbook?",
        "ground_truth": "A function is a group of statements performing a specific task. When a program gets bigger in size and its complexity grows, functions help keep track of which piece of code is doing what. A function can be reused by the programmer in a given program any number of times.",
    },
    {
        "question": "How does the if-else conditional work in Python?",
        "ground_truth": "If-else and elif statements are multiway decisions taken by the program due to certain conditions. The syntax uses if(condition) followed by code block, elif(condition) for additional conditions, and else for the default case. For example: if(a>9): print('greater') else: print('lesser').",
    },
    {
        "question": "What are sets in Python?",
        "ground_truth": "A set is a collection of non-repetitive elements in Python. Sets are created using set(). Sets do not allow duplicate values.",
    },
    {
        "question": "What position was offered in the Xoodrip offer letter?",
        "ground_truth": "The position offered was Software Developer Intern at Xoodrip Private Limited.",
    },
    {
        "question": "What is the salary mentioned in the Xoodrip offer letter?",
        "ground_truth": "The salary mentioned is Unpaid Internship.",
    },
    {
        "question": "What is the work mode and internship duration mentioned in the offer letter?",
        "ground_truth": "The work mode is Remote and the internship duration is 3 months, starting from October 17, 2025.",
    },
    {
        "question": "How does a while loop work in Python?",
        "ground_truth": "A while loop checks a condition. If it evaluates to true, the body of the loop is executed. The process of condition check and execution is continued until the condition becomes False. If the condition never becomes false, the loop keeps getting executed forever.",
    },
    {
        "question": "What dictionary methods are described in the Python handbook?",
        "ground_truth": "The dictionary methods described are: items() which returns a list of (key, value) tuples, keys() which returns a list containing dictionary's keys, update() which updates the dictionary with supplied key-value pairs, and get() which returns the value of the specified key.",
    },
]


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

    # Step 1: Classify intent
    intent = await classify_intent(question)
    if intent == "chitchat":
        result["answer"] = "[Classified as chitchat - skipping retrieval]"
        return result

    # Step 2: Rewrite query (no chat history for eval)
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


async def collect_evaluation_data():
    """
    Run the RAG pipeline on all test questions and collect the data
    needed for RAGAS evaluation.
    """
    logger.info(f"Running RAG pipeline on {len(EVAL_DATASET)} test questions...")

    all_results = []
    for i, item in enumerate(EVAL_DATASET):
        q = item["question"]
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(EVAL_DATASET)}] Question: {q}")
        logger.info(f"{'='*60}")

        try:
            result = await run_rag_pipeline(q)
            all_results.append({
                "question": q,
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "citations": result.get("citations", []),
                "retrieval_time_ms": result["retrieval_time_ms"],
                "generation_time_ms": result["generation_time_ms"],
                "chunk_scores": result["chunk_scores"],
                "chunk_metadata": result["chunk_metadata"],
            })

            logger.info(f"  Answer ({len(result['answer'])} chars): {result['answer'][:200]}...")
            logger.info(f"  Contexts retrieved: {len(result['contexts'])}")
            logger.info(f"  Retrieval: {result['retrieval_time_ms']:.0f}ms | Generation: {result['generation_time_ms']:.0f}ms")

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            all_results.append({
                "question": q,
                "ground_truth": item["ground_truth"],
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "retrieval_time_ms": 0,
                "generation_time_ms": 0,
                "chunk_scores": [],
                "chunk_metadata": [],
            })

        # Small delay to respect rate limits
        await asyncio.sleep(2)

    return all_results


def run_ragas_evaluation(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation on the collected results.
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

        class FastEmbedLangchain(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return embed_texts(texts)
            def embed_query(self, text: str) -> list[float]:
                return embed_query(text)

        logger.info("\n" + "=" * 60)
        logger.info("Running RAGAS evaluation...")
        logger.info("=" * 60)

        # Set up Groq as the judge LLM
        from app.config import settings

        groq_llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
            model="qwen/qwen3.8-27b",
            temperature=0.0,
            max_tokens=2000,
        )
        evaluator_llm = LangchainLLMWrapper(groq_llm)
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

        # Run evaluation
        eval_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            run_config=RunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=30),
        )

        # Convert to dict
        scores = eval_result.to_pandas().to_dict(orient="records")

        # Compute aggregates
        aggregate = {}
        metric_names = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference", "context_recall", "factual_correctness(mode=f1)"]
        for metric in metric_names:
            values = [s.get(metric) for s in scores if s.get(metric) is not None and not (isinstance(s.get(metric), float) and s.get(metric) != s.get(metric))]
            if values:
                aggregate[metric] = round(sum(values) / len(values), 4)

        return {
            "per_question": scores,
            "aggregate": aggregate,
            "num_samples": len(samples),
            "evaluation_method": "ragas",
        }

    except ImportError as e:
        logger.warning(f"RAGAS or langchain-groq not installed: {e}")
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
        print(f"\n{'_' * 60}")
        print(f"Q{i+1}: {raw['question']}")
        gt_preview = raw['ground_truth'][:100].encode('ascii', 'replace').decode()
        ans_preview = raw['answer'][:100].encode('ascii', 'replace').decode()
        print(f"  Ground Truth: {gt_preview}...")
        print(f"  Answer:       {ans_preview}...")
        print(f"  Chunks:       {len(raw['contexts'])} retrieved")

        for key, val in score.items():
            if key not in ("question", "user_input", "response", "retrieved_contexts", "reference"):
                if isinstance(val, float):
                    emoji = "[OK]" if val >= 0.7 else ("[WARN]" if val >= 0.5 else "[LOW]")
                    print(f"  {emoji} {key}: {val:.4f}")
                elif isinstance(val, (int, bool)):
                    print(f"     {key}: {val}")

    # Aggregate
    print(f"\n{'=' * 70}")
    print("  AGGREGATE SCORES")
    print(f"{'=' * 70}")
    for metric, val in eval_results.get("aggregate", {}).items():
        if isinstance(val, float) and val <= 1.0:
            emoji = "[OK]" if val >= 0.7 else ("[WARN]" if val >= 0.5 else "[LOW]")
            print(f"  {emoji} {metric}: {val:.4f}")
        else:
            print(f"       {metric}: {val}")

    print(f"\n  Total samples evaluated: {eval_results.get('num_samples', 0)}")
    method = eval_results.get("evaluation_method", "ragas")
    print(f"  Evaluation method: {method}")
    print("=" * 70)


async def main():
    """Main entry point."""
    logger.info("Starting RAGAS Evaluation of DocuChat RAG Pipeline")
    logger.info(f"   User ID: {USER_ID}")
    logger.info(f"   Test questions: {len(EVAL_DATASET)}")
    logger.info(f"   Timestamp: {datetime.now().isoformat()}")

    # Step 1: Run RAG pipeline on all test questions
    raw_results = await collect_evaluation_data()

    # Save raw results
    raw_path = os.path.join(os.path.dirname(__file__), "ragas_raw_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nRaw results saved to: {raw_path}")

    # Step 2: Run RAGAS evaluation
    eval_results = run_ragas_evaluation(raw_results)

    # Save evaluation results
    eval_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Evaluation results saved to: {eval_path}")

    # Step 3: Pretty-print results
    print_results(eval_results, raw_results)

    return eval_results


if __name__ == "__main__":
    asyncio.run(main())
