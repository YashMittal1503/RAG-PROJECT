import sys, os
sys.path.insert(0, os.path.abspath("scripts"))
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from benchmark_svd_vs_baseline import SAMPLE_CORPUS, BENCHMARK_QUERIES
from app.services.svd_summarizer import build_recursive_svd_tree
from app.services import embedding, reranker
import numpy as np

leaf_embeddings = embedding.embed_texts(SAMPLE_CORPUS)
leaf_chunks = [
    {
        "id": str(i),
        "content": text,
        "chunk_type": "text",
        "tree_level": 0,
        "is_root": False,
        "page_number": (i // 6) + 1,
        "filename": "doc.pdf",
        "vector": leaf_embeddings[i],
    }
    for i, text in enumerate(SAMPLE_CORPUS)
]
summaries, meta = build_recursive_svd_tree(leaf_chunks, leaf_embeddings, cluster_size=6, tau=0.95, filename="doc.pdf")
flattened = leaf_chunks + summaries

for q_item in BENCHMARK_QUERIES:
    q = q_item["query"]
    q_vec = embedding.embed_query(q)
    scores = []
    for node in flattened:
        sim = float(np.dot(q_vec, node["vector"]) / (np.linalg.norm(q_vec) * np.linalg.norm(node["vector"])))
        scores.append({**node, "score": sim})
    scores.sort(key=lambda x: x["score"], reverse=True)
    reranked = reranker.rerank_chunks(query=q, chunks=scores[:10], top_k=5)
    print("\nQUERY:", q)
    for rank, r in enumerate(reranked, 1):
        kind = "ROOT" if r.get("is_root") else (f"SUM-L{r.get('tree_level')}" if r.get("tree_level", 0) > 0 else f"LEAF-P{r.get('page_number')}")
        print(f"  {rank}. [{kind}] score={r.get('score', 0):.4f} | {r.get('content', '')[:100]}...")
