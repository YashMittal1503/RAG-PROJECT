# DocTalk — Enterprise-Grade Dual-Engine Document & Tabular RAG System

[![CI](https://github.com/YashMittal1503/RAG-PROJECT/actions/workflows/ci.yml/badge.svg)](https://github.com/YashMittal1503/RAG-PROJECT/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.3-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant-Hybrid%20Search%20(RRF)-red?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP%20SQL-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![4-Tier LLM Fallback](https://img.shields.io/badge/LLM%20Orchestrator-4--Tier%20Failover-success)](https://github.com/YashMittal1503/RAG-PROJECT)
[![Mistral AI](https://img.shields.io/badge/Mistral%20AI-Tier%202%20LLM-FD6F00?logo=mistralai&logoColor=white)](https://mistral.ai/)
[![Pydantic Logfire](https://img.shields.io/badge/Logfire-Observability-E92063?logo=pydantic&logoColor=white)](https://logfire.pydantic.dev/)
[![Tests: 78 Passing](https://img.shields.io/badge/Tests-78%20Passing-brightgreen)](backend/tests/)

An enterprise-grade, full-stack Retrieval-Augmented Generation (RAG) and Text-to-SQL platform built for high reliability, zero downtime, and strict data grounding. Enables users to upload multiple unstructured documents (PDF, TXT) and structured spreadsheets (CSV, XLSX), query them with natural language, and receive sub-second token-streamed answers backed by validated citations and transparent SQL execution.

---

## The Core USP: Resilient 4-Tier Cross-Provider Orchestration & Multi-Key Pools

In production RAG systems, relying on a single LLM API key or a single provider guarantees frequent outages: token-per-minute (TPM) limits get exhausted, provider outages happen, and streaming connections drop.

**DocTalk's primary architectural differentiator is an autonomous 4-Tier Cross-Provider Fallback Orchestrator (`app.services.llm_provider`) with per-tier multi-key rotation and intelligent rate-limit quarantine:**

```
[ Incoming Generation Prompt ]
              │
              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ TIER 1: Groq Cloud LPUs (Ultra-fast, sub-second TTFT)                  │
  │ Models: gpt-oss-120b, qwen3.8-27b (free limits are per model)          │
  │ Key Pool: GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3...              │
  │ ↳ Round-Robin selection across active keys                             │
  │ ↳ On 429/TPM: key quarantined for 60s for that model only              │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ (Tier 1 rate-limited / failed)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ TIER 2: NVIDIA NIM (free, ~40 RPM, large open models)                  │
  │ Models: nemotron-3-ultra-550b, minimax-m3.1-flash (NVIDIA_*_MODEL)    │
  │ Key Pool: NVIDIA_API_KEY, NVIDIA_API_KEY_2...                          │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ (Tier 2 rate-limited / failed)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ TIER 3: Mistral AI                                                      │
  │ Models: ministral-3b-2512, mistral-small-latest                       │
  │ Key Pool: MISTRAL_API_KEY, MISTRAL_API_KEY_2, MISTRAL_API_KEY_3...     │
  │ ↳ Round-Robin selection; automatic 60s quarantine on 429               │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ (Tier 3 rate-limited / failed)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ TIER 4: Google Gemini Flash (Massive 1M+ token context window)         │
  │ Models: gemini-flash-latest, gemini-3.6-flash                          │
  │ Key Pool: GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3...       │
  │ ↳ Round-Robin selection; automatic 60s quarantine on 429               │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ (Tier 4 rate-limited / failed)
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ TIER 5: OpenRouter (Universal fail-safe multi-model fallback)          │
  │ Models: deepseek/deepseek-chat, meta-llama/llama-3.3-70b-instruct      │
  │ Key Pool: OPENROUTER_API_KEY, OPENROUTER_API_KEY_2...                  │
  └────────────────────────────────────────────────────────────────────────┘
```

### Key Capabilities of the LLM Orchestrator:
1. **Multi-Key Pooling Per Tier:** Up to 3+ API keys configured per tier (`_KEY`, `_KEY_2`, `_KEY_3` or comma-separated lists) distributed in round-robin sequence.
2. **Per-Model 429 / TPM Quarantine:** If a key hits a rate limit (HTTP 429, RPM or TPM), it is quarantined for 60 seconds *for that model only*. Groq's free limits are per model, so the key keeps serving the other models.
3. **Healthy Keys First, Across Every Tier:** Every ready key on every tier (Groq → NVIDIA NIM → Mistral → Gemini → OpenRouter) is tried before any key that is still cooling down.
4. **All-Quarantined Soonest Recovery:** If every key everywhere is cooling down, keys are retried in order of soonest recovery.
5. **Dead Models Are Skipped:** A model that returns 404 / "decommissioned" is skipped for the rest of the request without quarantining the key.
6. **Zero SDK Retry Freezes:** All clients are instantiated with `max_retries=0`. Instead of freezing for 10–60 seconds in SDK exponential backoff loops, rate limits trigger immediate failover in milliseconds.
7. **Mid-Stream Failure Resilience:** If an error occurs before streaming begins, it seamlessly re-routes to the next provider. If a provider aborts mid-stream, the system appends a non-breaking inline notification without dropping the SSE connection.
7. **Universal Credential Masking:** Every API key is automatically masked (`gsk_...9cYD`, `AIza...3a4f`) across all Logfire spans and console logs to prevent secret leakage.

---

## Architectural Highlights

### 1. Dual-Engine Pipeline: Hybrid RAG + Analytical Text-to-SQL
- **Unstructured Documents (PDF / TXT):** Extracted via PyMuPDF, split with token-bounded sentence chunking, dual-embedded locally via FastEmbed for dense vectors (`BAAI/bge-small-en-v1.5`) and sparse lexical vectors (`Qdrant/bm25`), and indexed into user-isolated Qdrant Cloud collections.
- **Tabular Datasets (CSV / XLSX):** Ingested into an embedded columnar **DuckDB** engine per user. Natural language queries are translated into `SELECT` SQL, executed in a sandboxed read-only connection, and answered with a short interpretation plus the exact result table.

### 2. SOTA Hybrid Retrieval with Qdrant Reciprocal Rank Fusion (RRF)
- **Dense + BM25 Sparse Hybrid Search:** Queries Qdrant concurrently with 384-dimensional dense vectors (semantic intent) and sparse BM25 vectors (exact keywords, model numbers, IDs, acronyms).
- **Reciprocal Rank Fusion (RRF):** Server-side rank fusion via Qdrant's `query_points` API merges semantic and lexical results without score calibration mismatches.
- **FlashRank Cross-Encoder Reranking:** Local ONNX cross-encoder (`ms-marco-TinyBERT-L-2-v2`) performs cross-attention reranking on the top 10 candidates down to the 5 highest-signal chunks.

### 3. Extractive Contextual Compression & Token Budgeting
- **Distractor Sentence Pruning:** Evaluates retrieved 300–500 token chunks using query keyword overlap, density weighting, and bigram phrase matching to extract only the most query-relevant sentences.
- **Narrative Continuity:** Automatically preserves adjacent context windows ($\pm 1$ sentence) and connects non-contiguous excerpts with `[...]`.
- **Token Efficiency & Rate-Limit Shield:** Cuts prompt context tokens by **25%–45%**, reducing Groq TPM pressure and improving Time-to-First-Token (TTFT).
- **Guaranteed Zero Data Loss:** Short chunks ($\le 4$ sentences), spreadsheet summary chunks, and chunks without query overlap safely bypass compression. Citation anchors (`[Page X]`, `[Rows Y-Z]`) remain intact.

### 4. Conversational Intelligence & Deterministic Verification
- **LLM Query Classification:** one fast call per question decides the intent (chitchat vs retrieval), the scope (whole document, one section, document metadata, or a targeted fact) and whether it aggregates over many records. It drives chitchat routing, summary-chunk retrieval for whole-document questions, and Text-to-SQL planning. Measured with `python -m eval classify` on a labelled question set.
- **Context-Aware Query Rewriting:** Rewrites multi-turn conversational follow-ups (e.g. `"tell me more"`, `"continue"`, `"what about his salary?"`) into standalone search queries under 20 words.
- **Mathematical Citation Validation:** Algorithmic regex verification verifies that every cited page (`[Page X]`), row range (`[Rows Y-Z]`), or summary (`[Summary]`) genuinely exists in the retrieved context chunks, preventing hallucinated citations.
- **Adaptive Multi-Turn Chat Auto-Naming:** Generates clean 2-to-5 word chat titles from user inquiries and assistant responses. Defeats initial conversational noise (`"hi"`) by remaining `"New conversation"` until substantive questions are asked, refining the title across turns 1–3, persisting to PostgreSQL, and streaming real-time updates via SSE with zero TTFT delay. Supports manual inline renaming.

### 5. Evaluation Harness with RAGAS
- **Hand-verified golden set:** every item has a gold answer and verbatim evidence quotes, checked against the corpus by `python -m eval validate`. LLM-drafted candidates (`python -m eval draft`) are tagged for review and never used unchecked.
- **Layer-by-layer scoring:** judge correctness, RAGAS faithfulness / answer relevancy / context precision / context recall, evidence recall@k and MRR, citations, refusals, routing, Text-to-SQL execution accuracy, latency.
- **Isolated eval tenant:** its own Qdrant collection and DuckDB file; never touches real users' data.
- **Comparable runs:** bootstrap confidence intervals, paired run-vs-run comparison, per-category and per-document breakdowns, and a cross-provider judge (Gemini by default) with key rotation and on-disk caching.

### 6. Production Security & Data Isolation
- **Magic Byte File Signature Validation:** Inspects binary headers via `python-magic` to defeat extension spoofing (e.g. executable disguised as PDF).
- **Macro Malware Defense:** Explicitly blocks macro-enabled spreadsheet files (`.xlsm`, `.xlsb`, `.xltm`, `.docm`) to prevent macro injection.
- **Connection-Safe Scoped SSE Streaming:** Scoped database sessions release PostgreSQL connections immediately before streaming, preventing pool starvation during long generations.
- **Multi-Tenant Data Isolation:** Per-user Qdrant collections (`docs_{user_id}`) and per-user DuckDB files (`{user_id}.duckdb`).

---

## End-to-End System Architecture

```mermaid
flowchart TD
    User([User / Next.js 16 Frontend])

    subgraph Ingestion Pipeline
        Upload[Upload Documents / Spreadsheets] --> Validate[Magic Byte & Extension Validation]
        Validate --> Route{File Type?}
        Route -->|PDF / TXT| ParseText[PyMuPDF & Token-Bounded Chunking]
        ParseText --> DualEmbed[FastEmbed: Dense bge-small + Sparse BM25]
        DualEmbed --> Qdrant[(Qdrant Cloud: docs_user_id)]
        Route -->|CSV / XLSX| DuckStore[(DuckDB Embedded OLAP: user_id.duckdb)]
    end

    subgraph Query & Hybrid Retrieval Pipeline
        User -->|SSE Stream Query| ChatRouter[FastAPI Chat Router]
        ChatRouter --> Intent{Intent Classifier}
        Intent -->|Chitchat| DirectLLM[Stream Direct Response]
        Intent -->|Factual Query| Rewrite[Contextual Query Rewriter]
        Rewrite --> DataCheck{Tabular Data Available?}
        DataCheck -->|Yes| TextToSQL[DuckDB Text-to-SQL Pipeline]
        TextToSQL --> GenSQL[LLM Generates SQL SELECT]
        GenSQL --> ExecSQL[Execute in Sandboxed Read-Only DuckDB]
        ExecSQL --> StreamSQL[Stream Interpretation + Exact Markdown Result Table]
        DataCheck -->|No / Vector RAG| HybridSearch[Qdrant Hybrid Search: Dense + BM25 Prefetch]
        HybridSearch --> RRF[Server-Side Reciprocal Rank Fusion RRF]
        RRF --> CrossEncoder[FlashRank Cross-Encoder Top-5]
        CrossEncoder --> Compression[Extractive Contextual Compression -25% to -45% Tokens]
        Compression --> RouterLLM[4-Tier Multi-Key LLM Fallback Orchestrator]
        RouterLLM -->|Tier 1| Groq[Groq LPU Pool: 3+ Keys]
        RouterLLM -->|Tier 2 Fallback| NVIDIA[NVIDIA NIM Pool: ~40 RPM]
        RouterLLM -->|Tier 3 Fallback| Mistral[Mistral AI Pool]
        RouterLLM -->|Tier 4 Fallback| Gemini[Gemini Flash Pool: 3+ Keys]
        RouterLLM -->|Tier 5 Fallback| OpenRouter[OpenRouter Pool]
        RouterLLM --> CitCheck[Deterministic Citation Validation]
        CitCheck --> StreamAns[Stream Tokens & Page Citations]
    end

    subgraph Evaluation Harness
        Golden[(Golden JSONL + corpus manifest)] --> EvalTenant[Isolated eval tenant: Qdrant + DuckDB]
        EvalTenant --> Vanilla[Vanilla pipeline adapter]
        Vanilla --> Scorers[Judge correctness + RAGAS + retrieval / SQL / routing metrics]
        Scorers --> Reports[Reports with 95% CIs and paired comparison]
    end

    DirectLLM --> User
    StreamSQL --> User
    StreamAns --> User
```

---

## Benchmark Results

Scores come from the evaluation harness in [`backend/eval/`](backend/eval/README.md) and are published per run in `eval/runs/<run>/report.md`. The golden set is being built; numbers will appear here once it has been run.

Earlier README figures came from the retired `ragas_eval.py`, which scored LLM-generated single-chunk questions with a 3B judge. They aren't comparable with harness results, so they were removed (the last run is recorded in the harness README for reference).

---

## Detailed Component Breakdown

### 1. Dual-Embedding Ingestion & Hybrid Search
```
Document Upload ──> PyMuPDF Parsing ──> Token-Bounded Sentence Chunking (500 tokens, 50 overlap)
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
            FastEmbed Dense Embedding                    FastEmbed Sparse BM25
            BAAI/bge-small-en-v1.5                       Qdrant/bm25
            (384-dimensional dense vectors)              (Token frequency vectors)
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              ▼
                               Qdrant Cloud Point Struct
                               vector: {"": dense, "bm25": sparse}
```

At query time, Qdrant executes native server-side **Reciprocal Rank Fusion (RRF)**:
$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
where $r_m(d)$ is the rank of document $d$ in retrieval method $m$ (dense vs. sparse). The combined top 10 candidates are then scored by **FlashRank** cross-encoder (`ms-marco-TinyBERT-L-2-v2`) to produce the top 5 chunks.

### 2. Extractive Contextual Compression
Retrieved chunks often contain several paragraphs of background information surrounding the specific answer sentence. Sending all raw chunks consumes excessive prompt tokens, increases generation latency, and risks triggering 429 TPM limits on Groq.

```
[ Raw Retrieved Chunk (400 tokens) ]
├── Sentence 1: Background history of company... (Irrelevant)
├── Sentence 2: Founder biography... (Irrelevant)
├── Sentence 3: In Q3 2024, revenue grew 42% to $18.4M.  <── [MATCH: Score 4.5]
├── Sentence 4: Operating margins reached 28%.           <── [WINDOW EXPANSION ±1]
└── Sentence 5: Unrelated product roadmap... (Irrelevant)
                    │
                    ▼
[ Compressed Context (95 tokens) ]
"[...] In Q3 2024, revenue grew 42% to $18.4M. Operating margins reached 28%."
↳ 76% Token Reduction | Citation tag [Page 14] Preserved
```

### 3. In-Process DuckDB Text-to-SQL
When a user uploads a `.csv` or `.xlsx` file:
1. Parsed DataFrames are stored as named tables (`t_{doc_id}_{sheet}`) in `{user_id}.duckdb`. Column names become plain SQL identifiers (original headers kept as column comments) and ISO-format date columns are stored as real `DATE`/`TIMESTAMP` values.
2. At question time the schema is described to the LLM as Markdown: types, value ranges, the exact allowed values of category columns, missing-value share and sample rows.
3. The query classifier decides the path: whole-dataset questions ("what's in my data?") are answered from that profile without SQL; others get a DuckDB `SELECT`/`WITH` query, generated with a hint about whether the question aggregates. A failing query is repaired once using the error message.
4. Queries run in a **sandboxed**, read-only DuckDB connection: file and network access are disabled at the engine level (so a prompt can't make the SQL read server files), one statement only, a 15-second timeout and a 100-row cap.
5. The LLM streams a short interpretation, then the exact result is appended as a Markdown table, and the frontend shows the executed SQL.

---

## Tech Stack

| Layer | Technology | Details |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python 3.12, Pydantic v2) | Async REST API & Server-Sent Events (SSE) |
| **Vector Database** | Qdrant Cloud | Isolated per-user collections with Dense + BM25 Sparse & RRF |
| **Dense Embedding** | FastEmbed (`BAAI/bge-small-en-v1.5`) | 384-dimensional dense vectors, local ONNX CPU inference |
| **Sparse Embedding** | FastEmbed (`Qdrant/bm25`) | Lexical BM25 token frequencies, zero external API costs |
| **Cross-Encoder Reranker** | FlashRank (`ms-marco-TinyBERT-L-2-v2`) | Local CPU cross-attention reranker (top-10 to top-5) |
| **Contextual Compression** | Custom Extractive Engine | Distractor sentence pruning with window expansion ($\pm 1$) |
| **Tabular OLAP Engine** | DuckDB | In-process analytical columnar Text-to-SQL |
| **Tier 1 LLM (Primary)** | Groq Cloud LPUs | Sub-second inference, 3+ key round-robin pool, 60s quarantine |
| **Tier 2 LLM (Free, large models)** | NVIDIA NIM (`nemotron-3-ultra-550b`, `minimax-m3.1-flash`) | Free ~40 RPM, OpenAI-compatible; also the recommended eval judge |
| **Tier 3 LLM (High-Throughput)**| Mistral AI (`ministral-3b-2512`) | Multi-key pool for fast generations |
| **Tier 4 LLM (Large-Context)** | Google Gemini Flash (`gemini-flash-latest`) | 1M+ token context window, 3+ key project pool |
| **Tier 5 LLM (Universal)** | OpenRouter (`deepseek-chat`, `llama-3.3-70b`) | Universal fallback tier with multi-key support |
| **Evaluation** | RAGAS 0.4 + custom harness | Judge: Gemini Flash by default (NVIDIA / Mistral / Groq selectable; NVIDIA recommended), multi-key rotation |
| **Observability** | Pydantic Logfire | End-to-end distributed tracing with automatic key masking |
| **Relational Database** | Supabase PostgreSQL | SQLAlchemy 2.0 Async + Alembic migrations |
| **Authentication & Storage** | Supabase Auth & Storage | JWT session auth & document storage |
| **Frontend Framework** | Next.js 16.3 (React 19, Turbopack) | Modern App Router dashboard and streaming chat UI |
| **Styling & UI** | Tailwind CSS v4, Lucide React | Glassmorphism dark interface, responsive Markdown tables |
| **CI / CD Pipeline** | GitHub Actions | Ubuntu runner executing 78 backend unit tests + Next.js build |

---

## Automated Testing & CI/CD

The project maintains a rigorous test suite of **78 unit and integration tests** that execute in under 5 seconds with **zero external API dependencies** (using mocks, local embeddings, and isolated in-memory fixtures):

```bash
cd backend
.\venv\Scripts\pytest -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.1.1, pluggy-1.6.0
collected 78 items

backend/tests/test_chat_title.py ........                                [ 10%]
backend/tests/test_chunking.py .........................                 [ 42%]
backend/tests/test_citation_validation.py ..........                     [ 55%]
backend/tests/test_contextual_compression.py .....                       [ 61%]
backend/tests/test_file_validation.py ........                           [ 71%]
backend/tests/test_hybrid_search.py .....                                [ 78%]
backend/tests/test_llm_fallback.py ........                              [ 88%]
backend/tests/test_llm_key_rotation.py .......                           [ 97%]
backend/tests/test_parsing.py ..                                         [100%]

============================== 78 passed in 4.85s ==============================
```

### Test Suite Directory:
- `tests/test_chat_title.py`: Normalization/cleaning of LLM title outputs (2–5 words), social pleasantry detection, substantive topic synthesis from question + answer, error fallbacks, and session PATCH endpoint.
- `tests/test_llm_key_rotation.py`: Multi-key round-robin rotation, 429 quarantine isolation, cooldown recovery, and all-keys-quarantined graceful fallback.
- `tests/test_llm_fallback.py`: cascading cross-provider fallback (Groq $\rightarrow$ NVIDIA NIM $\rightarrow$ Mistral $\rightarrow$ Gemini $\rightarrow$ OpenRouter); `tests/test_llm_provider_tiers.py`: per-model cooldowns, dead-model skipping, healthy-keys-first ordering for streaming and non-streaming calls, mid-stream failure recovery.
- `tests/test_contextual_compression.py`: Sentence keyword scoring, density bonus, window expansion ($\pm 1$), short-chunk bypass, zero-match fallback, and citation preservation.
- `tests/test_hybrid_search.py`: Dense + BM25 sparse vector generation, Qdrant payload schema indexing, and RRF prefetch validation.
- `tests/test_file_validation.py`: Magic byte binary header identification, macro-enabled file blocking, and spoofed extension defense.
- `tests/test_chunking.py`: Token-bounded sentence splitting, abbreviation preservation (`Dr.`, `e.g.`), and page-boundary-aware chunking.
- `tests/test_citation_validation.py`: Regex claim verification for `[Page X]`, `[Rows Y-Z]`, and `[Summary]` citation tags.

---

## Evaluation Harness

Architecture comparisons and regression checks use the evaluation harness in [`backend/eval/`](backend/eval/README.md). It runs on a hand-verified golden dataset with an isolated eval tenant, and reports per-category and per-document metrics (judge correctness, RAGAS faithfulness / answer relevancy / context precision / context recall, evidence recall, MRR, citations, abstention, routing, Text-to-SQL accuracy, latency) with confidence intervals and paired run-vs-run comparison.

```bash
cd backend
pip install -r requirements-eval.txt
python -m eval run --system vanilla
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Supabase Project (PostgreSQL, Auth, Storage)
- Qdrant Cloud Cluster URL & API Key
- API Key(s) for LLM Tiers:
  - **Groq Cloud API Key(s)** (Primary Tier 1)
  - **NVIDIA NIM API Key(s)** (Tier 2; free at build.nvidia.com; recommended eval judge)
  - **Mistral AI API Key(s)** (Tier 3; optional eval judge)
  - **Google Gemini API Key(s)** (Tier 4 Fallback; default eval judge)
  - **OpenRouter API Key(s)** (Tier 5 Fallback)

---

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Fill in your API credentials:
   ```dotenv
   # ── Supabase & Database ──────────────────────────────────────────────────
   DATABASE_URL=postgresql+asyncpg://postgres.xxx:password@aws-0-region.pooler.supabase.com:6543/postgres
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-service-role-key
   SUPABASE_JWT_SECRET=your-jwt-secret

   # ── Qdrant Cloud ──────────────────────────────────────────────────────────
   QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
   QDRANT_API_KEY=your-qdrant-api-key

   # ── Tier 1: Groq Cloud LPU Pool (Primary) ────────────────────────────────
   GROQ_API_KEY=gsk_key_1
   GROQ_API_KEY_2=gsk_key_2
   GROQ_API_KEY_3=gsk_key_3
   GROQ_MODEL=openai/gpt-oss-120b

   # ── Tier 2: NVIDIA NIM (free, ~40 RPM: https://build.nvidia.com) ────────
   NVIDIA_API_KEY=nvapi-your_key

   # ── Tier 3: Mistral AI (High-Throughput; optional eval judge) ───────────
   MISTRAL_API_KEY=your_mistral_key_1
   MISTRAL_API_KEY_2=your_mistral_key_2
   MISTRAL_API_KEY_3=your_mistral_key_3
   MISTRAL_MODEL=ministral-3b-2512

   # ── Tier 4: Google Gemini Flash (Large Context Fallback) ─────────────────
   GEMINI_API_KEY=AIza_key_1
   GEMINI_API_KEY_2=AIza_key_2
   GEMINI_API_KEY_3=AIza_key_3
   GEMINI_MODEL=gemini-flash-latest

   # ── Tier 4: OpenRouter (Universal Fallback) ──────────────────────────────
   OPENROUTER_API_KEY=sk-or-v1-key_1
   OPENROUTER_API_KEY_2=sk-or-v1-key_2
   OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

   # ── Observability ────────────────────────────────────────────────────────
   LOGFIRE_TOKEN=your_logfire_token_optional
   FRONTEND_URL=http://localhost:3000
   ```

5. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Start the FastAPI backend server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

---

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm ci
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env.local
   ```
   Set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

4. **Start the Next.js development server:**
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
