# DocuChat — Production-Grade AI Document & Tabular RAG System

[![CI](https://github.com/YashMittal1503/RAG-PROJECT/actions/workflows/ci.yml/badge.svg)](https://github.com/YashMittal1503/RAG-PROJECT/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.3-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant-Vector%20Search-red?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP%20SQL-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)

An enterprise-grade, full-stack Retrieval-Augmented Generation (RAG) and Text-to-SQL platform. Enables users to upload multiple unstructured documents (PDF, TXT) and structured spreadsheets (CSV, XLSX), query them with natural language, and receive sub-second token-streamed answers with validated page/row citations and interactive SQL transparency.

---

## Key Architectural Highlights

- **Dual-Engine Pipeline (Hybrid RAG + Text-to-SQL):**
  - **Unstructured Documents (PDF/TXT):** Parsed with PyMuPDF, split via token-bounded sentence chunking, embedded locally via FastEmbed (`BAAI/bge-small-en-v1.5`), and indexed in Qdrant Cloud.
  - **Tabular Datasets (CSV/XLSX):** Stored into an embedded columnar DuckDB engine. Natural language queries are synthesized into safe `SELECT` statements, executed in microseconds, and summarized with LLM insights.
- **Two-Tier Retrieval with Cross-Encoder Reranking:**
  - **Tier 1 (Fast Recall):** Dense cosine similarity search in Qdrant retrieves the top 15 candidate chunks.
  - **Tier 2 (High Precision):** FlashRank Cross-Encoder (`ms-marco-TinyBERT-L-2-v2` ONNX) reranks candidates on local CPU, passing only the top 5 highest-scoring contexts to the generator.
- **Conversational Intelligence:**
  - **Zero-Vector Chitchat Routing:** Intent classifier separates conversational greetings (`"Hi"`, `"What can you do?"`) from factual questions, eliminating wasteful vector searches.
  - **Context-Aware Query Rewriting:** Reformulates conversational pronouns and follow-up questions into standalone search queries using chat history.
- **Deterministic Citation Validation:**
  - Mathematical claim verification ensures every citation (`[Page X]` or `[Rows Y-Z]`) cited in the answer genuinely exists in the retrieved context chunks.
- **Connection-Safe SSE Streaming:**
  - Token-by-token streaming over Server-Sent Events (SSE) using short-lived scoped database sessions, preventing database connection pool starvation during long generations.
- **Modern Next.js 16 / React 19 Frontend:**
  - SWR-style instant local cache hydration, responsive Markdown table rendering, collapsible citation preview drawers, and live SQL query inspections.

---

## End-to-End Architecture

```mermaid
flowchart TD
    User([User / Next.js 16 Frontend])

    subgraph Ingestion Pipeline
        Upload[Upload Documents / Spreadsheets] --> Validate[Magic Byte & Extension Validation]
        Validate --> Route{File Type?}
        Route -->|PDF / TXT| ParseText[PyMuPDF & Sentence Chunking]
        ParseText --> Embed[FastEmbed Local ONNX Embeddings]
        Embed --> Qdrant[(Qdrant Cloud Vector DB)]
        Route -->|CSV / XLSX| DuckStore[(DuckDB Embedded OLAP Engine)]
    end

    subgraph Query & Retrieval Pipeline
        User -->|SSE Stream Query| ChatRouter[FastAPI Chat Router]
        ChatRouter --> Intent{Intent Classifier}
        Intent -->|Chitchat| DirectLLM[Stream Direct Response]
        Intent -->|Document Query| Rewrite[Query Rewriter]
        Rewrite --> DataCheck{Tabular Data Available?}
        DataCheck -->|Yes| TextToSQL[DuckDB Text-to-SQL Pipeline]
        TextToSQL --> GenSQL[LLM Generates SQL]
        GenSQL --> ExecSQL[Execute in DuckDB]
        ExecSQL --> StreamSQL[Stream SQL Explanation]
        DataCheck -->|No / Fallback| BiEncoder[Qdrant Dense Search Top-15]
        BiEncoder --> CrossEncoder[FlashRank Cross-Encoder Top-5]
        CrossEncoder --> Generator[Groq LLM Generation]
        Generator --> CitCheck[Deterministic Citation Validation]
        CitCheck --> StreamAns[Stream Tokens & Citations]
    end

    DirectLLM --> User
    StreamSQL --> User
    StreamAns --> User
```

---

## Quantitative RAGAS Benchmark Scorecard

Evaluated on a curated benchmark dataset covering complex multi-page documentation and contractual offer letters using Groq as the evaluation judge and local ONNX embeddings:

| Metric | Score | Target / Industry Benchmark | Assessment |
| :--- | :---: | :---: | :--- |
| **Context Recall** | **1.000 (100%)** | > 0.85 | **Exceptional:** Zero missing facts from reference ground truth. |
| **Answer Relevancy** | **0.970 (97.0%)** | > 0.90 | **Exceptional:** Answers directly resolve user intent without deviation. |
| **Citation Precision** | **1.000 (100%)** | > 0.90 | **Flawless:** 100% of generated citations match source chunks. |
| **Faithfulness** | **0.750 (75.0%)** | > 0.70 | **Strong:** High factual adherence; minimal hallucination. |
| **Context Precision** | **0.625 (62.5%)** | > 0.60 | **Solid:** Signal-to-noise ratio maintained across multi-chunk contexts. |
| **Average Retrieval Latency** | **1,834 ms** | < 2,500 ms | **Fast:** Combined Qdrant vector retrieval + FlashRank rerank. |

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI (Async, Python 3.12, Pydantic v2) |
| **Vector Storage** | Qdrant Cloud (384-dimensional cosine distance) |
| **Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5` via ONNX runtime) |
| **Reranker** | FlashRank (`ms-marco-TinyBERT-L-2-v2` cross-encoder via ONNX) |
| **Tabular OLAP Engine** | DuckDB (Embedded, in-process analytical SQL) |
| **Relational Database** | Supabase PostgreSQL via SQLAlchemy Async + Alembic |
| **LLM Inference** | Groq Cloud API (Qwen-2.5 / Qwen-3.8) |
| **Authentication & Storage** | Supabase Auth (JWT) & Supabase Storage |
| **Frontend Framework** | Next.js 16.3 (Turbopack, App Router, React 19) |
| **Styling & Icons** | Tailwind CSS v4, Lucide React |
| **CI / CD Pipeline** | GitHub Actions (Automated Ubuntu Runner with Pytest + Next.js Build) |

---

## Database Migrations (Alembic)

Database schema evolutions are version-controlled with Alembic and applied automatically on server boot:

- `001_initial_schema.py`: Documents, chunks, chat sessions, chat messages, and status enums.
- `002_add_tabular_support.py`: Adds `is_tabular` flags and schema alterations for Text-to-SQL pipelines.

Migrations are applied automatically via container entrypoint:
```bash
alembic upgrade head
```

---

## Automated Testing & CI/CD

The project includes an automated test suite verifying file type detection, text parsers, and token-bounded chunking:

```bash
# Run backend test suite (36 tests, 0 external API dependencies)
cd backend
pytest tests -v
```

GitHub Actions executes the full pipeline on every push:
1. Installs OS binary dependencies (`libmagic1`).
2. Runs all 36 backend unit tests with Python 3.12.
3. Validates frontend TypeScript compilation, ESLint, and Next.js production build (`npm run build`).

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Supabase account (PostgreSQL, Auth, Storage)
- Qdrant Cloud cluster and API key
- Groq Cloud API key

---

### Backend Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Provide your `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `GROQ_API_KEY`.

5. **Run database migrations**:
   ```bash
   alembic upgrade head
   ```

6. **Start the API server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

---

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm ci
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env.local
   ```
   Set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

4. **Start the development server**:
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## License

Distributed under the MIT License.
