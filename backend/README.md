# DocTalk — Backend

FastAPI service for DocTalk: document ingestion, hybrid retrieval, Text-to-SQL over spreadsheets, multi-provider LLM orchestration and the evaluation harness. See the [root README](../README.md) for the architecture.

## Layout

| Path | Contents |
|---|---|
| `app/main.py` | App factory, lifespan (DB/auth warm-up, stale-job recovery), Logfire, CORS |
| `app/routers/` | `documents` (upload/list/status/retry/delete), `chat` (sessions + SSE query), `health` |
| `app/services/query.py` | Query classification, document routing, query rewriting, retrieval, contextual compression, answer streaming, citation checks |
| `app/services/llm_provider.py` | Multi-key, multi-provider LLM fallback (Groq → NVIDIA NIM → Mistral → Gemini → OpenRouter) |
| `app/services/tabular_qa.py`, `tabular_store.py` | Text-to-SQL planning and the sandboxed DuckDB store |
| `app/services/ingestion.py`, `parsing.py`, `chunking.py` | Parse (PyMuPDF / pymupdf4llm / RapidOCR) → chunk → embed → index |
| `app/services/embedding.py`, `vector_store.py`, `reranker.py` | FastEmbed dense + BM25, Qdrant hybrid search with RRF, FlashRank reranking |
| `app/services/doc_profiles.py` | Distinctive-term profiles used to route questions to the right document |
| `eval/` | Evaluation harness ([README](eval/README.md)) |
| `tests/` | Pytest suite, no network or API keys needed |

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env           # fill in Supabase, Qdrant and at least one LLM provider key
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>.

## Tests

```bash
pytest tests -q
```

## Docker

```bash
docker build -t doctalk-backend .
docker run --env-file .env -p 8000:10000 doctalk-backend   # the container listens on $PORT (default 10000)
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check and model-load status |
| POST | `/api/documents/upload` | Upload up to 10 files (multipart) |
| GET | `/api/documents` | List the user's documents |
| GET | `/api/documents/{id}/status` | Poll ingestion status |
| POST | `/api/documents/{id}/retry` | Re-run ingestion for a failed document |
| DELETE | `/api/documents/{id}` | Delete a document, its vectors, tables and stored file |
| POST / GET | `/api/chat/sessions` | Create / list chat sessions |
| GET / PATCH / DELETE | `/api/chat/sessions/{id}` | Get / rename / delete a session |
| GET | `/api/chat/sessions/{id}/messages` | Message history |
| POST | `/api/chat/sessions/{id}/query` | Ask a question (SSE stream: `token`, `sql_query`, `citations`, `title`, `done`, `error`) |
