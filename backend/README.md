# RAG Chatbot — Backend

FastAPI backend for document Q&A with streaming answers and source citations.

## Tech Stack

- **Framework**: FastAPI (async)
- **Database**: PostgreSQL (Supabase) via SQLAlchemy + Alembic
- **Vector DB**: Qdrant Cloud (384-dim, cosine similarity)
- **Embeddings**: FastEmbed (ONNX) — BAAI/bge-small-en-v1.5
- **LLM**: Groq API — llama-3.3-70b-versatile
- **Storage**: Supabase Storage (raw uploaded files)
- **Auth**: Supabase Auth (JWT verification)

## Local Development

```bash
# 1. Create a virtual environment
python -m venv venv 
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your credentials

# 4. Run database migrations
alembic upgrade head

# 5. Start the dev server
uvicorn app.main:app --reload --port 8000
```

## Running Tests

```bash
pytest tests/ -v
```

## Deploy to Render

1. Push this `backend/` directory to a Git repository
2. Create a **Web Service** on [render.com](https://render.com)
3. Connect the repo, select **Docker** environment
4. Set instance type to **Free**
5. Add all environment variables from `.env.example`
6. Render will build the Docker image and deploy automatically

The app binds to `$PORT` (provided by Render) automatically.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/documents/upload` | Upload documents |
| GET | `/api/documents` | List user's documents |
| GET | `/api/documents/{id}/status` | Poll document status |
| DELETE | `/api/documents/{id}` | Delete a document |
| POST | `/api/chat/sessions` | Create chat session |
| GET | `/api/chat/sessions` | List chat sessions |
| GET | `/api/chat/sessions/{id}/messages` | Get messages |
| POST | `/api/chat/sessions/{id}/query` | Query (SSE stream) |
