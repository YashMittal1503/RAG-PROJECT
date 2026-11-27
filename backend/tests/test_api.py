"""
API surface smoke tests against the real FastAPI app (app.main), without its startup hooks: no embedding
model, database, Supabase or Qdrant. Catches broken imports, missing auth on a route, and upload validation
regressions before they ship.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.auth import get_current_user
from app.database import get_db
from app.main import app

USER = "11111111-2222-4333-8444-555555555555"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)   # no lifespan: startup (model load, DB recovery) doesn't run
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def signed_in():
    async def _refresh(doc):   # what the database's column defaults fill in on insert
        if doc.is_tabular is None:
            doc.is_tabular = False

    db = MagicMock(add=MagicMock(), commit=AsyncMock(), refresh=AsyncMock(side_effect=_refresh))

    async def _db():
        yield db

    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_db] = _db
    return db


@pytest.mark.anyio
async def test_health_needs_no_auth(client):
    r = await client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_every_data_route_requires_a_token(client):
    """Any route under /api other than health must refuse a request with no Authorization header."""
    sample = {"session_id": "00000000-0000-4000-8000-000000000001", "doc_id": "00000000-0000-4000-8000-000000000002"}
    checked = 0
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api") or path == "/api/health":
            continue
        for method in sorted(getattr(route, "methods", None) or []):
            if method in ("HEAD", "OPTIONS"):
                continue
            r = await client.request(method, path.format(**sample))
            assert r.status_code in (401, 403), f"{method} {path} answered {r.status_code} without a token"
            checked += 1
    assert checked >= 8   # the document and chat routes were actually exercised


@pytest.mark.anyio
async def test_upload_rejects_disguised_and_oversized_files_before_storing_them(client, signed_in, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_file_size_mb", 1)
    files = [
        ("files", ("notes.pdf", b"MZ\x90\x00 this is a windows executable", "application/pdf")),   # wrong magic bytes
        ("files", ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain")),
    ]
    with patch("app.routers.documents.storage.upload_file", new=AsyncMock()) as upload, \
         patch("app.routers.documents.ingest_document", new=MagicMock()) as ingest:
        r = await client.post("/api/documents/upload", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] == [] and {e["filename"] for e in body["errors"]} == {"notes.pdf", "big.txt"}
    assert "1MB" in next(e["error"] for e in body["errors"] if e["filename"] == "big.txt")
    upload.assert_not_awaited()
    ingest.assert_not_called()
    signed_in.add.assert_not_called()


@pytest.mark.anyio
async def test_upload_rejects_too_many_files(client, signed_in, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_batch_size", 2)
    files = [("files", (f"f{i}.txt", b"hello", "text/plain")) for i in range(3)]
    r = await client.post("/api/documents/upload", files=files)
    assert r.status_code == 400 and "Maximum 2 files" in r.json()["detail"]


@pytest.mark.anyio
async def test_valid_upload_is_stored_under_the_callers_user_id_and_queued(client, signed_in):
    with patch("app.routers.documents.storage.upload_file", new=AsyncMock(return_value="u/doc/notes.txt")) as upload, \
         patch("app.routers.documents.ingest_document", new=AsyncMock()) as ingest:
        r = await client.post("/api/documents/upload", files=[("files", ("notes.txt", b"plain text notes", "text/plain"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["errors"] == [] and body["documents"][0]["filename"] == "notes.txt"
    assert body["documents"][0]["status"] == "queued"
    assert upload.await_args.kwargs["user_id"] == USER          # files land in the signed-in user's own folder
    assert ingest.call_args.kwargs["user_id"] == USER
    signed_in.commit.assert_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("filename, stored_as", [
    ("../../22222222-0000-4000-8000-000000000000/report.pdf", "report.pdf"),
    (r"..\..\other\report.pdf", "report.pdf"),   # Windows-style separators
    ("/etc/passwd", "passwd"),
    ("..", "upload"),
    ("notes.txt", "notes.txt"),
])
async def test_storage_path_stays_inside_the_users_folder(filename, stored_as):
    import uuid

    from app.services import storage

    bucket = MagicMock()
    doc = uuid.UUID("00000000-0000-4000-8000-0000000000aa")
    with patch.object(storage, "_storage", return_value=bucket):
        path = await storage.upload_file(USER, doc, filename, b"x", "text/plain")
    assert path == f"{USER}/{doc}/{stored_as}"
