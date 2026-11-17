"""
Evaluation corpus: a YAML manifest of documents, and ingestion into an isolated
eval tenant (its own Qdrant collection), so evaluation never touches real users' data.

Manifest format (paths are relative to the manifest file):

    documents:
      - id: gatsby                      # stable id used by GoldItem.docs / evidence.doc
        title: The Great Gatsby
        path: corpus/the_great_gatsby.txt
        url: https://...                # optional: downloaded if `path` is missing
        sha256: <hex>                   # optional: verified after download / before use
        doc_type: novel                 # free-form: novel, paper, contract, report...

Ingestion reuses the app's own parsing, chunking and embedding code, writing the
same Qdrant payload shape as app/services/ingestion.py, but skips Supabase
Storage and Postgres.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Fixed namespace so doc uuids are stable across machines and runs.
_DOC_NAMESPACE = uuid.UUID("6f1c2a4e-3b0d-4c55-9a8e-5d7e0b1f2a33")
DEFAULT_EVAL_USER_ID = "00000000-0000-4000-8000-0000000e7a11"
SUPPORTED_TYPES = {"pdf", "txt"}


def eval_user_id() -> str:
    """The tenant id used for evaluation. Override with EVAL_USER_ID; never point it at a real user."""
    return os.environ.get("EVAL_USER_ID", DEFAULT_EVAL_USER_ID)


@dataclass
class CorpusDoc:
    id: str
    title: str
    path: Path
    url: str | None = None
    sha256: str | None = None
    doc_type: str = "document"

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def file_type(self) -> str:
        return self.path.suffix.lower().lstrip(".")

    @property
    def uuid(self) -> str:
        return str(uuid.uuid5(_DOC_NAMESPACE, self.id))


def load_manifest(path: str | Path) -> list[CorpusDoc]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    docs = []
    for d in data.get("documents", []):
        doc = CorpusDoc(
            id=d["id"],
            title=d.get("title", d["id"]),
            path=(path.parent / d["path"]).resolve(),
            url=d.get("url"),
            sha256=d.get("sha256"),
            doc_type=d.get("doc_type", "document"),
        )
        if doc.file_type not in SUPPORTED_TYPES:
            raise ValueError(f"{doc.id}: unsupported file type '{doc.file_type}' (supported: {sorted(SUPPORTED_TYPES)})")
        docs.append(doc)
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate document ids in {path}")
    return docs


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ensure_files(docs: list[CorpusDoc]) -> None:
    """Download missing files that have a URL, and verify checksums."""
    for doc in docs:
        if not doc.path.exists():
            if not doc.url:
                raise FileNotFoundError(f"{doc.id}: {doc.path} is missing and has no url to download from")
            import httpx

            logger.info(f"Downloading {doc.id} from {doc.url}")
            doc.path.parent.mkdir(parents=True, exist_ok=True)
            resp = httpx.get(doc.url, follow_redirects=True, timeout=120)
            resp.raise_for_status()
            doc.path.write_bytes(resp.content)
        if doc.sha256 and _sha256(doc.path) != doc.sha256:
            raise ValueError(f"{doc.id}: sha256 mismatch for {doc.path} (file changed; update the manifest on purpose)")


def read_pages(doc: CorpusDoc) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs exactly as the app's parsers produce them."""
    from app.services.parsing import parse_pdf, parse_txt

    raw = doc.path.read_bytes()
    if doc.file_type == "pdf":
        return [(p.page_number, p.text) for p in parse_pdf(raw)]
    return [(1, parse_txt(raw))]


def doc_texts(docs: list[CorpusDoc]) -> dict[str, str]:
    """Full text per doc id, for validating evidence quotes."""
    return {d.id: "\n\n".join(t for _, t in read_pages(d)) for d in docs}


def corpus_hash(docs: list[CorpusDoc]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.id):
        h.update(d.id.encode())
        h.update(_sha256(d.path).encode())
    return h.hexdigest()[:12]


def _state_file() -> Path:
    return Path(__file__).resolve().parent / "runs" / ".index_state.json"


def _index_signature(docs: list[CorpusDoc], user_id: str) -> str:
    from app.config import settings
    from app.services import chunking

    # Re-index when the corpus, the embedding model or the chunking parameters change.
    parts = [
        corpus_hash(docs),
        user_id,
        settings.embedding_model,
        str(chunking.TARGET_CHUNK_TOKENS),
        str(chunking.MAX_CHUNK_TOKENS),
        str(chunking.OVERLAP_FRACTION),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


async def index_corpus(docs: list[CorpusDoc], user_id: str | None = None, force: bool = False) -> dict:
    """Parse, chunk, embed and upsert every document into the eval tenant's collection."""
    from app.config import settings
    from app.services import embedding, vector_store
    from app.services.chunking import chunk_text
    from app.services.parsing import PageText

    user_id = user_id or eval_user_id()
    ensure_files(docs)
    signature = _index_signature(docs, user_id)
    state_path = _state_file()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not force and state.get(user_id) == signature:
        logger.info(f"Eval corpus already indexed for tenant {user_id} (signature {signature}); skipping.")
        return {"skipped": True, "signature": signature}

    await vector_store.ensure_collection(user_id, force_recreate=True)
    totals: dict[str, int] = {}
    for doc in docs:
        pages = [PageText(page_number=n, text=t) for n, t in read_pages(doc)]
        chunks = chunk_text(pages, filename=doc.filename)
        if not chunks:
            raise ValueError(f"{doc.id}: no text extracted")
        # Stable chunk ids so repeated indexing of the same corpus produces the same points.
        for c in chunks:
            c.id = str(uuid.uuid5(_DOC_NAMESPACE, f"{doc.id}:{c.chunk_index}"))

        bs = settings.embed_batch_size
        for start in range(0, len(chunks), bs):
            batch = chunks[start : start + bs]
            texts = [c.content for c in batch]
            dense = embedding.embed_texts(texts)
            sparse = embedding.embed_sparse_texts(texts)
            points = [
                {
                    "id": c.id,
                    "vector": v,
                    "sparse_vector": sv,
                    "payload": {
                        "doc_id": doc.uuid,
                        "chunk_type": c.chunk_type,
                        "filename": doc.filename,
                        "page_number": c.page_number,
                        "row_range_start": c.row_range_start,
                        "row_range_end": c.row_range_end,
                        "content": c.content,
                    },
                }
                for c, v, sv in zip(batch, dense, sparse)
            ]
            await vector_store.upsert_chunks(user_id, points)
        totals[doc.id] = len(chunks)
        logger.info(f"Indexed {doc.id}: {len(chunks)} chunks")

    state[user_id] = signature
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    return {"skipped": False, "signature": signature, "chunks": totals}
