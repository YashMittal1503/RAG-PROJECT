"""
Supabase Storage service for uploading and downloading document files.

Files are stored in a private Supabase Storage bucket named "documents".
Each file is stored under the user's UUID to enforce data isolation:
    documents/{user_id}/{doc_id}/{filename}
"""

import asyncio
import logging
from uuid import UUID

from supabase import create_client

from app.config import settings

logger = logging.getLogger(__name__)

# The Supabase client is created on first use and reused. (Not at import: importing the app must not need
# valid credentials, or tests and tooling can't load it.) We use the service-role key (not the anon key)
# because the backend needs full access to Storage regardless of RLS policies.
_supabase_client = None


def _storage():
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(settings.supabase_url, settings.supabase_key)
    return _supabase_client.storage.from_(BUCKET_NAME)

BUCKET_NAME = "documents"


async def upload_file(
    user_id: str,
    doc_id: UUID,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> str:
    """
    Upload a file to Supabase Storage.

    Returns the storage path (relative to the bucket root).
    """
    # Keep only the base name: a client-supplied "../../<other user>/x.pdf" must not escape this user's folder
    safe_name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip().lstrip(".") or "upload"
    storage_path = f"{user_id}/{doc_id}/{safe_name}"

    try:
        await asyncio.to_thread(
            _storage().upload,
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    except Exception as e:
        logger.error(f"Failed to upload {storage_path}: {e}")
        raise RuntimeError(f"Failed to upload file to storage: {e}")

    return storage_path


async def download_file(storage_path: str) -> bytes:
    """
    Download a file from Supabase Storage.

    Returns the raw file bytes.
    """
    try:
        response = await asyncio.to_thread(
            _storage().download,
            storage_path,
        )
        return response
    except Exception as e:
        logger.error(f"Failed to download {storage_path}: {e}")
        raise RuntimeError(f"Failed to download file from storage: {e}")


async def delete_file(storage_path: str) -> None:
    """Delete a file from Supabase Storage."""
    try:
        await asyncio.to_thread(
            _storage().remove,
            [storage_path],
        )
    except Exception as e:
        # Log but don't raise — if the file is already gone, that's fine
        logger.warning(f"Failed to delete {storage_path}: {e}")
