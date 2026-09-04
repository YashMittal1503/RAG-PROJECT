"""
Supabase Storage service for uploading and downloading document files.

Files are stored in a private Supabase Storage bucket named "documents".
Each file is stored under the user's UUID to enforce data isolation:
    documents/{user_id}/{doc_id}/{filename}
"""

import logging
from uuid import UUID

from supabase import create_client

from app.config import settings

logger = logging.getLogger(__name__)

# The Supabase client is created once and reused.
# We use the service-role key (not the anon key) because the backend
# needs full access to Storage regardless of RLS policies.
_supabase = create_client(settings.supabase_url, settings.supabase_key)

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
    storage_path = f"{user_id}/{doc_id}/{filename}"

    try:
        _supabase.storage.from_(BUCKET_NAME).upload(
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
        response = _supabase.storage.from_(BUCKET_NAME).download(storage_path)
        return response
    except Exception as e:
        logger.error(f"Failed to download {storage_path}: {e}")
        raise RuntimeError(f"Failed to download file from storage: {e}")


async def delete_file(storage_path: str) -> None:
    """Delete a file from Supabase Storage."""
    try:
        _supabase.storage.from_(BUCKET_NAME).remove([storage_path])
    except Exception as e:
        # Log but don't raise — if the file is already gone, that's fine
        logger.warning(f"Failed to delete {storage_path}: {e}")
