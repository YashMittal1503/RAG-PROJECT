"""
File type validation using magic bytes (file signatures).

Validates uploaded files by reading their binary headers, not just
their file extension, which can be trivially spoofed.
"""

import magic

# Mapping of allowed MIME types to our internal file type names
ALLOWED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/csv": "csv",
    # XLSX files are ZIP containers — python-magic identifies the specific Office type
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    # Some systems identify XLSX as generic zip first; we handle this in detection
    "application/zip": "xlsx",
}

# Explicitly blocked MIME types (macro-enabled formats)
BLOCKED_MIME_TYPES: set[str] = {
    "application/vnd.ms-excel.sheet.macroEnabled.12",
    "application/vnd.ms-excel.sheet.macroenabled.12",
}

# Blocked extensions as a secondary check
BLOCKED_EXTENSIONS: set[str] = {".xlsm", ".xlsb", ".xltm", ".docm"}


def validate_file_type(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Validate a file's type by its magic bytes (binary header).

    Args:
        file_bytes: The raw file content (or at least the first 2048 bytes).
        filename:   The original filename (used only for extension checks).

    Returns:
        A tuple of (file_type, mime_type), e.g. ("pdf", "application/pdf").

    Raises:
        ValueError: If the file type is not allowed or is explicitly blocked.
    """
    # Check extension against blocklist first (fast reject)
    lower_name = filename.lower()
    for ext in BLOCKED_EXTENSIONS:
        if lower_name.endswith(ext):
            raise ValueError(
                f"Macro-enabled files ({ext}) are not supported for security reasons. "
                f"Please save as a standard .xlsx format and re-upload."
            )

    # Detect MIME type from file content (reads magic bytes)
    # We use from_buffer because we already have the bytes in memory
    mime_type = magic.from_buffer(file_bytes[:2048], mime=True)

    # Check against blocklist
    if mime_type in BLOCKED_MIME_TYPES:
        raise ValueError(
            "Macro-enabled spreadsheet files are not supported for security reasons. "
            "Please save as a standard .xlsx format and re-upload."
        )

    # Handle the ambiguous zip/xlsx case:
    # XLSX files are technically ZIP containers. python-magic sometimes
    # identifies them as "application/zip". If the extension is .xlsx
    # and magic says it's a zip, we accept it as xlsx.
    if mime_type == "application/zip" and lower_name.endswith(".xlsx"):
        return "xlsx", mime_type

    # For CSV: python-magic often misidentifies CSV as text/plain.
    # If the extension is .csv and magic says text/plain, treat as CSV.
    if mime_type == "text/plain" and lower_name.endswith(".csv"):
        return "csv", mime_type

    # Look up the MIME type in our allowed list
    file_type = ALLOWED_MIME_TYPES.get(mime_type)
    if file_type is None:
        raise ValueError(
            f"Unsupported file type: {mime_type}. "
            f"We accept PDF, TXT, CSV, and XLSX files only."
        )

    return file_type, mime_type
