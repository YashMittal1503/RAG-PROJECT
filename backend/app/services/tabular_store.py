"""
DuckDB-based tabular data store for the Text-to-SQL pipeline.

When users upload spreadsheets (XLSX/CSV), the parsed DataFrames are
persisted into a per-user DuckDB file instead of being chunked and
embedded.  At query time the LLM generates SQL, which is executed here.

Storage layout:
    backend/data/tabular/{user_id}.duckdb   — one file per user

Table naming:
    t_{doc_id_short}_{sheet_sanitized}
    e.g. t_a1b2c3d4_Sheet1
"""

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ── Storage root ──────────────────────────────────────────────────────────
# Relative to the backend working directory
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tabular"


def _ensure_data_dir() -> None:
    """Create the tabular data directory if it doesn't exist."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _db_path(user_id: str) -> str:
    """Return the DuckDB file path for a given user."""
    _ensure_data_dir()
    safe_id = user_id.replace("-", "_")
    return str(_DATA_DIR / f"{safe_id}.duckdb")


def _sanitize_name(name: str) -> str:
    """Sanitize a sheet/table name to be a valid SQL identifier."""
    # Replace non-alphanumeric chars with underscores, collapse multiples
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized.lower() or "sheet"


def _table_prefix(doc_id: uuid.UUID) -> str:
    """Short prefix derived from the doc UUID (first 8 hex chars)."""
    return str(doc_id).replace("-", "")[:8]


def _table_name(doc_id: uuid.UUID, sheet_name: str) -> str:
    """Full table name for a specific sheet of a document."""
    return f"t_{_table_prefix(doc_id)}_{_sanitize_name(sheet_name)}"


# ── Public API ────────────────────────────────────────────────────────────

def store_dataframe(
    user_id: str,
    doc_id: uuid.UUID,
    sheet_name: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Persist a DataFrame into the user's DuckDB as a named table.

    Returns metadata about the stored table:
        {"table_name": ..., "rows": ..., "columns": [...]}
    """
    table = _table_name(doc_id, sheet_name)
    db_file = _db_path(user_id)

    # Sanitize column names — DuckDB is flexible but let's be safe
    clean_cols = []
    seen: set[str] = set()
    for col in df.columns:
        c = re.sub(r"[^a-zA-Z0-9_]", "_", str(col)).strip("_") or "col"
        # Deduplicate
        base = c
        i = 1
        while c in seen:
            c = f"{base}_{i}"
            i += 1
        seen.add(c)
        clean_cols.append(c)
    df = df.copy()
    df.columns = clean_cols

    conn = duckdb.connect(db_file)
    try:
        # Drop if exists (idempotent re-upload)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        # Register the DataFrame and create table from it
        conn.register("__df", df)
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM __df")
        conn.unregister("__df")
        conn.execute("CHECKPOINT")  # Flush to disk
        logger.info(f"Stored {len(df)} rows into {table} for user {user_id[:8]}…")
    finally:
        conn.close()

    return {
        "table_name": table,
        "rows": len(df),
        "columns": clean_cols,
    }


def get_table_schema(user_id: str, doc_id: uuid.UUID | None = None) -> str:
    """
    Return a human-readable schema description for the LLM prompt.

    If doc_id is provided, only returns schema for that document's tables.
    Otherwise returns schema for ALL tables in the user's DuckDB.

    Format:
        Table: t_a1b2c3d4_sheet1 (245 rows)
        Columns:
          - id (INTEGER)
          - name (VARCHAR)
          - revenue (DOUBLE)
        Sample rows (first 3):
          | 1 | Alice | 50000.0 |
          | 2 | Bob   | 62000.0 |
    """
    db_file = _db_path(user_id)
    if not os.path.exists(db_file):
        return ""

    conn = duckdb.connect(db_file, read_only=True)
    try:
        # Get all tables
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        # Filter by doc_id if provided
        if doc_id is not None:
            prefix = f"t_{_table_prefix(doc_id)}_"
            table_names = [t for t in table_names if t.startswith(prefix)]

        if not table_names:
            return ""

        schema_parts: list[str] = []

        for tname in table_names:
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]

            # Get column info
            cols = conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{tname}' ORDER BY ordinal_position"
            ).fetchall()

            col_lines = "\n".join(f"  - {c[0]} ({c[1]})" for c in cols)

            # Get sample rows (first 3)
            sample = conn.execute(f"SELECT * FROM {tname} LIMIT 3").fetchall()
            sample_lines = ""
            if sample:
                sample_lines = "\nSample rows (first 3):\n" + "\n".join(
                    "  | " + " | ".join(str(v) for v in row) + " |"
                    for row in sample
                )

            schema_parts.append(
                f"Table: {tname} ({row_count} rows)\n"
                f"Columns:\n{col_lines}{sample_lines}"
            )

        return "\n\n".join(schema_parts)

    finally:
        conn.close()


def execute_sql(user_id: str, sql: str) -> dict[str, Any]:
    """
    Execute a SELECT query against the user's DuckDB.

    Returns:
        {
            "columns": ["col1", "col2", ...],
            "rows": [[val1, val2, ...], ...],
            "row_count": N,
            "truncated": bool
        }

    Raises ValueError for non-SELECT queries or execution errors.
    """
    # Safety: only allow SELECT statements
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    # Block dangerous statements that could be hidden in subqueries
    upper = cleaned.upper()
    for forbidden in ("DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"):
        # Check for the keyword at word boundaries (not inside identifiers)
        if re.search(rf"\b{forbidden}\b", upper):
            raise ValueError(f"Query contains forbidden keyword: {forbidden}")

    db_file = _db_path(user_id)
    if not os.path.exists(db_file):
        raise ValueError("No tabular data found. Please upload a spreadsheet first.")

    # Convert any strict CAST(...) to TRY_CAST(...) to gracefully handle dirty data or blank spaces
    cleaned = re.sub(r"\bCAST\s*\(", "TRY_CAST(", cleaned, flags=re.IGNORECASE)

    conn = duckdb.connect(db_file, read_only=True)
    try:
        result = conn.execute(cleaned)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        # Cap output to 100 rows to prevent oversized LLM context
        truncated = len(rows) > 100
        if truncated:
            rows = rows[:100]

        return {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }
    except duckdb.Error as e:
        raise ValueError(f"SQL execution error: {e}")
    finally:
        conn.close()


def delete_tables(user_id: str, doc_id: uuid.UUID) -> None:
    """
    Drop all tables associated with a document from the user's DuckDB.
    Called when a document is deleted.
    """
    db_file = _db_path(user_id)
    if not os.path.exists(db_file):
        return

    prefix = f"t_{_table_prefix(doc_id)}_"

    conn = duckdb.connect(db_file)
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()

        dropped = 0
        for (tname,) in tables:
            if tname.startswith(prefix):
                conn.execute(f"DROP TABLE IF EXISTS {tname}")
                dropped += 1

        if dropped:
            conn.execute("CHECKPOINT")
            logger.info(f"Dropped {dropped} table(s) for doc {doc_id} from user {user_id[:8]}…")
    finally:
        conn.close()


def list_tables(user_id: str) -> list[str]:
    """List all table names in the user's DuckDB."""
    db_file = _db_path(user_id)
    if not os.path.exists(db_file):
        return []

    conn = duckdb.connect(db_file, read_only=True)
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        return [t[0] for t in tables]
    finally:
        conn.close()


def has_tabular_data(user_id: str) -> bool:
    """Check if the user has any tabular data stored in DuckDB."""
    return len(list_tables(user_id)) > 0
