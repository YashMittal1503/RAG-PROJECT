"""
DuckDB-based tabular data store for the Text-to-SQL pipeline.

When users upload spreadsheets (XLSX/CSV), the parsed DataFrames are
persisted into a per-user DuckDB file instead of being chunked and
embedded.  At query time the LLM generates SQL, which is executed here.

Storage layout:
    backend/data/tabular/{user_id}.duckdb   — one file per user

Table naming:
    t_{doc_id_short}_{sheet_sanitized}
    e.g. t_a1b2c3d4_sheet1

Security model for LLM-written SQL (execute_sql):
  1. The connection is read-only AND sandboxed: enable_external_access=false blocks
     every file/network function (read_text, read_csv, COPY, ATTACH, INSTALL, httpfs...)
     and lock_configuration=true stops a query from switching that back on. Without
     this, a prompt like "run SELECT * FROM read_text('.env')" leaks server secrets.
  2. Exactly one statement, starting with SELECT or WITH (CTEs are allowed).
  3. A keyword deny-list as defence in depth, checked with string literals removed,
     so WHERE status = 'Deleted' is not mistaken for a DELETE.
  4. A wall-clock timeout (conn.interrupt) and a row cap read with fetchmany, so a
     runaway query can't hang a worker or load millions of rows into memory.
"""

import logging
import math
import os
import re
import threading
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

# ── Storage root ──────────────────────────────────────────────────────────
# Relative to the backend working directory
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tabular"

MAX_RESULT_ROWS = 100          # rows returned to the caller (and shown in the UI)
QUERY_TIMEOUT_S = 15.0         # LLM-written queries are interrupted after this
CATEGORY_MAX_DISTINCT = 15     # text columns with at most this many values list them in the schema
SAMPLE_ROWS = 3
MAX_SCHEMA_COLUMNS = 60        # very wide sheets: describe the first N columns

# Statement keywords only: the sandboxed read-only connection is the real protection, and
# broader lists block legitimate SQL (the replace() function, a column named "call"...).
_FORBIDDEN = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "INSTALL", "PRAGMA",
)


def _q(identifier: str) -> str:
    """Double-quote an identifier (tables created before sanitization changes may need it)."""
    return '"' + str(identifier).replace('"', '""') + '"'


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


def _sanitize_column(name: Any) -> str:
    """Column name the LLM can use unquoted: letters, digits, underscores, not starting with a digit."""
    c = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    c = re.sub(r"_+", "_", c).strip("_") or "col"
    if c[0].isdigit():
        c = f"col_{c}"
    return c


def _table_prefix(doc_id: uuid.UUID) -> str:
    """Short prefix derived from the doc UUID (first 8 hex chars)."""
    return str(doc_id).replace("-", "")[:8]


def _table_name(doc_id: uuid.UUID, sheet_name: str) -> str:
    """Full table name for a specific sheet of a document."""
    return f"t_{_table_prefix(doc_id)}_{_sanitize_name(sheet_name)}"


def _connect(db_file: str, read_only: bool = False, sandboxed: bool = False) -> duckdb.DuckDBPyConnection:
    """Open DuckDB connection with configured memory limits and thread allocation.

    sandboxed=True (for anything that runs or describes LLM-written SQL) disables all
    file and network access and locks the configuration.
    """
    config = {
        "memory_limit": settings.duckdb_memory_limit,
        "threads": str(settings.duckdb_threads),
    }
    if sandboxed:
        config["enable_external_access"] = False
        config["lock_configuration"] = True
    return duckdb.connect(db_file, read_only=read_only, config=config)


# ── Value formatting & Markdown tables ────────────────────────────────────

def json_safe(v: Any) -> Any:
    """Convert DuckDB values (Decimal, date, NaN...) into JSON-friendly Python values."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v)


def format_value(v: Any) -> str:
    """Human-readable cell: thousands separators, at most 4 decimals, '—' for NULL."""
    v = json_safe(v)
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return f"{int(v):,}"
        text = f"{v:,.4f}".rstrip("0").rstrip(".")
        return text if text not in ("-0", "") else "0"
    return str(v)


def _cell(v: Any) -> str:
    """Markdown-safe cell: escape pipes, flatten newlines, cap very long text."""
    s = format_value(v).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return s if len(s) <= 120 else s[:117] + "..."


def markdown_table(columns: list[str], rows: list[list], max_rows: int | None = None) -> str:
    """GitHub-flavoured Markdown table (header + separator line), renderable by the chat UI."""
    shown = rows if max_rows is None else rows[:max_rows]
    numeric = [
        all(isinstance(json_safe(r[i]), (int, float)) and not isinstance(r[i], bool) for r in shown if r[i] is not None)
        and any(r[i] is not None for r in shown)
        for i in range(len(columns))
    ]
    header = "| " + " | ".join(_cell(c) for c in columns) + " |"
    sep = "| " + " | ".join("---:" if n else "---" for n in numeric) + " |"
    body = ["| " + " | ".join(_cell(v) for v in r) + " |" for r in shown]
    return "\n".join([header, sep, *body])


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$")


def _coerce_iso_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns that hold ISO dates (2025-03-01, 2025-03-01 14:30) into real dates,
    so SQL can filter and compare them. Ambiguous formats like 03/04/2025 are left as text:
    day/month order can't be guessed safely."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        values = df[col].dropna().astype(str).str.strip()
        if values.empty or not values.map(lambda v: bool(_ISO_DATE.match(v))).all():
            continue
        parsed = pd.to_datetime(values, errors="coerce", format="ISO8601")
        if parsed.notna().all():
            df[col] = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
            if (df[col].dropna().dt.normalize() == df[col].dropna()).all():
                # No time part anywhere: store as DATE rather than TIMESTAMP (see store_dataframe).
                df.attrs.setdefault("date_columns", []).append(col)
    return df


# ── Public API ────────────────────────────────────────────────────────────

def store_dataframe(
    user_id: str,
    doc_id: uuid.UUID,
    sheet_name: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Persist a DataFrame into the user's DuckDB as a named table.

    Column names are sanitized into plain SQL identifiers; when that changes a name,
    the original spreadsheet header is kept as a column comment and shown in the schema.

    Returns metadata about the stored table:
        {"table_name": ..., "rows": ..., "columns": [...]}
    """
    table = _table_name(doc_id, sheet_name)
    db_file = _db_path(user_id)

    clean_cols: list[str] = []
    seen: set[str] = set()
    for col in df.columns:
        c = base = _sanitize_column(col)
        i = 1
        while c.lower() in seen:
            c = f"{base}_{i}"
            i += 1
        seen.add(c.lower())
        clean_cols.append(c)
    originals = {c: str(o) for c, o in zip(clean_cols, df.columns) if str(o) != c}
    df = df.copy()
    df.columns = clean_cols
    df = _coerce_iso_dates(df)
    date_cols = set(df.attrs.get("date_columns", []))

    conn = _connect(db_file)
    try:
        # Drop if exists (idempotent re-upload)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        # Register the DataFrame and create table from it
        conn.register("__df", df)
        select = ", ".join(
            f"CAST({_q(c)} AS DATE) AS {_q(c)}" if c in date_cols else _q(c) for c in df.columns
        )
        conn.execute(f"CREATE TABLE {table} AS SELECT {select} FROM __df")
        conn.unregister("__df")
        for col, original in originals.items():
            try:
                literal = "'" + original.replace("'", "''") + "'"
                conn.execute(f"COMMENT ON COLUMN {table}.{_q(col)} IS {literal}")
            except duckdb.Error:  # comments are a nicety; never fail an upload over one
                break
        conn.execute("CHECKPOINT")  # Flush to disk
        logger.info(f"Stored {len(df)} rows into {table} for user {user_id[:8]}…")
    finally:
        conn.close()

    return {
        "table_name": table,
        "rows": len(df),
        "columns": clean_cols,
    }


def _describe_table(conn: duckdb.DuckDBPyConnection, tname: str) -> str:
    """Markdown description of one table: columns with types, value ranges, categories and sample rows."""
    row_count = conn.execute(f"SELECT COUNT(*) FROM {_q(tname)}").fetchone()[0]
    cols = conn.execute(
        "SELECT column_name, data_type, comment FROM duckdb_columns() "
        "WHERE table_name = ? ORDER BY column_index",
        [tname],
    ).fetchall()
    summary = {}
    try:
        res = conn.execute(f"SUMMARIZE {_q(tname)}")
        names = [d[0] for d in res.description]
        for r in res.fetchall():
            row = dict(zip(names, r))
            summary[row["column_name"]] = {"min": row.get("min"), "max": row.get("max"),
                                           "distinct": row.get("approx_unique"), "nulls": row.get("null_percentage")}
    except duckdb.Error:
        pass

    lines = [f"### Table `{tname}` ({row_count:,} rows)", "",
             "| column | type | values | nulls |", "|---|---|---|---|"]
    for name, dtype, comment in cols[:MAX_SCHEMA_COLUMNS]:
        s = summary.get(name, {})
        label = f"`{name}`" + (f" (header: {comment})" if comment else "")
        values = ""
        is_text = dtype.upper() in ("VARCHAR", "TEXT") or dtype.upper().startswith("VARCHAR")
        if is_text and s.get("distinct") is not None and s["distinct"] <= CATEGORY_MAX_DISTINCT:
            distinct = conn.execute(
                f"SELECT DISTINCT {_q(name)} FROM {_q(tname)} WHERE {_q(name)} IS NOT NULL "
                f"ORDER BY 1 LIMIT {CATEGORY_MAX_DISTINCT + 1}"
            ).fetchall()
            if len(distinct) <= CATEGORY_MAX_DISTINCT:
                values = "one of: " + ", ".join(f"'{str(v[0])}'" for v in distinct)
        if not values and s.get("min") is not None:
            if is_text:
                values = f"~{s['distinct']:,} distinct, e.g. '{str(s['min'])[:40]}'"
            else:
                values = f"{format_value(s['min'])} to {format_value(s['max'])}"
        nulls = s.get("nulls")
        nulls_txt = f"{float(nulls):.0f}%" if nulls not in (None, "") else ""
        lines.append(f"| {_cell(label)} | {dtype} | {_cell(values)} | {nulls_txt} |")
    if len(cols) > MAX_SCHEMA_COLUMNS:
        lines.append(f"\n({len(cols) - MAX_SCHEMA_COLUMNS} more columns not shown)")

    result = conn.execute(f"SELECT * FROM {_q(tname)} LIMIT {SAMPLE_ROWS}")
    sample_cols = [d[0] for d in result.description][:MAX_SCHEMA_COLUMNS]
    sample = [list(r)[:MAX_SCHEMA_COLUMNS] for r in result.fetchall()]
    if sample:
        lines += ["", f"Sample rows (first {len(sample)}):", "", markdown_table(sample_cols, sample)]
    return "\n".join(lines)


def get_table_schema(user_id: str, doc_id: uuid.UUID | None = None) -> str:
    """
    Return a Markdown schema description for the LLM prompt.

    If doc_id is provided, only returns schema for that document's tables.
    Otherwise returns schema for ALL tables in the user's DuckDB.

    For every table: row count; each column's type, value range (numbers/dates) or
    allowed values (low-cardinality text, so filters use the exact spelling), null
    share and original header; plus a few sample rows as a Markdown table.
    """
    db_file = _db_path(user_id)
    if not os.path.exists(db_file):
        return ""

    conn = _connect(db_file, read_only=True, sandboxed=True)
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        table_names = [t[0] for t in tables]

        # Filter by doc_id if provided
        if doc_id is not None:
            prefix = f"t_{_table_prefix(doc_id)}_"
            table_names = [t for t in table_names if t.startswith(prefix)]

        if not table_names:
            return ""
        return "\n\n".join(_describe_table(conn, t) for t in table_names)
    finally:
        conn.close()


def _strip_literals_and_comments(sql: str) -> str:
    """SQL with string literals, quoted identifiers and comments blanked out (for keyword checks)."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return re.sub(r'"(?:[^"]|"")*"', '""', sql)


def validate_select(sql: str) -> str:
    """Return the cleaned single SELECT/WITH statement, or raise ValueError."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("Empty query.")
    bare = _strip_literals_and_comments(cleaned)
    if ";" in bare:
        raise ValueError("Only a single SQL statement is allowed.")
    first = bare.lstrip("( \n\t").split(None, 1)[0].upper() if bare.strip() else ""
    if first not in ("SELECT", "WITH"):
        raise ValueError("Only SELECT queries are allowed.")
    upper = bare.upper()
    for forbidden in _FORBIDDEN:
        if re.search(rf"\b{forbidden}\b", upper):
            raise ValueError(f"Query contains forbidden keyword: {forbidden}")
    return cleaned


def _try_cast_outside_literals(sql: str) -> str:
    """CAST( -> TRY_CAST( so dirty cells become NULL instead of failing the query (not inside strings)."""
    parts = re.split(r"('(?:[^']|'')*')", sql)
    return "".join(p if i % 2 else re.sub(r"(?<!TRY_)\bCAST\s*\(", "TRY_CAST(", p, flags=re.IGNORECASE)
                   for i, p in enumerate(parts))


def execute_sql(user_id: str, sql: str, timeout_s: float = QUERY_TIMEOUT_S) -> dict[str, Any]:
    """
    Execute a SELECT query against the user's DuckDB (sandboxed, read-only, time-limited).

    Returns:
        {
            "columns": ["col1", "col2", ...],
            "rows": [[val1, val2, ...], ...],   # JSON-safe values, at most MAX_RESULT_ROWS
            "row_count": N,                      # rows returned
            "truncated": bool                    # True if the query produced more rows
        }

    Raises ValueError for disallowed queries, execution errors and timeouts.
    """
    cleaned = _try_cast_outside_literals(validate_select(sql))

    db_file = _db_path(user_id)
    if not os.path.exists(db_file):
        raise ValueError("No tabular data found. Please upload a spreadsheet first.")

    conn = _connect(db_file, read_only=True, sandboxed=True)
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        result = conn.execute(cleaned)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchmany(MAX_RESULT_ROWS + 1)
        truncated = len(rows) > MAX_RESULT_ROWS
        rows = rows[:MAX_RESULT_ROWS]
        return {
            "columns": columns,
            "rows": [[json_safe(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }
    except duckdb.InterruptException:
        raise ValueError(f"SQL execution error: the query took longer than {timeout_s:.0f}s and was stopped.")
    except duckdb.Error as e:
        raise ValueError(f"SQL execution error: {e}")
    finally:
        timer.cancel()
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

    conn = _connect(db_file)
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

    conn = _connect(db_file, read_only=True, sandboxed=True)
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        return [t[0] for t in tables]
    finally:
        conn.close()


def table_columns_by_doc(user_id: str) -> dict[str, list[tuple[str, list[str]]]]:
    """{doc table prefix: [(table name, [column names...]), ...]} for every table in the user's DuckDB.

    Cheap (catalog only, no data scan); used to tell the document router what each spreadsheet contains.
    """
    db_file = _db_path(user_id)
    if not os.path.exists(db_file):
        return {}
    conn = _connect(db_file, read_only=True, sandboxed=True)
    try:
        rows = conn.execute(
            "SELECT table_name, column_name FROM duckdb_columns() WHERE schema_name = 'main' "
            "ORDER BY table_name, column_index"
        ).fetchall()
    finally:
        conn.close()
    tables: dict[str, list[str]] = {}
    for tname, col in rows:
        tables.setdefault(tname, []).append(col)
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for tname, cols in tables.items():
        m = re.match(r"^t_([0-9a-f]{8})_", tname)
        if m:
            out.setdefault(m.group(1), []).append((tname, cols))
    return out


def has_tabular_data(user_id: str) -> bool:
    """Check if the user has any tabular data stored in DuckDB."""
    return len(list_tables(user_id)) > 0
