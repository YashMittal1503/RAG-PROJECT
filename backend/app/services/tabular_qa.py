"""
Text-to-SQL question answering over the user's spreadsheets (DuckDB).

Flow (shared by the chat endpoint and the eval harness, so both run the same code):

  plan_tabular_answer(...)
    - whole-dataset questions (classify_query scope "global": "summarize my data",
      "what does this spreadsheet contain") skip SQL and answer from the data profile;
    - everything else: generate SQL (told whether the question needs aggregating),
      execute it in the sandbox, and on an error ask the LLM to repair the query once,
      giving it the error message;
    - if SQL still fails, fall back to the profile-based answer, which is told that the
      computation failed and must not invent numbers.
  stream_tabular_answer(plan, ...)
    - SQL route: the LLM writes a short interpretation, then the exact result is
      appended as a Markdown table (rendered by the chat UI). The model never retypes
      the numbers, so it can't mistype them, and long results don't cost output tokens.
    - overview route: an answer grounded in the Markdown data profile.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator
from uuid import UUID

import logfire

from app.services import tabular_store
from app.services.llm_provider import (
    call_llm_with_cross_provider_fallback as call_llm_with_fallback,
    stream_llm_with_cross_provider_fallback as stream_llm_with_fallback,
)

logger = logging.getLogger(__name__)

LLM_RESULT_ROWS = 50        # rows of a result the answering LLM sees
DISPLAY_RESULT_ROWS = 50    # rows shown to the user in the appended table

SQL_GENERATION_PROMPT = """You write DuckDB SQL that answers a user's question about their spreadsheet data.

You get the schema as Markdown: each table's columns with types, value ranges, the exact allowed values of category columns, the original spreadsheet headers, and sample rows.

RULES:
1. Output ONE read-only query: SELECT, or WITH ... SELECT. Put it in a ```sql code block. No explanation.
2. Use the exact table and column names from the schema.
3. Filtering text: if the schema lists a column's allowed values, use the exact spelling shown. Otherwise use ILIKE '%term%'.
4. Aggregates: always alias them meaningfully (total_revenue, avg_age, customer_count) and include the grouping/label column next to them.
5. Top-N / "which ... the most": ORDER BY the measure DESC and LIMIT N (LIMIT 1 for a single winner); also return the measure.
6. Percentages and rates: round(100.0 * part / total, 2). Averages: round(avg(x), 2).
7. Dates: DATE columns compare with DATE 'YYYY-MM-DD' literals; use date_trunc, year(), month(), strftime() as needed. For text columns that hold numbers, use TRY_CAST(TRIM(col) AS DOUBLE).
8. Listing records: select only the columns needed to answer and add LIMIT 50 unless the user asked for a specific number.
9. If the question can't be answered from these tables at all, output exactly: CANNOT_ANSWER"""

SQL_ANSWER_PROMPT = """You are a data analyst answering a question about the user's spreadsheet data.
You get the user's question and the exact result of a query that answers it, as a Markdown table.

RULES:
1. Answer the question directly in the first sentence, using numbers exactly as they appear in the result (bold the key ones).
2. Then add up to 3 short bullet points of insight (highest / lowest, notable gaps, share of total) ONLY if they follow directly from the result. Never invent numbers or compute anything that isn't simple arithmetic on the shown values.
3. The full result table is shown to the user automatically right after your answer. Do NOT reproduce the table.
4. If the result is empty, say that no matching data was found and suggest how to rephrase.
5. If the result is marked as truncated, say the answer is based on the first rows shown.
6. Call the data "your spreadsheet". Don't mention SQL, queries, tables or databases."""

TABULAR_OVERVIEW_PROMPT = """You are a data analyst describing the user's uploaded spreadsheet.
You get a profile of the data as Markdown: tables, row counts, each column's type, value range or categories, share of missing values, and a few sample rows.

RULES:
1. Answer the user's question from the profile: what the data is about (infer the domain from the columns), its size, what each important column tracks, value ranges and categories, and data-quality notes (missing values).
2. Only state numbers that appear in the profile (row counts, ranges, category lists). Never invent totals, averages, trends or correlations: the profile doesn't contain them.
3. End with 3 specific quantitative questions the user could ask next, using real column names in plain words.
4. Use clean Markdown with short sections and bullet points. Call it "your spreadsheet" (or its filename if given).
5. Don't mention SQL, DuckDB, table names like t_..., or databases."""

SQL_FAILED_NOTE = (
    "NOTE: computing the exact answer to this question failed. Say briefly that you couldn't "
    "calculate it, give whatever the profile does show that is relevant, and suggest a simpler "
    "or more specific way to ask."
)


# ── Thin async wrappers over the store ────────────────────────────────────

async def check_tabular_data(user_id: str) -> bool:
    """Whether the user has any spreadsheet tables (runs in a thread)."""
    return await asyncio.to_thread(tabular_store.has_tabular_data, user_id)


PROFILE_MAX_CHARS = 2400   # per spreadsheet, in the document router's catalog


_ID_COLUMN = re.compile(r"(^|_)(id|code|number|no)$", re.I)


def _short_sheet(tname: str) -> str:
    return re.sub(r"^t_[0-9a-f]{8}_", "", tname)


def tabular_profile(tables: list[tuple[str, list[str]]], max_chars: int | None = None) -> str:
    """One line naming a spreadsheet's sheets and their columns, e.g. 'customers (Customer_Name, State); sales (...)'.

    Every sheet name is always listed (a 29-sheet workbook must still show its 'customers' sheet); columns
    per sheet are cut down evenly until the line fits.
    """
    if not tables:
        return ""
    max_chars = max_chars or PROFILE_MAX_CHARS
    names = [_short_sheet(t) for t, _ in tables]
    # Descriptive columns first: an ID/code/number column says little about what a sheet holds.
    tables = [(t, sorted(cols, key=lambda c: bool(_ID_COLUMN.search(c)))) for t, cols in tables]
    widest = max(len(cols) for _, cols in tables)
    for per_sheet in range(widest, 0, -1):
        parts = []
        for name, (_, cols) in zip(names, tables):
            shown = ", ".join(cols[:per_sheet]) + (", ..." if len(cols) > per_sheet else "")
            parts.append(f"{name} ({shown})")
        text = "; ".join(parts)
        if len(text) <= max_chars:
            return text
    text = ", ".join(names)   # even one column per sheet is too long: sheet names only
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip(" ,") + "..."


async def attach_tabular_profiles(user_id: str, docs: list[dict]) -> list[dict]:
    """Add a 'tabular_profile' (sheets + columns) to each spreadsheet in `docs`, in place.

    The document router only sees filenames otherwise, so a question like "which 3 states have the most
    customers?" can't be matched to the spreadsheet that holds a customers sheet with a State column.
    """
    if not any(d.get("is_tabular") for d in docs):
        return docs
    try:
        by_prefix = await asyncio.to_thread(tabular_store.table_columns_by_doc, user_id)
    except Exception as e:  # routing still works from filenames alone
        logger.warning(f"Could not read spreadsheet columns for routing: {e}")
        return docs
    for d in docs:
        if d.get("is_tabular"):
            tables = by_prefix.get(str(d["id"]).replace("-", "")[:8])
            if tables:
                d["tabular_profile"] = tabular_profile(tables)
    return docs


async def get_schema_for_prompt(user_id: str, doc_id: UUID | None = None) -> str:
    """Markdown schema + data profile, optionally for one document's tables (runs in a thread)."""
    return await asyncio.to_thread(tabular_store.get_table_schema, user_id, doc_id)


async def execute_sql_query(user_id: str, sql: str) -> dict:
    """Run SQL in the user's sandboxed DuckDB (runs in a thread)."""
    with logfire.span("🦆 Execute DuckDB SQL | '{sql_short}'", sql_short=sql[:50], sql=sql) as span:
        result = await asyncio.to_thread(tabular_store.execute_sql, user_id, sql)
        span.set_attribute("row_count", result.get("row_count", 0))
        span.set_attribute("truncated", result.get("truncated", False))
        return result


async def get_tabular_tables(user_id: str) -> list[str]:
    return await asyncio.to_thread(tabular_store.list_tables, user_id)


# ── SQL generation ────────────────────────────────────────────────────────

def extract_sql(content: str) -> str | None:
    """Pull one SELECT / WITH statement out of LLM output (code fence or raw)."""
    if not content or "CANNOT_ANSWER" in content:
        return None
    fence = re.search(r"```(?:sql|duckdb)?\s*([\s\S]*?)```", content, re.IGNORECASE)
    text = fence.group(1) if fence else content
    m = re.search(r"\b((?:WITH|SELECT)\b[\s\S]*)", text, re.IGNORECASE)
    if not m:
        return None
    sql = m.group(1).strip()
    sql = re.sub(r"```[\s\S]*$", "", sql).strip()
    # Keep only the first statement if the model added more.
    sql = sql.split(";")[0].strip()
    return sql or None


def _history_text(chat_history: list[dict] | None, turns: int = 4) -> str:
    if not chat_history:
        return ""
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}" for m in chat_history[-turns:]]
    return "Recent conversation (for follow-up questions):\n" + "\n".join(lines) + "\n\n"


async def generate_sql_query(
    question: str,
    schema_info: str,
    chat_history: list[dict] | None = None,
    *,
    aggregation: bool | None = None,
    failed_sql: str | None = None,
    error: str | None = None,
) -> str | None:
    """
    Ask the LLM for a DuckDB query. `aggregation` (from classify_query) tells it whether
    to aggregate or list records. With `failed_sql` + `error` it repairs a failed query.
    Returns None if the question can't be answered with SQL.
    """
    hint = ""
    if aggregation is True:
        hint = "This question needs an aggregate over many rows (count / sum / average / min / max / ranking): return the aggregated result, not raw rows.\n\n"
    elif aggregation is False:
        hint = "This question is about specific records or values: return the matching rows with only the columns needed.\n\n"
    repair = ""
    if failed_sql and error:
        repair = (
            f"Your previous query failed.\nQuery:\n```sql\n{failed_sql}\n```\nError: {error[:500]}\n"
            "Write a corrected query.\n\n"
        )
    messages: list[dict] = [
        {"role": "system", "content": SQL_GENERATION_PROMPT},
        {"role": "user", "content": (
            f"Schema:\n\n{schema_info}\n\n{_history_text(chat_history)}{hint}{repair}"
            f"Question: {question}"
        )},
    ]
    with logfire.span("📝 Generate DuckDB SQL | '{q_short}'", q_short=question[:50], question=question,
                      repair=bool(repair)) as span:
        try:
            response = await call_llm_with_fallback(messages=messages, max_tokens=600, temperature=0.0, fast=True)
            msg = response.choices[0].message
            content = (msg.content or "").strip()
            reasoning = (getattr(msg, "reasoning", None) or "").strip()
            sql = extract_sql(content) or (extract_sql(reasoning) if not content else None)
            span.set_attribute("sql_generated", bool(sql))
            if sql:
                span.set_attribute("sql", sql)
                logger.info(f"Generated SQL{' (repair)' if repair else ''}: {sql}")
            else:
                logger.info(f"No SQL for '{question[:80]}' (CANNOT_ANSWER or non-SELECT output)")
            return sql
        except Exception as e:
            logger.warning(f"SQL generation failed: {e}")
            span.set_attribute("error", str(e))
            return None


# ── Result formatting ─────────────────────────────────────────────────────

def is_scalar(result: dict) -> bool:
    return len(result.get("rows", [])) == 1 and len(result.get("columns", [])) == 1


def format_sql_result(result: dict, max_rows: int = LLM_RESULT_ROWS) -> str:
    """The result as a Markdown table (header + separator), for the answering LLM."""
    rows = result.get("rows", [])
    if not rows:
        return "The query returned no rows."
    text = tabular_store.markdown_table(result["columns"], rows, max_rows=max_rows)
    shown = min(len(rows), max_rows)
    if result.get("truncated") or len(rows) > max_rows:
        text += f"\n\n(truncated: showing the first {shown} rows; the full result has more)"
    else:
        text += f"\n\n({shown} row{'s' if shown != 1 else ''})"
    return text


def result_table_for_user(result: dict, max_rows: int = DISPLAY_RESULT_ROWS) -> str | None:
    """Markdown table appended to the answer in the UI. None for empty or single-value results."""
    rows = result.get("rows", [])
    if not rows or is_scalar(result):
        return None
    table = tabular_store.markdown_table(result["columns"], rows, max_rows=max_rows)
    if result.get("truncated") or len(rows) > max_rows:
        table += f"\n\n*Showing the first {min(len(rows), max_rows)} rows.*"
    return table


# ── Plan + answer ─────────────────────────────────────────────────────────

@dataclass
class TabularPlan:
    route: str                         # "sql" or "tabular_overview"
    schema_info: str
    sql: str | None = None
    result: dict | None = None
    sql_error: str | None = None
    attempts: int = 0                  # SQL executions tried (0, 1 or 2)
    timings_ms: dict[str, float] = field(default_factory=dict)


async def plan_tabular_answer(
    user_id: str,
    question: str,
    doc_id: UUID | None,
    classification: Any = None,
    chat_history: list[dict] | None = None,
    max_attempts: int = 2,
) -> TabularPlan | None:
    """Decide and run the data step for a spreadsheet question. None when there's no schema."""
    import time

    def _tick(plan: TabularPlan, stage: str, t0: float) -> None:
        plan.timings_ms[stage] = plan.timings_ms.get(stage, 0.0) + round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    schema_info = await get_schema_for_prompt(user_id, doc_id)
    if not schema_info:
        return None
    plan = TabularPlan(route="tabular_overview", schema_info=schema_info)
    _tick(plan, "schema", t0)

    if classification is not None and getattr(classification, "is_overview", False):
        return plan  # "summarize / what's in my data": the profile answers it; no SQL to get wrong

    aggregation = getattr(classification, "aggregation", None) if classification is not None else None
    failed_sql, error = None, None
    for _ in range(max_attempts):
        t0 = time.perf_counter()
        sql = await generate_sql_query(question, schema_info, chat_history, aggregation=aggregation,
                                       failed_sql=failed_sql, error=error)
        _tick(plan, "sql_generate", t0)
        if not sql:
            break
        plan.sql = sql
        plan.attempts += 1
        t0 = time.perf_counter()
        try:
            plan.result = await execute_sql_query(user_id, sql)
            _tick(plan, "sql_execute", t0)
            plan.route, plan.sql_error = "sql", None
            return plan
        except Exception as e:
            _tick(plan, "sql_execute", t0)
            failed_sql, error = sql, str(e)
            plan.sql_error = f"{type(e).__name__}: {e}"
            logger.warning(f"SQL attempt {plan.attempts} failed: {e}")
    return plan


async def generate_sql_answer_stream(question: str, result_text: str) -> AsyncGenerator[str, None]:
    """Stream the LLM's interpretation of a query result (the table itself is appended separately)."""
    messages = [
        {"role": "system", "content": SQL_ANSWER_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nResult:\n{result_text}"},
    ]
    async for token in stream_llm_with_fallback(messages=messages, max_tokens=600, temperature=0.1):
        yield token


async def generate_tabular_overview_stream(
    question: str,
    schema_info: str,
    filename: str = "",
    sql_failed: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream an answer grounded in the data profile (overviews, or when SQL failed)."""
    with logfire.span("📊 Stream Tabular Overview | {filename}", filename=filename or "Spreadsheet", question=question) as span:
        note = f"\n\n{SQL_FAILED_NOTE}" if sql_failed else ""
        messages = [
            {"role": "system", "content": TABULAR_OVERVIEW_PROMPT},
            {"role": "user", "content": (
                f"Spreadsheet: {filename or 'your spreadsheet'}\n\nData profile:\n{schema_info}\n\n"
                f"Question: {question}{note}"
            )},
        ]
        token_count = 0
        async for token in stream_llm_with_fallback(messages=messages, max_tokens=1000, temperature=0.2):
            token_count += 1
            yield token
        span.set_attribute("tokens_generated", token_count)


async def stream_tabular_answer(plan: TabularPlan, question: str, filename: str = "") -> AsyncGenerator[str, None]:
    """Tokens of the final answer for a plan: interpretation + exact table, or a profile-based answer."""
    if plan.route == "sql" and plan.result is not None:
        with logfire.span("🦆 Stream SQL Answer | '{q_short}'", q_short=question[:50], sql_query=plan.sql):
            async for token in generate_sql_answer_stream(question, format_sql_result(plan.result)):
                yield token
            table = result_table_for_user(plan.result)
            if table:
                yield "\n\n" + table
        return
    async for token in generate_tabular_overview_stream(question, plan.schema_info, filename,
                                                        sql_failed=plan.sql_error is not None):
        yield token
