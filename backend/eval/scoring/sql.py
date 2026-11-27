"""
Text-to-SQL metrics: did the generated SQL run, and does its result equal the gold rows?

Result match is execution accuracy, the standard Text-to-SQL metric (as in Spider/BIRD):
compare what the query returned, not the SQL text, because many different queries are
equally correct. To avoid penalizing harmless differences:
  - column order and column names are ignored (each row is compared as a multiset of values);
  - extra columns are allowed: "which vessel carried the most?" answered with (vessel, tonnes)
    still matches gold rows [["Iron Duke"]]; missing columns are not;
  - row order is ignored unless the item sets `ordered: true` (ORDER BY / top-N questions);
  - numbers match within a relative tolerance, so 49.99999 equals 50.0 and 2 equals 2.0;
  - strings are compared case-insensitively with surrounding whitespace removed.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from eval.schema import SQLResult

REL_TOL = 1e-4
_ROUND_DIGITS = 4
_MAX_EXTRA_COLUMN_TRIES = 500    # bound the column-subset search on very wide results


def _norm_value(v) -> tuple:
    """Hashable, tolerance-friendly form of one cell. Numeric strings count as numbers."""
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("num", float(v))
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return ("null",)
        return ("num", _round_sig(f))
    s = str(v).strip()
    try:
        return ("num", _round_sig(float(s.replace(",", ""))))
    except ValueError:
        return ("str", s.lower())


def _round_sig(f: float) -> float:
    """Round to a few significant digits so values within REL_TOL compare equal after hashing."""
    if f == 0:
        return 0.0
    digits = _ROUND_DIGITS - int(math.floor(math.log10(abs(f)))) - 1
    return round(f, digits)


def _norm_row(row: list) -> tuple:
    return tuple(sorted(_norm_value(v) for v in row))


def _same(p: list[tuple], g: list[tuple], ordered: bool) -> bool:
    return p == g if ordered else Counter(p) == Counter(g)


def rows_match(pred_rows: list[list], gold_rows: list[list], ordered: bool = False) -> bool:
    """True if the predicted rows equal the gold rows, allowing extra predicted columns."""
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    g = [_norm_row(r) for r in gold_rows]
    if _same([_norm_row(r) for r in pred_rows], g, ordered):
        return True
    width, gold_width = len(pred_rows[0]), len(gold_rows[0])
    if width <= gold_width or any(len(r) != width for r in pred_rows):
        return False
    for tries, cols in enumerate(combinations(range(width), gold_width)):
        if tries >= _MAX_EXTRA_COLUMN_TRIES:
            break
        if _same([_norm_row([r[c] for c in cols]) for r in pred_rows], g, ordered):
            return True
    return False


def score_sql(
    expected_route: str,
    sql: str | None,
    sql_result: SQLResult | None,
    gold_rows: list[list] | None,
    ordered: bool = False,
) -> dict:
    """sql_executed applies to items routed to SQL; sql_result_match to items with gold rows."""
    applies = expected_route == "sql" or gold_rows is not None
    out: dict = {"sql_executed": None, "sql_result_match": None}
    if not applies:
        return out
    executed = sql is not None and sql_result is not None
    out["sql_executed"] = 1.0 if executed else 0.0
    if gold_rows is not None:
        out["sql_result_match"] = (
            1.0 if executed and not sql_result.truncated and rows_match(sql_result.rows, gold_rows, ordered) else 0.0
        )
    return out
