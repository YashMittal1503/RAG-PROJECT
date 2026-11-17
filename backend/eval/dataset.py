"""Loading, filtering, hashing and validating golden datasets (JSONL, one GoldItem per line)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from eval.schema import GoldItem


def load_dataset(path: str | Path) -> list[GoldItem]:
    """Parse a JSONL dataset. Blank lines and lines starting with // are skipped."""
    items: list[GoldItem] = []
    errors: list[str] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(GoldItem.model_validate_json(line))
        except ValidationError as e:
            errors.append(f"line {lineno}: {e.errors()[0]['msg']}")
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: invalid JSON ({e})")
    if errors:
        raise ValueError(f"{len(errors)} invalid item(s) in {path}:\n  " + "\n  ".join(errors))

    seen: set[str] = set()
    dupes = [i.id for i in items if i.id in seen or seen.add(i.id)]
    if dupes:
        raise ValueError(f"Duplicate item ids in {path}: {sorted(set(dupes))}")
    return items


def filter_items(
    items: list[GoldItem],
    split: str | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[GoldItem]:
    out = [
        i for i in items
        if (split is None or i.split == split)
        and (not categories or i.category in categories)
        and (not tags or set(tags) & set(i.tags))
    ]
    return out[:limit] if limit else out


def dataset_hash(path: str | Path) -> str:
    """Short content hash, recorded with every run so results are tied to an exact dataset version."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def validate_against_corpus(items: list[GoldItem], doc_texts: dict[str, str]) -> list[str]:
    """Return problems: unknown doc ids, and evidence quotes not found in their document."""
    from eval.scoring.retrieval import quote_in_context

    problems: list[str] = []
    for item in items:
        for d in item.docs + ([item.expected_doc] if item.expected_doc else []):
            if d not in doc_texts:
                problems.append(f"{item.id}: unknown doc '{d}'")
        for ev in item.evidence:
            text = doc_texts.get(ev.doc)
            if text is None:
                problems.append(f"{item.id}: evidence references unknown doc '{ev.doc}'")
            elif not quote_in_context(ev.quote, text):
                problems.append(f"{item.id}: evidence quote not found in '{ev.doc}': {ev.quote[:60]!r}")
    return problems
