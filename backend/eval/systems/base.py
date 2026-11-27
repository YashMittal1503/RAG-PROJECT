"""
The interface every architecture implements so the harness can compare them.

To add an architecture (SVD-RAG, HippoRAG 2, long-context...):
  1. Subclass RAGSystem in eval/systems/<name>.py.
  2. `index()` builds whatever the method needs from the corpus (idempotent).
  3. `answer()` returns a Prediction, including the contexts given to the
     generator, so retrieval can be scored on text.
  4. Register it in SYSTEMS below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eval.corpus import CorpusDoc
from eval.schema import GoldItem, Prediction


class RAGSystem(ABC):
    name: str = "base"

    def __init__(self, docs: list[CorpusDoc], **options: Any):
        self.docs = docs
        self.options = options

    def config(self) -> dict[str, Any]:
        """Everything that affects results; saved with each run for reproducibility."""
        return {"name": self.name, **self.options}

    @abstractmethod
    async def index(self, force: bool = False) -> dict: ...

    @abstractmethod
    async def answer(self, item: GoldItem) -> Prediction: ...


def get_system(name: str, docs: list[CorpusDoc], **options: Any) -> RAGSystem:
    from eval.systems.vanilla import VanillaRAG

    systems: dict[str, type[RAGSystem]] = {VanillaRAG.name: VanillaRAG}
    if name not in systems:
        raise ValueError(f"Unknown system '{name}'. Available: {sorted(systems)}")
    return systems[name](docs, **options)
