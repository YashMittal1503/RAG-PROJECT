"""
Data models shared by every part of the evaluation harness.

GoldItem   — one hand-verified question in the golden dataset (JSONL line).
Prediction — what a RAG system produced for one GoldItem.
ItemScore  — every metric computed for one (GoldItem, Prediction) pair.

These models deliberately do not import anything from `app`, so the scorers and
their tests run offline without model downloads or API keys.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Route = Literal["rag", "sql", "tabular_overview", "chitchat", "clarify", "no_documents"]
Category = Literal[
    "single_fact",
    "implicit_inference",
    "character_relationship",
    "multi_hop",
    "temporal",
    "global_summary",
    "unanswerable",
    "multi_turn",
    "routing",
    "tabular_sql",
    "table_figure",
    "other",
]
Difficulty = Literal["easy", "medium", "hard"]
QueryScope = Literal["global", "section", "metadata", "targeted"]
Split = Literal["dev", "holdout"]


class Evidence(BaseModel):
    """A verbatim span of a corpus document that supports the gold answer."""

    doc: str                      # corpus document id (from the corpus manifest)
    quote: str                    # exact (or near-exact) text from the document
    page: int | None = None       # 1-indexed page, when the source has pages

    @field_validator("quote")
    @classmethod
    def _quote_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("evidence quote must not be blank")
        return v


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GoldItem(BaseModel):
    """One question in the golden dataset."""

    id: str
    question: str
    docs: list[str] = Field(default_factory=list)          # corpus doc ids in scope
    history: list[Turn] = Field(default_factory=list)       # prior turns, for multi-turn items
    answer: str = ""                                        # canonical short answer
    acceptable: list[str] = Field(default_factory=list)     # aliases that also count as correct
    evidence: list[Evidence] = Field(default_factory=list)
    answerable: bool = True
    category: Category = "other"
    hops: int = 1
    needs_inference: bool = False
    expected_route: Route = "rag"
    expected_doc: str | None = None                         # doc the router should pick
    expected_scope: QueryScope | None = None                # what classify_query should say: global / section / metadata / targeted
    expected_aggregation: bool | None = None                # whether the question aggregates over many records
    gold_sql_rows: list[list] | None = None                 # for tabular items: expected result rows
    gold_sql: str | None = None                             # optional reference query (documentation only)
    ordered: bool = False                                   # tabular items: row order matters (ORDER BY questions)
    key_points: list[str] = Field(default_factory=list)     # summary items: facts a complete answer must cover
    difficulty: Difficulty = "medium"
    split: Split = "dev"
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _consistency(self) -> "GoldItem":
        if self.answerable and not self.answer.strip() and self.expected_route == "rag":
            raise ValueError(f"{self.id}: answerable RAG items need a gold answer")
        if self.answerable and self.expected_route == "rag" and not self.evidence:
            raise ValueError(f"{self.id}: answerable RAG items need at least one evidence span")
        if self.answerable and self.expected_route == "sql" and self.gold_sql_rows is None and not self.answer.strip():
            raise ValueError(f"{self.id}: answerable SQL items need gold_sql_rows or a gold answer")
        if not self.answerable and self.category not in ("unanswerable", "multi_turn", "routing", "other"):
            raise ValueError(f"{self.id}: unanswerable items should use category 'unanswerable'")
        return self

    @property
    def answers(self) -> list[str]:
        """All strings that count as a correct answer."""
        return [a for a in [self.answer, *self.acceptable] if a.strip()]


class RetrievedContext(BaseModel):
    """One unit of context a system gave the generator (chunk, graph passage, tree node...)."""

    text: str
    doc: str | None = None        # corpus doc id if known
    page: int | None = None
    score: float | None = None


class Citation(BaseModel):
    doc: str | None = None
    page: int | None = None


class SQLResult(BaseModel):
    """What a Text-to-SQL route executed and got back."""

    columns: list[str] = Field(default_factory=list)
    rows: list[list] = Field(default_factory=list)
    truncated: bool = False


class Prediction(BaseModel):
    """A system's output for one GoldItem."""

    item_id: str
    answer: str
    contexts: list[RetrievedContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    route: str | None = None
    intent: str | None = None                    # query classification: retrieve / chitchat
    query_scope: str | None = None               # global / section / metadata / targeted
    aggregation: bool | None = None
    sql: str | None = None                       # query a Text-to-SQL route generated
    sql_result: SQLResult | None = None          # its result, when it executed
    sql_error: str | None = None                 # why the SQL failed (the route then falls back)
    chosen_doc: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.values()))


class ItemScore(BaseModel):
    """All metrics for one item. None means 'not applicable to this item'."""

    item_id: str
    category: str
    answerable: bool
    difficulty: str
    doc: str | None = None                      # the item's document (expected_doc, or its only doc)
    # answer quality
    judge_label: Literal["correct", "partial", "wrong"] | None = None
    judge_score: float | None = None            # 1 / 0.5 / 0
    judge_reason: str | None = None
    exact_match: float | None = None
    token_f1: float | None = None
    # RAGAS: faithfulness, answer_relevancy, context_precision, context_recall (None = not applicable / failed)
    ragas: dict[str, float | None] = Field(default_factory=dict)
    unsupported_claims: list[str] = Field(default_factory=list)   # statements RAGAS faithfulness judged unsupported
    # summary completeness: share of gold key points the answer covers
    key_point_recall: float | None = None
    missed_key_points: list[str] = Field(default_factory=list)
    # tabular / Text-to-SQL
    sql_executed: float | None = None           # tabular items: generated SQL ran successfully
    sql_result_match: float | None = None       # items with gold_sql_rows: result equals the gold rows
    # retrieval
    evidence_recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float | None = None
    ndcg_at_10: float | None = None
    # citations
    citation_precision: float | None = None
    citation_recall: float | None = None
    # abstention
    refused: bool = False
    correct_refusal: float | None = None        # unanswerable items only
    false_refusal: float | None = None          # answerable items only
    # routing
    route_correct: float | None = None
    doc_correct: float | None = None
    scope_correct: float | None = None          # items with expected_scope
    aggregation_correct: float | None = None    # items with expected_aggregation
    # cost
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    scoring_errors: list[str] = Field(default_factory=list)   # judge/RAGAS calls that failed; re-run `rescore` to fill them
