"""Offline tests for the RAGAS metrics, the shared judge config and key-point recall.

RAGAS metrics run for real (ragas 0.4 collections API) against a scripted fake LLM,
so these tests exercise our adapter *and* ragas' own scoring logic, with no API calls.
"""

import json

import pytest

pytest.importorskip("ragas")

from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402

import eval.scoring.judge as judge_mod  # noqa: E402
from eval.schema import GoldItem, Prediction, RetrievedContext  # noqa: E402
from eval.scoring import score_item  # noqa: E402
from eval.scoring.judge import Judge, JudgeConfig, parse_key_points  # noqa: E402
from eval.scoring.ragas_metrics import (  # noqa: E402
    CONTEXT_PRECISION_TOP_K,
    RagasScorer,
    _build_llm_class,
    applicable_metrics,
)


class FakeLLM(InstructorBaseRagasLLM):
    """Answers each RAGAS structured call by response-model name, from scripted values."""

    model = "fake-llm"

    def __init__(self, precision_verdicts=(1, 0, 1, 0, 1, 0, 1), fail=()):
        self.calls: list[str] = []
        self.precision_verdicts = list(precision_verdicts)
        self.fail = set(fail)

    async def agenerate(self, prompt, response_model):
        name = response_model.__name__
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"boom in {name}")
        if name == "StatementGeneratorOutput":
            return response_model(statements=["Elias is the narrator.", "Elias had a wife named Ruth."])
        if name == "NLIStatementOutput":
            return response_model(statements=[
                {"statement": "Elias is the narrator.", "reason": "stated", "verdict": 1},
                {"statement": "Elias had a wife named Ruth.", "reason": "not in context", "verdict": 0},
            ])
        if name == "AnswerRelevanceOutput":
            return response_model(question="What is the narrator's name?", noncommittal=0)
        if name == "ContextPrecisionOutput":
            i = sum(1 for c in self.calls if c == name) - 1
            return response_model(reason="r", verdict=self.precision_verdicts[i])
        if name == "ContextRecallOutput":
            return response_model(classifications=[
                {"statement": "The narrator is Elias.", "reason": "in context", "attributed": 1},
                {"statement": "He arrived in 1931.", "reason": "missing", "attributed": 0},
            ])
        raise AssertionError(f"unexpected response model {name}")

    def generate(self, prompt, response_model):
        raise NotImplementedError


class FakeEmbeddings(BaseRagasEmbedding):
    def embed_text(self, text, **kw):
        return [1.0, 0.0, 0.0]

    async def aembed_text(self, text, **kw):
        return [1.0, 0.0, 0.0]


ITEM = GoldItem(id="r1", question="What is the narrator's name?", docs=["lighthouse"], answer="Elias",
                evidence=[{"doc": "lighthouse", "page": 1, "quote": "Morning, Elias!"}])
PRED = Prediction(item_id="r1", answer="Elias is the narrator. His wife Ruth helped him.",
                  contexts=[RetrievedContext(text="Morning, Elias!", doc="lighthouse", page=1),
                            RetrievedContext(text="The storm came on Saturday night.", doc="lighthouse", page=1)])


def _scorer(**kw):
    llm = kw.pop("llm", FakeLLM())
    return RagasScorer(llm=llm, embeddings=FakeEmbeddings(), **kw), llm


# ── which metrics apply ────────────────────────────────────────────────

def test_applicable_metrics_rules():
    all_m = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    assert applicable_metrics(ITEM, PRED, refused=False, enabled=all_m) == list(all_m)
    assert applicable_metrics(ITEM, PRED, refused=True, enabled=all_m) == ["context_precision", "context_recall"]
    unanswerable = GoldItem(id="u", question="Wife?", answerable=False, category="unanswerable")
    assert applicable_metrics(unanswerable, PRED, refused=False, enabled=all_m) == ["faithfulness", "answer_relevancy"]
    assert applicable_metrics(ITEM, PRED.model_copy(update={"contexts": []}), False, all_m) == []
    assert applicable_metrics(ITEM, PRED.model_copy(update={"error": "x"}), False, all_m) == []
    assert applicable_metrics(ITEM, PRED, False, ("faithfulness",)) == ["faithfulness"]


# ── scores from the real ragas metric code ──────────────────────────────

@pytest.mark.anyio
async def test_all_four_metrics_score_and_report_unsupported_claims():
    scorer, llm = _scorer(llm=FakeLLM(precision_verdicts=[0, 1]))
    out = await scorer.score(ITEM, PRED, refused=False)
    r = out["ragas"]
    assert r["faithfulness"] == 0.5
    assert out["unsupported_claims"] == ["Elias had a wife named Ruth."]
    assert r["answer_relevancy"] == pytest.approx(1.0)      # identical fake embeddings, committal answer
    assert r["context_precision"] == pytest.approx(0.5)     # relevant context ranked 2nd of 2
    assert r["context_recall"] == 0.5                       # 1 of 2 reference statements attributed


@pytest.mark.anyio
async def test_context_precision_is_capped_at_top_k():
    many = PRED.model_copy(update={"contexts": [RetrievedContext(text=f"ctx {i}") for i in range(9)]})
    scorer, llm = _scorer(metrics=("context_precision",))
    await scorer.score(ITEM, many, refused=False)
    assert llm.calls.count("ContextPrecisionOutput") == CONTEXT_PRECISION_TOP_K


@pytest.mark.anyio
async def test_one_failing_metric_does_not_lose_the_others():
    scorer, _ = _scorer(llm=FakeLLM(fail={"ContextRecallOutput"}))
    out = await scorer.score(ITEM, PRED, refused=False)
    assert out["ragas"]["context_recall"] is None
    assert out["ragas"]["faithfulness"] == 0.5


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        RagasScorer(llm=FakeLLM(), embeddings=FakeEmbeddings(), metrics=("bleu",))


# ── key rotation, retries and cache ─────────────────────────────────────

class _Inner:
    def __init__(self, name, fail_first=False):
        self.name, self.fail_first, self.calls = name, fail_first, 0

    async def agenerate(self, prompt, response_model):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("429 rate limited")
        return response_model(statements=[f"from {self.name}"])


@pytest.mark.anyio
async def test_rotating_llm_rotates_retries_and_caches(tmp_path, monkeypatch):
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput

    import eval.scoring.ragas_metrics as rm

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(rm.asyncio, "sleep", _no_sleep)
    a, b = _Inner("a", fail_first=True), _Inner("b")
    llm = _build_llm_class()([a, b], "m", tmp_path, "0.4.3")
    assert isinstance(llm, InstructorBaseRagasLLM)

    r1 = await llm.agenerate("prompt-1", StatementGeneratorOutput)   # a fails (429), retried on b
    assert r1.statements == ["from b"] and a.calls == 1 and b.calls == 1
    r2 = await llm.agenerate("prompt-1", StatementGeneratorOutput)   # cached: no new calls
    assert r2.statements == ["from b"] and a.calls + b.calls == 2
    await llm.agenerate("prompt-2", StatementGeneratorOutput)        # new prompt goes to the next key
    assert a.calls + b.calls == 3


# ── score_item wiring ───────────────────────────────────────────────────

class _ScriptedClient:
    def __init__(self):
        self.calls = []
        self.chat = type("Chat", (), {})()
        self.chat.completions = self

    async def create(self, model, messages, temperature, max_tokens):
        system = messages[0]["content"]
        self.calls.append(system[:30])
        if "KEY POINTS" in system:
            reply = json.dumps({"points": [{"index": 1, "covered": True}, {"index": 2, "covered": False}]})
        else:
            reply = '{"label":"correct","reason":"ok"}'

        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        return type("Resp", (), {"choices": [_Choice()]})()


@pytest.mark.anyio
async def test_score_item_combines_judge_key_points_and_ragas(tmp_path):
    client = _ScriptedClient()
    judge = Judge(model="fake", cache_dir=tmp_path, client=client)
    scorer, _ = _scorer()
    item = ITEM.model_copy(update={"category": "global_summary",
                                   "key_points": ["Elias is the narrator", "The light never failed"]})

    s = await score_item(item, PRED, judge, scorer)
    assert s.judge_label == "correct"
    assert s.key_point_recall == 0.5 and s.missed_key_points == ["The light never failed"]
    assert s.ragas["faithfulness"] == 0.5 and s.unsupported_claims == ["Elias had a wife named Ruth."]
    assert set(s.ragas) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}

    await score_item(item, PRED, judge, scorer)                      # judge calls are cached
    assert len(client.calls) == 2

    s = await score_item(item, PRED, judge, None)                    # --no-ragas keeps the judge
    assert s.ragas == {} and s.judge_label == "correct"


@pytest.mark.anyio
async def test_refusal_gets_no_faithfulness_or_relevancy():
    scorer, llm = _scorer()
    unanswerable = GoldItem(id="u", question="Wife's name?", answerable=False, category="unanswerable")
    refusal = PRED.model_copy(update={"answer": "This information is not available in the provided document."})
    s = await score_item(unanswerable, refusal, None, scorer)
    assert s.refused and s.ragas == {} and llm.calls == []


def test_parse_key_points_handles_missing_and_out_of_range():
    raw = '{"points":[{"index":1,"covered":true},{"index":3,"covered":"true"},{"index":9,"covered":true}]}'
    assert parse_key_points(raw, 3) == [True, False, True]


# ── shared judge configuration ──────────────────────────────────────────

@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr(judge_mod, "_load_dotenv", lambda: None)
    for prefix in ("GEMINI", "MISTRAL", "GROQ"):
        for suffix in ("", "_2", "_3", "S"):
            monkeypatch.delenv(f"{prefix}_API_KEY{suffix}", raising=False)
    for name in ("EVAL_JUDGE_PROVIDER", "EVAL_JUDGE_MODEL", "EVAL_JUDGE_API_KEY", "EVAL_JUDGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_judge_config_resolves_provider_model_and_key_pool(clean_env):
    clean_env.setenv("MISTRAL_API_KEY", "m1")
    clean_env.setenv("MISTRAL_API_KEYS", "m2, m1 ,m3")
    cfg = JudgeConfig.resolve("mistral")
    assert cfg.provider == "mistral" and cfg.model == "mistral-small-latest"
    assert cfg.base_url == "https://api.mistral.ai/v1" and cfg.keys == ["m1", "m2", "m3"]

    clean_env.setenv("GEMINI_API_KEY", "g1")
    clean_env.setenv("EVAL_JUDGE_MODEL", "gemini-2.5-pro")
    cfg = JudgeConfig.resolve()
    assert cfg.provider == "gemini" and cfg.model == "gemini-2.5-pro" and cfg.keys == ["g1"]

    clean_env.setenv("EVAL_JUDGE_API_KEY", "override")
    assert JudgeConfig.resolve().keys == ["override"]


def test_judge_config_errors(clean_env):
    with pytest.raises(ValueError):
        JudgeConfig.resolve("openai")
    with pytest.raises(RuntimeError):
        JudgeConfig.resolve("groq")
