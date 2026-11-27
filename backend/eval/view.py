"""
HTML dashboard for eval runs: one self-contained file, no server, no external assets.

  python -m eval view eval/runs/<run_id>                    # -> eval/runs/<run_id>/report.html
  python -m eval view eval/runs/<run_a> eval/runs/<run_b>   # compare B against A -> <run_b>/compare_<run_a>.html
  python -m eval view --index                               # -> eval/runs/index.html (every run, headline numbers)

`run` and `rescore` also write report.html automatically. The page shows headline score
cards with confidence intervals, every metric with a one-line meaning, breakdowns by
category and document, and a filterable list of items you can open to see why each one
passed or failed (gold answer, system answer, judge reason, unsupported claims, missed key
points, SQL, retrieved chunks).
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from eval.report import HEADLINE, LOWER_IS_BETTER, METRICS, compare, load_meta, load_scores, summarize

# Plain-language meaning of each metric, shown under its name.
DESCRIPTIONS = {
    "correctness (judge)": "LLM judge: does the answer match the gold answer? correct = 1, partial = 0.5",
    "answer match": "Gold answer (or an alias) appears in the answer, after normalising",
    "token F1": "Word overlap between answer and gold answer",
    "key-point recall": "Summaries: share of the gold key points the answer covers",
    "faithfulness (RAGAS)": "Share of the answer's claims supported by the retrieved text (catches made-up facts)",
    "answer relevancy (RAGAS)": "Does the answer address the question that was asked?",
    "context precision@5 (RAGAS)": "Are the useful chunks ranked at the top of the first 5?",
    "context recall (RAGAS)": "Share of the gold answer's facts present in the retrieved text",
    "evidence recall@5": "Gold evidence quote found in the top 5 retrieved chunks",
    "evidence recall@10": "Gold evidence quote found in the top 10 retrieved chunks",
    "MRR": "1 / rank of the first chunk containing the evidence (1 = ranked first)",
    "nDCG@10": "Ranking quality of the evidence chunks in the top 10",
    "citation precision": "Share of cited pages that hold gold evidence",
    "citation recall": "Share of gold evidence pages that were cited",
    "correct refusal (unanswerable)": "Unanswerable questions: the system said it doesn't know",
    "false refusal (answerable)": "Answerable questions the system refused (lower is better)",
    "route accuracy": "Went to the right pipeline (RAG, SQL, spreadsheet overview, chitchat)",
    "doc routing accuracy": "Picked the right document",
    "query scope accuracy": "Classifier scope (global / section / metadata / targeted) was right",
    "aggregation detection accuracy": "Classifier aggregation flag was right",
    "SQL executed": "Spreadsheet questions: the generated SQL ran without error",
    "SQL result match": "SQL result equals the gold rows (order-insensitive unless the question needs it)",
    "error rate": "Items that crashed (lower is better)",
}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run_data(run_dir: Path) -> dict:
    scores = load_scores(run_dir)
    items = {i["id"]: i for i in _jsonl(run_dir / "items.jsonl")}
    preds = {p["item_id"]: p for p in _jsonl(run_dir / "predictions.jsonl")}
    rows = []
    for s in scores:
        it, p = items.get(s.item_id, {}), preds.get(s.item_id, {})
        rows.append({
            "score": s.model_dump(mode="json"),
            "item": {k: it.get(k) for k in ("question", "answer", "acceptable", "key_points", "answerable", "history",
                                            "expected_route", "expected_scope", "gold_sql_rows", "split", "notes")},
            "pred": {k: p.get(k) for k in ("answer", "route", "query_scope", "aggregation", "sql", "sql_result",
                                            "sql_error", "chosen_doc", "error")},
            "contexts": [{"doc": c.get("doc"), "page": c.get("page"), "text": (c.get("text") or "")[:1200]}
                         for c in (p.get("contexts") or [])[:10]],
        })
    return {"name": run_dir.name, "meta": load_meta(run_dir), "summary": summarize(scores), "rows": rows}


def _page(title: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    return (_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data))


def write_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run = _run_data(run_dir)
    out = run_dir / "report.html"
    out.write_text(_page(f"Eval: {run['meta'].get('system', '?')} {run_dir.name}",
                         {"mode": "run", "run": run, "metrics": list(METRICS), "headline": HEADLINE,
                          "lower": sorted(LOWER_IS_BETTER), "desc": DESCRIPTIONS}), encoding="utf-8")
    return out


LATEST_REPORT = Path(__file__).resolve().parent / "report.html"   # eval/report.html


def _is_full(run_dir: Path) -> bool:
    """Scored with both the judge and RAGAS (not an offline routing-only run)."""
    try:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(meta.get("judge_model") and meta.get("ragas")) and (run_dir / "scores.jsonl").exists()


def write_latest(runs_dir: str | Path, out: str | Path | None = None) -> Path | None:
    """Write the newest fully scored run's dashboard to eval/report.html (a fixed place to look).

    Offline runs (--no-judge / --no-ragas) are skipped, so they never replace a full report with empty
    correctness columns. Returns None when no full run exists yet.
    """
    runs = sorted((d for d in Path(runs_dir).iterdir() if d.is_dir() and _is_full(d)), key=lambda d: d.name, reverse=True)
    if not runs:
        return None
    src = runs[0] / "report.html"
    if not src.exists():
        write_report(runs[0])
    out = Path(out) if out else LATEST_REPORT
    out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def write_comparison(run_a: str | Path, run_b: str | Path) -> Path:
    a, b = Path(run_a), Path(run_b)
    ra, rb = _run_data(a), _run_data(b)
    cmp = compare(load_scores(a), load_scores(b))
    out = b / f"compare_{a.name}.html"
    out.write_text(_page(f"Compare: {b.name} vs {a.name}",
                         {"mode": "compare", "run": rb, "base": ra, "cmp": cmp, "metrics": list(METRICS),
                          "headline": HEADLINE, "lower": sorted(LOWER_IS_BETTER), "desc": DESCRIPTIONS}), encoding="utf-8")
    return out


def write_index(runs_dir: str | Path) -> Path:
    runs_dir = Path(runs_dir)
    runs = []
    for d in sorted((p for p in runs_dir.iterdir() if (p / "summary.json").exists()), key=lambda p: p.name, reverse=True):
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        meta = load_meta(d)
        runs.append({"name": d.name, "meta": {k: meta.get(k) for k in ("system", "dataset", "judge_model", "started_at", "run_id")},
                     "n": summary["overall"]["n_items"],
                     "headline": {h: (summary["overall"].get(h) or {}).get("mean") for h in HEADLINE},
                     "has_report": (d / "report.html").exists()})
    out = runs_dir / "index.html"
    out.write_text(_page("Eval runs", {"mode": "index", "runs": runs, "headline": HEADLINE,
                                       "lower": sorted(LOWER_IS_BETTER), "desc": DESCRIPTIONS}), encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#667085;--line:#e4e7ec;--good:#1a7f4b;--goodbg:#e6f4ec;
--mid:#9a6700;--midbg:#fdf3d7;--bad:#b42318;--badbg:#fde8e7;--accent:#3b5bdb;--code:#f1f3f5}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1a1e24;--ink:#e6e8eb;--mute:#98a2b3;--line:#2c323b;
--good:#4ade80;--goodbg:#13301f;--mid:#facc15;--midbg:#332b0d;--bad:#f87171;--badbg:#3a1616;--accent:#8ea2ff;--code:#22272e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1240px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.sub{color:var(--mute);font-size:13px}.sub code{background:var(--code);padding:1px 5px;border-radius:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{font-size:12px;color:var(--mute);font-weight:600}.card .v{font-size:28px;font-weight:700;margin:2px 0}
.card .d{font-size:12px;color:var(--mute)}.card .ci{font-size:12px;color:var(--mute)}
.good{color:var(--good)}.mid{color:var(--mid)}.bad{color:var(--bad)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:auto}
table{border-collapse:collapse;width:100%}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:12px;color:var(--mute);font-weight:600;position:sticky;top:0;background:var(--card);white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.mdesc{color:var(--mute);font-size:12px}
.cell-good{background:var(--goodbg)}.cell-mid{background:var(--midbg)}.cell-bad{background:var(--badbg)}
.tag{display:inline-block;font-size:11px;font-weight:600;padding:1px 7px;border-radius:99px;white-space:nowrap}
.t-pass{background:var(--goodbg);color:var(--good)}.t-partial{background:var(--midbg);color:var(--mid)}
.t-fail,.t-error{background:var(--badbg);color:var(--bad)}.t-na{background:var(--code);color:var(--mute)}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
select,input{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
input[type=search]{flex:1;min-width:180px}tr.item{cursor:pointer}tr.item:hover{background:var(--code)}
tr.detail td{background:var(--bg);padding:14px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px}
.box h4{margin:0 0 6px;font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.03em}
.box pre{white-space:pre-wrap;word-break:break-word;margin:0;font:12px/1.5 ui-monospace,Consolas,monospace}
.ans{white-space:pre-wrap;word-break:break-word}ul{margin:4px 0;padding-left:18px}
details summary{cursor:pointer;color:var(--accent)}.ctx{border-top:1px dashed var(--line);padding-top:6px;margin-top:6px}
.tabs{display:flex;gap:6px;margin:16px 0 0}.tabs button{background:var(--card);border:1px solid var(--line);color:var(--ink);
border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit}.tabs button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
a{color:var(--accent)}.legend{font-size:12px;color:var(--mute);margin-top:6px}
</style></head><body><main id="app"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const LOWER = new Set(D.lower || []);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = v => v == null || isNaN(v) ? '—' : (v * 100).toFixed(1) + '%';
const level = (name, v) => { if (v == null) return ''; const x = LOWER.has(name) ? 1 - v : v; return x >= 0.8 ? 'good' : x >= 0.5 ? 'mid' : 'bad'; };
const app = document.getElementById('app');

// Metric extractors over a serialized ItemScore (mirrors eval/report.py METRICS).
const M = {
  'correctness (judge)': s => s.judge_score, 'answer match': s => s.exact_match, 'token F1': s => s.token_f1,
  'key-point recall': s => s.key_point_recall, 'faithfulness (RAGAS)': s => (s.ragas||{}).faithfulness,
  'answer relevancy (RAGAS)': s => (s.ragas||{}).answer_relevancy, 'context precision@5 (RAGAS)': s => (s.ragas||{}).context_precision,
  'context recall (RAGAS)': s => (s.ragas||{}).context_recall, 'evidence recall@5': s => (s.evidence_recall_at_k||{})['5'],
  'evidence recall@10': s => (s.evidence_recall_at_k||{})['10'], 'MRR': s => s.mrr, 'SQL result match': s => s.sql_result_match,
  'SQL executed': s => s.sql_executed, 'correct refusal (unanswerable)': s => s.correct_refusal, 'route accuracy': s => s.route_correct,
};

function status(s) {
  if (s.error) return 'error';
  const v = [s.judge_score, s.sql_result_match, s.correct_refusal, s.route_correct].find(x => x != null);
  if (v == null) return 'na';
  return v >= 1 ? 'pass' : v > 0 ? 'partial' : 'fail';
}

function header(run, extra) {
  const m = run.meta || {}, o = run.summary.overall;
  return `<h1>${esc(m.system || 'system')} <span class="sub">${esc(run.name)}</span></h1>
  <div class="sub">${o.n_items} items · dataset <code>${esc(m.dataset)}</code> · judge <code>${esc(m.judge_model || 'off')}</code>
  ${m.ragas ? '· RAGAS ' + esc(m.ragas.ragas_version) : '· RAGAS off'} ${o.latency_ms ? '· latency p50 ' + Math.round(o.latency_ms.p50) + ' ms, p95 ' + Math.round(o.latency_ms.p95) + ' ms' : ''}
  · ~${Math.round(o.tokens.prompt_mean)} prompt / ${Math.round(o.tokens.completion_mean)} completion tokens per item ${extra || ''}</div>`;
}

function cards(summary, cmp) {
  return '<div class="cards">' + D.headline.map(h => {
    const m = summary.overall[h]; const c = cmp && cmp.metrics[h];
    const delta = c ? `<div class="ci ${c.significant ? (c.b_better ? 'good' : 'bad') : ''}">Δ ${(c.delta >= 0 ? '+' : '') + (c.delta * 100).toFixed(1)} pts ${c.significant ? (c.b_better ? '▲ better' : '▼ worse') : '(not significant)'}</div>` : '';
    return `<div class="card"><div class="k">${esc(h)}</div><div class="v ${m ? level(h, m.mean) : ''}">${m ? pct(m.mean) : '—'}</div>
      <div class="ci">${m ? `95% CI ${pct(m.ci95[0])}–${pct(m.ci95[1])} · n=${m.n}` : 'not measured'}</div>${delta}
      <div class="d">${esc(D.desc[h] || '')}</div></div>`;
  }).join('') + '</div>';
}

function metricsTable(summary, base, cmp) {
  const rows = D.metrics.map(name => {
    const m = summary.overall[name]; if (!m && !(base && base.overall[name])) return '';
    const b = base && base.overall[name]; const c = cmp && cmp.metrics[name];
    const verdict = c ? (c.significant ? `<span class="${c.b_better ? 'good' : 'bad'}">${c.b_better ? 'better' : 'worse'}</span>` : '<span class="sub">no sig. change</span>') : '';
    return `<tr><td>${esc(name)}<div class="mdesc">${esc(D.desc[name] || '')}</div></td>
      ${base ? `<td class="num">${b ? pct(b.mean) : '—'}</td>` : ''}
      <td class="num ${m ? 'cell-' + level(name, m.mean) : ''}">${m ? pct(m.mean) : '—'}</td>
      <td class="num">${m ? pct(m.ci95[0]) + ' – ' + pct(m.ci95[1]) : ''}</td><td class="num">${m ? m.n : ''}</td>
      ${cmp ? `<td class="num">${c ? (c.delta >= 0 ? '+' : '') + (c.delta * 100).toFixed(1) + ' pts' : ''}</td><td>${verdict}</td>` : ''}</tr>`;
  }).join('');
  return `<div class="panel"><table><tr><th>Metric</th>${base ? '<th>Base</th>' : ''}<th>${base ? 'This run' : 'Value'}</th><th>95% CI</th><th>n</th>${cmp ? '<th>Δ</th><th>Verdict</th>' : ''}</tr>${rows}</table></div>`;
}

function breakdown(groups, label) {
  const cols = D.headline;
  const rows = Object.entries(groups).map(([g, blk]) => `<tr><td>${esc(g)}</td><td class="num">${blk.n_items}</td>` +
    cols.map(h => { const m = blk[h]; return `<td class="num ${m ? 'cell-' + level(h, m.mean) : ''}">${m ? pct(m.mean) : '—'}</td>`; }).join('') + '</tr>').join('');
  return `<div class="panel"><table><tr><th>${label}</th><th>n</th>${cols.map(h => `<th title="${esc(D.desc[h] || '')}">${esc(h.replace(' (RAGAS)', '').replace(' (judge)', ''))}</th>`).join('')}</tr>${rows}</table></div>
  <div class="legend">Green ≥ 80%, amber 50–80%, red &lt; 50%. Hover a column for its meaning.</div>`;
}

function list(items) { return items && items.length ? '<ul>' + items.map(x => `<li>${esc(x)}</li>`).join('') + '</ul>' : '<span class="sub">none</span>'; }

function detail(r, baseRow) {
  const s = r.score, it = r.item, p = r.pred;
  const ragas = Object.entries(s.ragas || {}).map(([k, v]) => `${k}: ${pct(v)}`).join(' · ') || 'not scored';
  const retr = Object.entries(s.evidence_recall_at_k || {}).map(([k, v]) => `@${k}: ${pct(v)}`).join(' · ');
  const sqlRes = p.sql_result ? `<pre>${esc(p.sql_result.columns.join(' | '))}\n${p.sql_result.rows.slice(0, 15).map(x => x.join(' | ')).map(esc).join('\n')}</pre>` : '';
  const hist = (it.history || []).map(t => `<div><b>${esc(t.role)}:</b> ${esc(t.content)}</div>`).join('');
  const ctx = r.contexts.map((c, i) => `<div class="ctx"><b>#${i + 1}</b> ${esc(c.doc || '')} p.${esc(c.page ?? '?')}<pre>${esc(c.text)}</pre></div>`).join('');
  return `<div class="grid2"><div>
    <div class="box"><h4>Question</h4>${hist ? hist + '<hr>' : ''}<div class="ans">${esc(r.question)}</div></div>
    <div class="box"><h4>Gold answer</h4><div class="ans">${it.answerable === false ? '<i>Unanswerable: the system should say it does not know.</i>' : esc(it.answer)}</div>
      ${(it.acceptable || []).length ? '<div class="sub">also accepted: ' + esc(it.acceptable.join(', ')) + '</div>' : ''}
      ${(it.key_points || []).length ? '<h4 style="margin-top:8px">Key points</h4>' + list(it.key_points) : ''}
      ${it.gold_sql_rows ? '<h4 style="margin-top:8px">Gold rows</h4><pre>' + esc(it.gold_sql_rows.map(x => x.join(' | ')).join('\n')) + '</pre>' : ''}</div>
    ${baseRow ? `<div class="box"><h4>Base run answer</h4><div class="ans">${esc(baseRow.pred.answer)}</div></div>` : ''}
  </div><div>
    <div class="box"><h4>System answer</h4><div class="ans">${esc(p.answer || p.error || '')}</div></div>
    <div class="box"><h4>Why this score</h4>
      <div><b>Judge:</b> ${esc(s.judge_label || '—')} ${s.judge_reason ? '— ' + esc(s.judge_reason) : ''}</div>
      <div><b>RAGAS:</b> ${ragas}</div>${retr ? `<div><b>Evidence recall:</b> ${retr} · MRR ${s.mrr == null ? '—' : s.mrr.toFixed(2)}</div>` : ''}
      ${s.key_point_recall != null ? `<div><b>Key-point recall:</b> ${pct(s.key_point_recall)}</div><div>Missed:</div>${list(s.missed_key_points)}` : ''}
      ${(s.unsupported_claims || []).length ? '<div><b>Unsupported claims:</b></div>' + list(s.unsupported_claims) : ''}
      <div><b>Route:</b> ${esc(p.route)} (expected ${esc(it.expected_route)}) · <b>scope:</b> ${esc(p.query_scope)}${it.expected_scope ? ' (expected ' + esc(it.expected_scope) + ')' : ''} · <b>doc:</b> ${esc(p.chosen_doc || '—')}</div>
      <div><b>Latency:</b> ${Math.round(s.total_ms)} ms · <b>tokens:</b> ${s.prompt_tokens} / ${s.completion_tokens}</div></div>
    ${p.sql || p.sql_error ? `<div class="box"><h4>SQL</h4><pre>${esc(p.sql || '')}</pre>${p.sql_error ? '<div class="bad">' + esc(p.sql_error) + '</div>' : ''}${sqlRes}</div>` : ''}
    ${ctx ? `<div class="box"><details><summary>Retrieved chunks (${r.contexts.length})</summary>${ctx}</details></div>` : ''}
  </div></div>`;
}

function itemsSection(run, base) {
  const baseBy = base ? Object.fromEntries(base.rows.map(r => [r.score.item_id, r])) : null;
  const rows = run.rows.map(r => ({...r, question: r.item.question || r.score.item_id, st: status(r.score),
    bst: baseBy && baseBy[r.score.item_id] ? status(baseBy[r.score.item_id].score) : null}));
  const cats = [...new Set(rows.map(r => r.score.category))].sort(), docs = [...new Set(rows.map(r => r.score.doc || '—'))].sort();
  const opt = (id, label, vals) => `<select id="${id}"><option value="">${label}: all</option>${vals.map(v => `<option>${esc(v)}</option>`).join('')}</select>`;
  setTimeout(() => {
    const tbody = document.getElementById('items');
    const draw = () => {
      const f = {c: cat.value, d: doc.value, s: st.value, q: q.value.toLowerCase()};
      const shown = rows.filter(r => (!f.c || r.score.category === f.c) && (!f.d || (r.score.doc || '—') === f.d) &&
        (!f.s || r.st === f.s || (f.s === 'changed' && r.bst && r.bst !== r.st)) && (!f.q || (r.question + ' ' + r.score.item_id).toLowerCase().includes(f.q)));
      document.getElementById('count').textContent = `${shown.length} of ${rows.length} items`;
      tbody.innerHTML = shown.map((r, i) => `<tr class="item" data-i="${rows.indexOf(r)}"><td><span class="tag t-${r.st}">${r.st}</span>${r.bst && r.bst !== r.st ? ` <span class="sub">was ${r.bst}</span>` : ''}</td>
        <td>${esc(r.question)}<div class="mdesc">${esc(r.score.item_id)}</div></td><td>${esc(r.score.category)}</td><td>${esc(r.score.doc || '—')}</td>
        <td class="num">${pct(r.score.judge_score)}</td><td class="num">${pct((r.score.ragas || {}).faithfulness)}</td>
        <td class="num">${pct((r.score.evidence_recall_at_k || {})['10'])}</td><td class="num">${Math.round(r.score.total_ms / 100) / 10}s</td></tr>`).join('');
    };
    tbody.onclick = e => {
      const tr = e.target.closest('tr.item'); if (!tr) return;
      if (tr.nextElementSibling && tr.nextElementSibling.classList.contains('detail')) { tr.nextElementSibling.remove(); return; }
      const r = rows[+tr.dataset.i]; const d = document.createElement('tr'); d.className = 'detail';
      d.innerHTML = `<td colspan="8">${detail(r, baseBy && baseBy[r.score.item_id])}</td>`; tr.after(d);
    };
    const cat = document.getElementById('f-cat'), doc = document.getElementById('f-doc'), st = document.getElementById('f-st'), q = document.getElementById('f-q');
    [cat, doc, st].forEach(x => x.onchange = draw); q.oninput = draw; draw();
  });
  return `<div class="filters">${opt('f-cat', 'Category', cats)}${opt('f-doc', 'Document', docs)}
    <select id="f-st"><option value="">Status: all</option><option>pass</option><option>partial</option><option>fail</option><option>error</option><option>na</option>${base ? '<option value="changed">changed vs base</option>' : ''}</select>
    <input id="f-q" type="search" placeholder="Search questions…"><span class="sub" id="count" style="align-self:center"></span></div>
    <div class="panel"><table><thead><tr><th>Status</th><th>Question</th><th>Category</th><th>Doc</th><th>Judge</th><th>Faithful</th><th>Ev.@10</th><th>Time</th></tr></thead><tbody id="items"></tbody></table></div>
    <div class="legend">Status uses the judge score, else SQL match, else refusal, else routing. Click a row for the full story.</div>`;
}

function tabs(sections) {
  const ids = Object.keys(sections);
  setTimeout(() => {
    const show = id => { ids.forEach(k => { document.getElementById('sec-' + k).style.display = k === id ? '' : 'none';
      document.getElementById('tab-' + k).classList.toggle('on', k === id); }); try { localStorage.setItem('evalTab', id); } catch (e) {} };
    ids.forEach(k => document.getElementById('tab-' + k).onclick = () => show(k));
    let saved = null; try { saved = localStorage.getItem('evalTab'); } catch (e) {}
    show(ids.includes(saved) ? saved : ids[0]);
  });
  return `<div class="tabs">${ids.map(k => `<button id="tab-${k}">${esc(sections[k][0])}</button>`).join('')}</div>` +
    ids.map(k => `<section id="sec-${k}">${sections[k][1]}</section>`).join('');
}

if (D.mode === 'index') {
  const rows = D.runs.map(r => `<tr><td>${r.has_report ? `<a href="${esc(r.name)}/report.html">${esc(r.name)}</a>` : esc(r.name)}</td>
    <td>${esc(r.meta.system)}</td><td>${esc((r.meta.dataset || '').split(/[\\/]/).pop())}</td><td class="num">${r.n}</td>` +
    D.headline.map(h => `<td class="num ${r.headline[h] != null ? 'cell-' + level(h, r.headline[h]) : ''}">${pct(r.headline[h])}</td>`).join('') + '</tr>').join('');
  app.innerHTML = `<h1>Eval runs</h1><div class="sub">${D.runs.length} runs, newest first. Open a run for details; compare two with <code>python -m eval view RUN_A RUN_B</code>.</div>
    <h2>Runs</h2><div class="panel"><table><tr><th>Run</th><th>System</th><th>Dataset</th><th>n</th>${D.headline.map(h => `<th title="${esc(D.desc[h] || '')}">${esc(h.replace(' (RAGAS)', '').replace(' (judge)', ''))}</th>`).join('')}</tr>${rows}</table></div>`;
} else {
  const run = D.run, base = D.base, cmp = D.cmp;
  app.innerHTML = header(run, base ? `<br>Compared against base <code>${esc(base.name)}</code> on ${cmp.n_common_items} shared items` : '') +
    cards(run.summary, cmp) + tabs({
      metrics: ['All metrics', '<h2>All metrics</h2>' + metricsTable(run.summary, base && base.summary, cmp)],
      cat: ['By category', '<h2>By category</h2>' + breakdown(run.summary.by_category, 'Category')],
      doc: ['By document', '<h2>By document</h2>' + breakdown(run.summary.by_document || {}, 'Document')],
      items: ['Items', '<h2>Items</h2>' + itemsSection(run, base)],
    });
}
</script></body></html>
"""
