# src/reporting/report_builder.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.persistence import state_store as store
from src.metrics.metrics import summarize_run, metrics_snapshot

# write a full report file for a run
def generate_report(
    run_id: int,
    *,
    out_dir: str = "reports",
    fmt: str = "html", #html or md
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    run = store.get_run(run_id) or {}
    cal_rows = store.get_calibrations(run_id)
    dec_rows = store.get_decisions(run_id, limit=1000000)

    # parse decisions
    traces = []
    for r in dec_rows:
        try:
            traces.append(json.loads(r["trace_json"]))
        except Exception:
            continue

    snapshot = metrics_snapshot(_as_traces(traces), pseudo_labels={})

    # collect current thresholds from learner states
    thresholds = {}
    briers = {}
    for row in cal_rows:
        name = row["learner_name"]
        st = store.load_learner_state(name)
        thresholds[name] = None if st is None else (st.calibration.threshold if st.calibration else None)
        briers[name] = None if st is None else st.calibration.brier_score

    # build sections
    started = run.get("started_at")
    ended = run.get("ended_at")
    cfg = _pretty_json(run.get("config_json", "{}"))

    cal_table = _calibration_table(cal_rows, thresholds, briers)
    run_table = _run_table(snapshot.get("run", {}))
    examples = _select_examples(traces)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if fmt.lower() == "md":
        content = _render_markdown(run_id, started, ended, cfg, cal_table, run_table, examples, snapshot)
        path = os.path.join(out_dir, f"run_{run_id}_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    else:
        content = _render_html(run_id, started, ended, cfg, cal_table, run_table, examples, snapshot)
        path = os.path.join(out_dir, f"run_{run_id}_{ts}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

# convert dict traces to a light object with required fields
def _as_traces(traces_json: List[Dict[str, Any]]):
    from types import SimpleNamespace
    out = []
    for obj in traces_json:
        ln = {}
        for k, v in obj.get("learners", {}).items():
            ln[k] = SimpleNamespace(prob=float(v.get("prob", 0.0)))
        out.append(
            SimpleNamespace(
                pair_key=obj.get("pair_key"),
                a_id=obj.get("a_id"),
                b_id=obj.get("b_id"),
                final_label=obj.get("final_label"),
                agreed_learners=obj.get("agreed_learners", []),
                escalation_steps=obj.get("escalation_steps", []),
                learner_outputs=ln,
            )
        )
    return out

# pretty JSON for config snapshot
def _pretty_json(s: str) -> str:
    try:
        return json.dumps(json.loads(s or "{}"), indent=2, ensure_ascii=False)
    except Exception:
        return s or ""

# build calibration summary table rows
def _calibration_table(cal_rows: List[Dict[str, Any]], thresholds: Dict[str, Any], briers: Dict[str, Any]):
    rows = []
    for r in cal_rows:
        name = r["learner_name"]
        method = r.get("method") or "-"
        params = r.get("params_json") or "{}"
        reliability = r.get("reliability_json") or "[]"
        try:
            rb = json.loads(reliability)
            rb_str = ", ".join(f"{x.get('prob_center'):.2f}:{x.get('observed_pos_rate', 0):.2f}({x.get('count',0)})" for x in rb[:6])
        except Exception:
            rb_str = "-"
        rows.append({
            "learner": name,
            "method": method,
            "threshold": _fmt_num(thresholds.get(name)),
            "brier": _fmt_num(briers.get(name)),
            "params": params,
            "reliability": rb_str,
        })
    return rows

# build run metrics table
def _run_table(run: Dict[str, Any]):
    return [
        {"metric": "Total pairs", "value": str(run.get("total_pairs", 0))},
        {"metric": "Duplicates", "value": str(run.get("duplicates", 0))},
        {"metric": "Non-duplicates", "value": str(run.get("non_duplicates", 0))},
        {"metric": "Uncertain", "value": str(run.get("uncertain", 0))},
        {"metric": "Consensus rate", "value": f"{100.0 * float(run.get('consensus_rate', 0.0)):.1f}%"},
        {"metric": "Escalations", "value": f"{100.0 * float(run.get('escalations_pct', 0.0)):.1f}%"},
    ]

# pick example decisions for the report
def _select_examples(traces: List[Dict[str, Any]], k_easy: int = 5, k_hard: int = 5, k_unc: int = 5):
    easy = []
    hard = []
    uncertain = []
    for tr in traces:
        label = tr.get("final_label")
        learners = tr.get("learners", {})
        probs = [float(v.get("prob", 0.0)) for v in learners.values()]
        maxp = max(probs) if probs else 0.0
        minp = min(probs) if probs else 0.0
        esc = len(tr.get("escalation_steps", [])) > 0
        if label == "DUPLICATE" and not esc and maxp >= 0.95:
            easy.append(tr)
        elif label == "DUPLICATE" and esc:
            hard.append(tr)
        elif label == "UNCERTAIN":
            uncertain.append(tr)
    easy = easy[:k_easy]
    hard = hard[:k_hard]
    uncertain = uncertain[:k_unc]
    return {"easy": easy, "hard": hard, "uncertain": uncertain}

# html rendering
def _render_html(run_id: int, started: Optional[str], ended: Optional[str], cfg: str, cal_table, run_table, examples, snapshot) -> str:
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px; color: #1f2937; }
      h1 { font-size: 24px; margin: 0 0 12px 0; }
      h2 { font-size: 18px; margin-top: 24px; }
      code, pre { background: #f9fafb; padding: 8px; border-radius: 6px; display: block; overflow-x: auto; }
      table { border-collapse: collapse; width: 100%; margin-top: 8px; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; font-size: 13px; }
      th { background: #f3f4f6; }
      .pill { display: inline-block; padding: 2px 8px; border-radius: 9999px; background:#eef2ff; color:#3730a3; font-size:12px; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    </style>
    """
    header = f"<h1>Duplicate Detection Report — Run {run_id}</h1>"
    meta = f"<p><span class='pill'>Started</span> { _esc(started or '-') } &nbsp;&nbsp; <span class='pill'>Ended</span> { _esc(ended or '-') }</p>"

    run_table_html = _table_html(["Metric", "Value"], [[r["metric"], r["value"]] for r in run_table])

    cal_rows = []
    for r in cal_table:
        cal_rows.append([r["learner"], r["method"], r["threshold"], r["brier"], r["reliability"]])
    cal_html = _table_html(["Learner", "Method", "Threshold", "Brier", "Reliability (center:obs(count))"], cal_rows)

    ex_html = _examples_html(examples)

    cfg_html = f"<pre><code>{_esc(cfg)}</code></pre>"

    clusters_html = _clusters_html(snapshot.get("clusters", []))

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Run {run_id} Report</title>{css}</head>
<body>
{header}
{meta}

<h2>Run summary</h2>
{run_table_html}

<h2>Calibration snapshot</h2>
{cal_html}

<h2>Examples</h2>
{ex_html}

<h2>Clusters (from DUPLICATE traces)</h2>
{clusters_html}

<h2>Config</h2>
{cfg_html}
</body></html>"""

# markdown rendering
def _render_markdown(run_id: int, started: Optional[str], ended: Optional[str], cfg: str, cal_table, run_table, examples, snapshot) -> str:
    lines = []
    lines.append(f"# Duplicate Detection Report — Run {run_id}")
    lines.append("")
    lines.append(f"- **Started:** {started or '-'}")
    lines.append(f"- **Ended:** {ended or '-'}")
    lines.append("")
    lines.append("## Run summary")
    lines.append(_table_md(["Metric", "Value"], [[r["metric"], r["value"]] for r in run_table]))
    lines.append("")
    lines.append("## Calibration snapshot")
    cal_rows = [[r["learner"], r["method"], str(r["threshold"]), str(r["brier"]), r["reliability"]] for r in cal_table]
    lines.append(_table_md(["Learner", "Method", "Threshold", "Brier", "Reliability (center:obs(count))"], cal_rows))
    lines.append("")
    lines.append("## Examples")
    lines.extend(_examples_md(examples))
    lines.append("")
    lines.append("## Clusters (from DUPLICATE traces)")
    lines.append(_clusters_md(snapshot.get("clusters", [])))
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(cfg)
    lines.append("```")
    return "\n".join(lines)

# examples as HTML
def _examples_html(examples: Dict[str, List[Dict[str, Any]]]) -> str:
    out = []
    for title, rows in [("Easy positives", examples.get("easy", [])),
                        ("Hard but resolved (escalated)", examples.get("hard", [])),
                        ("Uncertain", examples.get("uncertain", []))]:
        out.append(f"<h3>{_esc(title)}</h3>")
        if not rows:
            out.append("<p><i>None</i></p>")
            continue
        hdr = ["Doc A", "Doc B", "Final", "Agreed", "Escalation", "Per-learner probs"]
        body = []
        for tr in rows:
            learners = ", ".join(f"{_esc(k)}={float(v.get('prob',0.0)):.2f}" for k, v in tr.get("learners", {}).items())
            body.append([
                _esc(tr.get("a_id","")), _esc(tr.get("b_id","")),
                _esc(tr.get("final_label","")),
                _esc(", ".join(tr.get("agreed_learners",[]))),
                _esc(" → ".join(tr.get("escalation_steps",[]))),
                _esc(learners),
            ])
        out.append(_table_html(hdr, body))
    return "\n".join(out)

# examples as Markdown
def _examples_md(examples: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = []
    for title, rows in [("Easy positives", examples.get("easy", [])),
                        ("Hard but resolved (escalated)", examples.get("hard", [])),
                        ("Uncertain", examples.get("uncertain", []))]:
        lines.append(f"### {title}")
        if not rows:
            lines.append("_None_")
            lines.append("")
            continue
        hdr = ["Doc A", "Doc B", "Final", "Agreed", "Escalation", "Per-learner probs"]
        body = []
        for tr in rows:
            learners = ", ".join(f"{k}={float(v.get('prob',0.0)):.2f}" for k, v in tr.get("learners", {}).items())
            body.append([tr.get("a_id",""), tr.get("b_id",""), tr.get("final_label",""),
                         ", ".join(tr.get("agreed_learners",[])), " → ".join(tr.get("escalation_steps",[])), learners])
        lines.append(_table_md(hdr, body))
        lines.append("")
    return lines

# clusters html
def _clusters_html(clusters: List[Dict[str, Any]]) -> str:
    if not clusters:
        return "<p><i>No clusters found</i></p>"
    hdr = ["#", "Size", "Members", "Avg prob (simhash|minhash|embed)", "Dispersion (min..max)"]
    body = []
    for c in clusters:
        avg = f"{_fmt_num(c.get('avg_simhash_prob'))}|{_fmt_num(c.get('avg_minhash_prob'))}|{_fmt_num(c.get('avg_embedding_prob'))}"
        disp = f"{_disp(c.get('dispersion_simhash'))} | {_disp(c.get('dispersion_minhash'))} | {_disp(c.get('dispersion_embedding'))}"
        body.append([str(c.get("cluster_index")), str(c.get("size")), ", ".join(c.get("members",[])), avg, disp])
    return _table_html(hdr, body)

# clusters md
def _clusters_md(clusters: List[Dict[str, Any]]) -> str:
    if not clusters:
        return "_No clusters found_"
    hdr = ["#", "Size", "Members", "Avg prob (simhash|minhash|embed)", "Dispersion (min..max)"]
    body = []
    for c in clusters:
        avg = f"{_fmt_num(c.get('avg_simhash_prob'))}|{_fmt_num(c.get('avg_minhash_prob'))}|{_fmt_num(c.get('avg_embedding_prob'))}"
        disp = f"{_disp(c.get('dispersion_simhash'))} | {_disp(c.get('dispersion_minhash'))} | {_disp(c.get('dispersion_embedding'))}"
        body.append([str(c.get("cluster_index")), str(c.get("size")), ", ".join(c.get("members",[])), avg, disp])
    return _table_md(hdr, body)

# simple HTML table
def _table_html(headers: List[str], rows: List[List[str]]) -> str:
    thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{_esc(str(x))}</td>" for x in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"

# simple MD table
def _table_md(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
    return "\n".join([head, sep, body])

def _fmt_num(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def _disp(mm: Optional[Dict[str, Any]]) -> str:
    if not mm:
        return "-"
    mn = _fmt_num(mm.get("min"))
    mx = _fmt_num(mm.get("max"))
    return f"{mn}..{mx}"

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
