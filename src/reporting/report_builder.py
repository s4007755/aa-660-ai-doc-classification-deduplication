# src/reporting/report_builder.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.persistence import state_store as store
from src.metrics.metrics import metrics_snapshot

try:
    from src.storage import sqlite_store
except Exception:
    sqlite_store = None


# Public API
def generate_report(
    run_id: int,
    *,
    out_dir: str = "reports",
    fmt: str = "html",  # "html" or "md"
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Run + decisions + calibrations from persistence
    run = store.get_run(run_id) or {}
    cal_rows = store.get_calibrations(run_id)
    dec_rows = store.get_decisions(run_id, limit=1_000_000)

    # Parse decision traces back from JSON
    traces_json: List[Dict[str, Any]] = []
    for r in dec_rows:
        try:
            traces_json.append(json.loads(r["trace_json"]))
        except Exception:
            continue

    # Build metrics snapshot from traces
    snapshot = metrics_snapshot(_as_traces(traces_json), pseudo_labels={})
    labels = _load_doc_labels()

    # Collect current thresholds/briers from learner states
    thresholds: Dict[str, Optional[float]] = {}
    briers: Dict[str, Optional[float]] = {}
    for row in cal_rows:
        name = row["learner_name"]
        st = store.load_learner_state(name)
        thr = None
        br = None
        if st is not None and getattr(st, "calibration", None) is not None:
            thr = st.calibration.threshold
            br = st.calibration.brier_score
        thresholds[name] = thr
        briers[name] = br

    # Build sections
    started = run.get("started_at")
    ended = run.get("ended_at")
    notes = run.get("notes") or ""
    cfg_str = _pretty_json(run.get("config_json", "{}"))
    cal_table = _calibration_table(cal_rows, thresholds, briers)
    run_table = _run_table(snapshot.get("run", {}))
    per_learner_rows = _per_learner_rows(snapshot.get("per_learner", {}))
    examples = _select_examples(traces_json, labels=labels)
    clusters = snapshot.get("clusters", [])

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if fmt.lower() == "md":
        content = _render_markdown(
            run_id, started, ended, notes, cfg_str, cal_table, run_table, per_learner_rows, examples, clusters, labels
        )
        path = os.path.join(out_dir, f"run_{run_id}_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)
    else:
        content = _render_html(
            run_id, started, ended, notes, cfg_str, cal_table, run_table, per_learner_rows, examples, clusters, labels
        )
        path = os.path.join(out_dir, f"run_{run_id}_{ts}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)


# Internal helpers
def _as_traces(traces_json: List[Dict[str, Any]]):
    from types import SimpleNamespace
    out = []
    for obj in traces_json:
        ln = {}
        for k, v in (obj.get("learners") or {}).items():
            try:
                ln[k] = SimpleNamespace(prob=float(v.get("prob", 0.0)))
            except Exception:
                ln[k] = SimpleNamespace(prob=0.0)
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


def _pretty_json(s: str) -> str:
    try:
        return json.dumps(json.loads(s or "{}"), indent=2, ensure_ascii=False)
    except Exception:
        return s or ""


def _calibration_table(cal_rows: List[Dict[str, Any]], thresholds: Dict[str, Any], briers: Dict[str, Any]):
    rows = []
    for r in cal_rows:
        name = r["learner_name"]
        method = r.get("method") or "-"
        reliability = r.get("reliability_json") or "[]"
        try:
            rb = json.loads(reliability)
            rb_str = ", ".join(
                f"{x.get('prob_center'):.2f}:{x.get('observed_pos_rate', 0):.2f}({x.get('count',0)})"
                for x in rb[:6]
            )
        except Exception:
            rb_str = "-"
        rows.append(
            {
                "learner": name,
                "method": method,
                "threshold": _fmt_num(thresholds.get(name)),
                "brier": _fmt_num(briers.get(name)),
                "reliability": rb_str,
            }
        )
    return rows


def _run_table(run: Dict[str, Any]):
    cons = run.get("consensus_rate", run.get("consensus_pct", 0.0))
    esc = run.get("escalations_pct", run.get("escalations_rate", 0.0))
    total = run.get("total_pairs", run.get("pairs_scored", run.get("pairs", 0)))
    return [
        {"metric": "Total pairs", "value": str(total)},
        {"metric": "Duplicates", "value": str(run.get("duplicates", 0))},
        {"metric": "Non-duplicates", "value": str(run.get("non_duplicates", 0))},
        {"metric": "Uncertain", "value": str(run.get("uncertain", 0))},
        {"metric": "Consensus rate", "value": f"{100.0 * float(cons or 0.0):.1f}%"},
        {"metric": "Escalations", "value": f"{100.0 * float(esc or 0.0):.1f}%"},
        {"metric": "Clusters", "value": str(run.get("clusters", "-"))},
    ]


def _per_learner_rows(per_learner: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    if not isinstance(per_learner, dict):
        return rows
    for name, m in per_learner.items():
        n = m.get("n")
        pos = m.get("pos_rate")
        auc = m.get("auc")
        brier = m.get("brier")
        thr = m.get("threshold")
        tgt = m.get("target_precision") or m.get("target")
        rows.append(
            [
                str(name),
                str(n if n is not None else "-"),
                _fmt_num(pos),
                _fmt_num(auc),
                _fmt_num(brier),
                _fmt_num(thr),
                _fmt_num(tgt),
            ]
        )
    order = {"simhash": 0, "minhash": 1, "embedding": 2}
    rows.sort(key=lambda r: order.get(r[0].split()[0].lower(), 99))
    return rows


def _select_examples(
    traces: List[Dict[str, Any]],
    *,
    k_easy: int = 5,
    k_hard: int = 5,
    k_unc: int = 5,
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    labels = labels or {}
    easy: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []
    for tr in traces:
        label = tr.get("final_label")
        learners = tr.get("learners", {})
        probs = []
        for v in learners.values():
            try:
                probs.append(float(v.get("prob", 0.0)))
            except Exception:
                probs.append(0.0)
        maxp = max(probs) if probs else 0.0
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
    for bucket in (easy, hard, uncertain):
        for tr in bucket:
            tr["a_name"] = _pretty_doc(tr.get("a_id"), labels)
            tr["b_name"] = _pretty_doc(tr.get("b_id"), labels)
    return {"easy": easy, "hard": hard, "uncertain": uncertain}


def _load_doc_labels() -> Dict[str, str]:
    labels: Dict[str, str] = {}
    try:
        if sqlite_store and hasattr(sqlite_store, "get_all_document_files"):
            rows = sqlite_store.get_all_document_files()
            for r in rows:
                did = r.get("doc_id")
                name = r.get("filename") or os.path.basename(r.get("filepath") or "") or None
                if did and name and did not in labels:
                    labels[did] = name
    except Exception:
        pass
    return labels


def _pretty_doc(doc_id: Optional[str], labels: Dict[str, str]) -> str:
    if not doc_id:
        return "—"
    name = labels.get(doc_id)
    return f"{name} ({doc_id[:8]})" if name else doc_id


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


# Rendering

def _render_html(
    run_id: int,
    started: Optional[str],
    ended: Optional[str],
    notes: str,
    cfg: str,
    cal_table,
    run_table,
    per_learner_rows: List[List[str]],
    examples: Dict[str, List[Dict[str, Any]]],
    clusters: List[Dict[str, Any]],
    labels: Dict[str, str],
) -> str:
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
      .muted { color:#6b7280; }
    </style>
    """
    header = f"<h1>Duplicate Detection Report — Run {run_id}</h1>"
    meta = (
        f"<p><span class='pill'>Started</span> { _esc(started or '-') } "
        f"&nbsp;&nbsp; <span class='pill'>Ended</span> { _esc(ended or '-') }"
        f"{' &nbsp;&nbsp; <span class=\"pill\">Notes</span> ' + _esc(notes) if notes else ''}</p>"
    )

    # Run summary
    run_table_html = _table_html(["Metric", "Value"], [[r["metric"], r["value"]] for r in run_table])

    # Per-learner metrics
    pl_headers = ["Learner", "N", "PosRate", "AUC", "Brier", "Threshold", "Target precision"]
    pl_html = _table_html(pl_headers, per_learner_rows) if per_learner_rows else "<p class='muted'>No per-learner metrics.</p>"

    # Calibration snapshot
    cal_rows = []
    for r in cal_table:
        cal_rows.append([r["learner"], r["method"], r["threshold"], r["brier"], r["reliability"]])
    cal_html = _table_html(["Learner", "Method", "Threshold", "Brier", "Reliability (center:obs(count))"], cal_rows)

    # Examples
    ex_html = _examples_html(examples)

    # Clusters
    clusters_html = _clusters_html(clusters, labels)

    # Config
    cfg_html = f"<pre><code>{_esc(cfg)}</code></pre>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Run {run_id} Report</title>{css}</head>
<body>
{header}
{meta}

<h2>Run summary</h2>
{run_table_html}

<h2>Per-learner metrics</h2>
{pl_html}

<h2>Calibration snapshot</h2>
{cal_html}

<h2>Examples</h2>
{ex_html}

<h2>Clusters (from DUPLICATE traces)</h2>
{clusters_html}

<h2>Config</h2>
{cfg_html}
</body></html>"""


def _render_markdown(
    run_id: int,
    started: Optional[str],
    ended: Optional[str],
    notes: str,
    cfg: str,
    cal_table,
    run_table,
    per_learner_rows: List[List[str]],
    examples: Dict[str, List[Dict[str, Any]]],
    clusters: List[Dict[str, Any]],
    labels: Dict[str, str],
) -> str:
    lines = []
    lines.append(f"# Duplicate Detection Report — Run {run_id}")
    lines.append("")
    lines.append(f"- **Started:** {started or '-'}")
    lines.append(f"- **Ended:** {ended or '-'}")
    if notes:
        lines.append(f"- **Notes:** {notes}")
    lines.append("")
    lines.append("## Run summary")
    lines.append(_table_md(["Metric", "Value"], [[r["metric"], r["value"]] for r in run_table]))
    lines.append("")
    lines.append("## Per-learner metrics")
    if per_learner_rows:
        lines.append(
            _table_md(
                ["Learner", "N", "PosRate", "AUC", "Brier", "Threshold", "Target precision"],
                per_learner_rows,
            )
        )
    else:
        lines.append("_No per-learner metrics._")
    lines.append("")
    lines.append("## Calibration snapshot")
    cal_rows = [[r["learner"], r["method"], str(r["threshold"]), str(r["brier"]), r["reliability"]] for r in cal_table]
    lines.append(_table_md(["Learner", "Method", "Threshold", "Brier", "Reliability (center:obs(count))"], cal_rows))
    lines.append("")
    lines.append("## Examples")
    lines.extend(_examples_md(examples))
    lines.append("")
    lines.append("## Clusters (from DUPLICATE traces)")
    lines.append(_clusters_md(clusters, labels))
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(cfg)
    lines.append("```")
    return "\n".join(lines)


# HTML/MD section builders

def _examples_html(examples: Dict[str, List[Dict[str, Any]]]) -> str:
    out = []
    for title, rows in [
        ("Easy positives", examples.get("easy", [])),
        ("Hard but resolved (escalated)", examples.get("hard", [])),
        ("Uncertain", examples.get("uncertain", [])),
    ]:
        out.append(f"<h3>{_esc(title)}</h3>")
        if not rows:
            out.append("<p><i>None</i></p>")
            continue
        hdr = ["Doc A", "Doc B", "Final", "Agreed", "Escalation", "Per-learner probs"]
        body = []
        for tr in rows:
            learners = ", ".join(f"{_esc(k)}={float(v.get('prob',0.0)):.2f}" for k, v in (tr.get("learners") or {}).items())
            body.append(
                [
                    _esc(tr.get("a_name") or tr.get("a_id") or ""),
                    _esc(tr.get("b_name") or tr.get("b_id") or ""),
                    _esc(tr.get("final_label", "")),
                    _esc(", ".join(tr.get("agreed_learners", []))),
                    _esc(" → ".join(tr.get("escalation_steps", []))),
                    _esc(learners),
                ]
            )
        out.append(_table_html(hdr, body))
    return "\n".join(out)


def _examples_md(examples: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    lines = []
    for title, rows in [
        ("Easy positives", examples.get("easy", [])),
        ("Hard but resolved (escalated)", examples.get("hard", [])),
        ("Uncertain", examples.get("uncertain", [])),
    ]:
        lines.append(f"### {title}")
        if not rows:
            lines.append("_None_")
            lines.append("")
            continue
        hdr = ["Doc A", "Doc B", "Final", "Agreed", "Escalation", "Per-learner probs"]
        body = []
        for tr in rows:
            learners = ", ".join(f"{k}={float(v.get('prob',0.0)):.2f}" for k, v in (tr.get("learners") or {}).items())
            body.append(
                [
                    tr.get("a_name") or tr.get("a_id") or "",
                    tr.get("b_name") or tr.get("b_id") or "",
                    tr.get("final_label", ""),
                    ", ".join(tr.get("agreed_learners", [])),
                    " → ".join(tr.get("escalation_steps", [])),
                    learners,
                ]
            )
        lines.append(_table_md(hdr, body))
        lines.append("")
    return lines


def _clusters_html(clusters: List[Dict[str, Any]], labels: Dict[str, str]) -> str:
    if not clusters:
        return "<p><i>No clusters found</i></p>"
    hdr = ["#", "Size", "Members", "Avg prob (simhash|minhash|embed)", "Dispersion (min..max)"]
    body = []
    for c in clusters:
        members = [ _pretty_doc(m, labels) for m in c.get("members", []) ]
        avg = f"{_fmt_num(c.get('avg_simhash_prob'))}|{_fmt_num(c.get('avg_minhash_prob'))}|{_fmt_num(c.get('avg_embedding_prob'))}"
        disp = f"{_disp(c.get('dispersion_simhash'))} | {_disp(c.get('dispersion_minhash'))} | {_disp(c.get('dispersion_embedding'))}"
        body.append([str(c.get("cluster_index")), str(c.get("size")), ", ".join(members), avg, disp])
    return _table_html(hdr, body)


def _clusters_md(clusters: List[Dict[str, Any]], labels: Dict[str, str]) -> str:
    if not clusters:
        return "_No clusters found_"
    hdr = ["#", "Size", "Members", "Avg prob (simhash|minhash|embed)", "Dispersion (min..max)"]
    body = []
    for c in clusters:
        members = [ _pretty_doc(m, labels) for m in c.get("members", []) ]
        avg = f"{_fmt_num(c.get('avg_simhash_prob'))}|{_fmt_num(c.get('avg_minhash_prob'))}|{_fmt_num(c.get('avg_embedding_prob'))}"
        disp = f"{_disp(c.get('dispersion_simhash'))} | {_disp(c.get('dispersion_minhash'))} | {_disp(c.get('dispersion_embedding'))}"
        body.append([str(c.get("cluster_index")), str(c.get("size")), ", ".join(members), avg, disp])
    return _table_md(hdr, body)


def _table_html(headers: List[str], rows: List[List[str]]) -> str:
    thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{_esc(str(x))}</td>" for x in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _table_md(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
    return "\n".join([head, sep, body])
