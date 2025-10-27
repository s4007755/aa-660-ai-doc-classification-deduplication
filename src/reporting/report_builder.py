# src/reporting/report_builder.py
from __future__ import annotations

import io
import json
import os
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.persistence import state_store as store
from src.metrics.metrics import metrics_snapshot

try:
    from src.storage import sqlite_store
except Exception:
    sqlite_store = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# small helpers
def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _pretty_json(s: Any) -> str:
    if isinstance(s, dict):
        try:
            return json.dumps(s, indent=2, ensure_ascii=False)
        except Exception:
            return str(s)
    try:
        return json.dumps(json.loads(s or "{}"), indent=2, ensure_ascii=False)
    except Exception:
        return str(s or "")

def _fmt_num(x: Any, nd: int = 3) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _disp(mm: Optional[Dict[str, Any]]) -> str:
    if not mm:
        return ""
    mn = _fmt_num(_get(mm, "min"))
    mx = _fmt_num(_get(mm, "max"))
    return f"{mn}..{mx}"

def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# public API
def generate_report(
    run_id: int,
    *,
    out_dir: str = "reports",
    fmt: str = "html",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Run + decisions + calibrations from persistence
    run = store.get_run(run_id) or {}
    cal_rows = _safe_get_calibrations(run_id)
    dec_rows = _safe_get_decisions(run_id)

    # Parse decision traces from JSON
    traces_json: List[Dict[str, Any]] = []
    for r in dec_rows:
        try:
            traces_json.append(json.loads(r.get("trace_json") or "{}"))
        except Exception:
            continue

    thresholds, briers = _collect_thresholds_briers(cal_rows)

    # build pseudo labels so charts & per-learner metrics have data
    pseudo = _pseudo_from_traces(traces_json)

    snapshot = metrics_snapshot(
        _as_traces_with_thresholds(traces_json, thresholds),
        pseudo_labels=pseudo,
    )

    labels = _load_doc_labels()

    # Build sections for the report
    started = run.get("started_at")
    ended = run.get("ended_at")
    notes = run.get("notes") or ""
    cfg_str = _pretty_json(run.get("config_json", "{}"))
    cal_table = _calibration_table(cal_rows, thresholds, briers)
    run_table = _run_table(snapshot.get("run", {}), traces_json)
    per_learner_rows = _per_learner_rows(snapshot.get("per_learner", {}) or {}, thresholds)
    examples = _select_examples(traces_json, labels=labels)
    clusters = snapshot.get("clusters", [])

    # Charts per learner (reliability, ROC, PR, threshold sweep, histogram)
    charts = snapshot.get("charts", {}) or {}
    chart_imgs = _render_all_charts(charts)

    # File name: stable
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if fmt.lower() == "md":
        content = _render_markdown(
            run_id, started, ended, notes, cfg_str, cal_table, run_table,
            per_learner_rows, examples, clusters, labels
        )
        filename = f"run_{run_id}_{ts}.md"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)
    else:
        content = _render_html(
            run_id, started, ended, notes, cfg_str, cal_table, run_table,
            per_learner_rows, examples, clusters, labels, chart_imgs, traces_json
        )
        filename = f"run_{run_id}_{ts}.html"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)


# persistence helpers
def _safe_get_calibrations(run_id: int) -> List[Dict[str, Any]]:
    try:
        if hasattr(store, "get_calibrations_for_run"):
            rows = store.get_calibrations_for_run(run_id)
            if rows:
                return rows
        if hasattr(store, "get_calibrations"):
            return store.get_calibrations(run_id)
    except Exception:
        pass
    return []

def _safe_get_decisions(run_id: int) -> List[Dict[str, Any]]:
    try:
        return store.get_decisions(run_id, limit=1_000_000)
    except Exception:
        return []

def _collect_thresholds_briers(cal_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    thresholds: Dict[str, Optional[float]] = {}
    briers: Dict[str, Optional[float]] = {}

    for row in cal_rows:
        name = row.get("learner_name") or row.get("learner") or ""
        if not name:
            continue

        thr = None
        br = None

        try:
            params = row.get("params_json") or row.get("params")
            if isinstance(params, str):
                params = json.loads(params)
            if isinstance(params, dict):
                thr = params.get("threshold", params.get("thr", params.get("cutoff")))
        except Exception:
            pass

        if thr is None and isinstance(row.get("threshold"), (int, float)):
            thr = float(row["threshold"])
        if br is None and isinstance(row.get("brier"), (int, float)):
            br = float(row["brier"])
        if br is None and isinstance(row.get("brier_score"), (int, float)):
            br = float(row["brier_score"])

        thresholds[name] = thr if isinstance(thr, (int, float)) else None
        briers[name] = br if isinstance(br, (int, float)) else None

    # prefer values saved on learner state if present
    for row in cal_rows:
        name = row.get("learner_name") or row.get("learner") or ""
        if not name:
            continue
        try:
            st = store.load_learner_state(name)
        except Exception:
            st = None
        if st is None:
            continue
        cal = _get(st, "calibration")
        st_thr = _get(cal, "threshold") or _get(cal, "thr") or _get(cal, "cutoff")
        st_br = _get(cal, "brier_score", _get(cal, "brier"))
        if st_thr is None:
            st_thr = _get(st, "threshold")
        if st_br is None:
            st_br = _get(st, "brier_score", _get(st, "brier"))
        if isinstance(st_thr, (int, float)):
            thresholds[name] = float(st_thr)
        if isinstance(st_br, (int, float)):
            briers[name] = float(st_br)

    return thresholds, briers


# snapshot helpers
def _pseudo_from_traces(traces_json: List[Dict[str, Any]]) -> Dict[str, int]:
    pseudo: Dict[str, int] = {}
    for t in traces_json:
        pk = t.get("pair_key")
        fl = (t.get("final_label") or "").upper()
        if pk and fl in ("DUPLICATE", "NON_DUPLICATE"):
            pseudo[pk] = 1 if fl == "DUPLICATE" else 0
    return pseudo

def _as_traces_with_thresholds(traces_json: List[Dict[str, Any]], thresholds: Dict[str, Optional[float]]):
    from types import SimpleNamespace
    out = []
    for obj in traces_json:
        ln: Dict[str, Any] = {}
        learners_map = obj.get("learners") or {}
        for k, v in learners_map.items():
            try:
                prob = float(_get(v, "prob", 0.0))
            except Exception:
                prob = 0.0
            thr = thresholds.get(k)
            ln[k] = SimpleNamespace(prob=prob, threshold=(float(thr) if isinstance(thr, (int, float)) else None))
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


# table builders
def _calibration_table(cal_rows: List[Dict[str, Any]], thresholds: Dict[str, Any], briers: Dict[str, Any]):
    rows = []
    for r in cal_rows:
        name = r.get("learner_name") or r.get("learner") or "-"
        method = r.get("method") or "-"
        reliability = r.get("reliability_json") or r.get("reliability") or "[]"
        try:
            rb = json.loads(reliability) if isinstance(reliability, str) else reliability
            rb_str = ", ".join(
                f"{_get(x,'prob_center',0.0):.2f}:{_get(x,'observed_pos_rate',0.0):.2f}({_get(x,'count',0)})"
                for x in (rb or [])[:6]
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


def _run_table(run: Dict[str, Any], traces_json: List[Dict[str, Any]]):
    pairs = len(traces_json)
    exact = 0
    near = 0
    unc = 0
    for t in traces_json:
        fl = (t.get("final_label") or "").upper()
        if fl == "DUPLICATE":
            if (t.get("dup_kind") or "").upper() == "EXACT":
                exact += 1
            else:
                near += 1
        elif fl == "UNCERTAIN":
            unc += 1

    cons = run.get("consensus_rate", run.get("consensus_pct", 0.0))
    esc = run.get("escalations_pct", run.get("escalations_rate", 0.0))
    return [
        {"metric": "Total pairs", "value": str(pairs or run.get("total_pairs") or run.get("pairs_scored") or run.get("pairs") or 0)},
        {"metric": "Exact duplicates", "value": str(exact)},
        {"metric": "Near duplicates", "value": str(near)},
        {"metric": "Duplicates (total)", "value": str(exact + near)},
        {"metric": "Uncertain", "value": str(unc)},
        {"metric": "Consensus rate", "value": f"{100.0 * float(cons or 0.0):.1f}%"},
        {"metric": "Escalations", "value": f"{100.0 * float(esc or 0.0):.1f}%"},
    ]


def _per_learner_rows(per_learner: Dict[str, Any], thresholds: Dict[str, Optional[float]]) -> List[List[str]]:
    rows: List[List[str]] = []
    if not isinstance(per_learner, dict):
        return rows
    for name, m in per_learner.items():
        rows.append(
            [
                str(name),
                str(_get(m, "n", "")),
                _fmt_num(_get(m, "pos_rate")),
                _fmt_num(_get(m, "auc")),
                _fmt_num(_get(m, "brier")),
                _fmt_num(thresholds.get(name)),
            ]
        )
    # keep consistent order
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
    labels = labels or []
    easy: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []
    for tr in traces:
        label = tr.get("final_label")
        learners = tr.get("learners", {}) or {}
        probs = []
        for v in learners.values():
            try:
                probs.append(float(_get(v, "prob", 0.0)))
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
        return ""
    name = labels.get(doc_id)
    return f"{name} ({doc_id[:8]})" if name else doc_id


# chart helpers
def _render_all_charts(charts: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not _HAVE_MPL:
        return out
    for name, chart in (charts or {}).items():
        imgs: Dict[str, str] = {}

        # Reliability
        rel = chart.get("reliability") or []
        xs = [_get(r, "expected_pos_rate", 0.0) for r in rel]
        ys = [_get(r, "observed_pos_rate", 0.0) for r in rel]
        imgs["reliability"] = _line_to_data_uri(
            xs, ys,
            "Predicted probability", "Observed positive rate",
            "Reliability curve", diag=True
        )

        # ROC
        roc = chart.get("roc") or {}
        imgs["roc"] = _line_to_data_uri(
            roc.get("fpr") or [0, 1], roc.get("tpr") or [0, 1],
            "False Positive Rate", "True Positive Rate",
            f"ROC (AUC={_fmt_num(roc.get('auc'))})", diag=True
        )

        # PR
        pr = chart.get("pr") or {}
        imgs["pr"] = _line_to_data_uri(
            pr.get("recall") or [0, 1], pr.get("precision") or [1, 0],
            "Recall", "Precision", "Precision–Recall"
        )

        # Threshold sweep
        ts = chart.get("thr_sweep") or {}
        imgs["thr"] = _multi_to_data_uri(
            ts.get("thresholds") or [0, 1],
            [ts.get("precision") or [1, 1], ts.get("recall") or [0, 1], ts.get("f1") or [0, 1]],
            ["Precision", "Recall", "F1"],
            "Threshold", "Score", "Threshold sweep", ylim=(0.0, 1.05)
        )

        # Histogram
        hist = chart.get("hist") or {}
        imgs["hist"] = _hist_to_data_uri(
            hist.get("bin_edges") or [0.0, 1.0],
            hist.get("pos") or [0],
            hist.get("neg") or [0],
            "Calibrated probability", "Count", "Score distribution"
        )

        out[name] = imgs
    return out


def _line_to_data_uri(xs, ys, xlabel, ylabel, title, diag=False) -> str:
    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=110)
    if diag:
        ax.plot([min(xs + [0]), max(xs + [1])], [min(ys + [0]), max(ys + [1])], linestyle="--", alpha=0.4)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _multi_to_data_uri(xs, y_list, labels, xlabel, ylabel, title, ylim=None) -> str:
    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=110)
    for y, lab in zip(y_list, labels):
        ax.plot(xs, y, label=lab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _hist_to_data_uri(edges, pos, neg, xlabel, ylabel, title) -> str:
    edges = list(edges)
    centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(edges)-1)] if len(edges) > 1 else [0.5]
    width = (edges[1]-edges[0]) * 0.9 if len(edges) > 1 else 0.5
    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=110)
    ax.bar(centers, neg[: len(centers)], width=width, alpha=0.6, label="negatives")
    ax.bar(centers, pos[: len(centers)], width=width, alpha=0.6, label="positives")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# HTML / MD rendering
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
    chart_imgs: Dict[str, Dict[str, str]],
    traces_json: List[Dict[str, Any]],
) -> str:
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px; color: #1f2937; }
      h1 { font-size: 24px; margin: 0 0 12px 0; }
      h2 { font-size: 18px; margin-top: 24px; }
      h3 { font-size: 15px; margin-top: 16px; }
      code, pre { background: #f9fafb; padding: 8px; border-radius: 6px; display: block; overflow-x: auto; }
      table { border-collapse: collapse; width: 100%; margin-top: 8px; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; font-size: 13px; vertical-align: top; }
      th { background: #f3f4f6; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
      .pill { display: inline-block; padding: 2px 8px; border-radius: 9999px; background:#eef2ff; color:#3730a3; font-size:12px; }
      .muted { color:#6b7280; }
      img.chart { width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 6px; background: #fff; }
    </style>
    """
    header = f"<h1>Duplicate Detection Report — Run {run_id}</h1>"
    meta = (
        f"<p><span class='pill'>Started</span> { _esc(started or '') } "
        f"&nbsp;&nbsp; <span class='pill'>Ended</span> { _esc(ended or '') }"
        f"{' &nbsp;&nbsp; <span class=\"pill\">Notes</span> ' + _esc(notes) if notes else ''}</p>"
    )

    # Run summary
    run_table_html = _table_html(["Metric", "Value"], [[r["metric"], r["value"]] for r in run_table])

    # Per-learner metrics
    pl_headers = ["Learner", "N", "PosRate", "AUC", "Brier", "Threshold"]
    pl_html = _table_html(pl_headers, per_learner_rows) if per_learner_rows else "<p class='muted'>No per-learner metrics.</p>"

    # Calibration snapshot
    cal_rows = []
    for r in cal_table:
        cal_rows.append([r["learner"], r["method"], r["threshold"], r["brier"], r["reliability"]])
    cal_html = _table_html(["Learner", "Method", "Threshold", "Brier", "Reliability (center:obs(count))"], cal_rows)

    # Charts
    charts_html_parts = []
    for learner, imgs in chart_imgs.items():
        charts_html_parts.append(f"<h3>{_esc(learner.capitalize())}</h3>")
        charts_html_parts.append("<div class='grid'>")
        for key in ("reliability", "roc", "pr", "thr", "hist"):
            if imgs.get(key):
                charts_html_parts.append(f"<div><img class='chart' src='{imgs[key]}' alt='{learner} {key}'></div>")
        charts_html_parts.append("</div>")
    charts_html = "\n".join(charts_html_parts) or "<p class='muted'>Charts unavailable.</p>"

    # Examples
    ex_html = _examples_html(examples)

    # Decision traces
    traces_html = _traces_html(traces_json, labels)

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

<h2>Charts</h2>
{charts_html}

<h2>Examples</h2>
{ex_html}

<h2>Decision traces</h2>
{traces_html}

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
    if started:
        lines.append(f"- **Started:** {started}")
    if ended:
        lines.append(f"- **Ended:** {ended}")
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
                ["Learner", "N", "PosRate", "AUC", "Brier", "Threshold"],
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
def _derive_agreed_from_trace(tr: Dict[str, Any]) -> List[str]:
    agreed: List[str] = []
    for name, v in (tr.get("learners") or {}).items():
        try:
            thr = v.get("threshold")
            prob = float(_get(v, "prob", 0.0))
            if isinstance(thr, (int, float)) and prob >= float(thr):
                agreed.append(name)
        except Exception:
            pass
    return sorted(agreed)

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
            learners = ", ".join(f"{_esc(k)}={float(_get(v,'prob',0.0)):.2f}" for k, v in (tr.get("learners") or {}).items())
            agreed = tr.get("agreed_learners") or _derive_agreed_from_trace(tr)
            escal = " → ".join(tr.get("escalation_steps", []))
            body.append(
                [
                    _esc(tr.get("a_name") or tr.get("a_id") or ""),
                    _esc(tr.get("b_name") or tr.get("b_id") or ""),
                    _esc(tr.get("final_label", "")),
                    _esc(", ".join(agreed)),
                    _esc(escal),
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
            learners = ", ".join(f"{k}={float(_get(v,'prob',0.0)):.2f}" for k, v in (tr.get("learners") or {}).items())
            agreed = tr.get("agreed_learners") or _derive_agreed_from_trace(tr)
            escal = " → ".join(tr.get("escalation_steps", []))
            body.append(
                [
                    tr.get("a_name") or tr.get("a_id") or "",
                    tr.get("b_name") or tr.get("b_id") or "",
                    tr.get("final_label", ""),
                    ", ".join(agreed),
                    escal,
                    learners,
                ]
            )
        lines.append(_table_md(hdr, body))
        lines.append("")
    return lines


def _traces_html(traces: List[Dict[str, Any]], labels: Dict[str, str]) -> str:
    if not traces:
        return "<p class='muted'>No traces stored.</p>"
    hdr = ["Doc A", "Doc B", "Final", "Dup kind", "Agreed", "Escalation", "Per-learner (prob)"]
    body = []
    for t in traces[:1000]:
        a = _pretty_doc(t.get("a_id"), labels)
        b = _pretty_doc(t.get("b_id"), labels)
        agreed = t.get("agreed_learners") or _derive_agreed_from_trace(t)
        esc = " → ".join(t.get("escalation_steps", []))
        learners = t.get("learners") or {}
        plist = []
        for k, v in learners.items():
            try:
                plist.append(f"{k}:{float(_get(v,'prob',0.0)):.3f}")
            except Exception:
                plist.append(f"{k}:?")
        body.append([a, b, t.get("final_label",""), t.get("dup_kind",""), ", ".join(agreed), esc, ", ".join(plist)])
    return _table_html(hdr, body)


def _clusters_html(clusters: List[Dict[str, Any]], labels: Dict[str, str]) -> str:
    if not clusters:
        return "<p><i>No clusters found</i></p>"
    hdr = ["#", "Size", "Members", "Avg prob (simhash|minhash|embed)", "Dispersion (min..max)"]
    body = []
    for c in clusters:
        members = [_pretty_doc(m, labels) for m in c.get("members", [])]
        avg = f"{_fmt_num(c.get('avg_simhash_prob'))}|{_fmt_num(c.get('avg_minhash_prob'))}|{_fmt_num(c.get('avg_embedding_prob'))}"
        disp = f"{_disp(c.get('dispersion_simhash'))} | {_disp(c.get('dispersion_minhash'))} | {_disp(c.get('dispersion_embedding'))}"
        body.append([str(c.get("cluster_index")), str(c.get("size")), ", ".join(members), avg, disp])
    return _table_html(hdr, body)


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
