# src/gui/widgets/metrics_panel.py
from __future__ import annotations

from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import ttk

_HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

from .metrics_tables import MetricsTables

class MetricsPanel(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # Top summary
        self.box_summary = ttk.LabelFrame(self, text="Run Summary")
        self.box_summary.pack(fill=tk.X, padx=8, pady=(8, 6))

        self.var_pairs = tk.StringVar(value="Pairs: —")
        self.var_dup = tk.StringVar(value="Duplicate: —")
        self.var_near = tk.StringVar(value="Near-duplicate: —")
        self.var_non = tk.StringVar(value="Non-duplicate: —")
        self.var_unc = tk.StringVar(value="Uncertain: —")
        self.var_cons = tk.StringVar(value="Consensus: —")
        self.var_esc = tk.StringVar(value="Escalations: —")

        row = ttk.Frame(self.box_summary)
        row.pack(fill=tk.X, padx=8, pady=4)
        for i, v in enumerate(
            [self.var_pairs, self.var_dup, self.var_near, self.var_non, self.var_unc, self.var_cons, self.var_esc]
        ):
            ttk.Label(row, textvariable=v).grid(row=0 if i < 4 else 1, column=i % 4, sticky="w", padx=(0, 16))

        # Middle: learner tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.tables_tab = MetricsTables(self.nb)

        # Bottom: consensus / escalations
        self.box_diag = ttk.LabelFrame(self, text="Consensus & Escalations")
        self.box_diag.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.var_agree = tk.StringVar(value="Agreement: —")
        self.var_voters = tk.StringVar(value="Voter share: —")
        self.var_escal = tk.StringVar(value="Escalations: —")
        ttk.Label(self.box_diag, textvariable=self.var_agree).pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Label(self.box_diag, textvariable=self.var_voters).pack(anchor="w", padx=8, pady=(2, 0))
        ttk.Label(self.box_diag, textvariable=self.var_escal).pack(anchor="w", padx=8, pady=(2, 8))

        self._last_snapshot = None
        self._doc_labels: Dict[str, str] = {}

    # Public entry
    def update_metrics(self, run_summary: Dict[str, Any], snapshot: Dict[str, Any], *, doc_labels: Optional[Dict[str, str]] = None):
        self._last_snapshot = snapshot
        self._doc_labels = dict(doc_labels or {})
        self._fill_summary(run_summary)
        self._build_learner_tabs(snapshot)
        self._fill_consensus(snapshot)
        if hasattr(self, "tables_tab"):
            self.tables_tab.update_tables(snapshot, doc_labels=self._doc_labels)

    # private

    def _fill_summary(self, run_summary: Dict[str, Any]):
        pairs = run_summary.get("pairs_scored") or run_summary.get("total_pairs") or run_summary.get("pairs") or "—"
        self.var_pairs.set(f"Pairs: {pairs}")
        self.var_dup.set(f"Duplicate: {run_summary.get('duplicates', '—')}")
        nd = run_summary.get("near_duplicates")
        if nd is not None:
            self.var_near.set(f"Near-duplicate: {nd}")
        self.var_non.set(f"Non-duplicate: {run_summary.get('non_duplicates', '—')}")
        self.var_unc.set(f"Uncertain: {run_summary.get('uncertain', '—')}")
        cr = run_summary.get("consensus_rate")
        self.var_cons.set(f"Consensus: {cr*100:.1f}%" if isinstance(cr, (int, float)) else "Consensus: —")
        er = run_summary.get("escalations_rate") or run_summary.get("escalations_pct")
        self.var_esc.set(f"Escalations: {er*100:.1f}%" if isinstance(er, (int, float)) else "Escalations: —")

    def _build_learner_tabs(self, snapshot: Dict[str, Any]):
        for tab_id in self.nb.tabs():
            self.nb.forget(tab_id)

        per_learner = snapshot.get("per_learner", {}) or {}
        charts = snapshot.get("charts", {}) or {}
        thresholds = snapshot.get("thresholds", {}) or {}

        for name in sorted(per_learner.keys()):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text=name.capitalize())

            # Top: table with AUC/Brier/ECE/threshold & PR@thr
            head = ttk.LabelFrame(tab, text="Summary")
            head.pack(fill=tk.X, padx=8, pady=(8, 6))

            pl = per_learner.get(name, {})
            thr = thresholds.get(name, {})
            row1 = ttk.Frame(head)
            row1.pack(fill=tk.X, padx=8, pady=4)
            ttk.Label(row1, text=f"AUC: {pl.get('auc', 0.0):.3f}").pack(side=tk.LEFT, padx=(0, 16))
            ttk.Label(row1, text=f"Brier: {pl.get('brier', 0.0):.3f}").pack(side=tk.LEFT, padx=(0, 16))
            ece = pl.get("ece")
            if isinstance(ece, (int, float)):
                ttk.Label(row1, text=f"ECE: {ece:.3f}").pack(side=tk.LEFT, padx=(0, 16))

            row2 = ttk.Frame(head)
            row2.pack(fill=tk.X, padx=8, pady=(0, 4))
            if thr.get("threshold") is not None:
                ttk.Label(row2, text=f"Threshold: {float(thr['threshold']):.3f}").pack(side=tk.LEFT, padx=(0, 16))
            if thr.get("precision") is not None:
                ttk.Label(row2, text=f"Precision@thr: {thr['precision']:.3f}").pack(side=tk.LEFT, padx=(0, 16))
            if thr.get("recall") is not None:
                ttk.Label(row2, text=f"Recall@thr: {thr['recall']:.3f}").pack(side=tk.LEFT, padx=(0, 16))
            if thr.get("f1") is not None:
                ttk.Label(row2, text=f"F1@thr: {thr['f1']:.3f}").pack(side=tk.LEFT, padx=(0, 16))
            if thr.get("near_band_share") is not None:
                ttk.Label(row2, text=f"Near-band volume: {thr['near_band_share']*100:.1f}%").pack(side=tk.LEFT, padx=(0, 16))

            # Middle: plots (Calibration / ROC / PR / Threshold sweep / Scores)
            body = ttk.Notebook(tab)
            body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

            caltab = ttk.Frame(body); body.add(caltab, text="Calibration")
            roctab = ttk.Frame(body); body.add(roctab, text="ROC")
            prtab  = ttk.Frame(body); body.add(prtab, text="PR")
            thrstab= ttk.Frame(body); body.add(thrstab, text="Threshold Sweep")
            histtab= ttk.Frame(body); body.add(histtab, text="Scores")

            chart = charts.get(name, {}) or {}
            self._plot_calibration(caltab, chart)
            self._plot_roc(roctab, chart)
            self._plot_pr(prtab, chart)
            self._plot_thr_sweep(thrstab, chart)
            self._plot_hist(histtab, chart)

        # Tables tab at the end
        self.nb.add(self.tables_tab, text="Tables")
        self.tables_tab.update_tables(snapshot, doc_labels=self._doc_labels)

    def _fill_consensus(self, snapshot: Dict[str, Any]):
        cons = snapshot.get("consensus", {}) or {}
        learners = cons.get("learners") or []
        mat = cons.get("agreement") or []
        voters = cons.get("voter_share") or {}
        if learners and mat:
            vals = []
            for i in range(len(learners)):
                for j in range(len(learners)):
                    if i == j:
                        continue
                    try:
                        vals.append(float(mat[i][j]))
                    except Exception:
                        pass
            if vals:
                self.var_agree.set(f"Agreement (avg pairwise): {100.0 * sum(vals) / len(vals):.1f}%")
        if voters:
            parts = [f"{k}: {v*100:.1f}%" for k, v in sorted(voters.items())]
            self.var_voters.set("Voter share (+): " + ", ".join(parts))
        esc = snapshot.get("escalations", {}) or {}
        rate = esc.get("rate")
        by_step = esc.get("by_step", {})
        if isinstance(rate, (int, float)):
            steps = ", ".join(f"{k}:{v}" for k, v in by_step.items()) if by_step else "—"
            self.var_escal.set(f"Escalations: {rate*100:.1f}%  |  Steps: {steps}")

    # plotting helpers
    def _clear_children(self, parent: tk.Widget):
        for w in parent.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

    def _plot_calibration(self, parent, chart: Dict[str, Any]):
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return
        rel = chart.get("reliability") or []
        xs = [r.get("expected_pos_rate", 0.0) for r in rel]
        ys = [r.get("observed_pos_rate", 0.0) for r in rel]
        fig = Figure(figsize=(5.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed positive rate")
        ax.set_title("Reliability curve")
        ax.grid(True, alpha=0.3)
        FigureCanvasTkAgg(fig, parent).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_roc(self, parent, chart: Dict[str, Any]):
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return
        roc = chart.get("roc") or {}
        fpr = roc.get("fpr") or [0.0, 1.0]
        tpr = roc.get("tpr") or [0.0, 1.0]
        auc = roc.get("auc")
        fig = Figure(figsize=(5.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC (AUC={auc:.3f})" if isinstance(auc, (int, float)) else "ROC")
        ax.grid(True, alpha=0.3)
        FigureCanvasTkAgg(fig, parent).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_pr(self, parent, chart: Dict[str, Any]):
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return
        pr = chart.get("pr") or {}
        precision = pr.get("precision") or [1.0, 0.0]
        recall = pr.get("recall") or [0.0, 1.0]
        fig = Figure(figsize=(5.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall")
        ax.grid(True, alpha=0.3)
        FigureCanvasTkAgg(fig, parent).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_thr_sweep(self, parent, chart: Dict[str, Any]):
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return
        ts = chart.get("thr_sweep") or {}
        th = ts.get("thresholds") or [0.0, 1.0]
        prec = ts.get("precision") or [1.0, 1.0]
        rec = ts.get("recall") or [0.0, 1.0]
        f1 = ts.get("f1") or [0.0, 1.0]
        fig = Figure(figsize=(5.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(th, prec, label="Precision")
        ax.plot(th, rec, label="Recall")
        ax.plot(th, f1, label="F1")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Threshold sweep")
        ax.legend()
        ax.grid(True, alpha=0.3)
        FigureCanvasTkAgg(fig, parent).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_hist(self, parent, chart: Dict[str, Any]):
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return
        hist = chart.get("hist") or {}
        edges = hist.get("bin_edges") or [0.0, 1.0]
        pos = hist.get("pos") or [0]
        neg = hist.get("neg") or [0]
        edges = list(edges)
        centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
        width = (edges[1] - edges[0]) * 0.9 if len(edges) > 1 else 0.05
        fig = Figure(figsize=(5.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(centers, neg, width=width, alpha=0.6, label="negatives")
        ax.bar(centers, pos, width=width, alpha=0.6, label="positives")
        ax.set_xlabel("Calibrated probability")
        ax.set_ylabel("Count")
        ax.set_title("Score distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        FigureCanvasTkAgg(fig, parent).get_tk_widget().pack(fill=tk.BOTH, expand=True)
