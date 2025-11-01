# src/gui/widgets/metrics_panel.py
from __future__ import annotations

from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import ttk

# Optional matplotlib embedding for inline plots inside Tk.
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
    """
    Composite panel showing:
      1) Top-level run summary KPIs (pairs, dup, consensus, escalations).
      2) Per-learner tabs with basics, settings, plots (reliability/ROC/PR/threshold sweep/histogram).
      3) Consensus and escalation diagnostics.

    Usage
    Call `update_metrics(run_summary, snapshot, doc_labels=...)` whenever
    a new run finishes or a snapshot is refreshed. The method will:
      * Fill the top summary row.
      * Rebuild learner tabs from `snapshot`.
      * Update consensus/escalation section.
      * Refresh the "Tables" tab.
    """

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # Top summary box
        self.box_summary = ttk.LabelFrame(self, text="Run Summary")
        self.box_summary.pack(fill=tk.X, padx=8, pady=(8, 6))

        # StringVars back the labels so updates remain cheap.
        self.var_pairs = tk.StringVar(value="Pairs: —")
        self.var_dup = tk.StringVar(value="Duplicate: —")
        self.var_near = tk.StringVar(value="Near-duplicate: —")
        self.var_non = tk.StringVar(value="Non-duplicate: —")
        self.var_unc = tk.StringVar(value="Uncertain: —")
        self.var_cons = tk.StringVar(value="Consensus: —")
        self.var_esc = tk.StringVar(value="Escalations: —")

        row = ttk.Frame(self.box_summary)
        row.pack(fill=tk.X, padx=8, pady=4)
        # Two rows of metrics
        for i, v in enumerate(
            [self.var_pairs, self.var_dup, self.var_near, self.var_non, self.var_unc, self.var_cons, self.var_esc]
        ):
            ttk.Label(row, textvariable=v).grid(row=0 if i < 4 else 1, column=i % 4, sticky="w", padx=(0, 16))

        # Middle: notebook with one tab per learner
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Tables tab is created once and re-used
        self.tables_tab = MetricsTables(self.nb)

        # Bottom: consensus and escalation diagnostics
        self.box_diag = ttk.LabelFrame(self, text="Consensus & Escalations")
        self.box_diag.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.var_agree = tk.StringVar(value="Agreement: —")
        self.var_voters = tk.StringVar(value="Voter share: —")
        self.var_escal = tk.StringVar(value="Escalations: —")
        ttk.Label(self.box_diag, textvariable=self.var_agree).pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Label(self.box_diag, textvariable=self.var_voters).pack(anchor="w", padx=8, pady=(2, 0))
        ttk.Label(self.box_diag, textvariable=self.var_escal).pack(anchor="w", padx=8, pady=(2, 8))

        # Internal cache of last snapshot
        self._last_snapshot = None
        self._doc_labels: Dict[str, str] = {}

    # Public entry
    def update_metrics(self, run_summary: Dict[str, Any], snapshot: Dict[str, Any], *, doc_labels: Optional[Dict[str, str]] = None):
        """
        Refresh the entire panel with a new run summary and learner snapshot.
        """
        self._last_snapshot = snapshot
        self._doc_labels = dict(doc_labels or {})
        self._fill_summary(run_summary)
        self._build_learner_tabs(snapshot)
        self._fill_consensus(snapshot)
        if hasattr(self, "tables_tab"):
            self.tables_tab.update_tables(snapshot, doc_labels=self._doc_labels)


    # Private

    def _fill_summary(self, run_summary: Dict[str, Any]):
        """
        Populate the top Run Summary KPIs. Handles variant field names and missing values.
        """
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
        """
        Recreate one tab per learner with:
          - Summary (AUC, Brier/ECE when calibrated, threshold slice).
          - Basics table (counts, means, confusion metrics).
          - Optional model settings snapshot.
          - Plots: reliability, ROC, PR, threshold sweep, histogram.
        """
        # Clear existing tabs.
        for tab_id in self.nb.tabs():
            self.nb.forget(tab_id)

        per_learner = snapshot.get("per_learner", {}) or {}
        charts = snapshot.get("charts", {}) or {}
        thresholds = snapshot.get("thresholds", {}) or {}
        settings = snapshot.get("settings", {}) or {}
        basics = snapshot.get("basics", {}) or {}
        use_calibrated = bool(snapshot.get("use_calibrated", False))

        for name in sorted(per_learner.keys()):
            tab = ttk.Frame(self.nb)
            self.nb.add(tab, text=name.capitalize())

            # Top: per-learner summary strip
            head = ttk.LabelFrame(tab, text="Summary")
            head.pack(fill=tk.X, padx=8, pady=(8, 6))

            pl = per_learner.get(name, {}) or {}
            thr = thresholds.get(name, {}) or {}

            row1 = ttk.Frame(head); row1.pack(fill=tk.X, padx=8, pady=4)

            auc_val = pl.get("auc")
            auc_txt = f"AUC: {auc_val:.3f}" if isinstance(auc_val, (int, float)) else "AUC: —"
            ttk.Label(row1, text=auc_txt).pack(side=tk.LEFT, padx=(0, 16))

            # Calibration quality metrics.
            if use_calibrated and pl.get("is_calibrated"):
                brier = pl.get("brier")
                if isinstance(brier, (int, float)):
                    ttk.Label(row1, text=f"Brier: {brier:.3f}").pack(side=tk.LEFT, padx=(0, 16))
                ece = pl.get("ece")
                if isinstance(ece, (int, float)):
                    ttk.Label(row1, text=f"ECE: {ece:.3f}").pack(side=tk.LEFT, padx=(0, 16))

            row2 = ttk.Frame(head); row2.pack(fill=tk.X, padx=8, pady=(0, 4))
            # Threshold slice KPIs
            th_val = thr.get("threshold")
            if use_calibrated and isinstance(th_val, (int, float)):
                ttk.Label(row2, text=f"Threshold: {float(th_val):.3f}").pack(side=tk.LEFT, padx=(0, 16))
                if isinstance(thr.get("precision"), (int, float)):
                    ttk.Label(row2, text=f"Precision@thr: {thr['precision']:.3f}").pack(side=tk.LEFT, padx=(0, 16))
                if isinstance(thr.get("recall"), (int, float)):
                    ttk.Label(row2, text=f"Recall@thr: {thr['recall']:.3f}").pack(side=tk.LEFT, padx=(0, 16))
                if isinstance(thr.get("f1"), (int, float)):
                    ttk.Label(row2, text=f"F1@thr: {thr['f1']:.3f}").pack(side=tk.LEFT, padx=(0, 16))
                if isinstance(thr.get("near_band_share"), (int, float)):
                    ttk.Label(row2, text=f"Near-band volume: {thr['near_band_share']*100:.1f}%").pack(side=tk.LEFT, padx=(0, 16))

            # Optional model settings mirror
            sett_map = settings.get(name, {}) or {}
            if sett_map:
                box_settings = ttk.LabelFrame(tab, text="Model settings")
                box_settings.pack(fill=tk.X, padx=8, pady=(0, 8))
                line = ttk.Frame(box_settings); line.pack(fill=tk.X, padx=8, pady=4)
                prefer = [
                    "mode", "hash_bits", "shingle_size", "tokenizer_mode",
                    "char_ngram", "pos_bucket", "whitening", "remove_top_pc",
                    "threshold_used", "model_fallback"
                ]
                shown = []
                # Prefer a subset ordering, then render the rest sorted for completeness.
                for k in prefer:
                    if k in sett_map:
                        ttk.Label(line, text=f"{k}: {sett_map[k]}").pack(side=tk.LEFT, padx=(0, 16))
                        shown.append(k)
                for k, v in sorted(sett_map.items()):
                    if k in shown:
                        continue
                    ttk.Label(line, text=f"{k}: {v}").pack(side=tk.LEFT, padx=(0, 16))

            # Middle: child notebook with basics and plots
            body = ttk.Notebook(tab)
            body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

            # Basics table
            basics_tab = ttk.Frame(body)
            body.add(basics_tab, text="Basics")

            # Minimal two-column table
            tree = ttk.Treeview(
                basics_tab,
                columns=("metric", "value"),
                show="headings",
                height=10,
            )
            tree.heading("metric", text="Metric")
            tree.heading("value", text="Value")
            tree.column("metric", width=220, anchor="w")
            tree.column("value", width=160, anchor="e")
            vsb = ttk.Scrollbar(basics_tab, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,0), pady=8)
            vsb.pack(side=tk.LEFT, fill=tk.Y, padx=(0,8), pady=8)

            b = basics.get(name, {}) or {}

            def _fmtv(x, prec=3):
                # Printer for table values.
                try:
                    if x is None: return "—"
                    if isinstance(x, float): return f"{x:.{prec}f}"
                    return str(x)
                except Exception:
                    return "—"

            # Rows list both core counts and threshold-slice metrics.
            rows = [
                ("Pairs (n) (Near Duplicate + Uncertain)", b.get("n")),
                ("Near-dup count", b.get("near_count")),
                ("Uncertain count", b.get("uncertain_count")),
                ("Near-dup rate", _fmtv(b.get("near_rate"))),

                ("Score mean", _fmtv(b.get("score_mean"))),
                ("Score std", _fmtv(b.get("score_std"))),
                ("Score min", _fmtv(b.get("score_min"))),
                ("Score max", _fmtv(b.get("score_max"))),

                ("Mean score (near-dup)", _fmtv(b.get("score_mean_near"))),
                ("Mean score (uncertain)", _fmtv(b.get("score_mean_uncertain"))),

                ("Has threshold", "yes" if b.get("has_threshold") else "no"),
                ("Threshold", _fmtv(b.get("threshold"))),
                ("% ≥ threshold", _fmtv( (b.get("pct_ge_threshold") * 100.0) if isinstance(b.get("pct_ge_threshold"), (int,float)) else None, prec=1) + "%" if isinstance(b.get("pct_ge_threshold"), (int,float)) else "—"),

                ("TP", b.get("tp")), ("FP", b.get("fp")),
                ("TN", b.get("tn")), ("FN", b.get("fn")),

                ("Precision@thr", _fmtv(b.get("precision_at_thr"))),
                ("Recall@thr", _fmtv(b.get("recall_at_thr"))),
                ("F1@thr", _fmtv(b.get("f1_at_thr"))),
            ]
            for m, v in rows:
                tree.insert("", tk.END, values=(m, v if v is not None else "—"))

            # Charts
            chart = charts.get(name, {}) or {}

            # Backfill flags when missing so downstream checks are simple/boolean.
            flags = chart.get("flags") or {}
            if not flags:
                flags = {}
                flags["is_calibrated"] = bool(chart.get("reliability"))
                roc = chart.get("roc") or {}
                fpr = roc.get("fpr") or []; tpr = roc.get("tpr") or []
                flags["roc_ok"] = (len(fpr) >= 2 and len(tpr) >= 2)
                pr = chart.get("pr") or {}
                precision = pr.get("precision") or []; recall = pr.get("recall") or []
                flags["pr_ok"] = (len(precision) >= 2 and len(recall) >= 2)
                ts = chart.get("thr_sweep") or {}
                ths = ts.get("thresholds") or []
                def _var(xs):
                    try:
                        return (isinstance(xs, list) and len(xs) >= 2 and len(set(xs)) > 1)
                    except Exception:
                        return False
                flags["thr_ok"] = (len(ths) >= 2) and (_var(ts.get("precision") or []) or _var(ts.get("recall") or []) or _var(ts.get("f1") or []))

            is_cal = bool(flags.get("is_calibrated", False))

            if is_cal and (chart.get("reliability") or []):
                caltab = ttk.Frame(body)
                body.add(caltab, text="Calibration")
                self._plot_calibration(caltab, chart)

            if bool(flags.get("roc_ok")):
                roctab = ttk.Frame(body)
                body.add(roctab, text="ROC")
                self._plot_roc(roctab, chart)

            if bool(flags.get("pr_ok")):
                prtab = ttk.Frame(body)
                body.add(prtab, text="PR")
                self._plot_pr(prtab, chart)

            if bool(flags.get("thr_ok")):
                thrstab = ttk.Frame(body)
                body.add(thrstab, text="Threshold Sweep")
                self._plot_thr_sweep(thrstab, chart)

            # Scores histogram
            hist = chart.get("hist") or {}
            edges = hist.get("bin_edges") or []
            pos = hist.get("pos") or []
            neg = hist.get("neg") or []
            if len(edges) >= 2 and (sum(pos) + sum(neg)) > 0:
                histtab = ttk.Frame(body)
                body.add(histtab, text="Scores")
                self._plot_hist(histtab, chart)

        self.nb.add(self.tables_tab, text="Tables")
        self.tables_tab.update_tables(snapshot, doc_labels=self._doc_labels)

    def _fill_consensus(self, snapshot: Dict[str, Any]):
        """
        Fill the consensus & escalation diagnostics section, plus a tiny outcome bar chart.
        """
        cons = snapshot.get("consensus", {}) or {}
        learners = cons.get("learners") or []
        mat = cons.get("agreement") or []
        voters = cons.get("voter_share") or {}

        # Average pairwise agreement across learners.
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

        # Voter share: who most often votes positive
        if voters:
            parts = [f"{k}: {v*100:.1f}%" for k, v in sorted(voters.items())]
            self.var_voters.set("Voter share (+): " + ", ".join(parts))

        # Escalations summary and per-step distribution
        esc = snapshot.get("escalations", {}) or {}
        rate = esc.get("rate")
        by_step = esc.get("by_step", {})
        if isinstance(rate, (int, float)):
            steps = ", ".join(f"{k}:{v}" for k, v in by_step.items()) if by_step else "—"
            self.var_escal.set(f"Escalations: {rate*100:.1f}%  |  Steps: {steps}")

        # Tiny bar chart for overall outcomes
        dk = snapshot.get("dup_kinds", {}) or {}
        if dk and _HAVE_MPL:
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                self._clear_children(self.box_diag)
                # re-add text rows
                ttk.Label(self.box_diag, textvariable=self.var_agree).pack(anchor="w", padx=8, pady=(6, 0))
                ttk.Label(self.box_diag, textvariable=self.var_voters).pack(anchor="w", padx=8, pady=(2, 0))
                ttk.Label(self.box_diag, textvariable=self.var_escal).pack(anchor="w", padx=8, pady=(2, 8))
                # chart
                labels = ["Exact", "Near", "Non", "Uncertain"]
                counts = [
                    int(dk.get("exact", 0)),
                    int(dk.get("near", 0)),
                    int(dk.get("non", 0)),
                    int(dk.get("uncertain", 0)),
                ]
                fig = Figure(figsize=(4.6, 1.8), dpi=100)
                ax = fig.add_subplot(111)
                ax.bar(labels, counts)
                ax.set_title("Pairs by outcome")
                ax.grid(True, axis="y", alpha=0.3)
                for i, c in enumerate(counts):
                    ax.text(i, c, str(c), ha="center", va="bottom")
                FigureCanvasTkAgg(fig, self.box_diag).get_tk_widget().pack(fill=tk.X, padx=8, pady=(0, 8))
            except Exception:
                pass


    # Plotting helpers

    def _clear_children(self, parent: tk.Widget):
        """
        Destroy all children of parent.
        """
        for w in parent.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

    def _plot_calibration(self, parent, chart: Dict[str, Any]):
        """
        Reliability curve: predicted probability vs observed positive rate.
        """
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
        """
        Receiver Operating Characteristic (ROC) curve with diagonal baseline.
        """
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return
        # in _plot_roc
        roc = chart.get("roc") or {}
        fpr = roc.get("fpr") or []
        tpr = roc.get("tpr") or []
        auc = roc.get("auc")
        if len(fpr) < 2 or len(tpr) < 2:
            ttk.Label(parent, text="Not enough variation to plot ROC").pack(padx=8, pady=8)
            return
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
        """
        Precision-Recall curve.
        """
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return
        # in _plot_pr
        pr = chart.get("pr") or {}
        precision = pr.get("precision") or []
        recall = pr.get("recall") or []
        if len(precision) < 2 or len(recall) < 2:
            ttk.Label(parent, text="Not enough variation to plot PR").pack(padx=8, pady=8)
            return
        fig = Figure(figsize=(5.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall")
        ax.grid(True, alpha=0.3)
        FigureCanvasTkAgg(fig, parent).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_thr_sweep(self, parent, chart: Dict[str, Any]):
        """
        Threshold sweep: precision/recall/F1 versus threshold.
        """
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
        """
        Score histogram.
        """
        self._clear_children(parent)
        if not _HAVE_MPL:
            ttk.Label(parent, text="Matplotlib not available").pack(padx=8, pady=8)
            return

        hist = chart.get("hist") or {}
        edges = hist.get("bin_edges") or [0.0, 1.0]
        pos = hist.get("pos") or [0]
        neg = hist.get("neg") or [0]
        meta = chart.get("hist_meta") or {}

        # Convert edges to bin centres for a bar plot.
        edges = list(edges)
        centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
        width = (edges[1] - edges[0]) * 0.9 if len(edges) > 1 else 0.05

        fig = Figure(figsize=(5.2, 3.4), dpi=100)
        ax = fig.add_subplot(111)

        ax.bar(centers, neg, width=width, alpha=0.6, label=meta.get("legend_neg", "Uncertain"))
        ax.bar(centers, pos, width=width, alpha=0.6, label=meta.get("legend_pos", "Near-duplicates"))

        ax.set_xlabel(meta.get("x_label", "Score"))
        ax.set_ylabel(meta.get("y_label", "Count"))
        ax.set_title(meta.get("title", "Score distribution"))
        ax.legend()
        ax.grid(True, alpha=0.3)

        FigureCanvasTkAgg(fig, parent).get_tk_widget().pack(fill=tk.BOTH, expand=True)
