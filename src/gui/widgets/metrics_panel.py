# src/gui/widgets/metrics_panel.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class MetricsPanel(ttk.Frame):
    def __init__(self, master, *, text: str = "Metrics"):
        super().__init__(master, padding=8)
        self._build_ui(text)

    def _build_ui(self, text: str):
        # title + export
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Label(header, text=text, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(header, text="Export JSON…", command=self._export_current).pack(side=tk.RIGHT)

        # summary box
        self.summary_frame = ttk.LabelFrame(self, text="Run summary")
        self.summary_frame.pack(fill=tk.X, pady=(8, 6))
        self._summary_labels: Dict[str, ttk.Label] = {}
        self._make_summary_grid(self.summary_frame)

        # notebook for per-learner metrics
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self._learner_tabs: Dict[str, Dict[str, Any]] = {}

        # placeholders
        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._last_run_summary: Optional[Dict[str, Any]] = None

    # summary grid
    def _make_summary_grid(self, parent: ttk.LabelFrame):
        grid_items = [
            ("pairs_scored", "Pairs scored"),
            ("duplicates", "Duplicates"),
            ("consensus_rate", "Consensus %"),
            ("escalations_rate", "Escalations %"),
            ("non_duplicates", "Non-duplicates"),
            ("uncertain", "Uncertain"),
            ("clusters", "Clusters"),
            ("epochs_run", "Self-learning epochs"),
        ]
        for i, (key, label) in enumerate(grid_items):
            r, c = divmod(i, 4)
            cell = ttk.Frame(parent)
            cell.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            ttk.Label(cell, text=label, foreground="#666").pack(anchor="w")
            val = ttk.Label(cell, text="—")
            val.pack(anchor="w")
            self._summary_labels[key] = val
        for c in range(4):
            parent.grid_columnconfigure(c, weight=1)

    # per-learner tab
    def _ensure_learner_tab(self, name: str):
        if name in self._learner_tabs:
            return
        frame = ttk.Frame(self.nb, padding=8)
        self.nb.add(frame, text=name)
        # top KPIs
        top = ttk.Frame(frame)
        top.pack(fill=tk.X)
        labels = {
            "auc": ttk.Label(top, text="AUC: —"),
            "brier": ttk.Label(top, text="Brier: —"),
            "threshold": ttk.Label(top, text="Threshold: —"),
            "target_precision": ttk.Label(top, text="Target precision: —"),
        }
        for k in ("auc", "brier", "threshold", "target_precision"):
            labels[k].pack(side=tk.LEFT, padx=(0, 16))

        # reliability table
        lf = ttk.LabelFrame(frame, text="Reliability (expected vs observed)")
        lf.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        cols = ("prob_center", "expected", "observed", "count")
        tree = ttk.Treeview(lf, columns=cols, show="headings", height=8, selectmode="browse")
        headings = [
            ("prob_center", "Bin prob"),
            ("expected", "Expected pos-rate"),
            ("observed", "Observed pos-rate"),
            ("count", "Count"),
        ]
        for cid, label in headings:
            tree.heading(cid, text=label)
            anchor = tk.CENTER if cid != "count" else tk.E
            w = 120 if cid != "count" else 80
            tree.column(cid, width=w, anchor=anchor, stretch=True)
        vs = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.LEFT, fill=tk.Y)

        self._learner_tabs[name] = {"frame": frame, "labels": labels, "tree": tree}

    # public API
    def update_metrics(self, *, run_summary: Optional[Dict[str, Any]] = None, snapshot: Optional[Dict[str, Any]] = None):
        if run_summary is not None:
            self._last_run_summary = run_summary
            self._render_summary(run_summary)
        if snapshot is not None:
            self._last_snapshot = snapshot
            self._render_learners(snapshot.get("learners") or {})

    # fill summary
    def _render_summary(self, s: Dict[str, Any]):
        def _pct(x):
            try:
                return f"{float(x) * 100.0:.1f}%"
            except Exception:
                return "—"

        mapping = {
            "pairs_scored": s.get("pairs_scored") or s.get("pairs") or s.get("total_pairs"),
            "duplicates": s.get("duplicates") or s.get("dups") or s.get("positives"),
            "non_duplicates": s.get("non_duplicates") or s.get("negatives"),
            "uncertain": s.get("uncertain"),
            "consensus_rate": _pct(s.get("consensus_rate") if isinstance(s.get("consensus_rate"), (int, float)) else s.get("consensus")),
            "escalations_rate": _pct(s.get("escalations_rate") if isinstance(s.get("escalations_rate"), (int, float)) else s.get("escalations")),
            "clusters": s.get("clusters") or s.get("num_clusters"),
            "epochs_run": s.get("epochs_run") or s.get("self_learning_epochs"),
        }
        for k, lbl in self._summary_labels.items():
            val = mapping.get(k)
            lbl.configure(text=("—" if val is None else str(val)))

    # fill per-learner tabs
    def _render_learners(self, learners: Dict[str, Any]):
        for name, info in learners.items():
            self._ensure_learner_tab(name)
            tab = self._learner_tabs[name]
            labels = tab["labels"]
            auc = info.get("auc")
            brier = info.get("brier")
            thr = info.get("threshold")
            tprec = info.get("target_precision")
            labels["auc"].configure(text=f"AUC: {auc:.3f}" if isinstance(auc, (int, float)) else "AUC: —")
            labels["brier"].configure(text=f"Brier: {brier:.3f}" if isinstance(brier, (int, float)) else "Brier: —")
            labels["threshold"].configure(text=f"Threshold: {thr:.3f}" if isinstance(thr, (int, float)) else "Threshold: —")
            labels["target_precision"].configure(text=f"Target precision: {tprec:.3f}" if isinstance(tprec, (int, float)) else "Target precision: —")
            self._fill_reliability(tab["tree"], info.get("reliability_bins") or [])

        # remove tabs for learners not present anymore
        present = set(learners.keys())
        stale = [n for n in self._learner_tabs.keys() if n not in present]
        for n in stale:
            tab = self._learner_tabs.pop(n)
            idx = self.nb.index(tab["frame"])
            self.nb.forget(idx)

        # select first tab if none selected
        if self.nb.index("end") > 0 and not self.nb.select():
            self.nb.select(0)

    def _fill_reliability(self, tree: ttk.Treeview, bins: Any):
        tree.delete(*tree.get_children())
        if not isinstance(bins, (list, tuple)):
            return
        for b in bins:
            try:
                pc = float(b.get("prob_center", float("nan")))
                ex = float(b.get("expected_pos_rate", float("nan")))
                ob = float(b.get("observed_pos_rate", float("nan")))
                ct = int(b.get("count", 0))
                tree.insert("", tk.END, values=(f"{pc:.3f}", f"{ex:.3f}", f"{ob:.3f}", f"{ct}"))
            except Exception:
                continue

    # export
    def _export_current(self):
        data = {
            "run_summary": self._last_run_summary,
            "metrics_snapshot": self._last_snapshot,
        }
        try:
            path = filedialog.asksaveasfilename(
                title="Export metrics JSON",
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
                initialfile="metrics_snapshot.json",
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Export", f"Saved metrics to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
