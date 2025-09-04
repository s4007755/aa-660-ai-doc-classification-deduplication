# src/gui/widgets/metrics_panel.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class MetricsPanel(ttk.Frame):
    """
    Compatible Metrics panel for app.py.

    Supports:
      - set_snapshot(snapshot_dict)
      - set_history(list_of_snapshots)
      - update_metrics(run_summary=..., snapshot=...)  # backward-friendly

    Expects the current snapshot shape from src/metrics/metrics.py::metrics_snapshot():
      {
        "run": {...},
        "per_learner": {...},
        "clusters": [...]
      }
    But also tolerates older / alternate key names.
    """

    def __init__(self, master, *, text: str = "Metrics"):
        super().__init__(master, padding=8)
        self._last_snapshot: Dict[str, Any] = {}
        self._last_history: List[Dict[str, Any]] = []

        self._build_ui(text)

    # ---------- UI ----------
    def _build_ui(self, text: str):
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Label(header, text=text, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(header, text="Export JSON…", command=self._export_current).pack(side=tk.RIGHT)

        # Summary box
        self.summary_frame = ttk.LabelFrame(self, text="Run summary")
        self.summary_frame.pack(fill=tk.X, pady=(8, 6))
        self._summary_labels: Dict[str, ttk.Label] = {}
        self._make_summary_grid(self.summary_frame)

        # Notebook for per-learner metrics
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self._learner_tabs: Dict[str, Dict[str, Any]] = {}

    def _make_summary_grid(self, parent: ttk.LabelFrame):
        grid_items = [
            ("pairs_scored", "Pairs"),
            ("duplicates", "Duplicates"),
            ("non_duplicates", "Non-duplicates"),
            ("uncertain", "Uncertain"),
            ("consensus_rate", "Consensus %"),
            ("escalations_rate", "Escalations %"),
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

    def _ensure_learner_tab(self, name: str):
        if name in self._learner_tabs:
            return
        frame = ttk.Frame(self.nb, padding=8)
        self.nb.add(frame, text=name)

        # Top KPIs
        top = ttk.Frame(frame)
        top.pack(fill=tk.X)
        labels = {
            "n": ttk.Label(top, text="N: —"),
            "pos_rate": ttk.Label(top, text="PosRate: —"),
            "auc": ttk.Label(top, text="AUC: —"),
            "brier": ttk.Label(top, text="Brier: —"),
            "threshold": ttk.Label(top, text="Threshold: —"),
            "target_precision": ttk.Label(top, text="Target precision: —"),
        }
        for k in ("n", "pos_rate", "auc", "brier", "threshold", "target_precision"):
            labels[k].pack(side=tk.LEFT, padx=(0, 16))

        # Reliability table
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

    # ---------- Public API (expected by app.py) ----------
    def set_snapshot(self, snapshot: Optional[Dict[str, Any]]):
        """Accepts the full metrics snapshot dict from metrics_snapshot()."""
        self._last_snapshot = snapshot or {}

        # Extract normalized views
        run = self._extract_run_summary(self._last_snapshot)
        per_learner = self._extract_per_learner(self._last_snapshot)
        clusters = self._extract_clusters(self._last_snapshot)

        # Render
        self._render_summary(run, clusters_count=len(clusters))
        self._render_learners(per_learner)

    def set_history(self, history: Optional[List[Dict[str, Any]]]):
        """Optional history support; stored for export or future diff UI."""
        self._last_history = history or []

    # Back-compat helper some code paths may call
    def update_metrics(self, *, run_summary: Optional[Dict[str, Any]] = None, snapshot: Optional[Dict[str, Any]] = None):
        if snapshot is not None:
            self.set_snapshot(snapshot)
        if run_summary is not None:
            # If someone passes a raw run_summary, render it over current snapshot-derived data
            clusters = self._extract_clusters(self._last_snapshot)
            self._render_summary(run_summary, clusters_count=len(clusters))

    # ---------- Rendering ----------
    def _render_summary(self, run_summary: Dict[str, Any], *, clusters_count: int):
        def _pct(x):
            try:
                return f"{float(x) * 100.0:.1f}%"
            except Exception:
                return "—"

        # map to UI keys; tolerate several naming schemes
        mapping = {
            "pairs_scored": run_summary.get("total_pairs")
                             or run_summary.get("pairs_scored")
                             or run_summary.get("pairs")
                             or 0,
            "duplicates": run_summary.get("duplicates") or 0,
            "non_duplicates": run_summary.get("non_duplicates") or 0,
            "uncertain": run_summary.get("uncertain") or 0,
            "consensus_rate": _pct(
                run_summary.get("consensus_rate")
                if isinstance(run_summary.get("consensus_rate"), (int, float))
                else run_summary.get("consensus")
            ),
            "escalations_rate": _pct(
                run_summary.get("escalations_pct")
                if isinstance(run_summary.get("escalations_pct"), (int, float))
                else run_summary.get("escalations_rate")
                if isinstance(run_summary.get("escalations_rate"), (int, float))
                else run_summary.get("escalations")
            ),
            "clusters": clusters_count,
            "epochs_run": run_summary.get("epochs_run") or run_summary.get("self_learning_epochs"),
        }
        for k, lbl in self._summary_labels.items():
            val = mapping.get(k)
            lbl.configure(text=("—" if val is None else str(val)))

    def _render_learners(self, per_learner: Dict[str, Any]):
        # Add/update tabs
        for name, info in sorted(per_learner.items()):
            self._ensure_learner_tab(name)
            tab = self._learner_tabs[name]
            labels = tab["labels"]

            # Support either 'reliability' or 'reliability_bins'
            bins = info.get("reliability")
            if bins is None:
                bins = info.get("reliability_bins")

            n = info.get("n")
            pr = info.get("pos_rate")
            auc = info.get("auc")
            brier = info.get("brier")
            thr = info.get("threshold")
            tprec = info.get("target_precision")

            labels["n"].configure(text=f"N: {int(n)}" if isinstance(n, (int, float)) else "N: —")
            labels["pos_rate"].configure(text=f"PosRate: {float(pr):.3f}" if isinstance(pr, (int, float)) else "PosRate: —")
            labels["auc"].configure(text=f"AUC: {float(auc):.3f}" if isinstance(auc, (int, float)) else "AUC: —")
            labels["brier"].configure(text=f"Brier: {float(brier):.4f}" if isinstance(brier, (int, float)) else "Brier: —")
            labels["threshold"].configure(text=f"Threshold: {float(thr):.3f}" if isinstance(thr, (int, float)) else "Threshold: —")
            labels["target_precision"].configure(text=f"Target precision: {float(tprec):.3f}" if isinstance(tprec, (int, float)) else "Target precision: —")

            self._fill_reliability(tab["tree"], bins)

        # Remove tabs that are no longer present
        present = set(per_learner.keys())
        stale = [n for n in list(self._learner_tabs.keys()) if n not in present]
        for n in stale:
            tab = self._learner_tabs.pop(n, None)
            if tab:
                try:
                    idx = self.nb.index(tab["frame"])
                    self.nb.forget(idx)
                except Exception:
                    pass

        # Ensure a selection exists
        try:
            if self.nb.index("end") > 0 and not self.nb.select():
                self.nb.select(0)
        except Exception:
            pass

    def _fill_reliability(self, tree: ttk.Treeview, bins: Any):
        # Clear table
        for iid in tree.get_children():
            tree.delete(iid)

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

    # ---------- Extractors / normalizers ----------
    def _extract_run_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer new shape
        run = snapshot.get("run")
        if isinstance(run, dict):
            return run
        # Or accept a flattened older shape
        return snapshot.get("run_summary") or {}

    def _extract_per_learner(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        pl = snapshot.get("per_learner")
        if isinstance(pl, dict):
            return pl
        # some older shapes: snapshot["learners"]
        return snapshot.get("learners") or {}

    def _extract_clusters(self, snapshot: Dict[str, Any]) -> List[Any]:
        cl = snapshot.get("clusters")
        return cl if isinstance(cl, list) else []

    # ---------- Export ----------
    def _export_current(self):
        data = {
            "metrics_snapshot": self._last_snapshot,
            "history": self._last_history,
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
