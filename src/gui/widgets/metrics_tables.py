# src/gui/widgets/metrics_tables.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import tkinter as tk
from tkinter import ttk

"""
MetricsTables
Reusable tables shown inside the Metrics view:

Tabs
 * Clusters: overview of duplicate clusters with per-learner probability
   summaries and score dispersion (min–max) per learner.
 * Thresholds: per-learner calibrated threshold slice (precision/recall/F1),
   rendered only when the snapshot indicates a calibrated view.
"""


def _fmt(x: Optional[float], prec: int = 3) -> str:
    """
    Format a numeric value to a fixed number of decimals.
    """
    try:
        if x is None:
            return "—"
        return f"{float(x):.{prec}f}"
    except Exception:
        return "—"


def _clear(tree: ttk.Treeview) -> None:
    """Remove all rows from a Treeview."""
    for iid in tree.get_children():
        tree.delete(iid)


class MetricsTables(ttk.Frame):
    """
    Notebook with two data tables:

      * Clusters: Cluster index, size, human-readable members, and per-learner
        average probabilities and dispersion (min–max).
      * Thresholds: Only present when calibration is active and a threshold
        report is available in the snapshot.
    """

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # Parent notebook holding both tables as tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Clusters tab and table
        self._tab_clusters = ttk.Frame(self.nb)
        self.nb.add(self._tab_clusters, text="Clusters")

        cl_wrap = ttk.Frame(self._tab_clusters, padding=6)
        cl_wrap.pack(fill=tk.BOTH, expand=True)

        # Tree columns:
        self.clusters = ttk.Treeview(
            cl_wrap,
            columns=(
                "idx", "size", "members",
                "avg_sim", "avg_min", "avg_emb",
                "disp_sim", "disp_min", "disp_emb"
            ),
            show="headings",
            height=12,
        )

        for key, label, width, anchor in [
            ("idx", "#", 60, "e"),
            ("size", "Size", 70, "e"),
            ("members", "Members", 520, "w"),
            ("avg_sim", "Avg simhash prob", 140, "e"),
            ("avg_min", "Avg minhash prob", 140, "e"),
            ("avg_emb", "Avg embedding prob", 160, "e"),
            ("disp_sim", "Disp simhash (min–max)", 170, "center"),
            ("disp_min", "Disp minhash (min–max)", 170, "center"),
            ("disp_emb", "Disp embed (min–max)", 170, "center"),
        ]:
            self.clusters.heading(key, text=label)
            self.clusters.column(key, width=width, anchor=anchor)

        # Vertical scrollbar for clusters table
        vsb1 = ttk.Scrollbar(cl_wrap, orient="vertical", command=self.clusters.yview)
        self.clusters.configure(yscrollcommand=vsb1.set)
        self.clusters.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb1.pack(side=tk.LEFT, fill=tk.Y)

        # Thresholds tab
        self._tab_thresholds = ttk.Frame(self.nb)
        th_wrap = ttk.Frame(self._tab_thresholds, padding=6)
        th_wrap.pack(fill=tk.BOTH, expand=True)

        # Columns reflect the calibrated decision slice per learner
        self.thresholds = ttk.Treeview(
            th_wrap,
            columns=("learner", "thr", "prec", "rec", "f1", "support", "near_band"),
            show="headings",
            height=12,
        )

        for key, label, width, anchor in [
            ("learner", "Learner", 160, "w"),
            ("thr", "Threshold", 110, "e"),
            ("prec", "Precision", 110, "e"),
            ("rec", "Recall", 110, "e"),
            ("f1", "F1", 90, "e"),
            ("support", "Support", 90, "e"),
            ("near_band", "Near-band share", 140, "e"),
        ]:
            self.thresholds.heading(key, text=label)
            self.thresholds.column(key, width=width, anchor=anchor)

        vsb2 = ttk.Scrollbar(th_wrap, orient="vertical", command=self.thresholds.yview)
        self.thresholds.configure(yscrollcommand=vsb2.set)
        self.thresholds.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb2.pack(side=tk.LEFT, fill=tk.Y)

        # Track whether the thresholds tab has been attached to the notebook.
        self._thresholds_added = False

    def update_tables(self, snapshot: Dict[str, Any], *, doc_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Redraw both tables from a snapshot.
        """
        self._fill_clusters(snapshot.get("clusters") or [], doc_labels=doc_labels or {})

        use_cal = bool(snapshot.get("use_calibrated", False))
        rep = snapshot.get("thresholds") or {}

        # Dynamically show/hide the Thresholds tab based on calibration mode and data presence.
        if use_cal and rep:
            if not self._thresholds_added:
                self.nb.add(self._tab_thresholds, text="Thresholds")
                self._thresholds_added = True
            self._fill_thresholds(rep)
        else:
            if self._thresholds_added:
                try:
                    self.nb.forget(self._tab_thresholds)
                except Exception:
                    pass
                self._thresholds_added = False


    # Fillers

    def _fill_clusters(self, rows: List[Dict[str, Any]], *, doc_labels: Dict[str, str]) -> None:
        """
        Populate the Clusters table.

        Each row is expected to contain:
          * cluster_index, size, members (list[str])
          * avg_simhash_prob / avg_minhash_prob / avg_embedding_prob
          () dispersion_* dicts with 'min' and 'max'
        """
        _clear(self.clusters)

        for r in rows:
            raw_members: List[str] = r.get("members", [])
            # Replace doc IDs with labels when provided.
            display_members: List[str] = []
            for mid in raw_members:
                label = doc_labels.get(mid) or mid
                display_members.append(label)
            members = ", ".join(display_members)
            # Truncate long member lists to keep the row height consistent.
            if len(members) > 240:
                members = members[:240] + "…"

            ds = r.get("dispersion_simhash", {})
            dm = r.get("dispersion_minhash", {})
            de = r.get("dispersion_embedding", {})

            self.clusters.insert(
                "",
                tk.END,
                values=(
                    r.get("cluster_index", "—"),
                    r.get("size", "—"),
                    members or "—",
                    _fmt(r.get("avg_simhash_prob")),
                    _fmt(r.get("avg_minhash_prob")),
                    _fmt(r.get("avg_embedding_prob")),
                    f"{_fmt(ds.get('min'))} – {_fmt(ds.get('max'))}",
                    f"{_fmt(dm.get('min'))} – {_fmt(dm.get('max'))}",
                    f"{_fmt(de.get('min'))} – {_fmt(de.get('max'))}",
                ),
            )

    def _fill_thresholds(self, rep: Dict[str, Any]) -> None:
        """
        Populate the Thresholds table from a calibrated thresholds report.
        """
        _clear(self.thresholds)

        for learner in sorted(rep.keys()):
            info = rep.get(learner, {})
            self.thresholds.insert(
                "",
                tk.END,
                values=(
                    learner,
                    _fmt(info.get("threshold")),
                    _fmt(info.get("precision")),
                    _fmt(info.get("recall")),
                    _fmt(info.get("f1")),
                    int(info.get("support", 0)),
                    _fmt(info.get("near_band_share")),
                ),
            )
