# src/gui/widgets/run_history.py
from __future__ import annotations

import json
import os
import sys
import webbrowser
from typing import Any, Callable, Dict, Iterable, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from src.persistence import state_store as _store
except Exception:
    _store = None

# Callback signatures
_LoadRuns = Optional[Callable[[], List[Dict[str, Any]]]]
_LoadDetails = Optional[Callable[[int], Dict[str, Any]]]
_OpenReport = Optional[Callable[[int], Optional[str]]]


class RunHistory(ttk.Frame):
    def __init__(
        self,
        master,
        *,
        load_runs: _LoadRuns = None,
        load_details: _LoadDetails = None,
        open_report: _OpenReport = None,
        text: str = "Run history",
    ):
        super().__init__(master, padding=8)
        self._load_runs_cb = load_runs or self._default_load_runs
        self._load_details_cb = load_details or self._default_load_details
        self._open_report_cb = open_report or self._default_open_report

        self._runs: List[Dict[str, Any]] = []
        self._run_by_id: Dict[int, Dict[str, Any]] = {}

        self._build_ui(text)
        self.refresh()

    # UI

    def _build_ui(self, text: str):
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        ttk.Label(top, text=text, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)

        ttk.Button(top, text="Refresh", command=self.refresh).pack(side=tk.RIGHT)
        self.entry_filter = ttk.Entry(top, width=28)
        self.entry_filter.pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(top, text="Filter", command=self._apply_filter).pack(side=tk.RIGHT)

        # Split pane
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # Left: runs table
        left = ttk.Frame(main, padding=(0, 0, 6, 0))
        main.add(left, weight=1)

        cols = ("run_id", "started_at", "ended_at", "status", "notes")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse", height=16)
        headings = [
            ("run_id", "Run"),
            ("started_at", "Started"),
            ("ended_at", "Ended"),
            ("status", "Status"),
            ("notes", "Notes"),
        ]
        widths = {"run_id": 80, "started_at": 160, "ended_at": 160, "status": 100, "notes": 320}
        for cid, label in headings:
            self.tree.heading(cid, text=label)
            anc = tk.CENTER if cid in {"run_id", "status"} else tk.W
            self.tree.column(cid, width=widths[cid], anchor=anc, stretch=True)
        vs = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Right: details
        right = ttk.Frame(main)
        main.add(right, weight=2)

        ab = ttk.Frame(right)
        ab.pack(fill=tk.X)
        ttk.Button(ab, text="Open report…", command=self._open_report).pack(side=tk.LEFT)
        ttk.Button(ab, text="Export config JSON…", command=self._export_config).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(ab, text="Export calibrations JSON…", command=self._export_calibrations).pack(side=tk.LEFT, padx=(6, 0))

        # Summary card
        self.card = ttk.LabelFrame(right, text="Summary")
        self.card.pack(fill=tk.X, pady=(8, 6))
        self._sum_labels: Dict[str, ttk.Label] = {}
        self._make_summary(self.card)

        # Notebook for Config / Calibrations
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Config tab
        self.tab_cfg = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.tab_cfg, text="Config")
        self.txt_cfg = tk.Text(self.tab_cfg, wrap="word", height=16)
        self.txt_cfg.pack(fill=tk.BOTH, expand=True)
        self.txt_cfg.configure(state="disabled")

        # Calibrations tab
        self.tab_cal = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.tab_cal, text="Calibrations")
        self.cal_tree = ttk.Treeview(
            self.tab_cal,
            columns=("learner", "method", "threshold", "reliability_bins"),
            show="headings",
            height=12,
            selectmode="browse",
        )
        for cid, label, w in [
            ("learner", "Learner", 160),
            ("method", "Method", 120),
            ("threshold", "Threshold", 100),
            ("reliability_bins", "Reliability bins", 140),
        ]:
            self.cal_tree.heading(cid, text=label)
            anc = tk.W if cid in {"learner", "method"} else tk.CENTER
            self.cal_tree.column(cid, width=w, anchor=anc, stretch=True)
        v2 = ttk.Scrollbar(self.tab_cal, orient=tk.VERTICAL, command=self.cal_tree.yview)
        self.cal_tree.configure(yscrollcommand=v2.set)
        self.cal_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v2.pack(side=tk.LEFT, fill=tk.Y)

    def _make_summary(self, parent: ttk.LabelFrame):
        fields = [
            ("run_id", "Run"),
            ("started_at", "Started"),
            ("ended_at", "Ended"),
            ("status", "Status"),
            ("notes", "Notes"),
            ("pairs_scored", "Pairs"),
            ("duplicates", "Duplicates"),
            ("non_duplicates", "Non-duplicates"),
            ("uncertain", "Uncertain"),
            ("consensus_rate", "Consensus %"),
            ("escalations_rate", "Escalations %"),
            ("clusters", "Clusters"),
            ("epochs_run", "Self-learn epochs"),
            ("report_path", "Report"),
        ]
        for i, (k, lab) in enumerate(fields):
            row, col = divmod(i, 2)
            box = ttk.Frame(parent)
            box.grid(row=row, column=col, sticky="nsew", padx=6, pady=4)
            ttk.Label(box, text=lab, foreground="#666").pack(anchor="w")
            val = ttk.Label(box, text="—")
            val.pack(anchor="w")
            self._sum_labels[k] = val
        for c in range(2):
            parent.grid_columnconfigure(c, weight=1)

    # Public API

    def refresh(self):
        try:
            runs = self._load_runs_cb() or []
        except Exception as e:
            messagebox.showerror("Run history", f"Failed to load runs:\n{e}")
            runs = []
        self._runs = sorted(runs, key=lambda r: (r.get("run_id") or 0), reverse=True)
        self._run_by_id = {int(r["run_id"]): r for r in self._runs if "run_id" in r}
        self._populate_runs(self._runs)

    # Internals

    def _populate_runs(self, rows: Iterable[Dict[str, Any]]):
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    r.get("run_id", "—"),
                    r.get("started_at", "—"),
                    r.get("ended_at", "—"),
                    r.get("status", "—"),
                    (r.get("notes") or "—")[:200],
                ),
            )

    def _apply_filter(self):
        q = (self.entry_filter.get() or "").strip().lower()
        if not q:
            self._populate_runs(self._runs)
            return
        flt = []
        for r in self._runs:
            hay = " ".join(
                str(r.get(k, "")) for k in ("run_id", "started_at", "ended_at", "status", "notes")
            ).lower()
            if q in hay:
                flt.append(r)
        self._populate_runs(flt)

    def _on_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            self._render_detail(None)
            return
        iid = sel[0]
        run_id = self.tree.set(iid, "run_id")
        try:
            run_id = int(run_id)
        except Exception:
            self._render_detail(None)
            return
        try:
            details = self._load_details_cb(run_id) or {}
        except Exception as e:
            messagebox.showerror("Run details", f"Failed to load details:\n{e}")
            details = {}
        self._render_detail(details)

    def _render_detail(self, d: Optional[Dict[str, Any]]):
        # Clear summary labels
        for k, lbl in self._sum_labels.items():
            lbl.configure(text="—")

        # Clear config text
        self.txt_cfg.configure(state="normal")
        self.txt_cfg.delete("1.0", tk.END)
        self.txt_cfg.configure(state="disabled")

        # Clear calibrations
        self.cal_tree.delete(*self.cal_tree.get_children())

        if not d:
            return

        # Summary values
        def _pct(x):
            try:
                return f"{float(x) * 100.0:.1f}%"
            except Exception:
                return "—"

        mapping = {
            "run_id": d.get("run_id"),
            "started_at": d.get("started_at"),
            "ended_at": d.get("ended_at"),
            "status": d.get("status"),
            "notes": d.get("notes"),
            "pairs_scored": d.get("pairs_scored") or d.get("pairs"),
            "duplicates": d.get("duplicates"),
            "non_duplicates": d.get("non_duplicates"),
            "uncertain": d.get("uncertain"),
            "consensus_rate": _pct(d.get("consensus_rate")),
            "escalations_rate": _pct(d.get("escalations_rate")),
            "clusters": d.get("clusters"),
            "epochs_run": d.get("epochs_run"),
            "report_path": d.get("report_path"),
        }
        for k, v in mapping.items():
            if k in self._sum_labels and v is not None:
                self._sum_labels[k].configure(text=str(v))

        # Config JSON
        cfg_json = d.get("config_json") or {}
        try:
            pretty = json.dumps(cfg_json if isinstance(cfg_json, dict) else json.loads(cfg_json), ensure_ascii=False, indent=2)
        except Exception:
            pretty = str(cfg_json)
        self.txt_cfg.configure(state="normal")
        self.txt_cfg.insert("1.0", pretty)
        self.txt_cfg.configure(state="disabled")

        # Calibrations list
        cals = d.get("calibrations") or []
        for row in cals:
            learner = row.get("learner_name") or row.get("learner") or "—"
            method = row.get("method") or "—"
            threshold = "—"
            try:
                params = row.get("params_json")
                if isinstance(params, str):
                    params = json.loads(params)
                thr = params.get("threshold") if isinstance(params, dict) else None
                if thr is None:
                    thr = row.get("threshold")
                if isinstance(thr, (int, float)):
                    threshold = f"{thr:.3f}"
            except Exception:
                pass
            try:
                rb = row.get("reliability_json")
                if isinstance(rb, str):
                    rb = json.loads(rb)
                rbcnt = len(rb) if isinstance(rb, list) else 0
            except Exception:
                rbcnt = 0
            self.cal_tree.insert("", tk.END, values=(learner, method, threshold, rbcnt))

    # Actions

    def _open_report(self):
        run = self._selected_run_details()
        if not run:
            return
        path = None
        try:
            path = self._open_report_cb(int(run.get("run_id")))
        except Exception:
            path = None
        path = path or run.get("report_path")
        if path and os.path.exists(path):
            self._open_path(path)
            return
        path = filedialog.askopenfilename(
            title="Open report…",
            initialdir=os.path.abspath("./reports"),
            filetypes=[("Reports", "*.html *.md *.txt"), ("All files", "*.*")],
        )
        if path:
            self._open_path(path)

    def _export_config(self):
        run = self._selected_run_details()
        if not run:
            return
        cfg = run.get("config_json") or {}
        try:
            payload = cfg if isinstance(cfg, dict) else json.loads(cfg)
        except Exception:
            payload = {"raw": str(cfg)}
        path = filedialog.asksaveasfilename(
            title="Export config JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile=f"run_{run.get('run_id','unknown')}_config.json",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Exported", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _export_calibrations(self):
        run = self._selected_run_details()
        if not run:
            return
        cals = run.get("calibrations") or []
        path = filedialog.asksaveasfilename(
            title="Export calibrations JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile=f"run_{run.get('run_id','unknown')}_calibrations.json",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cals, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Exported", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # Helpers

    def _selected_run_details(self) -> Optional[Dict[str, Any]]:
        sel = self.tree.selection()
        if not sel:
            return None
        run_id = self.tree.set(sel[0], "run_id")
        try:
            rid = int(run_id)
        except Exception:
            return None
        try:
            return self._load_details_cb(rid) or {}
        except Exception:
            return None

    def _open_path(self, path: str):
        try:
            if sys.platform.startswith("darwin"):
                os.system(f"open '{path}'")
            elif os.name == "nt":
                os.startfile(path)
            else:
                webbrowser.open(f"file://{os.path.abspath(path)}")
        except Exception as e:
            messagebox.showerror("Open report", str(e))

    # Defaults (state_store)

    def _default_load_runs(self) -> List[Dict[str, Any]]:
        if _store is None or not hasattr(_store, "list_runs"):
            return []
        return _store.list_runs()

    def _default_load_details(self, run_id: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if _store is None:
            return d
        # run meta
        if hasattr(_store, "get_run"):
            try:
                d.update(_store.get_run(run_id) or {})
            except Exception:
                pass
        # config
        if "config_json" not in d and hasattr(_store, "get_run_config"):
            try:
                d["config_json"] = _store.get_run_config(run_id)
            except Exception:
                pass
        # calibrations
        if hasattr(_store, "get_calibrations_for_run"):
            try:
                d["calibrations"] = _store.get_calibrations_for_run(run_id)
            except Exception:
                pass
        if hasattr(_store, "get_run_summary"):
            try:
                d.update(_store.get_run_summary(run_id) or {})
            except Exception:
                pass
        # possible stored report path
        if hasattr(_store, "get_report_path"):
            try:
                rp = _store.get_report_path(run_id)
                if rp:
                    d["report_path"] = rp
            except Exception:
                pass
        return d

    def _default_open_report(self, run_id: int) -> Optional[str]:
        if _store and hasattr(_store, "get_report_path"):
            try:
                return _store.get_report_path(run_id)
            except Exception:
                return None
        return None
