# src/app.py
from __future__ import annotations

import hashlib
import os
import random
import time
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Iterable
import sqlite3
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import psutil
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

import pandas as pd

# Storage and pipeline
from src.storage import sqlite_store
from src.persistence import state_store
from src.features.text_preproc import build_document_view, compute_corpus_stats, normalize_text
from src.learners.base import DocumentView, LearnerConfig
from src.ensemble.arbiter import ArbiterConfig
from src.pipelines.near_duplicate import (
    PipelineConfig,
    CandidateConfig,
    BootstrapConfig,
    SelfLearningConfig,
    run_intelligent_pipeline,
)
# GUI widgets
from src.gui.widgets.learner_card import LearnerCard
from src.gui.widgets.arbiter_panel import ArbiterPanel
from src.gui.widgets.metrics_panel import MetricsPanel
from src.gui.widgets.trace_viewer import TraceViewer
from src.gui.widgets.run_history import RunHistory

# Try to import ingestion
_ingestion_mod = None
try:
    from src.pipelines import ingestion as _ingestion_mod
except Exception:
    try:
        import ingestion as _ingestion_mod
    except Exception:
        _ingestion_mod = None


# small scrollable frame helper
class VScrollFrame(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # Mouse wheel support
        self.inner.bind_all("<MouseWheel>", self._on_mousewheel)
        self.inner.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.inner.bind_all("<Button-5>", self._on_mousewheel_linux)

        # Resize inner width with frame
        self.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width - self.vbar.winfo_width()))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(+3, "units")


class App(tk.Tk):
    # Main window
    def __init__(self):
        super().__init__()
        self.title("Duplicate Finder")
        self.geometry("1360x900")
        self.minsize(1200, 760)

        # init databases
        sqlite_store.init_db()
        state_store.ensure_state_schema()

        # runtime state
        self.docs: List[DocumentView] = []
        self.pipeline_result: Optional[Dict[str, Any]] = None
        self.running = False

        # timer state
        self._elapsed_job: Optional[str] = None
        self._run_start_ts: Optional[float] = None

        # NEW: resource monitor state
        self._res_job: Optional[str] = None
        self._proc = psutil.Process(os.getpid())

        self._build_ui()
        self._refresh_docs_from_db()
        self._refresh_docs_tab()

    # UI layout
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.tab_mode = ttk.Frame(nb)
        self.tab_traces = ttk.Frame(nb)
        self.tab_metrics = ttk.Frame(nb)
        self.tab_history = ttk.Frame(nb)
        self.tab_docs = ttk.Frame(nb)

        nb.add(self.tab_mode, text="Run")
        nb.add(self.tab_traces, text="Decision Traces")
        nb.add(self.tab_metrics, text="Metrics")
        nb.add(self.tab_history, text="Run History")
        nb.add(self.tab_docs, text="Documents")

        # Main tab
        self._build_mode_tab(self.tab_mode)
        # Decision Traces tab
        self.trace_view = TraceViewer(self.tab_traces, text="Decision Traces")
        self.trace_view.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # Metrics tab
        self.metrics_panel = MetricsPanel(self.tab_metrics)
        self.metrics_panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # History tab
        self.history = RunHistory(self.tab_history)
        self.history.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # Documents tab
        self._build_docs_tab(self.tab_docs)

    # Main content
    def _build_mode_tab(self, parent: ttk.Frame):
        # Toolbar
        bar = ttk.Frame(parent, padding=8)
        bar.pack(fill=tk.X)

        # Right side: run controls and files counter
        controls = ttk.Frame(bar)
        controls.pack(side=tk.RIGHT)

        # Files counter
        self.lbl_files = ttk.Label(controls, text="Files: —")
        self.lbl_files.pack(side=tk.LEFT, padx=(0, 12))

        # Left side: refresh files button
        ttk.Button(
            bar,
            text="Refresh files",
            command=lambda: [self._refresh_docs_from_db(), self._refresh_docs_tab()],
        ).pack(side=tk.LEFT, padx=(10, 0))

        self.btn_run = ttk.Button(controls, text="Run", command=self._on_run_clicked)
        self.btn_run.pack(side=tk.RIGHT)

        main = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # LEFT: scrollable frame
        left_wrap = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left_wrap, weight=1)
        main.add(right, weight=1)

        left_scroller = VScrollFrame(left_wrap)
        left_scroller.pack(fill=tk.BOTH, expand=True)
        left = left_scroller.inner

        # Profiles row
        prof = ttk.LabelFrame(left, text="Profile")
        prof.pack(fill=tk.X, padx=4, pady=(6, 6))
        ttk.Label(prof, text="Preset:").pack(side=tk.LEFT, padx=(8, 6))

        self.profile_var = tk.StringVar(value="Balanced")
        cb = ttk.Combobox(prof, textvariable=self.profile_var, width=22, values=[
            "Balanced",
            "High Precision",
            "Recall-Heavy",
            "Custom",
        ])
        cb.pack(side=tk.LEFT)
        ttk.Button(prof, text="Apply preset", command=self._apply_preset).pack(side=tk.LEFT, padx=(8, 8))

        # Advanced settings toggle
        toggle_row = ttk.Frame(left)
        toggle_row.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.var_show_adv = tk.BooleanVar(value=False)
        self.btn_toggle_adv = ttk.Checkbutton(
            toggle_row,
            text="Advanced settings ▸",
            variable=self.var_show_adv,
            command=self._toggle_advanced_section,
            style="Toolbutton",
        )
        self.btn_toggle_adv.pack(side=tk.LEFT, padx=(6, 0))

        # Advanced container
        self.adv_container = ttk.LabelFrame(left, text="Advanced")
        # contents of advanced:
        self.card_sim = LearnerCard(self.adv_container, learner_name="SimHash", kind="simhash", config=LearnerConfig(), collapsed=True)
        self.card_sim.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.card_min = LearnerCard(self.adv_container, learner_name="MinHash", kind="minhash", config=LearnerConfig(), collapsed=True)
        self.card_min.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.card_emb = LearnerCard(self.adv_container, learner_name="Embedding", kind="embedding", config=LearnerConfig(), collapsed=True)
        self.card_emb.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.arbiter_panel = ArbiterPanel(self.adv_container, config=ArbiterConfig(), text="Consensus & Escalation")
        self.arbiter_panel.pack(fill=tk.X, padx=4, pady=(0, 6))

        misc = ttk.LabelFrame(self.adv_container, text="Self-learning & Candidate Generation")
        misc.pack(fill=tk.X, padx=4, pady=(0, 12))

        # Self-learning
        self.var_sl_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(misc, text="Enable self-learning", variable=self.var_sl_enable).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(misc, text="Epochs").grid(row=0, column=1, sticky="e", padx=(16, 4))
        self.entry_sl_epochs = ttk.Entry(misc, width=6)
        self.entry_sl_epochs.insert(0, "2")
        self.entry_sl_epochs.grid(row=0, column=2, sticky="w", padx=(0, 8), pady=4)

        # Candidate config
        ttk.Label(misc, text="LSH threshold").grid(row=1, column=0, sticky="e", padx=(8, 4))
        self.entry_lsh_thr = ttk.Entry(misc, width=6)
        self.entry_lsh_thr.insert(0, "0.60")
        self.entry_lsh_thr.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(misc, text="Shingle size").grid(row=1, column=2, sticky="e", padx=(8, 4))
        self.entry_shingle = ttk.Entry(misc, width=6)
        self.entry_shingle.insert(0, "3")
        self.entry_shingle.grid(row=1, column=3, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(misc, text="Max cand/doc").grid(row=2, column=0, sticky="e", padx=(8, 4))
        self.entry_cand_doc = ttk.Entry(misc, width=8)
        self.entry_cand_doc.insert(0, "2000")
        self.entry_cand_doc.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(misc, text="Max total").grid(row=2, column=2, sticky="e", padx=(8, 4))
        self.entry_cand_total = ttk.Entry(misc, width=10)
        self.entry_cand_total.insert(0, "")
        self.entry_cand_total.grid(row=2, column=3, sticky="w", padx=(0, 8), pady=4)

        for c in range(4):
            misc.grid_columnconfigure(c, weight=1)

        # RIGHT: run status and quick metrics
        status = ttk.LabelFrame(right, text="Run status")
        status.pack(fill=tk.BOTH, expand=True, padx=4, pady=(6, 6))

        self.var_status = tk.StringVar(value="Idle")
        ttk.Label(status, textvariable=self.var_status).pack(anchor="w", padx=8, pady=(6, 0))

        # elapsed timer
        timer_row = ttk.Frame(status)
        timer_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Label(timer_row, text="Elapsed:").pack(side=tk.LEFT)
        self.var_elapsed = tk.StringVar(value="00:00:00")
        ttk.Label(timer_row, textvariable=self.var_elapsed).pack(side=tk.LEFT, padx=(6, 0))

        # Live counters
        counters = ttk.Frame(status)
        counters.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.var_cnt_pairs = tk.StringVar(value="Pairs: —")
        self.var_cnt_exact = tk.StringVar(value="Exact-duplicate: —")
        self.var_cnt_near  = tk.StringVar(value="Near-duplicate: —")
        self.var_cnt_unc   = tk.StringVar(value="Uncertain: —")
        ttk.Label(counters, textvariable=self.var_cnt_pairs).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Label(counters, textvariable=self.var_cnt_exact).grid(row=0, column=1, sticky="w", padx=(0, 12))
        ttk.Label(counters, textvariable=self.var_cnt_near).grid(row=0, column=2, sticky="w", padx=(0, 12))
        ttk.Label(counters, textvariable=self.var_cnt_unc).grid(row=1, column=0, sticky="w", padx=(0, 12))

        # Summary box
        self.txt_summary = tk.Text(status, height=16, wrap="word")
        self.txt_summary.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.txt_summary.configure(state="disabled")

        # Resource usage (toggle and panel)
        res_toggle_row = ttk.Frame(status)
        res_toggle_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.var_show_res = tk.BooleanVar(value=True)
        self.btn_toggle_res = ttk.Checkbutton(
            res_toggle_row,
            text="Resource usage ▾",
            variable=self.var_show_res,
            command=self._toggle_resource_section,
            style="Toolbutton",
        )
        self.btn_toggle_res.pack(side=tk.LEFT)

        self.res_container = ttk.LabelFrame(status, text="Resource usage")

        self.var_res_cpu = tk.StringVar(value="")
        self.var_res_mem = tk.StringVar(value="")
        self.var_res_io  = tk.StringVar(value="")
        self.var_res_gpu = tk.StringVar(value="")

        ttk.Label(self.res_container, textvariable=self.var_res_cpu).pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Label(self.res_container, textvariable=self.var_res_mem).pack(anchor="w", padx=8, pady=(2, 0))
        ttk.Label(self.res_container, textvariable=self.var_res_io).pack(anchor="w", padx=8, pady=(2, 6))
        if _NVML_OK:
            ttk.Label(self.res_container, textvariable=self.var_res_gpu).pack(anchor="w", padx=8, pady=(0, 8))

        # show by default
        self.res_container.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _toggle_advanced_section(self):
        show = bool(self.var_show_adv.get())
        self.btn_toggle_adv.configure(text="Advanced settings ▾" if show else "Advanced settings ▸")
        try:
            if show:
                self.adv_container.pack(fill=tk.X, padx=4, pady=(0, 8))
            else:
                self.adv_container.pack_forget()
        except Exception:
            pass

    # toggle for resource usage panel
    def _toggle_resource_section(self):
        show = bool(self.var_show_res.get())
        self.btn_toggle_res.configure(text="Resource usage ▾" if show else "Resource usage ▸")
        try:
            if show:
                self.res_container.pack(fill=tk.X, padx=8, pady=(0, 8), after=self.txt_summary)
            else:
                self.res_container.pack_forget()
        except Exception:
            pass

    # Documents tab
    def _build_docs_tab(self, parent: ttk.Frame):
        top = ttk.Frame(parent, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Add Files…", command=self._on_add_files).pack(side=tk.LEFT)
        ttk.Button(top, text="Add Folder…", command=self._on_add_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Delete Selected", command=self._on_delete_selected_docs).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Refresh", command=self._refresh_docs_tab).pack(side=tk.RIGHT)

        self.lbl_doc_count = ttk.Label(top, text="— files")
        self.lbl_doc_count.pack(side=tk.RIGHT, padx=(0, 12))

        # Table
        table_frame = ttk.Frame(parent, padding=(8, 0, 8, 8))
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("doc_id", "filename", "language", "filesize", "filepath")
        self.docs_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=18)
        self.docs_tree.heading("doc_id", text="Doc ID")
        self.docs_tree.heading("filename", text="Filename")
        self.docs_tree.heading("language", text="Lang")
        self.docs_tree.heading("filesize", text="Size (KB)")
        self.docs_tree.heading("filepath", text="Path")

        self.docs_tree.column("doc_id", width=420, anchor="w")
        self.docs_tree.column("filename", width=220, anchor="w")
        self.docs_tree.column("language", width=80, anchor="center")
        self.docs_tree.column("filesize", width=100, anchor="e")
        self.docs_tree.column("filepath", width=420, anchor="w")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.docs_tree.yview)
        self.docs_tree.configure(yscrollcommand=vsb.set)
        self.docs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        # Status bar
        self.doc_status = ttk.Label(parent, text="", anchor="w")
        self.doc_status.pack(fill=tk.X, padx=8, pady=(0, 8))

    # Load docs from DB
    def _refresh_docs_from_db(self):
        pairs = sqlite_store.get_docs_text(include_dirty=False)
        self.docs = [
            build_document_view(doc_id=did, text=(txt or ""), language=None, meta={})
            for (did, txt) in pairs
            if (txt or "").strip()
        ]

    # CSV ingestion
    def _on_add_csv(self):
        file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        df = pd.read_csv(file_path)
        text_col = "text"
        id_col = "id" if "id" in df.columns else None
        docs = []
        for i, row in df.iterrows():
            doc_id = str(row[id_col]) if id_col else str(i)
            text = row[text_col]
            if not text or not str(text).strip():
                continue
            docs.append(build_document_view(doc_id=doc_id, text=normalize_text(str(text)), language=None, meta={}))
        self.docs = docs
        self.lbl_docs.configure(text=f"Docs: {len(self.docs)}")
        self._csv_loaded = True
        self._refresh_docs_tab()


    # Fill docs tab table
    def _refresh_docs_tab(self):
    # clear existing
        for iid in self.docs_tree.get_children():
            self.docs_tree.delete(iid)

        if getattr(self, "_csv_loaded", False):
            # show CSV docs
            for d in self.docs:
                self.docs_tree.insert(
                    "", tk.END,
                    values=(d.doc_id, f"csv_row_{d.doc_id}", "—", len(d.text) // 1024, "CSV import")
                )
            self.lbl_doc_count.configure(text=f"{len(self.docs)} rows from CSV")
        else:
            # fall back to DB
            try:
                rows = sqlite_store.get_all_document_files()
            except Exception as e:
                messagebox.showerror("DB error", str(e))
                return
    # clear existing
        for iid in self.docs_tree.get_children():
            self.docs_tree.delete(iid)

        if getattr(self, "_csv_loaded", False):
            # show CSV docs
            for d in self.docs:
                self.docs_tree.insert(
                    "", tk.END,
                    values=(d.doc_id, f"csv_row_{d.doc_id}", "—", len(d.text) // 1024, "CSV import")
                )
            self.lbl_doc_count.configure(text=f"{len(self.docs)} rows from CSV")
        else:
            # fall back to DB
            try:
                rows = sqlite_store.get_all_document_files()
            except Exception as e:
                messagebox.showerror("DB error", str(e))
                return

            for r in rows:
                doc_id = r.get("doc_id", "")
                lang = r.get("language") or "—"
                size_kb = int((r.get("filesize") or 0) / 1024)
                fname = r.get("filename") or "—"
                fpath = r.get("filepath") or ""
                self.docs_tree.insert("", tk.END, values=(doc_id, fname, lang, size_kb, fpath))
            for r in rows:
                doc_id = r.get("doc_id", "")
                lang = r.get("language") or "—"
                size_kb = int((r.get("filesize") or 0) / 1024)
                fname = r.get("filename") or "—"
                fpath = r.get("filepath") or ""
                self.docs_tree.insert("", tk.END, values=(doc_id, fname, lang, size_kb, fpath))

            unique_docs = len({r.get("doc_id") for r in rows})
            total_files = len(rows)
            self.lbl_doc_count.configure(text=f"{total_files} files across {unique_docs} docs")
            self.lbl_files.configure(text=f"Files: {total_files}")
            unique_docs = len({r.get("doc_id") for r in rows})
            total_files = len(rows)
            self.lbl_doc_count.configure(text=f"{total_files} files across {unique_docs} docs")
            self.lbl_files.configure(text=f"Files: {total_files}")

            self._refresh_docs_from_db()
            self._refresh_docs_from_db()


    # helper: map doc_id -> filename (for trace labels)
    def _doc_labels_from_db(self) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        try:
            rows = sqlite_store.get_all_document_files()
        except Exception:
            return labels
        for r in rows:
            did = r.get("doc_id")
            name = r.get("filename") or os.path.basename(r.get("filepath") or "") or None
            if did and name and did not in labels:
                labels[did] = name
        return labels

    # helper: try to pull calibrated thresholds from result
    def _thresholds_from_result(self, res: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        candidates = []
        for k in ("calibration_snapshot", "calibration", "learner_states", "learners"):
            snap = res.get(k)
            if isinstance(snap, dict):
                candidates.append(snap)
        for snap in candidates:
            for name, info in snap.items():
                if isinstance(info, dict):
                    thr = (
                        info.get("threshold")
                        or (info.get("calibration") or {}).get("threshold")
                        or info.get("thr")
                    )
                    if thr is not None:
                        out[name.lower()] = float(thr)
        return out

    def _on_run_clicked(self):
        if self.running:
            return
        if len(self.docs) < 2:
            messagebox.showinfo("No data", "Need at least 2 normalized docs in DB. Ingest files first (Documents tab).")
            return

        # Build pipeline config from UI
        try:
            pconfig = self._make_pipeline_config()
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return

        # Background thread to avoid freezing UI
        self.running = True
        self.btn_run.configure(state=tk.DISABLED)
        self.var_status.set("Starting…")
        self._reset_counters()
        self._start_timer()
        # start resource monitor
        self._start_resource_monitor()

        def worker():
            err = None
            result = None
            try:
                _ = compute_corpus_stats(self.docs)

                def ui_progress(done: int, total: int, phase: str):
                    pct = (done / total * 100.0) if total else 0.0
                    self.after(0, lambda: self.var_status.set(f"{phase}… {done}/{total} ({pct:.1f}%)"))

                result = run_intelligent_pipeline(
                    self.docs,
                    config=pconfig,
                    run_notes=self.profile_var.get(),
                    progress_cb=ui_progress,
                )
            except Exception as ex:
                err = ex
            finally:
                self.after(0, lambda: self._finish_run(result, err))

        threading.Thread(target=worker, daemon=True).start()

    # Finish run on main thread
    def _finish_run(self, result: Optional[Dict[str, Any]], err: Optional[BaseException]):
        self.running = False
        self.btn_run.configure(state=tk.NORMAL)
        self._stop_timer()
        # stop resource monitor
        self._stop_resource_monitor()

        if err is not None:
            self.var_status.set("Error")
            messagebox.showerror("Pipeline failed", f"{err}\n\n{traceback.format_exc()}")
            return

        self.pipeline_result = result or {}
        self.var_status.set("Completed")

        # Update summary and counters
        self._render_summary(self.pipeline_result)
        self._update_counters_from_result(self.pipeline_result)

        # Decision Traces with filenames
        traces = self.pipeline_result.get("traces") or []
        self.trace_view.set_traces(traces, doc_labels=self._doc_labels_from_db())

        # MetricsPanel
        raw_snap = self.pipeline_result.get("metrics_snapshot") or {}
        run_summary = (self.pipeline_result.get("run_summary") or raw_snap.get("run") or {})

        if "escalations_pct" in run_summary and "escalations_rate" not in run_summary:
            run_summary = dict(run_summary)
            run_summary["escalations_rate"] = run_summary["escalations_pct"]

        per_learner = raw_snap.get("per_learner", {})
        snap_norm = {
            "per_learner": per_learner,
            "run": run_summary,
            "clusters": raw_snap.get("clusters", []),
            "charts": raw_snap.get("charts", {}),
            "thresholds": raw_snap.get("thresholds", {}),
            "consensus": raw_snap.get("consensus", {}),
            "escalations": raw_snap.get("escalations", {}),
        }
        self.metrics_panel.update_metrics(run_summary=run_summary, snapshot=snap_norm, doc_labels=self._doc_labels_from_db())

        # push numbers into learner cards header KPIs
        thresholds = self._thresholds_from_result(self.pipeline_result)

        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        # simhash
        sim_cfg = self.card_sim.get_config()
        sim_brier = _num((per_learner.get("simhash") or {}).get("brier"))
        self.card_sim.set_header_metrics(
            est_precision=sim_cfg.target_precision,
            threshold=thresholds.get("simhash"),
            brier=sim_brier,
            target_precision=sim_cfg.target_precision,
        )

        # minhash
        min_cfg = self.card_min.get_config()
        min_brier = _num((per_learner.get("minhash") or {}).get("brier"))
        self.card_min.set_header_metrics(
            est_precision=min_cfg.target_precision,
            threshold=thresholds.get("minhash"),
            brier=min_brier,
            target_precision=min_cfg.target_precision,
        )

        # embedding
        emb_cfg = self.card_emb.get_config()
        emb_brier = _num((per_learner.get("embedding") or {}).get("brier"))
        emb_thr = thresholds.get("embedding")
        if emb_thr is None:
            try:
                emb_thr = float(emb_cfg.extras.get("cosine_threshold"))
            except Exception:
                emb_thr = None
        self.card_emb.set_header_metrics(
            est_precision=emb_cfg.target_precision,
            threshold=emb_thr,
            brier=emb_brier,
            target_precision=emb_cfg.target_precision,
        )

        # Refresh history tab
        try:
            self.history.refresh()
        except Exception:
            pass

        messagebox.showinfo("Done", f"Scored pairs: {self.pipeline_result.get('pairs_scored', 0)}")

    # Summary text block
    def _render_summary(self, res: Dict[str, Any]):
        lines = []
        lines.append(f"Run ID: {res.get('run_id', '—')}")
        rs = res.get("run_summary") or {}

        # Prefer deriving exact/near from traces to avoid mismatches
        traces = res.get("traces") or []
        ex_calc = 0
        nr_calc = 0
        if traces:
            for tr in traces:
                try:
                    t = tr.as_dict() if hasattr(tr, "as_dict") else tr
                    if (t.get('final_label') or '').upper() == 'DUPLICATE':
                        if (t.get('dup_kind') or '').upper() == 'EXACT':
                            ex_calc += 1
                        else:
                            nr_calc += 1
                except Exception:
                    pass
            ex = ex_calc
            nr = nr_calc
        else:
            ex = rs.get('exact_duplicates', 0)
            nr = rs.get('near_duplicates', 0)

        # Pairs
        pairs = rs.get('pairs_scored') or rs.get('total_pairs') or rs.get('pairs') or '—'
        lines.append(f"Pairs: {pairs}")
        lines.append(f"Exact-duplicate: {ex}")
        lines.append(f"Near-duplicate: {nr}")
        try:
            lines.append(f"Duplicates (total): {int(ex) + int(nr)}")
        except Exception:
            pass

        lines.append(f"Uncertain: {rs.get('uncertain', '—')}")
        cr = rs.get("consensus_rate")
        er = rs.get("escalations_rate") or rs.get("escalations_pct")
        if isinstance(cr, (int, float)):
            lines.append(f"Consensus rate: {cr * 100:.1f}%")
        if isinstance(er, (int, float)):
            lines.append(f"Escalations: {er * 100:.1f}%")

        clusters = res.get("clusters") or []
        if clusters:
            lines.append(f"Clusters: {len(clusters)}")
            top = min(5, len(clusters))
            for i in range(top):
                lines.append(f"  - Cluster {i+1}: {len(clusters[i])} docs")

        self.txt_summary.configure(state="normal")
        self.txt_summary.delete("1.0", tk.END)
        self.txt_summary.insert("1.0", "\n".join(lines))
        self.txt_summary.configure(state="disabled")

    # Build PipelineConfig from UI
    def _make_pipeline_config(self) -> PipelineConfig:
        # Learners
        cfg_sim = self.card_sim.get_config()
        cfg_min = self.card_min.get_config()
        cfg_emb = self.card_emb.get_config()

        # Arbiter
        cfg_arb = self.arbiter_panel.get_config()

        # Candidates
        try:
            lsh_thr = float(self.entry_lsh_thr.get() or "0.6")
        except Exception:
            lsh_thr = 0.6
        try:
            shingle = int(self.entry_shingle.get() or "3")
        except Exception:
            shingle = 3
        try:
            per_doc = int(self.entry_cand_doc.get() or "2000")
        except Exception:
            per_doc = 2000
        try:
            total = int(self.entry_cand_total.get()) if (self.entry_cand_total.get() or "").strip() else None
        except Exception:
            total = None

        cfg_cand = CandidateConfig(
            use_lsh=True,
            shingle_size=shingle,
            num_perm=64,
            lsh_threshold=lsh_thr,
            max_candidates_per_doc=per_doc,
            max_total_candidates=total,
        )

        # Bootstrap and self-learning
        cfg_boot = BootstrapConfig(max_pos_pairs=50_000, max_neg_pairs=50_000)
        try:
            epochs = int(self.entry_sl_epochs.get() or "2")
        except Exception:
            epochs = 2
        cfg_sl = SelfLearningConfig(enabled=bool(self.var_sl_enable.get()), epochs=epochs)

        return PipelineConfig(
            simhash=cfg_sim,
            minhash=cfg_min,
            embedding=cfg_emb,
            arbiter=cfg_arb,
            candidates=cfg_cand,
            bootstrap=cfg_boot,
            self_learning=cfg_sl,
            persist=True,
        )

    def _apply_preset(self):
        preset = (self.profile_var.get() or "Balanced").lower()

        # Update learners
        sim_cfg = self.card_sim.get_config()
        min_cfg = self.card_min.get_config()
        emb_cfg = self.card_emb.get_config()

        # Update arbiter
        arb_cfg = self.arbiter_panel.get_config()

        if preset.startswith("balanced"):
            target = 0.98
            gray = 0.05
            epochs = "2"
        elif preset.startswith("high"):
            target = 0.995
            gray = 0.04
            epochs = "1"
        elif preset.startswith("recall"):
            target = 0.95
            gray = 0.06
            epochs = "3"
        else:
            return

        sim_cfg.target_precision = target
        min_cfg.target_precision = target
        emb_cfg.target_precision = target
        arb_cfg.gray_zone_margin = gray

        self.card_sim.set_config(sim_cfg)
        self.card_min.set_config(min_cfg)
        self.card_emb.set_config(emb_cfg)
        self.arbiter_panel.set_config(arb_cfg)

        self.entry_sl_epochs.delete(0, tk.END)
        self.entry_sl_epochs.insert(0, epochs)

    # Documents tab actions
    def _on_add_files(self):
        if _ingestion_mod is None:
            messagebox.showerror("Missing ingestion", "ingestion.py not found.")
            return
        paths = filedialog.askopenfilenames(
            title="Select files",
            filetypes=[("Documents", "*.pdf *.docx *.txt"), ("All files", "*.*")]
        )
        if not paths:
            return
        self._ingest_paths(list(paths))

    def _on_add_folder(self):
        if _ingestion_mod is None:
            messagebox.showerror("Missing ingestion", "ingestion.py not found.")
            return
        folder = filedialog.askdirectory(title="Select folder")
        if not folder:
            return
        exts = {".pdf", ".docx", ".txt"}
        paths: List[str] = []
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
        if not paths:
            messagebox.showinfo("No supported files", "No .pdf / .docx / .txt found.")
            return
        self._ingest_paths(paths)

    def _on_delete_selected_docs(self):
        selected = [self.docs_tree.item(iid, "values")[0] for iid in self.docs_tree.selection()]
        if not selected:
            return
        if not messagebox.askyesno("Delete documents", f"Delete {len(selected)} documents from DB?"):
            return
        try:
            sqlite_store.delete_documents(selected)
            self._refresh_docs_tab()
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))

    """def _ingest_paths(self, paths: Iterable[str]):
    """def _ingest_paths(self, paths: Iterable[str]):
        ps = list(paths)
        self.doc_status = getattr(self, "doc_status", None)
        if self.doc_status:
            self.doc_status.configure(text=f"Ingesting {len(ps)} files…")
            self.update_idletasks()

        def worker(plist: List[str]):
            ok = 0
            fail = 0
            for p in plist:
                try:
                    self._ingest_file(p)
                    ok += 1
                    if self.doc_status:
                        self.after(0, lambda o=ok, f=fail: self.doc_status.configure(text=f"Ingesting… ok={o} fail={f}"))
                except Exception:
                    fail += 1
                    if self.doc_status:
                        self.after(0, lambda o=ok, f=fail: self.doc_status.configure(text=f"Ingesting… ok={o} fail={f}"))
            if self.doc_status:
                self.after(0, lambda: [self._refresh_docs_tab(),
                                       self.doc_status.configure(text=f"Done. ok={ok}, fail={fail}")])

        threading.Thread(target=worker, args=(ps,), daemon=True).start()

    def _ingest_file(self, path: str):
        data = _ingestion_mod.extract_document(path)
        raw_text = data.get("raw_text") or ""
        meta = data.get("metadata") or {}

        # make one document per file-version
        import hashlib, os
        abs_path = os.path.abspath(path)
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except Exception:
            mtime_ns = 0

        doc_id = hashlib.sha1(f"{abs_path}|{mtime_ns}".encode("utf-8", errors="ignore")).hexdigest()

        norm = normalize_text(raw_text)

        # write to DB
        sqlite_store.upsert_document(
            doc_id=doc_id,
            raw_text=raw_text,
            normalized_text=norm,
            meta={
                "language": meta.get("language"),
                "filesize": meta.get("filesize") or 0,
            },
        )

        # file mapping
        sqlite_store.add_file_mapping(doc_id, path, mtime_ns)
"""
    def _ingest_paths(self, paths: Iterable[str]):
        ps = list(paths)
        if self.doc_status:
            self.doc_status.configure(text=f"Ingesting {len(ps)} files…")
            self.update_idletasks()

        def worker(plist: List[str]):
            docs_batch = []
            mappings_batch = []
            ok = 0
            fail = 0

            for p in plist:
                try:
                    doc_entry, mapping_entry = self._ingest_file(p, return_batch=True)
                    if doc_entry and mapping_entry:
                        docs_batch.append(doc_entry)
                        mappings_batch.append(mapping_entry)
                    ok += 1
                    if self.doc_status:
                        self.after(0, lambda o=ok, f=fail: self.doc_status.configure(text=f"Ingesting… ok={o} fail={f}"))
                except Exception:
                    fail += 1
                    if self.doc_status:
                        self.after(0, lambda o=ok, f=fail: self.doc_status.configure(text=f"Ingesting… ok={o} fail={f}"))

            # Batch insert at the end
            if docs_batch:
                sqlite_store.batch_upsert_documents(docs_batch)
            if mappings_batch:
                sqlite_store.batch_add_file_mappings(mappings_batch)

            if self.doc_status:
                self.after(0, lambda: [self._refresh_docs_tab(),
                                    self.doc_status.configure(text=f"Done. ok={ok}, fail={fail}")])

        threading.Thread(target=worker, args=(ps,), daemon=True).start()


    def _ingest_file(self, path: str, return_batch: bool = False):
        data = _ingestion_mod.extract_document(path)
        raw_text = data.get("raw_text") or ""
        meta = data.get("metadata") or {}

        abs_path = os.path.abspath(path)
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except Exception:
            mtime_ns = 0

        doc_id = hashlib.sha1(f"{abs_path}|{mtime_ns}".encode("utf-8", errors="ignore")).hexdigest()
        norm = normalize_text(raw_text)

        if return_batch:
            # Return tuples for batch insertion
            doc_entry = (doc_id, raw_text, norm, str({
                "language": meta.get("language"),
                "filesize": meta.get("filesize") or 0
            }))
            mapping_entry = (doc_id, abs_path, mtime_ns)
            return doc_entry, mapping_entry

        # legacy behavior: single insert
        sqlite_store.upsert_document(
            doc_id=doc_id,
            raw_text=raw_text,
            normalized_text=norm,
            meta={
                "language": meta.get("language"),
                "filesize": meta.get("filesize") or 0,
            },
        )
        sqlite_store.add_file_mapping(doc_id, abs_path, mtime_ns)


    # helpers: counters and timer

    def _reset_counters(self):
        self.var_cnt_pairs.set("Pairs: —")
        self.var_cnt_exact.set("Exact-duplicate: —")
        self.var_cnt_near.set("Near-duplicate: —")
        self.var_cnt_unc.set("Uncertain: —")

    def _update_counters_from_result(self, res: Dict[str, Any]):
        rs = res.get("run_summary") or {}
        pairs = rs.get('pairs_scored') or rs.get('total_pairs') or rs.get('pairs')
        if pairs is not None:
            self.var_cnt_pairs.set(f"Pairs: {pairs}")

        exact = rs.get('exact_duplicates')
        near  = rs.get('near_duplicates')
        if exact is None or near is None:
            exact = 0 if exact is None else exact
            near  = 0 if near  is None else near
            traces = res.get("traces") or []
            for tr in traces:
                try:
                    t = tr.as_dict() if hasattr(tr, "as_dict") else tr
                    fl = (t.get("final_label") or "").upper()
                    dk = (t.get("dup_kind") or "").upper()
                    if fl == "DUPLICATE":
                        if dk == "EXACT":
                            exact += 1
                        else:
                            near += 1
                except Exception:
                    pass

        self.var_cnt_exact.set(f"Exact-duplicate: {exact}")
        self.var_cnt_near.set(f"Near-duplicate: {near}")

        uc = rs.get('uncertain')
        if uc is not None:
            self.var_cnt_unc.set(f"Uncertain: {uc}")

    def _start_timer(self):
        self._run_start_ts = time.time()
        self._tick_timer()

    def _stop_timer(self):
        if self._elapsed_job:
            try:
                self.after_cancel(self._elapsed_job)
            except Exception:
                pass
        self._elapsed_job = None
        if self._run_start_ts is not None:
            elapsed = int(time.time() - self._run_start_ts)
            self.var_elapsed.set(self._fmt_hms(elapsed))
        self._run_start_ts = None

    def _tick_timer(self):
        if self._run_start_ts is None:
            return
        elapsed = int(time.time() - self._run_start_ts)
        self.var_elapsed.set(self._fmt_hms(elapsed))
        self._elapsed_job = self.after(1000, self._tick_timer)

    @staticmethod
    def _fmt_hms(seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # Resource monitor
    def _start_resource_monitor(self):
        try:
            psutil.cpu_percent(interval=None)
            self._proc.cpu_percent(interval=None)
        except Exception:
            pass
        self._poll_resources()

    def _stop_resource_monitor(self):
        if self._res_job:
            try:
                self.after_cancel(self._res_job)
            except Exception:
                pass
        self._res_job = None

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        try:
            n = float(n)
        except Exception:
            return f"{n}"
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024 or unit == "TiB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
            n /= 1024.0

    def _gpu_summary_text(self) -> Optional[str]:
        if not _NVML_OK:
            return None
        try:
            pynvml.nvmlInit()
            cnt = pynvml.nvmlDeviceGetCount()
            if cnt == 0:
                pynvml.nvmlShutdown()
                return None
            parts = []
            for i in range(cnt):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                used = mem.used / (1024**3)
                total = mem.total / (1024**3)
                try:
                    name = pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore")
                except Exception:
                    name = f"GPU{i}"
                parts.append(f"{name}: {util.gpu}% · VRAM {used:.1f}/{total:.1f} GiB")
            pynvml.nvmlShutdown()
            return " | ".join(parts) if parts else None
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return None

    def _poll_resources(self):
        try:
            with self._proc.oneshot():
                # psutil returns process CPU % on a single core scale
                raw_proc = self._proc.cpu_percent(interval=None)
                cores = psutil.cpu_count(logical=True) or 1
                norm = raw_proc / float(cores)  # normalize

                mem_info = self._proc.memory_full_info()
                rss = mem_info.rss
                total_ram = psutil.virtual_memory().total
                mem_percent = (rss / total_ram * 100.0) if total_ram else 0.0

                open_files_cnt = len(self._proc.open_files())
                threads_cnt = self._proc.num_threads()

            # Update labels
            self.var_res_cpu.set(f"App CPU: {norm:.1f}% of system  (raw {raw_proc:.1f}%)")
            self.var_res_mem.set(f"App RAM: {self._fmt_bytes(rss)}  ({mem_percent:.1f}%)")
            self.var_res_io.set(f"Open files: {open_files_cnt}   Threads: {threads_cnt}")

            # GPU only if available
            if _NVML_OK:
                gtxt = self._gpu_summary_text()
                if gtxt:
                    self.var_res_gpu.set(f"GPU: {gtxt}")
                else:
                    self.var_res_gpu.set("")

        except Exception:
            pass
        finally:
            self._res_job = self.after(1000, self._poll_resources)


def main():
    App().mainloop()


# import first 500 rows from a CSV file with text column field only
def import_rows_from_csv(csv_path="dataset/True.csv"):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if "text" not in df.columns:
        print("CSV must have a 'text' column.")
        print(f"Found columns: {df.columns.tolist()}")
        return

    # Sample 1500 random rows
    if len(df) < 1500:
        print(f"⚠️ CSV has only {len(df)} rows, sampling all available.")
        sampled_df = df.copy()
    else:
        sampled_df = df.sample(n=1500, random_state=random.randint(0, 9999))

    duplicates = sampled_df.sample(n=500, replace=False, random_state=random.randint(0, 9999))
    combined_df = pd.concat([sampled_df, duplicates], ignore_index=True).sample(frac=1).reset_index(drop=True)

    sqlite_store.init_db()

    fake_path = os.path.abspath(csv_path)
    for i, row in combined_df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        doc_id = hashlib.sha1(f"csv_row_{i}_{time.time_ns()}".encode()).hexdigest()
        norm = normalize_text(text)

        meta = {
            "language": "en",
            "filesize": len(text.encode("utf-8")),
        }

        sqlite_store.upsert_document(
            doc_id=doc_id,
            raw_text=text,
            normalized_text=norm,
            meta=meta,
        )

        mtime_ns = time.time_ns()
        sqlite_store.add_file_mapping(doc_id, fake_path, mtime_ns)

    print("Imported 2000 CSV rows into database.")

if __name__ == "__main__":
    # Comment out the below function as needed to insert more data from CSV
    # import_rows_from_csv() 
    main()
