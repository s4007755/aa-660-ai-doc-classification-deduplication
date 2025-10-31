# src/app.py
"""
GUI application for near-duplicate/exact-duplicate detection.

This module wires together:
- Persistence (SQLite-backed stores)
- Text preprocessing utilities
- Learners and ensemble arbitration
- A Tkinter-based desktop UI with tabs for running, inspecting traces,
  metrics, run history and document management.

The implementation focuses on being production-friendly:
- Background threads for long-running work
- Best-effort resource monitoring (CPU/RAM/IO/GPU via psutil/NVML)
- Pragmatic presets with a Custom escape hatch
- Careful separation between UI state and pipeline configuration
"""
from __future__ import annotations

import hashlib
import os
import random
import time
import threading
import traceback
from typing import Any, Dict, List, Optional, Iterable
import sqlite3
from pathlib import Path
import json

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


def _safe_nvml_snapshot(min_vram_gb: float = 6.0):
    """
    Probe NVIDIA GPUs and return a tuple (has_suitable_gpu, total_vram_gb).

    Best-effort: shields callers from NVML errors and guarantees a boolean/float
    result even when NVML is unavailable or fails.
    """
    if not _NVML_OK:
        return False, 0.0
    try:
        pynvml.nvmlInit()
        cnt = pynvml.nvmlDeviceGetCount()
        meets = False
        total = 0
        for i in range(cnt):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).total
            total += mem
            if mem / (1024**3) >= min_vram_gb:
                meets = True
        total_gb = total / (1024 ** 3)
        try: pynvml.nvmlShutdown()
        except Exception: pass
        return (cnt > 0 and meets), total_gb
    except Exception:
        try: pynvml.nvmlShutdown()
        except Exception: pass
        return False, 0.0


class VScrollFrame(ttk.Frame):
    """
    A simple vertically scrollable container implemented with a Canvas and Frame.

    Usage:
        host = VScrollFrame(parent)
        host.pack(fill=tk.BOTH, expand=True)
        inner = host.inner
    """
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
        self.canvas.bind("<Enter>", lambda e: (
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel),
            self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux),
            self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux),
        ))
        self.canvas.bind("<Leave>", lambda e: (
            self.canvas.unbind_all("<MouseWheel>"),
            self.canvas.unbind_all("<Button-4>"),
            self.canvas.unbind_all("<Button-5>"),
        ))

        # Resize inner width with frame
        self.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width - self.vbar.winfo_width()))

    def _on_mousewheel(self, event):
        """Cross-platform wheel handling (Windows/macOS delta convention)."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        """X11-style wheel handling (Button-4/5 for up/down)."""
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(+3, "units")


class App(tk.Tk):
    """
    Main Tkinter application for duplicate detection.

    Responsibilities:
    - Assemble the UI (tabs, panels, controls)
    - Manage in-memory state and persistence interactions
    - Build pipeline configuration from UI selections
    - Launch the pipeline on a worker thread and render results
    - Poll and display resource usage
    """
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

        # Resource monitor state
        self._res_job: Optional[str] = None
        self._proc = psutil.Process(os.getpid())

        self._build_ui()
        self._refresh_docs_from_db()
        self._refresh_docs_tab()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # UI layout
    def _build_ui(self):
        """Construct the top-level notebook and all tabs."""
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
        """Build the Run tab with presets, advanced options and status panel."""
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

        # Calibration toggle
        self.var_enable_calibration = tk.BooleanVar(value=False)
        self.chk_calibration = ttk.Checkbutton(
            bar,
            text="Calibration (Experimental)",
            variable=self.var_enable_calibration,
            command=self._sync_calib_visibility,
        )
        self.chk_calibration.pack(side=tk.LEFT, padx=(12, 0))

        self.btn_run = ttk.Button(controls, text="Run", command=self._on_run_clicked)
        self.btn_run.pack(side=tk.RIGHT)

        main = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Left: scrollable frame
        left_wrap = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left_wrap, weight=1)
        main.add(right, weight=1)

        left_scroller = VScrollFrame(left_wrap)
        left_scroller.pack(fill=tk.BOTH, expand=True)
        left = left_scroller.inner

        # Profile section
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

        # Advanced (Profile) toggle under the row
        prof_adv_row = ttk.Frame(left)
        prof_adv_row.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.var_show_adv = tk.BooleanVar(value=False)
        self.btn_toggle_adv = ttk.Checkbutton(
            prof_adv_row,
            text="Advanced settings (Profile) ▸",
            variable=self.var_show_adv,
            command=self._toggle_advanced_section,
            style="Toolbutton",
        )
        self.btn_toggle_adv.pack(side=tk.LEFT, padx=(6, 0))

        # Container for the profile’s advanced widgets
        self.adv_container = ttk.LabelFrame(left, text="Advanced")
        self.card_sim = LearnerCard(self.adv_container, learner_name="SimHash", kind="simhash", config=LearnerConfig(), collapsed=True)
        self.card_sim.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.card_min = LearnerCard(self.adv_container, learner_name="MinHash", kind="minhash", config=LearnerConfig(), collapsed=True)
        self.card_min.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.card_emb = LearnerCard(self.adv_container, learner_name="Embedding", kind="embedding", config=LearnerConfig(), collapsed=True)
        self.card_emb.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.arbiter_panel = ArbiterPanel(self.adv_container, config=ArbiterConfig(), text="Consensus & Escalation")
        self.arbiter_panel.pack(fill=tk.X, padx=4, pady=(0, 6))

        # Hide Target precision when calibration is off
        self._sync_calib_visibility()
        self.var_enable_calibration.trace_add("write", lambda *_: self._sync_calib_visibility())

        misc = ttk.LabelFrame(self.adv_container, text="Self-training (pseudo-labels) & Candidate Generation")
        misc.pack(fill=tk.X, padx=4, pady=(0, 12))

        # Self-learning
        self.var_sl_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(misc, text="Enable self-training", variable=self.var_sl_enable).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(misc, text="Epochs").grid(row=0, column=1, sticky="e", padx=(16, 4))
        self.entry_sl_epochs = ttk.Entry(misc, width=6)
        self.entry_sl_epochs.insert(0, "2")
        self.entry_sl_epochs.grid(row=0, column=2, sticky="w", padx=(0, 8), pady=4)

        # Candidate config
        ttk.Label(misc, text="LSH threshold").grid(row=1, column=0, sticky="e", padx=(8, 4))
        self.entry_lsh_thr = ttk.Entry(misc, width=6); self.entry_lsh_thr.insert(0, "0.60")
        self.entry_lsh_thr.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(misc, text="Shingle size").grid(row=1, column=2, sticky="e", padx=(8, 4))
        self.entry_shingle = ttk.Entry(misc, width=6); self.entry_shingle.insert(0, "3")
        self.entry_shingle.grid(row=1, column=3, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(misc, text="Max cand/doc").grid(row=2, column=0, sticky="e", padx=(8, 4))
        self.entry_cand_doc = ttk.Entry(misc, width=8); self.entry_cand_doc.insert(0, "2000")
        self.entry_cand_doc.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(misc, text="Max total").grid(row=2, column=2, sticky="e", padx=(8, 4))
        self.entry_cand_total = ttk.Entry(misc, width=10); self.entry_cand_total.insert(0, "")
        self.entry_cand_total.grid(row=2, column=3, sticky="w", padx=(0, 8), pady=4)

        for c in range(4):
            misc.grid_columnconfigure(c, weight=1)

        # Performance section
        perf = ttk.LabelFrame(left, text="Performance")
        perf.pack(fill=tk.X, padx=4, pady=(0, 6))

        top_perf_row = ttk.Frame(perf)
        top_perf_row.pack(fill=tk.X, padx=6, pady=(8, 0))

        ttk.Label(top_perf_row, text="Preset:").pack(side=tk.LEFT, padx=(2, 6))
        self.perf_var = tk.StringVar(value="Auto (detect)")
        self.cmb_perf = ttk.Combobox(
            top_perf_row, textvariable=self.perf_var, width=22,
            values=["Auto (detect)", "High-End", "Medium", "Light", "High-Throughput", "High-Recall", "Custom"]
        )
        self.cmb_perf.pack(side=tk.LEFT)
        ttk.Button(top_perf_row, text="Apply", command=self._apply_perf_preset).pack(side=tk.LEFT, padx=(8, 6))

        # Bottom row
        bottom_perf_row = ttk.Frame(perf)
        bottom_perf_row.pack(fill=tk.X, padx=6, pady=(6, 8))

        self.var_show_perf_adv = tk.BooleanVar(value=False)
        self.btn_toggle_perf_adv = ttk.Checkbutton(
            bottom_perf_row, text="Advanced settings (Performance) ▸",
            variable=self.var_show_perf_adv, command=self._toggle_perf_advanced_section,
            style="Toolbutton"
        )
        self.btn_toggle_perf_adv.pack(side=tk.LEFT)

        # Advanced Performance container
        self.perf_adv_container = ttk.LabelFrame(left, text="Custom Performance Profile")

        # Row 1: workers and embedding batch size
        row1 = ttk.Frame(self.perf_adv_container)
        row1.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(row1, text="Max CPU workers").grid(row=0, column=0, sticky="e")
        self.entry_perf_workers = ttk.Entry(row1, width=8)
        self.entry_perf_workers.insert(0, str(psutil.cpu_count(logical=True) or 4))
        self.entry_perf_workers.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(row1, text="Embedding batch size").grid(row=0, column=2, sticky="e")
        self.entry_perf_emb_batch = ttk.Entry(row1, width=8); self.entry_perf_emb_batch.insert(0, "64")
        self.entry_perf_emb_batch.grid(row=0, column=3, sticky="w", padx=(6, 0))

        # Row 2: MinHash knobs
        row2 = ttk.Frame(self.perf_adv_container)
        row2.pack(fill=tk.X, padx=8, pady=(8, 0))
        self.var_perf_use_minhash = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Use MinHash approximation", variable=self.var_perf_use_minhash)\
            .grid(row=0, column=0, sticky="w")
        ttk.Label(row2, text="MinHash permutations").grid(row=0, column=1, sticky="e", padx=(18, 0))
        self.entry_perf_minhash_perm = ttk.Entry(row2, width=8); self.entry_perf_minhash_perm.insert(0, "128")
        self.entry_perf_minhash_perm.grid(row=0, column=2, sticky="w", padx=(6, 0))

        # Row 3: candidate generator sizes
        row3 = ttk.Frame(self.perf_adv_container)
        row3.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(row3, text="Max candidates / doc").grid(row=0, column=0, sticky="e")
        self.entry_perf_cand_per_doc = ttk.Entry(row3, width=10); self.entry_perf_cand_per_doc.insert(0, "2000")
        self.entry_perf_cand_per_doc.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(row3, text="Max total candidates").grid(row=0, column=2, sticky="e")
        self.entry_perf_cand_total = ttk.Entry(row3, width=12); self.entry_perf_cand_total.insert(0, "")
        self.entry_perf_cand_total.grid(row=0, column=3, sticky="w", padx=(6, 0))

        # Row 4: Embedding model
        row4 = ttk.Frame(self.perf_adv_container)
        row4.pack(fill=tk.X, padx=8, pady=(8, 10))
        ttk.Label(row4, text="Embedding model").grid(row=0, column=0, sticky="e")
        self.cmb_perf_emb_model = ttk.Combobox(
            row4, width=28, values=[
                "fallback",
                "all-MiniLM-L6-v2",
                "multi-qa-MiniLM-L6-cos-v1",
                "paraphrase-multilingual-MiniLM-L12-v2",
            ]
        )
        self.cmb_perf_emb_model.set("fallback")
        self.cmb_perf_emb_model.grid(row=0, column=1, sticky="w", padx=(6, 18))

        for c in range(4):
            row1.grid_columnconfigure(c, weight=1)
            row2.grid_columnconfigure(c, weight=1)
            row3.grid_columnconfigure(c, weight=1)
            row4.grid_columnconfigure(c, weight=1)

        # ---- SimHash options
        sim_box = ttk.LabelFrame(self.perf_adv_container, text="SimHash options")
        sim_box.pack(fill=tk.X, padx=8, pady=(0, 10))

        rowS1 = ttk.Frame(sim_box); rowS1.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(rowS1, text="Hash bits").grid(row=0, column=0, sticky="e")
        self.entry_sim_bits = ttk.Entry(rowS1, width=8); self.entry_sim_bits.insert(0, "128")
        self.entry_sim_bits.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(rowS1, text="Mode").grid(row=0, column=2, sticky="e")
        self.cmb_sim_mode = ttk.Combobox(rowS1, width=16, values=["unigram", "wshingle", "cngram"])
        self.cmb_sim_mode.set("unigram")
        self.cmb_sim_mode.grid(row=0, column=3, sticky="w", padx=(6, 0))

        rowS2 = ttk.Frame(sim_box); rowS2.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(rowS2, text="Word shingle size").grid(row=0, column=0, sticky="e")
        self.entry_sim_wshingle = ttk.Entry(rowS2, width=8); self.entry_sim_wshingle.insert(0, "3")
        self.entry_sim_wshingle.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(rowS2, text="Char n-gram").grid(row=0, column=2, sticky="e")
        self.entry_sim_cngram = ttk.Entry(rowS2, width=8); self.entry_sim_cngram.insert(0, "5")
        self.entry_sim_cngram.grid(row=0, column=3, sticky="w", padx=(6, 0))

        rowS3 = ttk.Frame(sim_box); rowS3.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(rowS3, text="Positional bucket").grid(row=0, column=0, sticky="e")
        self.entry_sim_posbucket = ttk.Entry(rowS3, width=8); self.entry_sim_posbucket.insert(0, "0")
        self.entry_sim_posbucket.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(rowS3, text="Min token length").grid(row=0, column=2, sticky="e")
        self.entry_sim_minlen = ttk.Entry(rowS3, width=8); self.entry_sim_minlen.insert(0, "2")
        self.entry_sim_minlen.grid(row=0, column=3, sticky="w", padx=(6, 0))

        rowS4 = ttk.Frame(sim_box); rowS4.pack(fill=tk.X, pady=(8, 10))
        self.var_sim_norm_strict = tk.BooleanVar(value=False)
        ttk.Checkbutton(rowS4, text="Normalize strictly", variable=self.var_sim_norm_strict).grid(row=0, column=0, sticky="w")

        self.var_sim_strip_ids = tk.BooleanVar(value=False)
        ttk.Checkbutton(rowS4, text="Strip dates/IDs", variable=self.var_sim_strip_ids).grid(row=0, column=1, sticky="w", padx=(18, 0))

        ttk.Label(rowS4, text="Max token weight").grid(row=0, column=2, sticky="e")
        self.entry_sim_maxw = ttk.Entry(rowS4, width=8); self.entry_sim_maxw.insert(0, "255")
        self.entry_sim_maxw.grid(row=0, column=3, sticky="w", padx=(6, 0))

        for c in range(4):
            rowS1.grid_columnconfigure(c, weight=1)
            rowS2.grid_columnconfigure(c, weight=1)
            rowS3.grid_columnconfigure(c, weight=1)
            rowS4.grid_columnconfigure(c, weight=1)

        # Right: run status panel
        status = ttk.LabelFrame(right, text="Run status")
        status.pack(fill=tk.BOTH, expand=True, padx=4, pady=(6, 6))

        self.var_status = tk.StringVar(value="Idle")
        ttk.Label(status, textvariable=self.var_status).pack(anchor="w", padx=8, pady=(6, 0))

        # Elapsed timer
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

        # Resource usage
        res_toggle_row = ttk.Frame(status)
        res_toggle_row.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.res_toggle_row = res_toggle_row 
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

        # Show by default
        self.res_container.pack(fill=tk.X, padx=8, pady=(0, 8))
        if self.var_show_res.get():
            self._start_resource_monitor()

    def _toggle_advanced_section(self):
        """Expand/collapse the Profile advanced section."""
        show = bool(self.var_show_adv.get())
        # If opening Profile Advanced, close Performance Advanced
        if show and hasattr(self, 'var_show_perf_adv') and self.var_show_perf_adv.get():
            self.var_show_perf_adv.set(False)
            try:
                self.perf_adv_container.pack_forget()
            except Exception:
                pass
            self.btn_toggle_perf_adv.configure(text="Advanced settings (Performance) ▸")

        self.btn_toggle_adv.configure(
            text="Advanced settings (Profile) ▾" if show else "Advanced settings (Profile) ▸"
        )
        try:
            if show:
                self.adv_container.pack(fill=tk.X, padx=4, pady=(0, 8))
            else:
                self.adv_container.pack_forget()
        except Exception:
            pass

    def _toggle_perf_advanced_section(self):
        """Expand/collapse the Performance advanced section;"""
        show = bool(self.var_show_perf_adv.get())
        # If opening Performance Advanced, close Profile Advanced
        if show and hasattr(self, 'var_show_adv') and self.var_show_adv.get():
            self.var_show_adv.set(False)
            try:
                self.adv_container.pack_forget()
            except Exception:
                pass
            self.btn_toggle_adv.configure(text="Advanced settings (Profile) ▸")

        self.btn_toggle_perf_adv.configure(
            text="Advanced settings (Performance) ▾" if show else "Advanced settings (Performance) ▸"
        )
        try:
            if show:
                self.perf_adv_container.pack(fill=tk.X, padx=4, pady=(0, 8))
            else:
                self.perf_adv_container.pack_forget()
        except Exception:
            pass

    def _apply_torch_threading(self, max_threads: int):
        """
        Best-effort: set thread env and torch threading so heavy math
        will actually use the requested cores.
        """
        try:
            max_threads = int(max_threads)
        except Exception:
            return
        # Env knobs picked up by numpy/numexpr/openblas/mkl
        try:
            os.environ["OMP_NUM_THREADS"] = str(max_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
            os.environ["MKL_NUM_THREADS"] = str(max_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
            os.environ["PYTORCH_NUM_THREADS"] = str(max_threads)
            # Optional: tokenizer parallelism
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        except Exception:
            pass

        # Torch runtime knobs
        try:
            import torch
            # Use all cores for CPU ops
            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(max_threads)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max_threads)
        except Exception:
            pass

    def _apply_perf_preset(self):
        """
        Compute and apply performance knobs based on preset and corpus size.

        The method updates both the visible UI controls and an internal
        `_perf_overrides` dict that is consumed when building PipelineConfig.
        """
        preset = (self.perf_var.get() or "Auto (detect)").lower()
        n_docs = max(1, len(self.docs))
        Pphys = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 4
        P = Pphys
        RAM_GB = (psutil.virtual_memory().total or 0) / (1024**3)

        # GPU snapshot
        gpu_ok, VRAM_GB = _safe_nvml_snapshot(min_vram_gb=6.0)

        bucket = "small" if n_docs < 5_000 else ("medium" if n_docs < 50_000 else "large")

        # Defaults
        workers   = min(P, 8)
        emb_batch = 64
        mh_perm   = 128
        cand_doc  = 2000
        cand_total= None
        model     = "fallback"

        # SimHash defaults
        sim_bits       = 192 if bucket == "large" else 128
        sim_mode       = "unigram"
        sim_wshingle   = 3
        sim_cngram     = 5
        sim_posbucket  = 0
        sim_minlen     = 2
        sim_norm_strict= False
        sim_strip_ids  = False
        sim_maxw       = 255

        if preset.startswith("auto"):
            # Former _detect_best_settings logic:
            workers   = max(2, min(12, min(Pphys, 2 + (n_docs // 10_000))))
            if gpu_ok:
                emb_batch = 128 + (64 if VRAM_GB >= 12 else 0)
                model     = "all-MiniLM-L6-v2"
            else:
                emb_batch = 32 + (32 if RAM_GB >= 16 else 0) + (32 if RAM_GB >= 32 else 0)
                emb_batch = min(128, emb_batch)
                model     = "fallback"

            mh_perm   = 64 if bucket == "small" else (128 if bucket == "medium" else 192)
            cand_doc  = 1000 if bucket == "small" else (3000 if bucket == "medium" else 5000)
            cand_total= n_docs * cand_doc

            sim_bits     = 192 if bucket == "large" else 128
            sim_mode     = "unigram" if bucket == "small" else "wshingle"
            sim_wshingle = 3 if bucket != "large" else 4
            sim_cngram   = 5
            sim_posbucket= 0
            sim_minlen   = 2
            sim_norm_strict = False
            sim_strip_ids   = False
            sim_maxw        = 255

            self._set_threading_env(workers, torch_intra=max(1, Pphys // 2 if gpu_ok else workers), torch_inter=1)

        elif preset.startswith("light"):
            workers   = min(P, 4)
            emb_batch = 48 if RAM_GB >= 16 else 32
            mh_perm   = 64
            cand_doc  = 1000
            cand_total= min(100_000, 20 * n_docs)
            model     = "fallback"
            sim_bits  = 128
            sim_mode  = "unigram"
            self._set_threading_env(workers, torch_intra=1, torch_inter=1)

        elif preset.startswith("medium"):
            workers   = min(P, 8)
            if gpu_ok:
                model = "all-MiniLM-L6-v2"; emb_batch = 128
            else:
                model = "fallback"; emb_batch = 64 if RAM_GB < 16 else 96
            mh_perm   = 128
            cand_doc  = 2000 if bucket == "small" else (3000 if bucket == "medium" else 4000)
            cand_total= n_docs * 2000
            sim_bits  = 128 if bucket != "large" else 192
            sim_mode  = "unigram"
            self._set_threading_env(workers, torch_intra=max(1, P // 2), torch_inter=1)

        elif preset.startswith("high-throughput"):
            workers   = min(P, 12)
            model     = "fallback"
            emb_batch = 64 if RAM_GB < 16 else (96 if RAM_GB < 32 else 128)
            mh_perm   = 128 if bucket != "large" else 192
            cand_doc  = 3000 if bucket == "small" else (4000 if bucket == "medium" else 5000)
            cand_total= n_docs * 3000
            sim_bits  = 128 if bucket != "large" else 192
            sim_mode  = "wshingle" if bucket != "small" else "unigram"
            sim_wshingle = 3 if bucket != "large" else 4
            self._set_threading_env(workers, torch_intra=P, torch_inter=1)

        elif preset.startswith("high-recall"):
            workers   = min(P, 6)
            model     = "all-MiniLM-L6-v2" if gpu_ok else "fallback"
            emb_batch = 128 if (gpu_ok and VRAM_GB < 12) else (192 if gpu_ok else 64)
            mh_perm   = 192 if bucket == "large" else 128
            cand_doc  = 4000 if bucket != "large" else 6000
            cand_total= n_docs * 4000
            sim_bits  = 192 if bucket == "large" else 128
            sim_mode  = "wshingle" if bucket != "small" else "unigram"
            sim_wshingle = 3 if bucket != "large" else 4
            self._set_threading_env(workers, torch_intra=max(1, P // 2), torch_inter=1)

        else:
            # custom: read current UI
            try: workers   = int(self.entry_perf_workers.get() or workers)
            except: pass
            try: emb_batch = int(self.entry_perf_emb_batch.get() or emb_batch)
            except: pass
            try: mh_perm   = int(self.entry_perf_minhash_perm.get() or mh_perm)
            except: pass
            try: cand_doc  = int(self.entry_perf_cand_per_doc.get() or cand_doc)
            except: pass
            try:
                cand_total = int(self.entry_perf_cand_total.get()) if (self.entry_perf_cand_total.get() or "").strip() else None
            except:
                pass
            try: sim_bits  = int(self.entry_sim_bits.get() or sim_bits)
            except: pass
            sim_mode = self.cmb_sim_mode.get() or sim_mode
            try: sim_wshingle = int(self.entry_sim_wshingle.get() or sim_wshingle)
            except: pass
            try: sim_cngram = int(self.entry_sim_cngram.get() or sim_cngram)
            except: pass
            try: sim_posbucket = int(self.entry_sim_posbucket.get() or sim_posbucket)
            except: pass
            try: sim_minlen = int(self.entry_sim_minlen.get() or sim_minlen)
            except: pass
            sim_norm_strict = bool(self.var_sim_norm_strict.get())
            sim_strip_ids   = bool(self.var_sim_strip_ids.get())
            try: sim_maxw = int(self.entry_sim_maxw.get() or sim_maxw)
            except: pass
            model = self.cmb_perf_emb_model.get() or model
            self._set_threading_env(workers)

        # Push into UI
        self.entry_perf_workers.delete(0, tk.END); self.entry_perf_workers.insert(0, str(workers))
        self.entry_perf_emb_batch.delete(0, tk.END); self.entry_perf_emb_batch.insert(0, str(emb_batch))
        self.entry_perf_minhash_perm.delete(0, tk.END); self.entry_perf_minhash_perm.insert(0, str(mh_perm))
        self.entry_perf_cand_per_doc.delete(0, tk.END); self.entry_perf_cand_per_doc.insert(0, str(cand_doc))
        self.entry_perf_cand_total.delete(0, tk.END); self.entry_perf_cand_total.insert(0, "" if cand_total is None else str(cand_total))
        self.cmb_perf_emb_model.set(model)

        # SimHash UI
        self.entry_sim_bits.delete(0, tk.END); self.entry_sim_bits.insert(0, str(sim_bits))
        self.cmb_sim_mode.set(sim_mode)
        self.entry_sim_wshingle.delete(0, tk.END); self.entry_sim_wshingle.insert(0, str(sim_wshingle))
        self.entry_sim_cngram.delete(0, tk.END); self.entry_sim_cngram.insert(0, str(sim_cngram))
        self.entry_sim_posbucket.delete(0, tk.END); self.entry_sim_posbucket.insert(0, str(sim_posbucket))
        self.entry_sim_minlen.delete(0, tk.END); self.entry_sim_minlen.insert(0, str(sim_minlen))
        self.var_sim_norm_strict.set(sim_norm_strict)
        self.var_sim_strip_ids.set(sim_strip_ids)
        self.entry_sim_maxw.delete(0, tk.END); self.entry_sim_maxw.insert(0, str(sim_maxw))

        # Save overrides
        self._perf_overrides = {
            "max_workers": workers,
            "emb_batch": emb_batch,
            "use_minhash": bool(self.var_perf_use_minhash.get()),
            "minhash_num_perm": mh_perm,
            "cand_per_doc": cand_doc,
            "cand_total": cand_total,
            "model_name": model,
            "simhash_bits": sim_bits,
            "simhash_mode": sim_mode,
            "simhash_wshingle": sim_wshingle,
            "simhash_cngram": sim_cngram,
            "simhash_posbucket": sim_posbucket,
            "simhash_minlen": sim_minlen,
            "simhash_norm_strict": sim_norm_strict,
            "simhash_strip_ids": sim_strip_ids,
            "simhash_maxw": sim_maxw,
        }

        if preset.startswith("high"):
            self.var_show_perf_adv.set(True)
            self._toggle_perf_advanced_section()

        # Status line for Auto
        if preset.startswith("auto"):
            self.var_status.set(
                f"Auto chose: workers={workers}, batch={emb_batch}, perms={mh_perm}, model={model}; SimHash bits={sim_bits}, mode={sim_mode}"
            )

    def _toggle_resource_section(self):
        """Expand/collapse the resource usage panel and start/stop polling."""
        show = bool(self.var_show_res.get())
        self.btn_toggle_res.configure(text="Resource usage ▾" if show else "Resource usage ▸")
        try:
            if show:
                self.res_container.pack(fill=tk.X, padx=8, pady=(0, 8), after=self.res_toggle_row)
                self._start_resource_monitor()
            else:
                self.res_container.pack_forget()
                self._stop_resource_monitor()
        except Exception:
            pass

    def _on_close(self):
        """Gracefully stop monitors and destroy the window."""
        try:
            self._stop_resource_monitor()
        except Exception:
            pass
        self.destroy()   

    # Documents tab
    def _build_docs_tab(self, parent: ttk.Frame):
        """Create the Documents management tab with add/refresh/delete actions."""
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
        """Refresh in-memory normalized document views from the database."""
        pairs = sqlite_store.get_docs_text(include_dirty=False)
        self.docs = [
            build_document_view(doc_id=did, text=(txt or ""), language=None, meta={})
            for (did, txt) in pairs
            if (txt or "").strip()
        ]

    # CSV ingestion
    def _on_add_csv(self):
        """
        Import a CSV into in-memory docs only.

        This path does not persist to DB; it populates the UI table from a CSV
        sample so users can quickly try the pipeline without DB ingestion.
        """
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
        # in _on_add_csv()
        self.lbl_doc_count.configure(text=f"{len(self.docs)} rows from CSV")
        self._csv_loaded = True
        self._refresh_docs_tab()

    # Fill docs tab table
    def _refresh_docs_tab(self):
        """Populate the documents table from either a CSV load or the DB."""
        # clear existing table rows
        for iid in self.docs_tree.get_children():
            self.docs_tree.delete(iid)

        if getattr(self, "_csv_loaded", False):
            for d in self.docs:
                self.docs_tree.insert(
                    "", tk.END,
                    values=(d.doc_id, f"csv_row_{d.doc_id}", "—", len(d.text) // 1024, "CSV import")
                )
            self.lbl_doc_count.configure(text=f"{len(self.docs)} rows from CSV")
            self.lbl_files.configure(text=f"Files: {len(self.docs)}")
            return

        # Fall back to DB
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

        unique_docs = len({r.get("doc_id") for r in rows})
        total_files = len(rows)
        self.lbl_doc_count.configure(text=f"{total_files} files across {unique_docs} docs")
        self.lbl_files.configure(text=f"Files: {total_files}")

        # Keep the in-memory normalized views up to date
        self._refresh_docs_from_db()

    def _doc_labels_from_db(self) -> Dict[str, str]:
        """Build a map of document IDs to a representative filename."""
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
        """
        Extract per-learner thresholds from a pipeline result snapshot.
        """
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
    
    def _sync_calib_visibility(self):
        """Keep learner cards in sync with the Calibration checkbox."""
        on = bool(self.var_enable_calibration.get())
        self.card_sim.set_calibration_enabled(on)
        self.card_min.set_calibration_enabled(on)
        self.card_emb.set_calibration_enabled(on)

    def _bump_process_priority(self):
        """
        Try to raise process priority to get more CPU share during a run.
        Works on Windows/macOS/Linux best-effort
        """
        try:
            p = self._proc
            if os.name == "nt":
                # Windows priority class
                import psutil as _ps
                p.nice(_ps.HIGH_PRIORITY_CLASS)
            else:
                # nix: lower nice value -> higher priority, requires perms
                p.nice(-5)
        except Exception:
            pass

    def _on_run_clicked(self):
        """
        Validate the current state, build a PipelineConfig, and launch the run.

        Runs on a background thread, UI reflects progress via callbacks and
        switches to a completed/error state at the end.
        """
        if self.running:
            return
        if len(self.docs) < 2:
            messagebox.showinfo("No data", "Need at least 2 normalized docs in DB. Ingest files first (Documents tab).")
            return
        
        if (self.perf_var.get() or "").lower().startswith("auto"):
            self._apply_perf_preset()

        # Build pipeline config from UI
        try:
            pconfig = self._make_pipeline_config()
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return

        # Raise priority so the OS actually schedules us aggressively
        self._bump_process_priority()

        # Background thread to avoid freezing UI
        self.running = True
        self.btn_run.configure(state=tk.DISABLED)
        self.var_status.set("Starting…")
        self._reset_counters()
        self._start_timer()
        # start resource monitor
        self._start_resource_monitor()

        def worker():
            """Worker thread entry-point for the pipeline run."""
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

    def _set_threading_env(self, workers: int, torch_intra: Optional[int] = None, torch_inter: int = 1):
        """
        Apply sane default thread env vars for BLAS stacks and PyTorch interop.

        Args:
            workers: Requested worker count.
            torch_intra: Intra-op threads for torch (defaults to ~workers/2).
            torch_inter: Inter-op threads for torch (usually 1 is fine).
        """
        try:
            os.environ["OMP_NUM_THREADS"] = str(max(1, workers))
            os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, workers))
            os.environ["MKL_NUM_THREADS"] = str(max(1, workers))
            os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, workers))
        except Exception:
            pass

        try:
            import torch
            if torch_intra is None:
                torch_intra = max(1, workers // 2) if workers > 1 else 1
            torch.set_num_threads(max(1, torch_intra))
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max(1, torch_inter))
        except Exception:
            pass

    # Finish run on main thread
    def _finish_run(self, result: Optional[Dict[str, Any]], err: Optional[BaseException]):
        """
        Main-thread finalization: stop timers/monitors, render results or errors,
        update metrics panels and learner card KPIs, and push to history.
        """
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
            "basics": raw_snap.get("basics", {}),
            "use_calibrated": raw_snap.get("use_calibrated", False),
        }
        self.metrics_panel.update_metrics(run_summary=run_summary, snapshot=snap_norm, doc_labels=self._doc_labels_from_db())

        # push numbers into learner cards header KPIs
        thresholds = self._thresholds_from_result(self.pipeline_result)

        def _num(x):
            """Convert to float if possible"""
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
        """Render an at-a-glance textual summary into the summary Text widget."""
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

        clusters = res.get("clusters") or (res.get("metrics_snapshot") or {}).get("clusters") or []
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
        """
        Translate current UI selections and overrides into a PipelineConfig.

        Honors calibration toggle by forcing thresholds via the
        learners' `extras` dicts.
        """
        cfg_sim = self.card_sim.get_config()
        cfg_min = self.card_min.get_config()
        cfg_emb = self.card_emb.get_config()
        cfg_arb = self.arbiter_panel.get_config()

        # Candidate basics
        try: lsh_thr = float(self.entry_lsh_thr.get() or "0.6")
        except: lsh_thr = 0.6
        try: shingle = int(self.entry_shingle.get() or "3")
        except: shingle = 3
        try: per_doc_default = int(self.entry_cand_doc.get() or "2000")
        except: per_doc_default = 2000
        try:
            total_default = int(self.entry_cand_total.get()) if (self.entry_cand_total.get() or "").strip() else None
        except:
            total_default = None

        if not getattr(self, "_perf_overrides", None):
            self._apply_perf_preset()
        ov = dict(self._perf_overrides or {})

        cand_per_doc = int(ov.get("cand_per_doc", per_doc_default))
        cand_total = ov.get("cand_total", total_default)
        cfg_cand = CandidateConfig(
            use_lsh=True,
            shingle_size=shingle,
            num_perm=int(ov.get("minhash_num_perm", 128)),
            lsh_threshold=lsh_thr,
            max_candidates_per_doc=cand_per_doc,
            max_total_candidates=cand_total,
        )

        cfg_boot = BootstrapConfig(max_pos_pairs=50_000, max_neg_pairs=50_000)
        try: epochs = int(self.entry_sl_epochs.get() or "2")
        except: epochs = 2
        cfg_sl = SelfLearningConfig(enabled=bool(self.var_sl_enable.get()), epochs=epochs)
        calibration_enabled = bool(self.var_enable_calibration.get())

        # Embedding
        emb_ex = dict(cfg_emb.extras or {})
        if "max_workers" in ov: emb_ex["max_workers"] = int(ov["max_workers"])
        if "emb_batch" in ov:  emb_ex["batch_size"]  = int(ov["emb_batch"])
        if "model_name" in ov and (ov["model_name"] or "").strip():
            emb_ex["model_name"] = str(ov["model_name"]).strip()
        if not calibration_enabled:
            emb_ex["force_threshold"] = True
        cfg_emb.extras = emb_ex

        # MinHash
        min_ex = dict(cfg_min.extras or {})
        if "use_minhash" in ov:           min_ex["use_minhash"] = bool(ov["use_minhash"])
        if "minhash_num_perm" in ov:      min_ex["num_perm"]    = int(ov["minhash_num_perm"])
        if not calibration_enabled:
            min_ex["force_threshold"] = True
        cfg_min.extras = min_ex

        # SimHash
        sim_ex = dict(cfg_sim.extras or {})
        if "simhash_bits" in ov:          sim_ex["hash_bits"]        = int(ov["simhash_bits"])
        if "simhash_mode" in ov:          sim_ex["simhash_mode"]     = str(ov["simhash_mode"])
        if "simhash_wshingle" in ov:      sim_ex["shingle_size"]     = int(ov["simhash_wshingle"])
        if "simhash_cngram" in ov:        sim_ex["char_ngram"]       = int(ov["simhash_cngram"])
        if "simhash_posbucket" in ov:     sim_ex["pos_bucket"]       = int(ov["simhash_posbucket"])
        if "simhash_minlen" in ov:        sim_ex["min_token_len"]    = int(ov["simhash_minlen"])
        if "simhash_norm_strict" in ov:   sim_ex["normalize_strict"] = bool(ov["simhash_norm_strict"])
        if "simhash_strip_ids" in ov:     sim_ex["strip_dates_ids"]  = bool(ov["simhash_strip_ids"])
        if "simhash_maxw" in ov:          sim_ex["max_token_weight"] = int(ov["simhash_maxw"])
        if not calibration_enabled:
            sim_ex["force_threshold"] = True
        cfg_sim.extras = sim_ex

        return PipelineConfig(
            simhash=cfg_sim,
            minhash=cfg_min,
            embedding=cfg_emb,
            arbiter=cfg_arb,
            candidates=cfg_cand,
            bootstrap=cfg_boot,
            self_learning=cfg_sl,
            persist=True,
            disable_calibration=not calibration_enabled,
        )

    def _apply_preset(self):
        """
        Apply a high-level detection preset to learner configs and arbiter.

        Presets primarily adjust target precision, gray-zone margin, epochs,
        and initial decision thresholds.
        """
        preset = (self.profile_var.get() or "Balanced").lower()

        # Pull current configs
        sim_cfg = self.card_sim.get_config()
        min_cfg = self.card_min.get_config()
        emb_cfg = self.card_emb.get_config()
        arb_cfg = self.arbiter_panel.get_config()

        # Defaults
        if preset.startswith("balanced"):
            target = 0.98
            gray = 0.05
            epochs = "2"
            thr_sim = thr_min = 0.75
            thr_emb = 0.988
        elif preset.startswith("high"):
            target = 0.995
            gray = 0.04
            epochs = "1"
            thr_sim = thr_min = 0.88
            thr_emb = 0.994
        elif preset.startswith("recall"):
            target = 0.95
            gray = 0.06
            epochs = "3"
            thr_sim = thr_min = 0.60
            thr_emb = 0.975
        else:
            return

        # Apply target precision
        sim_cfg.target_precision = target
        min_cfg.target_precision = target
        emb_cfg.target_precision = target

        # Apply thresholds via extras
        sim_ex = dict(sim_cfg.extras or {}); sim_ex["decision_threshold"] = float(thr_sim); sim_cfg.extras = sim_ex
        min_ex = dict(min_cfg.extras or {}); min_ex["decision_threshold"] = float(thr_min); min_cfg.extras = min_ex
        emb_ex = dict(emb_cfg.extras or {}); emb_ex["cosine_threshold"] = float(thr_emb); emb_cfg.extras = emb_ex

        # Arbiter and epochs
        arb_cfg.gray_zone_margin = gray
        self.arbiter_panel.set_config(arb_cfg)
        self.entry_sl_epochs.delete(0, tk.END); self.entry_sl_epochs.insert(0, epochs)

        # Push updated configs back into the UI
        self.card_sim.set_config(sim_cfg)
        self.card_min.set_config(min_cfg)
        self.card_emb.set_config(emb_cfg)

    # Documents tab actions
    def _on_add_files(self):
        """
        Open a file picker and ingest selected documents into the database.

        Supported extensions: .pdf, .docx, .txt
        """
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
        """
        Recursively ingest a folder containing supported files into the database.

        Supported extensions: .pdf, .docx, .txt
        """
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
        """Delete selected document IDs and refresh the table."""
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

    def _ingest_paths(self, paths: Iterable[str]):
        """
        Ingest a list of absolute/relative file paths into the database.

        Work occurs on a background thread, UI status is updated incrementally.
        """
        ps = list(paths)
        if self.doc_status:
            self.doc_status.configure(text=f"Ingesting {len(ps)} files…")
            self.update_idletasks()

        def worker(plist: List[str]):
            """Background ingestion worker that batches DB writes for throughput."""
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
                if hasattr(sqlite_store, "batch_upsert_documents"):
                    sqlite_store.batch_upsert_documents(docs_batch)
                else:
                    for (doc_id, raw, norm, meta_s) in docs_batch:
                        sqlite_store.upsert_document(doc_id, raw, norm, json.loads(meta_s))

            if mappings_batch:
                if hasattr(sqlite_store, "batch_add_file_mappings"):
                    sqlite_store.batch_add_file_mappings(mappings_batch)
                else:
                    for (doc_id, abs_path, mtime_ns) in mappings_batch:
                        sqlite_store.add_file_mapping(doc_id, abs_path, mtime_ns)

            if self.doc_status:
                self.after(0, lambda: [self._refresh_docs_tab(),
                                    self.doc_status.configure(text=f"Done. ok={ok}, fail={fail}")])

        threading.Thread(target=worker, args=(ps,), daemon=True).start()

    def _ingest_file(self, path: str, return_batch: bool = False):
        """
        Extract raw text and metadata from a file and upsert into the DB.
        """
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
            doc_entry = (
                doc_id,
                raw_text,
                norm,
                json.dumps({
                    "language": meta.get("language"),
                    "filesize": meta.get("filesize") or 0,
                }),
            )

            mapping_entry = (doc_id, abs_path, mtime_ns)
            return doc_entry, mapping_entry

        # Legacy behavior: single insert
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

    # Helpers: counters and timer
    def _reset_counters(self):
        """Reset the live counters in the run status panel."""
        self.var_cnt_pairs.set("Pairs: —")
        self.var_cnt_exact.set("Exact-duplicate: —")
        self.var_cnt_near.set("Near-duplicate: —")
        self.var_cnt_unc.set("Uncertain: —")

    def _update_counters_from_result(self, res: Dict[str, Any]):
        """
        Update the live counters from the run result, compute from traces if
        not present in the summary for robustness.
        """
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
        """Start the elapsed wall-clock timer."""
        self._run_start_ts = time.time()
        self._tick_timer()

    def _stop_timer(self):
        """Stop the elapsed timer and finalize the display."""
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
        """Timer callback: update elapsed and reschedule self."""
        if self._run_start_ts is None:
            return
        elapsed = int(time.time() - self._run_start_ts)
        self.var_elapsed.set(self._fmt_hms(elapsed))
        self._elapsed_job = self.after(1000, self._tick_timer)

    @staticmethod
    def _fmt_hms(seconds: int) -> str:
        """Format seconds as HH:MM:SS."""
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # Resource monitor
    def _start_resource_monitor(self):
        """Begin periodic resource polling."""
        if self._res_job:
            return
        try:
            psutil.cpu_percent(interval=None)
            self._proc.cpu_percent(interval=None)
        except Exception:
            pass
        self._poll_resources()

    def _stop_resource_monitor(self):
        """Cancel the scheduled resource polling job if present."""
        if self._res_job:
            try:
                self.after_cancel(self._res_job)
            except Exception:
                pass
        self._res_job = None

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        """Human-readable binary size formatter (B, KiB, MiB, GiB, TiB)."""
        try:
            n = float(n)
        except Exception:
            return f"{n}"
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024 or unit == "TiB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
            n /= 1024.0

    def _gpu_summary_text(self) -> Optional[str]:
        """
        Build a concise GPU utilization/VRAM summary string, if NVML available.
        """
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
                    name_obj = pynvml.nvmlDeviceGetName(h)
                    name = name_obj.decode("utf-8", "ignore") if isinstance(name_obj, (bytes, bytearray)) else str(name_obj)
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
        """Poll process/system metrics and reflect them in the resource panel."""
        try:
            with self._proc.oneshot():
                # raw process CPU % (can be >100% because it's sum across cores)
                raw_proc = self._proc.cpu_percent(interval=None)
                cores = psutil.cpu_count(logical=True) or 1
                norm = raw_proc / float(cores)  # normalized to system-wide %

                mem_info = self._proc.memory_full_info()
                rss = mem_info.rss
                total_ram = psutil.virtual_memory().total
                mem_percent = (rss / total_ram * 100.0) if total_ram else 0.0

                open_files_cnt = len(self._proc.open_files())
                threads_cnt = self._proc.num_threads()

            # Update labels
            self.var_res_cpu.set(
                f"App CPU: {raw_proc:.1f}% raw  ({norm:.1f}% of system across {cores} cores)"
            )
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
    """Entry point: launch the Tkinter application loop."""
    App().mainloop()


# Import first 500 rows from a CSV file with text column field only
def import_rows_from_csv(csv_path="dataset/True.csv"):
    """
    Convenience utility to insert a randomized sample of CSV rows into the DB.
    """
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
        print(f"CSV has only {len(df)} rows, sampling all available.")
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
    main()
