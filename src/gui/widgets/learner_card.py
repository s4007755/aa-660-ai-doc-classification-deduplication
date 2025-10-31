from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List
import tkinter as tk
from tkinter import ttk

from src.learners.base import LearnerConfig, CalibrationParams

_ON_CHANGE = Optional[Callable[[LearnerConfig], None]]


class LearnerCard(ttk.Frame):
    """
    GUI card for a single learner's settings inside a profile editor.

    Purpose
    * Surfaces the learner's core knobs.
    * Exposes a threshold slider in the Core section that mirrors the current
      preset's threshold. When calibration is enabled, the slider is disabled and
      visually marked as controlled by calibration.
    * Shows a small KPI strip (estimated precision, threshold, Brier) for quick
      feedback when calibration data exists.

    Behavior
    * The Threshold control owns the value written back to extras:
        - Embedding -> `extras["cosine_threshold"]`
        - SimHash/MinHash -> `extras["decision_threshold"]`
    * When calibration is On:
        - Target precision row is visible/enabled.
        - Threshold slider is disabled and annotated.
    * When calibration is Off:
        - Target precision row is hidden.
        - Threshold slider is enabled, written back to extras via `get_config()`.
    """

    def __init__(
        self,
        master,
        *,
        learner_name: str,
        kind: Optional[str] = None,
        config: Optional[LearnerConfig] = None,
        on_change: _ON_CHANGE = None,
        collapsed: bool = True,
        show_min_conf: bool = True,
        text: Optional[str] = None,
        prefer_performance: bool = True,
        **kwargs,
    ):
        super().__init__(master, padding=8)
        self.learner_name = learner_name
        # Heuristically infer kind from common learner names, otherwise trust provided `kind`.
        inferred = (learner_name or "").strip().lower()
        if inferred in ("simhash", "minhash", "embedding"):
            inferred_kind = inferred
        else:
            inferred_kind = (kind or "simhash").strip().lower()
        self.kind = inferred_kind
        self._on_change = on_change
        self._desc_text = text
        self._show_min_conf = bool(show_min_conf)
        self._prefer_perf = bool(prefer_performance)

        cfg = config or LearnerConfig()

        # Tk control variables
        self.var_enabled = tk.BooleanVar(value=cfg.enabled)
        self.var_target_precision = tk.DoubleVar(value=float(cfg.target_precision))
        self.var_use_min_conf = tk.BooleanVar(value=(cfg.min_confidence is not None))
        self.var_min_conf = tk.DoubleVar(value=float(cfg.min_confidence or 0.0))
        self.var_max_pairs = tk.IntVar(value=int(cfg.max_pairs_per_epoch))
        self.var_random_state = tk.IntVar(value=int(cfg.random_state))

        # Threshold variable
        # Pick sensible defaults by kind, then apply any preset provided extras.
        thr_default = 0.988 if inferred_kind == "embedding" else 0.75
        ex0 = dict(cfg.extras or {})
        if inferred_kind == "embedding":
            thr_default = float(ex0.get("cosine_threshold", thr_default))
        else:
            thr_default = float(ex0.get("decision_threshold", thr_default))
        self.var_threshold = tk.DoubleVar(value=thr_default)
        self._thr_value_lbl: Optional[ttk.Label] = None

        # Internal UI bookkeeping
        self._calibration_enabled: bool = False
        self.extras_vars: Dict[str, tk.Variable] = {}
        self.min_conf_entry: Optional[ttk.Entry] = None
        self._tp_value_lbl: Optional[ttk.Label] = None
        self._tp_row_widgets: List[tk.Widget] = []
        self._thr_widgets: List[tk.Widget] = []
        self._thr_notes: List[tk.Widget] = []

        # Build UI and bind traces
        self._create_widgets(dict(cfg.extras or {}), collapsed)

        # Emit on any core variable change
        for v in (
            self.var_enabled,
            self.var_use_min_conf,
            self.var_min_conf,
            self.var_max_pairs,
            self.var_random_state,
        ):
            v.trace_add("write", lambda *_: self._emit_change())

        # Keep target precision label in sync and emit
        self.var_target_precision.trace_add(
            "write",
            lambda *_: (
                self._sync_scale_label(self._tp_value_lbl) if self._tp_value_lbl is not None else None,
                self._emit_change()
            )
        )

        # Keep threshold numeric label in sync and emit
        self.var_threshold.trace_add(
            "write",
            lambda *_: (
                self._thr_value_lbl.configure(text=f"{self.var_threshold.get():.3f}") if self._thr_value_lbl else None,
                self._emit_change()
            )
        )

    # UI
    def _create_widgets(self, extras: Dict[str, Any], collapsed: bool):
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X)

        ttk.Checkbutton(header, text=self.learner_name, variable=self.var_enabled).pack(side=tk.LEFT)
        ttk.Label(header, text=f"({self.kind})", foreground="#666").pack(side=tk.LEFT, padx=(6, 0))
        if self._desc_text:
            ttk.Label(header, text=self._desc_text, foreground="#888").pack(side=tk.LEFT, padx=(8, 0))

        # Calibration KPI strip
        self._calib = {
            "precision": ttk.Label(header, text="est. precision: —"),
            "threshold": ttk.Label(header, text="thr: —"),
            "brier": ttk.Label(header, text="brier: —"),
        }
        ttk.Separator(header, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._calib["precision"].pack(side=tk.LEFT, padx=(0, 8))
        self._calib["threshold"].pack(side=tk.LEFT, padx=(0, 8))
        self._calib["brier"].pack(side=tk.LEFT)

        # Core section
        core = ttk.LabelFrame(self, text="Core")
        core.pack(fill=tk.X, pady=(8, 4))

        row = 0
        # Target precision
        lbl_tp = ttk.Label(core, text="Target precision")
        lbl_tp.grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self._tp_value_lbl = ttk.Label(core, text=f"{self.var_target_precision.get():.3f}")
        scl_tp = ttk.Scale(
            core,
            from_=0.90,
            to=0.999,
            variable=self.var_target_precision,
            command=lambda *_: self._sync_scale_label(self._tp_value_lbl),
        )
        scl_tp.grid(row=row, column=1, sticky="we", padx=6)
        self._tp_value_lbl.grid(row=row, column=2, sticky="e", padx=6)
        self._tp_row_widgets = [lbl_tp, scl_tp, self._tp_value_lbl]

        # Threshold row:
        row += 1
        thr_label_txt = "Cosine threshold" if self.kind == "embedding" else "Decision threshold"
        lbl_thr = ttk.Label(core, text=thr_label_txt)
        lbl_thr.grid(row=row, column=0, sticky="w", padx=6, pady=4)

        # Reasonable ranges by kind
        if self.kind == "embedding":
            thr_from, thr_to = 0.90, 0.999
        else:
            thr_from, thr_to = 0.50, 0.99

        scl_thr = ttk.Scale(
            core,
            from_=thr_from,
            to=thr_to,
            variable=self.var_threshold,
        )
        scl_thr.grid(row=row, column=1, sticky="we", padx=6)

        self._thr_value_lbl = ttk.Label(core, text=f"{self.var_threshold.get():.3f}")
        self._thr_value_lbl.grid(row=row, column=2, sticky="e", padx=6)

        # Register threshold widgets for calibration toggle behavior
        self._thr_widgets.extend([lbl_thr, scl_thr, self._thr_value_lbl])
        thr_note = ttk.Label(core, text="(controlled by calibration)", foreground="#888")
        thr_note.grid(row=row, column=3, sticky="w", padx=(6, 0))
        self._thr_notes.append(thr_note)

        # Optional min-confidence row
        if self._show_min_conf:
            row += 1
            ttk.Label(core, text="Min confidence").grid(row=row, column=0, sticky="w", padx=6, pady=4)
            use_mc = ttk.Checkbutton(core, variable=self.var_use_min_conf, command=self._toggle_min_conf)
            use_mc.grid(row=row, column=1, sticky="w", padx=(6, 2))
            self.min_conf_entry = ttk.Entry(
                core,
                width=8,
                textvariable=self.var_min_conf,
                state=("normal" if self.var_use_min_conf.get() else "disabled"),
            )
            self.min_conf_entry.grid(row=row, column=2, sticky="e", padx=6)

        # Throughput/determinism
        row += 1
        ttk.Label(core, text="Max pairs/epoch").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(core, from_=1, to=10_000_000, increment=100, width=12, textvariable=self.var_max_pairs).grid(
            row=row, column=1, sticky="w", padx=6
        )
        ttk.Label(core, text="Random state").grid(row=row, column=2, sticky="e", padx=6)
        ttk.Entry(core, width=8, textvariable=self.var_random_state).grid(row=row, column=3, sticky="e", padx=6)

        for c in range(0, 4):
            core.grid_columnconfigure(c, weight=1 if c == 1 else 0)

        # Advanced section
        adv_frame = ttk.Frame(self)
        adv_header = ttk.Frame(adv_frame)
        adv_header.pack(fill=tk.X)
        self._adv_btn = ttk.Button(
            adv_header,
            text=("Show advanced ▸" if collapsed else "Hide advanced ▾"),
            command=self._toggle_advanced,
        )
        self._adv_btn.pack(side=tk.LEFT)
        self._adv_body = ttk.LabelFrame(adv_frame, text="Advanced")
        if not collapsed:
            self._adv_body.pack(fill=tk.X, pady=(6, 0))
        adv_frame.pack(fill=tk.X, pady=(6, 0))

        self._build_extras(self._adv_body, extras)

        self.set_calibration_enabled(self._calibration_enabled)

    def _build_extras(self, parent: ttk.LabelFrame, extras: Dict[str, Any]):
        """
        Build advanced controls by learner kind. These map 1:1 to extras keys.
        """
        def add_row(r: int, label: str, var: tk.Variable, widget: tk.Widget):
            lbl = ttk.Label(parent, text=label)
            lbl.grid(row=r, column=0, sticky="w", padx=6, pady=4)
            widget.grid(row=r, column=1, sticky="we", padx=6)
            parent.grid_columnconfigure(1, weight=1)

        r = 0
        k = self.kind

        if k == "simhash":
            # Keep SimHash extras minimal in perf-first mode, expose more when prefer_performance=False
            v1 = self._mk_int_var(extras.get("max_hamming", 5))
            add_row(r, "Max hamming", v1, ttk.Spinbox(parent, from_=0, to=64, textvariable=v1, width=8)); r += 1
            if not self._prefer_perf:
                v2 = self._mk_int_var(extras.get("max_token_weight", 255)); add_row(r, "Max token weight", v2, ttk.Spinbox(parent, from_=1, to=4096, textvariable=v2, width=8)); r += 1
                v3 = self._mk_int_var(extras.get("min_token_len", 2)); add_row(r, "Min token length", v3, ttk.Spinbox(parent, from_=1, to=16, textvariable=v3, width=8)); r += 1
                v4 = self._mk_bool_var(extras.get("normalize_strict", False)); add_row(r, "Normalize strictly", v4, ttk.Checkbutton(parent, variable=v4)); r += 1
                v5 = self._mk_bool_var(extras.get("strip_dates_ids", False)); add_row(r, "Strip dates/IDs", v5, ttk.Checkbutton(parent, variable=v5)); r += 1

        elif k == "minhash":
            if not self._prefer_perf:
                v1 = self._mk_int_var(extras.get("shingle_size", 3))
                add_row(r, "Shingle size", v1, ttk.Spinbox(parent, from_=1, to=10, textvariable=v1, width=8)); r += 1
                v2 = self._mk_int_var(extras.get("num_perm", 64))
                add_row(r, "Num permutations", v2, ttk.Spinbox(parent, from_=8, to=512, increment=8, textvariable=v2, width=8)); r += 1
                v3 = self._mk_float_var(extras.get("lsh_threshold", 0.6))
                add_row(r, "LSH threshold", v3, ttk.Entry(parent, textvariable=v3, width=8)); r += 1
                v4 = self._mk_bool_var(extras.get("normalize_strict", False))
                add_row(r, "Normalize strict", v4, ttk.Checkbutton(parent, variable=v4)); r += 1

        elif k == "embedding":
            # Model/batch size only when not favouring perf
            if not self._prefer_perf:
                v1 = self._mk_str_var(extras.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
                add_row(r, "Model name", v1, ttk.Entry(parent, textvariable=v1)); r += 1
                v2 = self._mk_int_var(extras.get("batch_size", 64))
                add_row(r, "Batch size", v2, ttk.Spinbox(parent, from_=1, to=2048, textvariable=v2, width=8)); r += 1
            v3 = self._mk_bool_var(extras.get("whiten", False)); add_row(r, "Whiten", v3, ttk.Checkbutton(parent, variable=v3)); r += 1
            v4 = self._mk_bool_var(extras.get("remove_top_pc", False)); add_row(r, "Remove top PC", v4, ttk.Checkbutton(parent, variable=v4)); r += 1
            v5 = self._mk_bool_var(extras.get("normalize_strict", False)); add_row(r, "Normalize strict", v5, ttk.Checkbutton(parent, variable=v5)); r += 1

        # Emit on any advanced variable change
        for _, var in self.extras_vars.items():
            var.trace_add("write", lambda *_: self._emit_change())

    # Calibration toggle behavior
    def set_calibration_enabled(self, on: bool) -> None:
        """
        Toggle UI according to calibration state.
        """
        self._calibration_enabled = bool(on)

        # Target precision row visibility
        for w in getattr(self, "_tp_row_widgets", []):
            try:
                if on:
                    w.grid()
                    if hasattr(w, "configure"):
                        try:
                            w.configure(state="normal")
                        except Exception:
                            pass
                else:
                    w.grid_remove()
            except Exception:
                pass

        # Threshold editor enablement
        for w in getattr(self, "_thr_widgets", []):
            try:
                if hasattr(w, "configure"):
                    w.configure(state=("disabled" if on else "normal"))
            except Exception:
                pass

        # Notes visibility
        for n in getattr(self, "_thr_notes", []):
            try:
                if on:
                    n.grid()
                else:
                    n.grid_remove()
            except Exception:
                pass

    # Header KPI setters
    def set_header_metrics(
        self,
        *,
        est_precision: Optional[float] = None,
        threshold: Optional[float] = None,
        brier: Optional[float] = None,
        target_precision: Optional[float] = None,
    ) -> None:
        """
        Update the KPI labels displayed in the header.
        """
        def fmt(x, pat="{:.3f}"):
            try:
                return pat.format(float(x))
            except Exception:
                return "—"

        if est_precision is None and target_precision is None:
            prec_txt = "est. precision: —"
        elif est_precision is None:
            prec_txt = f"est. precision: — (target {fmt(target_precision)})"
        else:
            prec_txt = f"est. precision: {fmt(est_precision)}" if target_precision is None \
                else f"est. precision: {fmt(est_precision)} (target {fmt(target_precision)})"

        self._calib["precision"].configure(text=prec_txt)
        self._calib["threshold"].configure(text=f"thr: {fmt(threshold)}")
        self._calib["brier"].configure(text=f"brier: {fmt(brier)}")

    def set_estimates(self, **kwargs):
        """Alias for `set_header_metrics`."""
        self.set_header_metrics(**kwargs)

    def set_kpis(self, **kwargs):
        """Alias for `set_header_metrics`."""
        self.set_header_metrics(**kwargs)

    # Calibration snapshot setters
    def set_calibration(self, cal: Optional[CalibrationParams]) -> None:
        """
        Inject a calibration snapshot to populate the header KPI strip.
        Passing None clears the KPIs.
        """
        if cal is None:
            self._calib["precision"].configure(text="est. precision: —")
            self._calib["threshold"].configure(text="thr: —")
            self._calib["brier"].configure(text="brier: —")
            return
        thr = cal.threshold if cal.threshold is not None else None
        brier = cal.brier_score if cal.brier_score is not None else None
        tprec = self.var_target_precision.get()
        self.set_header_metrics(est_precision=None, threshold=thr, brier=brier, target_precision=tprec)

    # get/set config
    def get_config(self) -> LearnerConfig:
        """
        Snapshot current UI state into a LearnerConfig.
        Threshold value is always taken from the Core slider.
        """
        extras = {k: self._coerce_var(v) for k, v in self.extras_vars.items()}

        # Write back the threshold owned by the Core slider
        if self.kind == "embedding":
            extras["cosine_threshold"] = float(self.var_threshold.get())
        else:
            extras["decision_threshold"] = float(self.var_threshold.get())

        # Optional min confidence
        min_conf: Optional[float] = None
        if self._show_min_conf and self.var_use_min_conf.get():
            min_conf = float(self.var_min_conf.get())

        return LearnerConfig(
            enabled=bool(self.var_enabled.get()),
            target_precision=float(self.var_target_precision.get()),
            min_confidence=min_conf,
            max_pairs_per_epoch=int(self.var_max_pairs.get()),
            random_state=int(self.var_random_state.get()),
            extras=extras,
        )

    def set_config(self, cfg: LearnerConfig) -> None:
        """
        Apply a LearnerConfig to the UI controls. Intended for preset application
        and restoring saved settings. Emits on_change at the end.
        """
        # Core
        self.var_enabled.set(bool(cfg.enabled))
        self.var_target_precision.set(float(cfg.target_precision))
        self.var_use_min_conf.set((cfg.min_confidence is not None) if self._show_min_conf else False)
        self.var_min_conf.set(float(cfg.min_confidence or 0.0))
        self.var_max_pairs.set(int(cfg.max_pairs_per_epoch))
        self.var_random_state.set(int(cfg.random_state))

        self._set_extras(cfg.extras or {})

        ex = cfg.extras or {}
        if self.kind == "embedding":
            self.var_threshold.set(float(ex.get("cosine_threshold", self.var_threshold.get())))
        else:
            self.var_threshold.set(float(ex.get("decision_threshold", self.var_threshold.get())))

        self._toggle_min_conf()

        # Refresh inline numeric labels
        if self._tp_value_lbl is not None:
            self._sync_scale_label(self._tp_value_lbl)
        if self._thr_value_lbl is not None:
            self._thr_value_lbl.configure(text=f"{self.var_threshold.get():.3f}")

        self._emit_change()

    # Internals
    def _toggle_advanced(self):
        """
        Collapse/expand the Advanced section.
        """
        if self._adv_body.winfo_ismapped():
            self._adv_body.pack_forget()
            self._adv_btn.configure(text="Show advanced ▸")
        else:
            self._adv_body.pack(fill=tk.X, pady=(6, 0))
            self._adv_btn.configure(text="Hide advanced ▾")

    def _toggle_min_conf(self):
        """
        Enable/disable the min-confidence entry field based on the checkbox.
        """
        state = ("normal" if (self._show_min_conf and self.var_use_min_conf.get()) else "disabled")
        if self.min_conf_entry is not None:
            self.min_conf_entry.configure(state=state)

    def _sync_scale_label(self, lbl: Optional[ttk.Label]):
        """
        Mirror the current scale value into the adjacent numeric label.
        """
        if lbl is None:
            return
        lbl.configure(text=f"{self.var_target_precision.get():.3f}")

    def _emit_change(self):
        """
        Emit a fresh LearnerConfig via the on_change callback.
        """
        if self._on_change:
            try:
                self._on_change(self.get_config())
            except Exception:
                pass

    # Extra var makers
    def _mk_int_var(self, v: Any) -> tk.IntVar:
        var = tk.IntVar(value=int(v))
        self.extras_vars[self._alloc_key()] = var
        return var

    def _mk_float_var(self, v: Any) -> tk.DoubleVar:
        var = tk.DoubleVar(value=float(v))
        self.extras_vars[self._alloc_key()] = var
        return var

    def _mk_str_var(self, v: Any) -> tk.StringVar:
        var = tk.StringVar(value=str(v))
        self.extras_vars[self._alloc_key()] = var
        return var

    def _mk_bool_var(self, v: Any) -> tk.BooleanVar:
        var = tk.BooleanVar(value=bool(v))
        self.extras_vars[self._alloc_key()] = var
        return var

    def _alloc_key(self) -> str:
        """
        Allocate a stable placeholder key for a new extras var.
        """
        return f"extra_{len(self.extras_vars)+1}"

    def _set_extras(self, extras: Dict[str, Any]):
        """
        Load known extras into the advanced section variables, preserving
        the stable order for each learner kind.
        """
        keys = self._known_extra_keys()
        # If counts match, rebind placeholder keys to named keys.
        if len(self.extras_vars) == len(keys):
            for (k, var), name in zip(list(self.extras_vars.items()), keys):
                self.extras_vars[name] = self.extras_vars.pop(k)
        # Push values into the bound variables.
        for k in keys:
            if k in self.extras_vars:
                v = extras.get(k)
                try:
                    if isinstance(self.extras_vars[k], tk.IntVar) and v is not None:
                        self.extras_vars[k].set(int(v))
                    elif isinstance(self.extras_vars[k], tk.DoubleVar) and v is not None:
                        self.extras_vars[k].set(float(v))
                    elif isinstance(self.extras_vars[k], tk.BooleanVar) and v is not None:
                        self.extras_vars[k].set(bool(v))
                    elif isinstance(self.extras_vars[k], tk.StringVar) and v is not None:
                        self.extras_vars[k].set(str(v))
                except Exception:
                    pass

    def _coerce_var(self, var: tk.Variable) -> Any:
        """
        Extract a Python value from a Tk variable with best-effort type coercion.
        """
        try:
            if isinstance(var, tk.IntVar): return int(var.get())
            if isinstance(var, tk.DoubleVar): return float(var.get())
            if isinstance(var, tk.BooleanVar): return bool(var.get())
            if isinstance(var, tk.StringVar): return str(var.get())
        except Exception:
            pass
        return var.get()

    def _known_extra_keys(self):
        """
        The canonical order of advanced extras per learner kind.
        """
        if self.kind == "simhash":
            if self._prefer_perf:
                return ["max_hamming"]
            return ["max_hamming", "max_token_weight", "min_token_len", "normalize_strict", "strip_dates_ids"]

        if self.kind == "minhash":
            if self._prefer_perf:
                return []
            return ["shingle_size", "num_perm", "lsh_threshold", "normalize_strict"]

        if self.kind == "embedding":
            if self._prefer_perf:
                return ["whiten", "remove_top_pc", "normalize_strict"]
            return ["model_name", "batch_size", "whiten", "remove_top_pc", "normalize_strict"]

        return []
