# src/gui/widgets/learner_card.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import tkinter as tk
from tkinter import ttk

from src.learners.base import LearnerConfig, CalibrationParams

_ON_CHANGE = Optional[Callable[[LearnerConfig], None]]

class LearnerCard(ttk.Frame):
    def __init__(
        self,
        master,
        *,
        learner_name: str,
        kind: str,
        config: LearnerConfig,
        on_change: _ON_CHANGE = None,
        collapsed: bool = True,
    ):
        super().__init__(master, padding=8)
        self.learner_name = learner_name
        self.kind = kind.lower().strip()
        self._on_change = on_change

        # control vars
        self.var_enabled = tk.BooleanVar(value=config.enabled)
        self.var_target_precision = tk.DoubleVar(value=float(config.target_precision))
        self.var_use_min_conf = tk.BooleanVar(value=(config.min_confidence is not None))
        self.var_min_conf = tk.DoubleVar(value=float(config.min_confidence or 0.0))
        self.var_max_pairs = tk.IntVar(value=int(config.max_pairs_per_epoch))
        self.var_random_state = tk.IntVar(value=int(config.random_state))

        # extras map
        extras = dict(config.extras or {})
        self.extras_vars: Dict[str, tk.Variable] = {}
        self._create_widgets(extras, collapsed)

        # wire generic change listeners
        for v in (self.var_enabled, self.var_target_precision, self.var_use_min_conf,
                  self.var_min_conf, self.var_max_pairs, self.var_random_state):
            v.trace_add("write", lambda *_: self._emit_change())

    # UI
    def _create_widgets(self, extras: Dict[str, Any], collapsed: bool):
        # header
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        ttk.Checkbutton(header, text=self.learner_name, variable=self.var_enabled).pack(side=tk.LEFT)
        ttk.Label(header, text=f"({self.kind})", foreground="#666").pack(side=tk.LEFT, padx=(6, 0))

        # calibration snapshot
        self._calib = {
            "precision": ttk.Label(header, text="est. precision: —"),
            "threshold": ttk.Label(header, text="thr: —"),
            "brier": ttk.Label(header, text="brier: —"),
        }
        ttk.Separator(header, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._calib["precision"].pack(side=tk.LEFT, padx=(0, 8))
        self._calib["threshold"].pack(side=tk.LEFT, padx=(0, 8))
        self._calib["brier"].pack(side=tk.LEFT)

        # core
        core = ttk.LabelFrame(self, text="Core")
        core.pack(fill=tk.X, pady=(8, 4))

        row = 0
        ttk.Label(core, text="Target precision").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(core, from_=0.90, to=0.999, variable=self.var_target_precision, command=lambda *_: self._sync_scale_label(tp_lbl)).grid(row=row, column=1, sticky="we", padx=6)
        tp_lbl = ttk.Label(core, text=f"{self.var_target_precision.get():.3f}")
        tp_lbl.grid(row=row, column=2, sticky="e", padx=6)

        row += 1
        ttk.Label(core, text="Min confidence").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        use_mc = ttk.Checkbutton(core, variable=self.var_use_min_conf, command=self._toggle_min_conf)
        use_mc.grid(row=row, column=1, sticky="w", padx=(6, 2))
        self.min_conf_entry = ttk.Entry(core, width=8, textvariable=self.var_min_conf, state=("normal" if self.var_use_min_conf.get() else "disabled"))
        self.min_conf_entry.grid(row=row, column=2, sticky="e", padx=6)

        row += 1
        ttk.Label(core, text="Max pairs/epoch").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(core, from_=1, to=10_000_000, increment=100, width=12, textvariable=self.var_max_pairs).grid(row=row, column=1, sticky="w", padx=6)
        ttk.Label(core, text="Random state").grid(row=row, column=2, sticky="e", padx=6)
        ttk.Entry(core, width=8, textvariable=self.var_random_state).grid(row=row, column=3, sticky="e", padx=6)

        for c in range(0, 4):
            core.grid_columnconfigure(c, weight=1 if c == 1 else 0)

        # advanced
        adv_frame = ttk.Frame(self)
        adv_header = ttk.Frame(adv_frame)
        adv_header.pack(fill=tk.X)
        self._adv_btn = ttk.Button(adv_header, text=("Show advanced ▸" if collapsed else "Hide advanced ▾"), command=self._toggle_advanced)
        self._adv_btn.pack(side=tk.LEFT)
        self._adv_body = ttk.LabelFrame(adv_frame, text="Advanced")
        if not collapsed:
            self._adv_body.pack(fill=tk.X, pady=(6, 0))
        adv_frame.pack(fill=tk.X, pady=(6, 0))

        # extras per kind
        self._build_extras(self._adv_body, extras)

    def _build_extras(self, parent: ttk.LabelFrame, extras: Dict[str, Any]):
        # helpers
        def add_row(r: int, label: str, var: tk.Variable, widget: tk.Widget):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
            widget.grid(row=r, column=1, sticky="we", padx=6)
            parent.grid_columnconfigure(1, weight=1)

        r = 0
        k = self.kind

        if k == "simhash":
            v1 = self._mk_int_var(extras.get("max_hamming", 5)); add_row(r, "Max hamming", v1, ttk.Spinbox(parent, from_=0, to=64, textvariable=v1, width=8)); r += 1
            v2 = self._mk_int_var(extras.get("max_token_weight", 255)); add_row(r, "Max token weight", v2, ttk.Spinbox(parent, from_=1, to=1024, textvariable=v2, width=8)); r += 1
            v3 = self._mk_int_var(extras.get("min_token_len", 2)); add_row(r, "Min token len", v3, ttk.Spinbox(parent, from_=1, to=10, textvariable=v3, width=8)); r += 1
            v4 = self._mk_bool_var(extras.get("normalize_strict", False)); add_row(r, "Normalize strict", v4, ttk.Checkbutton(parent, variable=v4)); r += 1
            v5 = self._mk_bool_var(extras.get("strip_dates_ids", False)); add_row(r, "Strip dates/IDs", v5, ttk.Checkbutton(parent, variable=v5)); r += 1

        elif k == "minhash":
            v1 = self._mk_int_var(extras.get("shingle_size", 3)); add_row(r, "Shingle size", v1, ttk.Spinbox(parent, from_=1, to=10, textvariable=v1, width=8)); r += 1
            v2 = self._mk_int_var(extras.get("num_perm", 64)); add_row(r, "Num permutations", v2, ttk.Spinbox(parent, from_=8, to=512, increment=8, textvariable=v2, width=8)); r += 1
            v3 = self._mk_float_var(extras.get("lsh_threshold", 0.6)); add_row(r, "LSH threshold", v3, ttk.Entry(parent, textvariable=v3, width=8)); r += 1
            v4 = self._mk_bool_var(extras.get("normalize_strict", False)); add_row(r, "Normalize strict", v4, ttk.Checkbutton(parent, variable=v4)); r += 1

        elif k == "embedding":
            v1 = self._mk_str_var(extras.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")); add_row(r, "Model name", v1, ttk.Entry(parent, textvariable=v1)); r += 1
            v2 = self._mk_int_var(extras.get("batch_size", 64)); add_row(r, "Batch size", v2, ttk.Spinbox(parent, from_=1, to=2048, textvariable=v2, width=8)); r += 1
            v3 = self._mk_float_var(extras.get("cosine_threshold", 0.92)); add_row(r, "Cosine threshold", v3, ttk.Entry(parent, textvariable=v3, width=8)); r += 1
            v4 = self._mk_bool_var(extras.get("whiten", False)); add_row(r, "Whiten", v4, ttk.Checkbutton(parent, variable=v4)); r += 1
            v5 = self._mk_bool_var(extras.get("remove_top_pc", False)); add_row(r, "Remove top PC", v5, ttk.Checkbutton(parent, variable=v5)); r += 1
            v6 = self._mk_bool_var(extras.get("normalize_strict", False)); add_row(r, "Normalize strict", v6, ttk.Checkbutton(parent, variable=v6)); r += 1

        for _, var in self.extras_vars.items():
            var.trace_add("write", lambda *_: self._emit_change())

    # calibration snapshot setters
    def set_calibration(self, cal: Optional[CalibrationParams]) -> None:
        if cal is None:
            self._calib["precision"].configure(text="est. precision: —")
            self._calib["threshold"].configure(text="thr: —")
            self._calib["brier"].configure(text="brier: —")
            return
        thr = cal.threshold if cal.threshold is not None else float("nan")
        brier = cal.brier_score if cal.brier_score is not None else float("nan")
        self._calib["precision"].configure(text=f"est. precision: target")  # shown as target; true est can be computed elsewhere
        self._calib["threshold"].configure(text=f"thr: {thr:.3f}")
        self._calib["brier"].configure(text=("brier: —" if (brier != brier) else f"brier: {brier:.3f}"))

    # get/set config
    def get_config(self) -> LearnerConfig:
        extras = {k: self._coerce_var(v) for k, v in self.extras_vars.items()}
        min_conf: Optional[float] = None
        if self.var_use_min_conf.get():
            min_conf = float(self.var_min_conf.get())
        cfg = LearnerConfig(
            enabled=bool(self.var_enabled.get()),
            target_precision=float(self.var_target_precision.get()),
            min_confidence=min_conf,
            max_pairs_per_epoch=int(self.var_max_pairs.get()),
            random_state=int(self.var_random_state.get()),
            extras=extras,
        )
        return cfg

    def set_config(self, cfg: LearnerConfig) -> None:
        self.var_enabled.set(bool(cfg.enabled))
        self.var_target_precision.set(float(cfg.target_precision))
        self.var_use_min_conf.set(cfg.min_confidence is not None)
        self.var_min_conf.set(float(cfg.min_confidence or 0.0))
        self.var_max_pairs.set(int(cfg.max_pairs_per_epoch))
        self.var_random_state.set(int(cfg.random_state))
        self._set_extras(cfg.extras or {})
        self._toggle_min_conf()
        self._emit_change()

    # internals
    def _toggle_advanced(self):
        if self._adv_body.winfo_ismapped():
            self._adv_body.pack_forget()
            self._adv_btn.configure(text="Show advanced ▸")
        else:
            self._adv_body.pack(fill=tk.X, pady=(6, 0))
            self._adv_btn.configure(text="Hide advanced ▾")

    def _toggle_min_conf(self):
        state = ("normal" if self.var_use_min_conf.get() else "disabled")
        self.min_conf_entry.configure(state=state)

    def _sync_scale_label(self, lbl: ttk.Label):
        lbl.configure(text=f"{self.var_target_precision.get():.3f}")

    def _emit_change(self):
        if self._on_change:
            try:
                self._on_change(self.get_config())
            except Exception:
                pass

    # extras var makers
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

    # extras key allocator: stable per row by label text order
    def _alloc_key(self) -> str:
        return f"extra_{len(self.extras_vars)+1}"

    def _set_extras(self, extras: Dict[str, Any]):
        keys = self._known_extra_keys()
        if len(self.extras_vars) == len(keys):
            for (k, var), name in zip(self.extras_vars.items(), keys):
                self.extras_vars[name] = self.extras_vars.pop(k)
        # set values
        for k in keys:
            if k in self.extras_vars:
                v = extras.get(k)
                if isinstance(self.extras_vars[k], tk.IntVar) and v is not None:
                    self.extras_vars[k].set(int(v))
                elif isinstance(self.extras_vars[k], tk.DoubleVar) and v is not None:
                    self.extras_vars[k].set(float(v))
                elif isinstance(self.extras_vars[k], tk.BooleanVar) and v is not None:
                    self.extras_vars[k].set(bool(v))
                elif isinstance(self.extras_vars[k], tk.StringVar) and v is not None:
                    self.extras_vars[k].set(str(v))

    def _coerce_var(self, var: tk.Variable) -> Any:
        try:
            if isinstance(var, tk.IntVar):
                return int(var.get())
            if isinstance(var, tk.DoubleVar):
                return float(var.get())
            if isinstance(var, tk.BooleanVar):
                return bool(var.get())
            if isinstance(var, tk.StringVar):
                return str(var.get())
        except Exception:
            pass
        return var.get()

    def _known_extra_keys(self):
        if self.kind == "simhash":
            return ["max_hamming", "max_token_weight", "min_token_len", "normalize_strict", "strip_dates_ids"]
        if self.kind == "minhash":
            return ["shingle_size", "num_perm", "lsh_threshold", "normalize_strict"]
        if self.kind == "embedding":
            return ["model_name", "batch_size", "cosine_threshold", "whiten", "remove_top_pc", "normalize_strict"]
        return []
