# src/gui/widgets/arbiter_panel.py
from __future__ import annotations

from typing import Callable, Optional, List
import tkinter as tk
from tkinter import ttk

from src.ensemble.arbiter import ArbiterConfig

_OnChange = Optional[Callable[[ArbiterConfig], None]]

_DEFAULT_STEPS = ["normalize_strict", "minhash_alt_shingle", "embed_whiten"]

class ArbiterPanel(ttk.LabelFrame):
    # Panel for editing ArbiterConfig
    def __init__(self, master, *, config: ArbiterConfig, on_change: _OnChange = None, text: str = "Arbiter / Consensus"):
        super().__init__(master, text=text, padding=8)
        self._on_change = on_change

        # vars
        self.var_require_agree = tk.IntVar(value=int(config.require_agreement))
        self.var_gray = tk.DoubleVar(value=float(config.gray_zone_margin))
        self.var_max_steps = tk.IntVar(value=int(config.max_escalation_steps))
        self.var_self_epochs = tk.IntVar(value=int(config.max_self_train_epochs))
        self.var_strong_margin = tk.DoubleVar(value=float(config.strong_margin))
        self.var_random_state = tk.IntVar(value=int(getattr(config, "random_state", 13)))

        # header
        hdr = ttk.Frame(self)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="Consensus rule").pack(side=tk.LEFT)
        rb = ttk.Frame(hdr); rb.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(rb, text="2 of 3", variable=self.var_require_agree, value=2, command=self._emit_change).pack(side=tk.LEFT)
        ttk.Radiobutton(rb, text="3 of 3", variable=self.var_require_agree, value=3, command=self._emit_change).pack(side=tk.LEFT, padx=(6, 0))

        # gray zone + margins
        rowf = ttk.LabelFrame(self, text="Decision thresholds")
        rowf.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(rowf, text="Gray-zone margin").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        gray_scale = ttk.Scale(rowf, from_=0.0, to=0.20, variable=self.var_gray, command=lambda *_: self._sync_label(l_gray))
        gray_scale.grid(row=0, column=1, sticky="we", padx=6)
        l_gray = ttk.Label(rowf, text=f"{self.var_gray.get():.3f}")
        l_gray.grid(row=0, column=2, sticky="e", padx=6)

        ttk.Label(rowf, text="Strong-margin (pseudo labels)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        sm_scale = ttk.Scale(rowf, from_=0.0, to=0.20, variable=self.var_strong_margin, command=lambda *_: self._sync_label(l_sm))
        sm_scale.grid(row=1, column=1, sticky="we", padx=6)
        l_sm = ttk.Label(rowf, text=f"{self.var_strong_margin.get():.3f}")
        l_sm.grid(row=1, column=2, sticky="e", padx=6)

        for c in (1,):
            rowf.grid_columnconfigure(c, weight=1)

        # escalation
        esc = ttk.LabelFrame(self, text="Escalation steps")
        esc.pack(fill=tk.BOTH, expand=True, pady=(6, 6))
        left = ttk.Frame(esc); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 3), pady=6)
        right = ttk.Frame(esc); right.pack(side=tk.LEFT, fill=tk.Y, padx=(3, 6), pady=6)

        self.listbox = tk.Listbox(left, height=6, activestyle="dotbox")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self._load_steps(config.escalation_order or _DEFAULT_STEPS)

        ttk.Button(right, text="Up", command=self._move_up).pack(fill=tk.X)
        ttk.Button(right, text="Down", command=self._move_down).pack(fill=tk.X, pady=(4, 0))
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self.entry_new = ttk.Entry(right)
        self.entry_new.pack(fill=tk.X)
        ttk.Button(right, text="Add", command=self._add_step).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(right, text="Remove", command=self._remove_step).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(right, text="Reset", command=self._reset_steps).pack(fill=tk.X, pady=(8, 0))

        # self-learning
        sl = ttk.LabelFrame(self, text="Self-learning")
        sl.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(sl, text="Max epochs").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Spinbox(sl, from_=0, to=50, textvariable=self.var_self_epochs, width=8, command=self._emit_change).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(sl, text="Max escalation steps").grid(row=0, column=2, sticky="e", padx=6)
        ttk.Spinbox(sl, from_=0, to=10, textvariable=self.var_max_steps, width=8, command=self._emit_change).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(sl, text="Random state").grid(row=0, column=4, sticky="e", padx=6)
        ttk.Entry(sl, width=8, textvariable=self.var_random_state).grid(row=0, column=5, sticky="w", padx=6)

        for v in (self.var_gray, self.var_strong_margin, self.var_self_epochs, self.var_max_steps, self.var_random_state):
            v.trace_add("write", lambda *_: self._emit_change())

    # get current config
    def get_config(self) -> ArbiterConfig:
        return ArbiterConfig(
            require_agreement=int(self.var_require_agree.get()),
            gray_zone_margin=float(self.var_gray.get()),
            max_escalation_steps=int(self.var_max_steps.get()),
            escalation_order=self._steps(),
            max_self_train_epochs=int(self.var_self_epochs.get()),
            strong_margin=float(self.var_strong_margin.get()),
            random_state=int(self.var_random_state.get()),
        )

    # set from config
    def set_config(self, cfg: ArbiterConfig) -> None:
        self.var_require_agree.set(int(cfg.require_agreement))
        self.var_gray.set(float(cfg.gray_zone_margin))
        self.var_max_steps.set(int(cfg.max_escalation_steps))
        self.var_self_epochs.set(int(cfg.max_self_train_epochs))
        self.var_strong_margin.set(float(cfg.strong_margin))
        self.var_random_state.set(int(getattr(cfg, "random_state", 13)))
        self._load_steps(cfg.escalation_order or _DEFAULT_STEPS)
        self._emit_change()

    # listbox helpers
    def _steps(self) -> List[str]:
        return [self.listbox.get(i) for i in range(self.listbox.size())]

    def _load_steps(self, steps: List[str]) -> None:
        self.listbox.delete(0, tk.END)
        for s in steps:
            self.listbox.insert(tk.END, s)

    def _move_up(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        if i == 0:
            return
        txt = self.listbox.get(i)
        self.listbox.delete(i)
        self.listbox.insert(i - 1, txt)
        self.listbox.selection_set(i - 1)
        self._emit_change()

    def _move_down(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        if i >= self.listbox.size() - 1:
            return
        txt = self.listbox.get(i)
        self.listbox.delete(i)
        self.listbox.insert(i + 1, txt)
        self.listbox.selection_set(i + 1)
        self._emit_change()

    def _add_step(self):
        s = self.entry_new.get().strip()
        if not s:
            return
        self.listbox.insert(tk.END, s)
        self.entry_new.delete(0, tk.END)
        self._emit_change()

    def _remove_step(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.listbox.delete(sel[0])
        self._emit_change()

    def _reset_steps(self):
        self._load_steps(_DEFAULT_STEPS)
        self._emit_change()

    # callbacks
    def _sync_label(self, lbl: ttk.Label):
        try:
            val = float(lbl.master.nametowidget(lbl.master.grid_slaves(row=lbl.grid_info()["row"], column=1)[0]._name).get())
        except Exception:
            val = float(lbl.cget("text") or 0.0)
        lbl.configure(text=f"{float(lbl.master.master.children.get(lbl.master._name, None) or self.var_gray.get()):.3f}")

    def _emit_change(self):
        if self._on_change:
            try:
                self._on_change(self.get_config())
            except Exception:
                pass
