# src/gui/widgets/trace_viewer.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from src.ensemble.arbiter import DecisionTrace


class TraceViewer(ttk.Frame):
    def __init__(self, master, *, text: str = "Decision Traces"):
        super().__init__(master, padding=8)
        self._build_ui(text)
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    # UI

    def _build_ui(self, text: str):
        # top bar
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        ttk.Label(top, text=text, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)

        self.entry_filter = ttk.Entry(top, width=28)
        self.entry_filter.pack(side=tk.RIGHT, padx=(6, 0))
        self.entry_filter.insert(0, "")
        ttk.Button(top, text="Filter", command=self._apply_filter).pack(side=tk.RIGHT)

        # main split
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # left: table
        left = ttk.Frame(main, padding=(0, 0, 6, 0))
        main.add(left, weight=1)

        cols = ("pair_key", "a_id", "b_id", "label", "reason", "agreed", "steps")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse", height=16)
        headings = [
            ("pair_key", "Pair"),
            ("a_id", "A"),
            ("b_id", "B"),
            ("label", "Final"),
            ("reason", "Reason"),
            ("agreed", "Agreed learners"),
            ("steps", "Escalation"),
        ]
        for cid, label in headings:
            self.tree.heading(cid, text=label)
            anchor = tk.W if cid in {"pair_key", "reason"} else (tk.CENTER if cid in {"label"} else tk.W)
            width = {
                "pair_key": 220, "a_id": 120, "b_id": 120,
                "label": 90, "reason": 160, "agreed": 160, "steps": 160,
            }[cid]
            self.tree.column(cid, width=width, anchor=anchor, stretch=True)
        vs = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # right: detail
        right = ttk.Frame(main)
        main.add(right, weight=2)

        # action bar
        ab = ttk.Frame(right)
        ab.pack(fill=tk.X)
        ttk.Button(ab, text="Copy JSON", command=self._copy_selected_json).pack(side=tk.LEFT)
        ttk.Button(ab, text="Save JSON…", command=self._save_selected_json).pack(side=tk.LEFT, padx=(6, 0))

        # summary card
        card = ttk.LabelFrame(right, text="Summary")
        card.pack(fill=tk.X, pady=(8, 6))
        self.lbl_pair = ttk.Label(card, text="Pair: —")
        self.lbl_pair.pack(anchor="w")
        self.lbl_label = ttk.Label(card, text="Final: —")
        self.lbl_label.pack(anchor="w")
        self.lbl_reason = ttk.Label(card, text="Reason: —")
        self.lbl_reason.pack(anchor="w")
        self.lbl_agreed = ttk.Label(card, text="Agreed learners: —")
        self.lbl_agreed.pack(anchor="w")
        self.lbl_steps = ttk.Label(card, text="Escalation steps: —")
        self.lbl_steps.pack(anchor="w")

        # per-learner tabs
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    # Public API

    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self._traces.clear()
        self._order.clear()
        self._render_detail(None)

    def set_traces(self, traces: Iterable[DecisionTrace | Dict[str, Any]]):
        self.clear()
        for tr in traces:
            self.add_trace(tr)

    def add_trace(self, trace: DecisionTrace | Dict[str, Any]):
        # normalize to dict via as_dict if needed
        if isinstance(trace, DecisionTrace):
            tr = trace.as_dict()
        else:
            tr = dict(trace)
        key = tr.get("pair_key") or f"{tr.get('a_id','?')}||{tr.get('b_id','?')}"
        self._traces[key] = tr
        self._order.append(key)
        self._insert_row(tr)

    def select_pair(self, pair_key: str):
        # select in tree and render
        for iid in self.tree.get_children(""):
            if self.tree.set(iid, "pair_key") == pair_key:
                self.tree.selection_set(iid)
                self.tree.see(iid)
                self._render_detail(self._traces.get(pair_key))
                break

    # Internals

    def _insert_row(self, tr: Dict[str, Any]):
        steps = ", ".join(tr.get("escalation_steps") or [])
        agreed = ", ".join(tr.get("agreed_learners") or [])
        vals = (
            tr.get("pair_key", "—"),
            tr.get("a_id", "—"),
            tr.get("b_id", "—"),
            tr.get("final_label", "—"),
            tr.get("reason", "—"),
            agreed or "—",
            steps or "—",
        )
        self.tree.insert("", tk.END, values=vals)

    def _apply_filter(self):
        q = (self.entry_filter.get() or "").strip().lower()
        self.tree.delete(*self.tree.get_children())
        for key in self._order:
            tr = self._traces[key]
            hay = " ".join([
                tr.get("pair_key", ""),
                tr.get("a_id", ""),
                tr.get("b_id", ""),
                tr.get("final_label", ""),
                tr.get("reason", ""),
                " ".join(tr.get("agreed_learners") or []),
                " ".join(tr.get("escalation_steps") or []),
            ]).lower()
            if not q or q in hay:
                self._insert_row(tr)

    def _on_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            self._render_detail(None)
            return
        iid = sel[0]
        key = self.tree.set(iid, "pair_key")
        self._render_detail(self._traces.get(key))

    def _render_detail(self, tr: Optional[Dict[str, Any]]):
        # clear tabs
        for i in reversed(range(self.nb.index("end") or 0)):
            self.nb.forget(i)

        if not tr:
            self.lbl_pair.configure(text="Pair: —")
            self.lbl_label.configure(text="Final: —")
            self.lbl_reason.configure(text="Reason: —")
            self.lbl_agreed.configure(text="Agreed learners: —")
            self.lbl_steps.configure(text="Escalation steps: —")
            return

        self.lbl_pair.configure(text=f"Pair: {tr.get('pair_key','—')}")
        self.lbl_label.configure(text=f"Final: {tr.get('final_label','—')}")
        self.lbl_reason.configure(text=f"Reason: {tr.get('reason','—')}")
        agreed = ", ".join(tr.get("agreed_learners") or [])
        self.lbl_agreed.configure(text=f"Agreed learners: {agreed or '—'}")
        steps = ", ".join(tr.get("escalation_steps") or [])
        self.lbl_steps.configure(text=f"Escalation steps: {steps or '—'}")

        # learners
        learners = tr.get("learners") or {}
        for name, info in learners.items():
            self._add_learner_tab(name, info)

        if self.nb.index("end") > 0:
            self.nb.select(0)

    def _add_learner_tab(self, name: str, info: Dict[str, Any]):
        frame = ttk.Frame(self.nb, padding=8)
        self.nb.add(frame, text=name)

        # top KPIs
        top = ttk.Frame(frame)
        top.pack(fill=tk.X)

        def _fmt(x, fmt="{:.3f}"):
            try:
                if x is None:
                    return "—"
                return fmt.format(float(x))
            except Exception:
                return "—"

        ttk.Label(top, text=f"raw: {_fmt(info.get('raw_score'))}").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(top, text=f"prob: {_fmt(info.get('prob'))}").pack(side=tk.LEFT, padx=(0, 12))
        thr = info.get("threshold")
        ttk.Label(top, text=f"thr: {_fmt(thr)}").pack(side=tk.LEFT, padx=(0, 12))

        # warnings
        warns = info.get("warnings") or []
        if warns:
            warn_box = ttk.LabelFrame(frame, text="Warnings")
            warn_box.pack(fill=tk.X, pady=(6, 0))
            ttk.Label(warn_box, text="; ".join(map(str, warns)), foreground="#a35").pack(anchor="w", padx=6, pady=4)

        # rationale
        rat = info.get("rationale") or {}
        rat_box = ttk.LabelFrame(frame, text="Rationale")
        rat_box.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        txt = tk.Text(rat_box, height=10, wrap="word")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", self._format_rationale(rat))
        txt.configure(state="disabled")

    def _format_rationale(self, rat: Any, indent: int = 0) -> str:
        pad = "  " * indent
        if isinstance(rat, dict):
            lines = []
            for k, v in rat.items():
                lines.append(f"{pad}{k}:")
                lines.append(self._format_rationale(v, indent + 1))
            return "\n".join(lines)
        if isinstance(rat, (list, tuple)):
            lines = []
            for i, v in enumerate(rat):
                prefix = f"{pad}- "
                if isinstance(v, (dict, list, tuple)):
                    lines.append(f"{prefix}")
                    lines.append(self._format_rationale(v, indent + 1))
                else:
                    lines.append(f"{prefix}{v}")
            return "\n".join(lines)
        try:
            if isinstance(rat, float):
                return f"{pad}{rat:.6f}"
            return f"{pad}{rat}"
        except Exception:
            return f"{pad}{str(rat)}"

    # Actions

    def _current_trace(self) -> Optional[Dict[str, Any]]:
        sel = self.tree.selection()
        if not sel:
            return None
        key = self.tree.set(sel[0], "pair_key")
        return self._traces.get(key)

    def _copy_selected_json(self):
        tr = self._current_trace()
        if not tr:
            return
        payload = json.dumps(tr, ensure_ascii=False, indent=2)
        try:
            self.clipboard_clear()
            self.clipboard_append(payload)
            self.update()
        except Exception as e:
            messagebox.showerror("Copy failed", str(e))

    def _save_selected_json(self):
        tr = self._current_trace()
        if not tr:
            return
        payload = json.dumps(tr, ensure_ascii=False, indent=2)
        try:
            path = filedialog.asksaveasfilename(
                title="Save trace JSON",
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
                initialfile=f"{(tr.get('pair_key') or 'trace').replace('|','_')}.json",
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
            messagebox.showinfo("Saved", f"Trace saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
