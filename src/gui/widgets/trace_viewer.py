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
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._doc_labels: Dict[str, str] = {}

        self._main_paned: Optional[ttk.PanedWindow] = None

        self._build_ui(text)

    # UI

    def _build_ui(self, text: str):
        # Top bar
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        ttk.Label(top, text=text, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)

        self.entry_filter = ttk.Entry(top, width=28)
        self.entry_filter.pack(side=tk.RIGHT, padx=(6, 0))
        self.entry_filter.insert(0, "")
        ttk.Button(top, text="Filter", command=self._apply_filter).pack(side=tk.RIGHT)

        # Main split
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self._main_paned = main
        main.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # Left: table
        left = ttk.Frame(main, padding=(0, 0, 6, 0))
        main.add(left, weight=3)

        cols = ("pair_key", "a_id", "b_id", "label", "exact_votes", "near_votes", "steps")
        self.tree = ttk.Treeview(
            left, columns=cols, show="headings", selectmode="browse", height=16
        )

        headings = [
            ("pair_key", "Pair"),
            ("a_id", "A"),
            ("b_id", "B"),
            ("label", "Final"),
            ("exact_votes", "Exact votes"),
            ("near_votes", "Near votes"),
            ("steps", "Escalations"),
        ]
        widths = {
            "pair_key": 140,
            "a_id": 200,
            "b_id": 200,
            "label": 120,
            "exact_votes": 140,
            "near_votes": 140,
            "steps": 120,
        }
        for cid, label in headings:
            self.tree.heading(cid, text=label)
            anchor = tk.W if cid in {"pair_key", "a_id", "b_id"} else (tk.CENTER if cid == "label" else tk.W)
            self.tree.column(cid, width=widths[cid], anchor=anchor, stretch=True)

        vs = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        hs = ttk.Scrollbar(left, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Right: detail
        right = ttk.Frame(main)
        main.add(right, weight=4)

        self.after_idle(self._initial_sash_position)

        self.bind("<Configure>", lambda _e: self.after_idle(self._enforce_min_right))

        # Action bar
        ab = ttk.Frame(right)
        ab.pack(fill=tk.X)
        ttk.Button(ab, text="Copy JSON", command=self._copy_selected_json).pack(side=tk.LEFT)
        ttk.Button(ab, text="Save JSON…", command=self._save_selected_json).pack(side=tk.LEFT, padx=(6, 0))

        # Summary card
        card = ttk.LabelFrame(right, text="Summary")
        card.pack(fill=tk.X, pady=(8, 6))
        self.lbl_pair = ttk.Label(card, text="Pair: —")
        self.lbl_final = ttk.Label(card, text="Final: —")
        self.lbl_ev = ttk.Label(card, text="Exact votes: —")
        self.lbl_nv = ttk.Label(card, text="Near votes: —")
        self.lbl_steps = ttk.Label(card, text="Escalation steps: —")
        for w in (self.lbl_pair, self.lbl_final, self.lbl_ev, self.lbl_nv, self.lbl_steps):
            w.pack(anchor="w")

        # Per-learner tabs
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _initial_sash_position(self):
        if not self._main_paned:
            return
        try:
            self._main_paned.sashpos(0, max(420, int(self.winfo_width() * 0.40)))
        except Exception:
            pass

    def _enforce_min_right(self):
        if not self._main_paned:
            return
        try:
            total = self._main_paned.winfo_width()
            desired = max(total - 480, int(total * 0.40))
            self._main_paned.sashpos(0, desired)
        except Exception:
            pass

    # Public API

    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self._traces.clear()
        self._order.clear()
        self._render_detail(None)
        self.update_idletasks()

    def set_traces(
        self,
        traces: Iterable[DecisionTrace | Dict[str, Any]],
        *,
        doc_labels: Optional[Dict[str, str]] = None,
    ):
        self.clear()
        if doc_labels:
            self._doc_labels = dict(doc_labels)
        for tr in traces:
            self.add_trace(tr)
        self.update_idletasks()

    def set_doc_labels(self, labels: Dict[str, str]):
        self._doc_labels = dict(labels or {})
        self._apply_filter()

    def add_trace(self, trace: DecisionTrace | Dict[str, Any]):
        t = trace.as_dict() if hasattr(trace, "as_dict") else dict(trace)
        key = t.get("pair_key") or f"{t.get('a_id','?')}||{t.get('b_id','?')}"
        self._traces[key] = t
        self._order.append(key)
        self._insert_row(t)

    def select_pair(self, pair_key: str):
        for iid in self.tree.get_children(""):
            if self.tree.set(iid, "pair_key") == pair_key:
                self.tree.selection_set(iid)
                self.tree.see(iid)
                self._render_detail(self._traces.get(pair_key))
                break

    # Internals

    def _fmt_doc(self, did: str) -> str:
        if not did or did == "—":
            return did or "—"
        name = self._doc_labels.get(did)
        return f"{name} ({did[:8]})" if name else did

    def _votes_from_trace_or_infer(self, tr: Dict[str, Any]) -> tuple[list[str], list[str]]:
        # 1) Arbiter-provided
        ev = list(tr.get("exact_voters") or [])
        nv = list(tr.get("near_voters") or [])
        if ev or nv:
            return ev, nv

        # 2) Fallback inference
        learners = tr.get("learners") or {}
        exact: List[str] = []
        near: List[str] = []
        for lname, info in learners.items():
            try:
                raw = float(info.get("raw_score"))
            except Exception:
                raw = None
            prob = info.get("prob")
            thr = info.get("threshold")
            voted_dup = False
            try:
                if prob is not None and thr is not None:
                    voted_dup = float(prob) >= float(thr)
            except Exception:
                voted_dup = False

            if raw is not None and abs(raw - 1.0) < 1e-9:
                exact.append(lname)
            elif voted_dup:
                near.append(lname)
        return exact, near

    def _final_text(self, tr: Dict[str, Any]) -> str:
        fl = (tr.get("final_label") or "").upper()
        if fl == "UNCERTAIN":
            return "Uncertain"
        if fl == "DUPLICATE":
            dk = (tr.get("dup_kind") or "").upper()
            if dk == "EXACT":
                return "Exact duplicate"
            if dk == "NEAR":
                return "Near duplicate"
            ev, _ = self._votes_from_trace_or_infer(tr)
            return "Exact duplicate" if ev else "Near duplicate"
        return "Uncertain"

    def _insert_row(self, tr: Dict[str, Any]):
        steps = ", ".join(tr.get("escalation_steps") or []) or "—"
        a_disp = self._fmt_doc(tr.get("a_id", "—"))
        b_disp = self._fmt_doc(tr.get("b_id", "—"))
        label_text = self._final_text(tr)
        exact_voters, near_voters = self._votes_from_trace_or_infer(tr)

        vals = (
            tr.get("pair_key", "—"),
            a_disp,
            b_disp,
            label_text,
            ", ".join(exact_voters) or "—",
            ", ".join(near_voters) or "—",
            steps,
        )
        self.tree.insert("", tk.END, values=vals)

    def _apply_filter(self):
        q = (self.entry_filter.get() or "").strip().lower()
        self.tree.delete(*self.tree.get_children())
        for key in self._order:
            tr = self._traces[key]
            a_id = tr.get("a_id", "")
            b_id = tr.get("b_id", "")
            a_disp = self._fmt_doc(a_id)
            b_disp = self._fmt_doc(b_id)
            hay = " ".join(
                [
                    tr.get("pair_key", ""),
                    a_id,
                    b_id,
                    a_disp,
                    b_disp,
                    self._final_text(tr),
                    " ".join(tr.get("escalation_steps") or []),
                    " ".join(tr.get("near_voters") or []),
                    " ".join(tr.get("exact_voters") or []),
                ]
            ).lower()
            if not q or q in hay:
                self._insert_row(tr)
        self.update_idletasks()

    def _on_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            self._render_detail(None)
            return
        iid = sel[0]
        key = self.tree.set(iid, "pair_key")
        self._render_detail(self._traces.get(key))

    def _render_detail(self, tr: Optional[Dict[str, Any]]):
        for i in reversed(range(self.nb.index("end") or 0)):
            self.nb.forget(i)

        if not tr:
            self.lbl_pair.configure(text="Pair: —")
            self.lbl_final.configure(text="Final: —")
            self.lbl_ev.configure(text="Exact votes: —")
            self.lbl_nv.configure(text="Near votes: —")
            self.lbl_steps.configure(text="Escalation steps: —")
            self.update_idletasks()
            return

        a_disp = self._fmt_doc(tr.get("a_id", "—"))
        b_disp = self._fmt_doc(tr.get("b_id", "—"))
        pair_key = tr.get("pair_key", "—")
        self.lbl_pair.configure(text=f"Pair: {a_disp}  ⇄  {b_disp}   ·   key: {pair_key}")

        self.lbl_final.configure(text=f"Final: {self._final_text(tr)}")

        ev, nv = self._votes_from_trace_or_infer(tr)
        self.lbl_ev.configure(text=f"Exact votes: {', '.join(ev) or '—'}")
        self.lbl_nv.configure(text=f"Near votes: {', '.join(nv) or '—'}")

        steps = ", ".join(tr.get("escalation_steps") or []) or "—"
        self.lbl_steps.configure(text=f"Escalation steps: {steps}")

        # learners
        learners = tr.get("learners") or {}
        for name, info in learners.items():
            self._add_learner_tab(name, info)
        if self.nb.index("end") > 0:
            self.nb.select(0)

        self.update_idletasks()

    def _add_learner_tab(self, name: str, info: Dict[str, Any]):
        frame = ttk.Frame(self.nb, padding=8)
        self.nb.add(frame, text=name)

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
        ttk.Label(top, text=f"thr:  {_fmt(info.get('threshold'))}").pack(side=tk.LEFT, padx=(0, 12))

        warns = info.get("warnings") or []
        if warns:
            warn_box = ttk.LabelFrame(frame, text="Warnings")
            warn_box.pack(fill=tk.X, pady=(6, 0))
            ttk.Label(warn_box, text="; ".join(map(str, warns)), foreground="#a35").pack(
                anchor="w", padx=6, pady=4
            )

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
            for v in rat:
                if isinstance(v, (dict, list, tuple)):
                    lines.append(f"{pad}-")
                    lines.append(self._format_rationale(v, indent + 1))
                else:
                    lines.append(f"{pad}- {v}")
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
