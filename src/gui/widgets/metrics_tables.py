# src/gui/widgets/metrics_tables.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import tkinter as tk
from tkinter import ttk

def _fmt(x: Optional[float], prec: int = 3) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{prec}f}"
    except Exception:
        return "—"

def _clear(tree: ttk.Treeview) -> None:
    for iid in tree.get_children():
        tree.delete(iid)

class MetricsTables(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Clusters
        self._tab_clusters = ttk.Frame(self.nb)
        self.nb.add(self._tab_clusters, text="Clusters")
        cl_wrap = ttk.Frame(self._tab_clusters, padding=6); cl_wrap.pack(fill=tk.BOTH, expand=True)
        self.clusters = ttk.Treeview(
            cl_wrap,
            columns=("idx","size","members","avg_sim","avg_min","avg_emb","disp_sim","disp_min","disp_emb"),
            show="headings", height=12
        )
        for k, label, w, a in [
            ("idx","#",60,"e"),("size","Size",70,"e"),("members","Members",520,"w"),
            ("avg_sim","Avg simhash prob",140,"e"),("avg_min","Avg minhash prob",140,"e"),
            ("avg_emb","Avg embedding prob",160,"e"),
            ("disp_sim","Disp simhash (min–max)",170,"center"),
            ("disp_min","Disp minhash (min–max)",170,"center"),
            ("disp_emb","Disp embed (min–max)",170,"center"),
        ]:
            self.clusters.heading(k, text=label); self.clusters.column(k, width=w, anchor=a)
        vsb1 = ttk.Scrollbar(cl_wrap, orient="vertical", command=self.clusters.yview)
        self.clusters.configure(yscrollcommand=vsb1.set)
        self.clusters.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb1.pack(side=tk.LEFT, fill=tk.Y)

        # Thresholds
        self._tab_thresholds = ttk.Frame(self.nb)
        self.nb.add(self._tab_thresholds, text="Thresholds")
        th_wrap = ttk.Frame(self._tab_thresholds, padding=6); th_wrap.pack(fill=tk.BOTH, expand=True)
        self.thresholds = ttk.Treeview(
            th_wrap,
            columns=("learner","thr","prec","rec","f1","support","near_band"),
            show="headings", height=12
        )
        for k, label, w, a in [
            ("learner","Learner",160,"w"),("thr","Threshold",110,"e"),
            ("prec","Precision",110,"e"),("rec","Recall",110,"e"),
            ("f1","F1",90,"e"),("support","Support",90,"e"),
            ("near_band","Near-band share",140,"e"),
        ]:
            self.thresholds.heading(k, text=label); self.thresholds.column(k, width=w, anchor=a)
        vsb2 = ttk.Scrollbar(th_wrap, orient="vertical", command=self.thresholds.yview)
        self.thresholds.configure(yscrollcommand=vsb2.set)
        self.thresholds.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb2.pack(side=tk.LEFT, fill=tk.Y)

        # Confusion
        self._tab_conf = ttk.Frame(self.nb)
        self.nb.add(self._tab_conf, text="Confusion")
        cf_wrap = ttk.Frame(self._tab_conf, padding=6); cf_wrap.pack(fill=tk.BOTH, expand=True)
        self.confusion = ttk.Treeview(
            cf_wrap,
            columns=("learner","tp","fp","tn","fn","prec","rec","f1"),
            show="headings", height=12
        )
        for k, label, w, a in [
            ("learner","Learner",160,"w"),("tp","TP",70,"e"),("fp","FP",70,"e"),
            ("tn","TN",70,"e"),("fn","FN",70,"e"),
            ("prec","Precision",110,"e"),("rec","Recall",110,"e"),("f1","F1",90,"e"),
        ]:
            self.confusion.heading(k, text=label); self.confusion.column(k, width=w, anchor=a)
        vsb3 = ttk.Scrollbar(cf_wrap, orient="vertical", command=self.confusion.yview)
        self.confusion.configure(yscrollcommand=vsb3.set)
        self.confusion.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb3.pack(side=tk.LEFT, fill=tk.Y)

        # Examples (FP/FN)
        self._tab_examples = ttk.Frame(self.nb)
        self.nb.add(self._tab_examples, text="Examples")
        ex_nb = ttk.Notebook(self._tab_examples); ex_nb.pack(fill=tk.BOTH, expand=True)

        self.fp_tree = ttk.Treeview(
            ex_nb,
            columns=("learner","pair_key","a_id","b_id","prob","thr"),
            show="headings", height=12
        )
        for k, label, w, a in [
            ("learner","Learner",120,"w"),("pair_key","Pair",180,"w"),
            ("a_id","A",220,"w"),("b_id","B",220,"w"),
            ("prob","Prob",90,"e"),("thr","Thr",90,"e"),
        ]:
            self.fp_tree.heading(k, text=label); self.fp_tree.column(k, width=w, anchor=a)
        fp_wrap = ttk.Frame(ex_nb, padding=6); fp_wrap.pack(fill=tk.BOTH, expand=True)
        vsb4 = ttk.Scrollbar(fp_wrap, orient="vertical", command=self.fp_tree.yview)
        self.fp_tree.configure(yscrollcommand=vsb4.set)
        self.fp_tree.pack(in_=fp_wrap, side=tk.LEFT, fill=tk.BOTH, expand=True); vsb4.pack(in_=fp_wrap, side=tk.LEFT, fill=tk.Y)
        ex_nb.add(fp_wrap, text="False Positives")

        self.fn_tree = ttk.Treeview(
            ex_nb,
            columns=("learner","pair_key","a_id","b_id","prob","thr"),
            show="headings", height=12
        )
        for k, label, w, a in [
            ("learner","Learner",120,"w"),("pair_key","Pair",180,"w"),
            ("a_id","A",220,"w"),("b_id","B",220,"w"),
            ("prob","Prob",90,"e"),("thr","Thr",90,"e"),
        ]:
            self.fn_tree.heading(k, text=label); self.fn_tree.column(k, width=w, anchor=a)
        fn_wrap = ttk.Frame(ex_nb, padding=6); fn_wrap.pack(fill=tk.BOTH, expand=True)
        vsb5 = ttk.Scrollbar(fn_wrap, orient="vertical", command=self.fn_tree.yview)
        self.fn_tree.configure(yscrollcommand=vsb5.set)
        self.fn_tree.pack(in_=fn_wrap, side=tk.LEFT, fill=tk.BOTH, expand=True); vsb5.pack(in_=fn_wrap, side=tk.LEFT, fill=tk.Y)
        ex_nb.add(fn_wrap, text="False Negatives")

    # public API
    def update_tables(self, snapshot: Dict[str, Any]) -> None:
        self._fill_clusters(snapshot.get("clusters") or [])
        self._fill_thresholds(snapshot.get("thresholds") or {})
        self._fill_confusion(snapshot.get("confusion") or {})
        ex = snapshot.get("examples") or {}
        self._fill_examples(ex.get("false_positives") or [], ex.get("false_negatives") or [])

    # fillers
    def _fill_clusters(self, rows: List[Dict[str, Any]]) -> None:
        _clear(self.clusters)
        for r in rows:
            members = ", ".join(r.get("members", []))
            if len(members) > 240: members = members[:240] + "…"
            ds = r.get("dispersion_simhash", {}); dm = r.get("dispersion_minhash", {}); de = r.get("dispersion_embedding", {})
            self.clusters.insert("", tk.END, values=(
                r.get("cluster_index","—"), r.get("size","—"), members or "—",
                _fmt(r.get("avg_simhash_prob")), _fmt(r.get("avg_minhash_prob")), _fmt(r.get("avg_embedding_prob")),
                f"{_fmt(ds.get('min'))} – {_fmt(ds.get('max'))}",
                f"{_fmt(dm.get('min'))} – {_fmt(dm.get('max'))}",
                f"{_fmt(de.get('min'))} – {_fmt(de.get('max'))}",
            ))

    def _fill_thresholds(self, rep: Dict[str, Any]) -> None:
        _clear(self.thresholds)
        for learner in sorted(rep.keys()):
            info = rep.get(learner, {})
            self.thresholds.insert("", tk.END, values=(
                learner,
                _fmt(info.get("threshold")), _fmt(info.get("precision")),
                _fmt(info.get("recall")), _fmt(info.get("f1")),
                int(info.get("support", 0)), _fmt(info.get("near_band_share")),
            ))

    def _fill_confusion(self, rep: Dict[str, Any]) -> None:
        _clear(self.confusion)
        for learner in sorted(rep.keys()):
            e = rep.get(learner, {})
            self.confusion.insert("", tk.END, values=(
                learner, int(e.get("tp", 0)), int(e.get("fp", 0)),
                int(e.get("tn", 0)), int(e.get("fn", 0)),
                _fmt(e.get("precision")), _fmt(e.get("recall")), _fmt(e.get("f1"))
            ))

    def _fill_examples(self, fp_rows: List[Dict[str, Any]], fn_rows: List[Dict[str, Any]]) -> None:
        _clear(self.fp_tree); _clear(self.fn_tree)
        for r in fp_rows:
            self.fp_tree.insert("", tk.END, values=(
                r.get("learner",""), r.get("pair_key",""), r.get("a_id",""), r.get("b_id",""),
                _fmt(r.get("prob")), _fmt(r.get("threshold"))
            ))
        for r in fn_rows:
            self.fn_tree.insert("", tk.END, values=(
                r.get("learner",""), r.get("pair_key",""), r.get("a_id",""), r.get("b_id",""),
                _fmt(r.get("prob")), _fmt(r.get("threshold"))
            ))
