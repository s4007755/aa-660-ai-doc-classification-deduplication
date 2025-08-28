from __future__ import annotations

import sys
import threading
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from src.pipelines.ingestion import extract_document
from src.pipelines.normalization import normalize_text
from src.pipelines.near_duplicate import (
    Document,
    detect_near_duplicates_with_scores,
)

from src.storage.sqlite_store import (
    init_db,
    upsert_document,
    add_file_mapping,
    mark_dirty,
    delete_documents,
    get_all_documents,
    insert_cluster,
    get_docs_text,
    get_docs_text_by_ids,
)

# Types
SUPPORTED = {".pdf", ".docx", ".txt"}


@dataclass
class ProcResult:
    doc_id: str
    filepath: str
    language: Optional[str]
    size: int
    normalized_text: str


# GUI
class DedupeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Duplicate Finder")
        self.geometry("1380x860")
        self.minsize(1120, 680)

        # init database
        init_db()

        # runtime state
        self.paths: List[Path] = []
        self.running = False
        self.status = tk.StringVar(value="")

        # processing toggles
        self.var_simhash = tk.BooleanVar(value=True)
        self.var_minhash = tk.BooleanVar(value=True)
        self.var_embed = tk.BooleanVar(value=True)
        self.var_model = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")

        self._build_ui()
        self._style_ui()
        self.refresh_db_tab()
        self.refresh_exact_tab()

    # UI
    def _build_ui(self):
        # Top row
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Add files or a folder").pack(side=tk.LEFT)

        # Buttons row
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(btns, text="Add Folder", command=self.on_add_folder).pack(side=tk.LEFT)
        ttk.Button(btns, text="Add Files", command=self.on_add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Clear Selection", command=self.on_clear_selection).pack(side=tk.LEFT, padx=5)
        ttk.Separator(btns, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Run behavior
        self.run_btn = ttk.Button(btns, text="Run", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT)

        # Explicit DB-only all-docs button
        ttk.Button(btns, text="Re-cluster (DB only)", command=self.on_recluster_db_all).pack(side=tk.LEFT, padx=5)

        ttk.Separator(btns, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(btns, text="Delete Selected From DB", command=self.on_delete_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Mark Selected as Dirty", command=self.on_mark_dirty).pack(side=tk.LEFT, padx=5)

        # Processing options
        opts = ttk.LabelFrame(self, text="Processing options", padding=8)
        opts.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Checkbutton(opts, text="SimHash", variable=self.var_simhash).pack(side=tk.LEFT)
        ttk.Checkbutton(opts, text="MinHash", variable=self.var_minhash).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(opts, text="Embeddings", variable=self.var_embed).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(opts, text="Model:").pack(side=tk.LEFT, padx=(20, 4))
        self.model_combo = ttk.Combobox(
            opts, textvariable=self.var_model, width=48, values=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L12-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2",
                "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            ]
        )
        self.model_combo.pack(side=tk.LEFT)

        # Progress and status
        prog_frame = ttk.Frame(self)
        prog_frame.pack(fill=tk.X, padx=10)
        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.status_label = ttk.Label(prog_frame, textvariable=self.status, width=60, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

        # Notebook
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        # Database tab
        self.db_tab = ttk.Frame(self.nb)
        self.nb.add(self.db_tab, text="Database")
        cols = ("doc_id", "file", "size", "lang", "dirty")
        self.tree_db = ttk.Treeview(self.db_tab, columns=cols, show="headings", height=18, selectmode="extended")
        headings = [
            ("doc_id", "DocID (sha256:12)", 180, tk.W),
            ("file", "File path", 900, tk.W),
            ("size", "Size (KB)", 100, tk.E),
            ("lang", "Lang", 80, tk.CENTER),
            ("dirty", "Dirty", 60, tk.CENTER),
        ]
        for cid, label, width, anchor in headings:
            self.tree_db.heading(cid, text=label)
            self.tree_db.column(cid, width=width, anchor=anchor, stretch=(cid not in {"size", "lang", "dirty"}))
        db_v = ttk.Scrollbar(self.db_tab, orient="vertical", command=self.tree_db.yview)
        db_h = ttk.Scrollbar(self.db_tab, orient="horizontal", command=self.tree_db.xview)
        self.tree_db.configure(yscrollcommand=db_v.set, xscrollcommand=db_h.set)
        self.tree_db.grid(row=0, column=0, sticky="nsew")
        db_v.grid(row=0, column=1, sticky="ns")
        db_h.grid(row=1, column=0, sticky="ew")
        self.db_tab.rowconfigure(0, weight=1)
        self.db_tab.columnconfigure(0, weight=1)

        # Tab: Exact duplicates
        self.exact_tab = ttk.Frame(self.nb)
        self.nb.add(self.exact_tab, text="Exact duplicates")
        self.tree_exact = ttk.Treeview(
            self.exact_tab,
            columns=("cluster", "count", "doc_id", "filepath"),
            show="headings",
            height=18,
            selectmode="browse"
        )
        for cid, label, width, anchor in [
            ("cluster", "Cluster #", 100, tk.CENTER),
            ("count", "Files", 80, tk.CENTER),
            ("doc_id", "DocID (sha256)", 280, tk.W),
            ("filepath", "File path", 900, tk.W),
        ]:
            self.tree_exact.heading(cid, text=label)
            self.tree_exact.column(cid, width=width, anchor=anchor, stretch=(cid in {"doc_id", "filepath"}))
        ex_v = ttk.Scrollbar(self.exact_tab, orient="vertical", command=self.tree_exact.yview)
        ex_h = ttk.Scrollbar(self.exact_tab, orient="horizontal", command=self.tree_exact.xview)
        self.tree_exact.configure(yscrollcommand=ex_v.set, xscrollcommand=ex_h.set)
        self.tree_exact.grid(row=0, column=0, sticky="nsew")
        ex_v.grid(row=0, column=1, sticky="ns")
        ex_h.grid(row=1, column=0, sticky="ew")
        self.exact_tab.rowconfigure(0, weight=1)
        self.exact_tab.columnconfigure(0, weight=1)

        # Tab: Near-duplicate clusters
        self.near_tab = ttk.Frame(self.nb)
        self.nb.add(self.near_tab, text="Near-duplicate clusters (latest run)")
        self.tree_near = ttk.Treeview(
            self.near_tab,
            columns=("cluster", "similarity", "doc_id", "filepath"),
            show="headings",
            height=18,
        )
        for cid, label, width, anchor in [
            ("cluster", "Cluster #", 100, tk.CENTER),
            ("similarity", "Composite", 120, tk.CENTER),
            ("doc_id", "DocID", 320, tk.W),
            ("filepath", "File path", 820, tk.W),
        ]:
            self.tree_near.heading(cid, text=label)
            self.tree_near.column(cid, width=width, anchor=anchor, stretch=(cid in {"doc_id", "filepath"}))
        nr_v = ttk.Scrollbar(self.near_tab, orient="vertical", command=self.tree_near.yview)
        nr_h = ttk.Scrollbar(self.near_tab, orient="horizontal", command=self.tree_near.xview)
        self.tree_near.configure(yscrollcommand=nr_v.set, xscrollcommand=nr_h.set)
        self.tree_near.grid(row=0, column=0, sticky="nsew")
        nr_v.grid(row=0, column=1, sticky="ns")
        nr_h.grid(row=1, column=0, sticky="ew")
        self.near_tab.rowconfigure(0, weight=1)
        self.near_tab.columnconfigure(0, weight=1)

    def _style_ui(self):
        style = ttk.Style(self)
        try:
            if "azure" in style.theme_names():
                style.theme_use("azure")
        except Exception:
            pass
        style.configure("Treeview", rowheight=22)

    # Actions
    def on_add_folder(self):
        folder = filedialog.askdirectory(title="Select a folder of documents")
        if not folder:
            return
        self.paths.append(Path(folder))
        self.set_status(f"Added folder: {folder}")

    def on_add_files(self):
        files = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[
                ("Supported", "*.pdf *.docx *.txt"),
                ("PDF", "*.pdf"),
                ("Word", "*.docx"),
                ("Text", "*.txt"),
                ("All", "*.*"),
            ],
        )
        if not files:
            return
        self.paths.extend([Path(f) for f in files])
        self.set_status(f"Added {len(files)} files")

    def on_clear_selection(self):
        self.paths = []
        self.set_status("Selection cleared")

    def on_delete_selected(self):
        items = self.tree_db.selection()
        if not items:
            messagebox.showinfo("Delete", "Select rows in the Database tab.")
            return
        doc_ids = [self.tree_db.set(it, "doc_id") for it in items]
        if not messagebox.askyesno("Confirm", f"Delete {len(doc_ids)} selected documents from DB?"):
            return
        delete_documents(doc_ids)
        self.refresh_db_tab()
        self.refresh_exact_tab()

    def on_mark_dirty(self):
        items = self.tree_db.selection()
        if not items:
            messagebox.showinfo("Mark Dirty", "Select rows in the Database tab.")
            return
        doc_ids = [self.tree_db.set(it, "doc_id") for it in items]
        mark_dirty(doc_ids)
        messagebox.showinfo("Mark Dirty", f"Marked {len(doc_ids)} docs dirty.")

    def on_run(self):
        if self.running:
            return

        selected_items = self.tree_db.selection()
        if selected_items:
            doc_ids = list({self.tree_db.set(it, "doc_id") for it in selected_items})
            self._start_thread(self._recluster_db_selected_worker, args=(doc_ids,))
            return

        files = self._iter_files_from_paths(self.paths)
        if files:
            self._start_thread(self._ingest_then_cluster_worker, args=(files,))
            return

        messagebox.showinfo("Nothing to run", "Select docs in the Database tab or add files/folders, then click Run.")

    def on_recluster_db_all(self):
        if self.running:
            return
        self._start_thread(self._recluster_db_all_worker)

    # Thread helpers
    def _start_thread(self, target, args: tuple = ()):
        self.running = True
        self.progress.configure(value=0, maximum=1)
        self.run_btn.configure(state=tk.DISABLED)
        threading.Thread(target=target, args=args, daemon=True).start()

    def _finish_run(self):
        self.after(0, lambda: self.run_btn.configure(state=tk.NORMAL))
        self.running = False

    # Workers
    def _ingest_then_cluster_worker(self, files: List[Path]):
        try:
            self.after(0, lambda: self.set_status("Reading and normalizing documents"))
            processed: List[ProcResult] = []
            for i, f in enumerate(files, 1):
                try:
                    doc = extract_document(str(f))
                    raw_text = doc["raw_text"] or ""
                    normalized = normalize_text(raw_text) or ""
                    meta = doc["metadata"] or {}
                    doc_id = meta["hash"]
                    upsert_document(
                        doc_id=doc_id,
                        raw_text=raw_text,
                        normalized_text=normalized,
                        meta={"language": meta.get("language"), "filesize": meta.get("filesize", 0)},
                    )
                    st = f.stat()
                    add_file_mapping(doc_id, str(f), int(st.st_mtime_ns))
                    processed.append(
                        ProcResult(
                            doc_id=doc_id,
                            filepath=str(f),
                            language=meta.get("language"),
                            size=int(meta.get("filesize") or 0),
                            normalized_text=normalized,
                        )
                    )
                except Exception as e:
                    self.after(0, lambda err=e, pf=f: self.set_status(f"Error: {Path(pf).name}: {err}"))
                finally:
                    self.after(0, lambda val=i, pf=f: (self.progress.configure(value=val),
                                                       self.set_status(f"Processed: {Path(pf).name}")))

            # cluster only ingested docs
            docs = [Document(doc_id=r.doc_id, normalized_text=r.normalized_text)
                    for r in processed if (r.normalized_text or "").strip()]
            self._cluster_docs(docs, source_tag="ingest")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.set_status("Error"))
        finally:
            self._finish_run()

    def _recluster_db_all_worker(self):
        try:
            self.after(0, lambda: self.set_status("Re-clustering all docs from DB"))
            pairs = get_docs_text(include_dirty=False)
        # build docs list
            docs = [Document(doc_id=did, normalized_text=txt or "") for (did, txt) in pairs if (txt or "").strip()]
            self._cluster_docs(docs, source_tag="db-all")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.set_status("Error"))
        finally:
            self._finish_run()

    def _recluster_db_selected_worker(self, doc_ids: List[str]):
        try:
            self.after(0, lambda: self.set_status(f"Re-clustering {len(doc_ids)} selected docs from DB"))
            pairs = get_docs_text_by_ids(doc_ids, include_dirty=False)
            docs = [Document(doc_id=did, normalized_text=txt or "") for (did, txt) in pairs if (txt or "").strip()]
            if not docs:
                self.after(0, lambda: self.set_status("No eligible selected docs"))
                return
            self._cluster_docs(docs, source_tag="db-selected")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.set_status("Error"))
        finally:
            self._finish_run()

    # Core clustering
    def _cluster_docs(self, docs: List[Document], *, source_tag: str):
        if not docs:
            self.after(0, lambda: self.set_status("No documents to cluster"))
            return

        self.after(0, lambda: self.set_status("Detecting near-duplicates"))
        infos, pairs = detect_near_duplicates_with_scores(
            docs,
            enable_simhash=self.var_simhash.get(),
            enable_minhash=self.var_minhash.get(),
            enable_embeddings=self.var_embed.get(),
            embed_model_name=self.var_model.get(),
        )

        # Render Near tab now
        self.tree_near.delete(*self.tree_near.get_children())
        cluster_idx = 0
        for info in infos:
            members = info.get("doc_ids", [])
            if len(members) <= 1:
                continue
            cluster_idx += 1

            # Compute composite from actual pairwise scores within the cluster
            members_set = set(members)
            cluster_pairs = [(a, b, s, j, c) for (a, b, s, j, c) in pairs if a in members_set and b in members_set]

            scores: List[float] = []
            for _a, _b, s, j, c in cluster_pairs:
                for v in (s, j, c):
                    if isinstance(v, (int, float)) and v == v:
                        scores.append(float(v))

            if scores:
                composite = sum(scores) / len(scores)
            else:
                # Fallback to metrics avgs if pairwise scores aren't present
                m = info.get("metrics", {})
                avgs = [float(m[k]["avg"]) for k in ("simhash", "minhash", "embedding") if k in m and m[k]["n_pairs"] > 0]
                if avgs and max(avgs) > 1.0:
                    avgs = [min(100.0, max(0.0, v)) / 100.0 for v in avgs]
                composite = float(sum(avgs) / len(avgs)) if avgs else 0.0

            if composite >= 1.0:
                pct_str = "100.0%"
            else:
                pct = math.floor(max(0.0, min(composite, 0.999999)) * 1000) / 10.0
                pct_str = f"{pct:.1f}%"

            id_to_paths: Dict[str, List[str]] = {}
            for d in get_all_documents():
                id_to_paths[d["doc_id"]] = d.get("filepaths") or []

            self.tree_near.insert("", tk.END, values=(cluster_idx, pct_str, "", ""))

            for did in members:
                paths = id_to_paths.get(did) or ["(path unknown)"]
                for p in paths:
                    self.tree_near.insert("", tk.END, values=(cluster_idx, "", did, p))

        # Persist clusters
        created_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for info in infos:
            if len(info["doc_ids"]) <= 1:
                continue
            cfg = {
                "created_at": created_ts,
                "source": source_tag,
                "thresholds": info.get("thresholds", {}),
                "config": info.get("config", {}),
            }
            members = set(info["doc_ids"])
            cluster_pairs = [(a, b, s, j, c) for (a, b, s, j, c) in pairs if a in members and b in members]
            insert_cluster(config=str(cfg), members=info["doc_ids"], pairwise=cluster_pairs)

        self.after(0, self.refresh_db_tab)
        self.after(0, self.refresh_exact_tab)
        self.after(0, lambda: self.set_status("Clustering complete"))

    # Exact duplicates tab
    def refresh_exact_tab(self):

        self.tree_exact.delete(*self.tree_exact.get_children())

        pairs: List[Tuple[str, str]] = get_docs_text(include_dirty=True)
        if not pairs:
            self.tree_exact.insert("", tk.END, values=("—", 0, "", ""))
            return

        id_to_paths: Dict[str, List[str]] = {}
        for d in get_all_documents():
            id_to_paths[d["doc_id"]] = d.get("filepaths") or []

        # Group by normalized text hash
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for did, txt in pairs:
            norm = txt or ""
            h = hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()
            groups.setdefault(h, []).append((did, norm))

        clusters_render: List[Tuple[int, List[Tuple[str, str]], List[Tuple[str, str]]]] = []

        for _h, members in groups.items():
            # Collect (doc_id, path) across all doc_ids in this text group
            expanded: List[Tuple[str, str]] = []
            for (did, _norm) in members:
                paths = id_to_paths.get(did) or []
                if not paths:
                    expanded.append((did, "(path unknown)"))
                else:
                    for p in paths:
                        expanded.append((did, p))
            # Deduplicate by (did, path)
            seen = set()
            expanded = [x for x in expanded if not (x in seen or seen.add(x))]

            expanded_count = len(expanded)
            if expanded_count <= 1:
                continue  # not a duplicate

            clusters_render.append((expanded_count, members, expanded))

        # Stable ordering
        clusters_render.sort(key=lambda x: (-x[0], x[1][0][0] if x[1] else ""))

        if not clusters_render:
            self.tree_exact.insert("", tk.END, values=("—", 0, "", ""))
            return

        # Render
        for idx, (expanded_count, members, expanded) in enumerate(clusters_render, 1):
            # Header row
            self.tree_exact.insert("", tk.END, values=(idx, expanded_count, "", ""))
            # Member rows: list each doc_id and each filepath
            for did, path in expanded:
                self.tree_exact.insert("", tk.END, values=(idx, "", did, path))

    # Helpers
    def refresh_db_tab(self):
        self.tree_db.delete(*self.tree_db.get_children())
        docs = get_all_documents()
        for d in docs:
            short = d["doc_id"][:12]
            size_kb = f"{int((d.get('filesize') or 0) / 1024)}"
            lang = d.get("language") or ""
            dirty = "Y" if d.get("is_dirty") else ""
            paths = d.get("filepaths") or [""]
            for path in paths:
                row_id = self.tree_db.insert("", tk.END, values=(short, path, size_kb, lang, dirty))
                self.tree_db.set(row_id, "doc_id", d["doc_id"])

    def _iter_files_from_paths(self, paths: List[Path]) -> List[Path]:
        out: List[Path] = []
        for p in paths:
            if p.is_dir():
                for q in p.rglob("*"):
                    if q.is_file() and q.suffix.lower() in SUPPORTED:
                        out.append(q)
            elif p.is_file() and p.suffix.lower() in SUPPORTED:
                out.append(p)
        uniq, seen = [], set()
        for f in out:
            ap = str(f.resolve())
            if ap not in seen:
                uniq.append(f); seen.add(ap)
        return uniq

    def set_status(self, msg: str):
        self.status.set(msg)
        self.update_idletasks()


if __name__ == "__main__":
    app = DedupeApp()
    app.mainloop()
