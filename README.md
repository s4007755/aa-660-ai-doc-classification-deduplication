# AA-660 Doc AI

Document deduplication with a friendly Tkinter GUI, fast duplicate/near-duplicate detection (SimHash, MinHash, Embeddings), an arbiter/consensus layer, lightweight SQLite storage and a one-click HTML report.

Built for day to day cleanup of messy document sets. Unsupervised by default, with a simple calibration bootstrap from exact duplicates.

## Features

- **Multi-signal duplicate detection**

  - SimHash (character/word shingles)

  - MinHash and LSH (Jaccard on shingles)

  - Sentence-Transformer embeddings (cosine similarity)

- **Arbiter & escalation**

  - Combines learners, applies thresholds

  - Sends uncertain cases to “escalation” (higher scrutiny)

- **Unsupervised calibration (bootstrap)**

  - Builds a temporary set of exact duplicates for per-learner calibration

  - Produces per-learner thresholds & Brier estimates

- **GUI workflow**

  - Documents tab: ingest files/folders

  - Main tab: configure & run pipeline

  - Decision Traces: per-pair reasoning (raw score, probability, escalation steps)

  - Metrics: run summary and per-learner snapshot

  - Run History: notes, quick metrics, open HTML report

- **Reports**

  - Export a professional HTML report with config snapshot, calibration table, examples and clusters

## Requirements

- Windows (for the packaged .exe) or Python environment for development

- Python 3.11 recommended for development (3.9+ supported)

- First run will download NLP models

## Quickstart (GUI)
1) Create & activate a virtual environment

- PowerShell (Windows):

python -m venv .venv
.\.venv\Scripts\Activate.ps1

- macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

2) Install deps
pip install -e .[dev]

3) Run the GUI
python -m src.app


Workflow inside the app:

1) Open Documents tab, Add Files… or Add Folder…

2) Go to Main tab, choose a profile (Balanced / High Precision / Recall-Heavy)

3) Click Run

4) Explore Decision Traces, Metrics and Run History

5) In Run History, click Open report to view the HTML report

## Quickstart - CLI (headless)

> Short, practical commands for running the pipeline without the GUI.  
>
> If you installed with `pip install -e .[dev]`, a console script **`dupfinder`** is available. Otherwise, use the module form `python -m src.cli_nd`.

**Run the pipeline on a folder and emit a report**
```bash
dupfinder run --docs ./docs --preset balanced --perf-preset auto --report-dir ./reports
# Fallback form (if console script not available):
python -m src.cli_nd run --docs ./docs --preset balanced --perf-preset auto --report-dir ./reports
```

**Generate a fresh HTML report for a past run**
```bash
dupfinder report 42 --out-dir ./reports
# Fallback:
python -m src.cli_nd report 42 --out-dir ./reports
```

**List documents in the DB**
```bash
dupfinder list --max 50
# Fallback:
python -m src.cli_nd list --max 50
```

**Notes**
- `--preset` accepts: `balanced`, `high`, `recall`.
- Reports are written under `reports/run_<RUN_ID>_<TIMESTAMP>.html`.
- The CLI mirrors the GUI’s defaults; flags are optional in simple cases.

## CLI quick reference

Run `dupfinder --help` or `dupfinder <cmd> --help` for full details.

### Commands

| Command | What it does |
|---|---|
| `run` | Run the pipeline on provided sources (folder / JSON / API) or the existing DB via `--use-db`. Writes an HTML report. |
| `run-db` | Run the pipeline using documents already in the SQLite DB. |
| `run-folder <FOLDER>` | Convenience: run the pipeline directly on a folder of `.pdf/.docx/.txt`. |
| `ingest <FOLDER>` | Add a folder of files into the DB (no pipeline run). |
| `list` | List documents currently in the DB (first N rows). |
| `wipe` | Delete **all** documents from the DB (dangerous). |
| `report <RUN_ID>` | Re-generate an HTML report for a previous run. |
| `import-csv <PATH>` | Import up to N rows from a CSV with a `text` column into the DB. |

### Common examples

**Run on a folder (balanced, calibration off by default)**
```bash
dupfinder run-folder ./docs --preset balanced --report-dir ./reports
```

**Run pulling docs from the DB**
```bash
dupfinder run-db --preset high --calibration --report-dir ./reports
```

**Run with in-memory sources**
```bash
# From a folder (without persisting to DB)
dupfinder run --docs ./docs --preset recall --report-dir ./reports

# From a JSON file (list or {items:[...]}) 
dupfinder run --json-file ./my_docs.json --preset balanced --report-dir ./reports

# From an HTTP API
dupfinder run --api-url "https://api.example.com/my-docs" --api-headers '{"Authorization":"Bearer TOKEN"}' --preset balanced --report-dir ./reports
```

**Generate a fresh report for a past run**
```bash
dupfinder report 42 --out-dir ./reports
```

**DB utilities**
```bash
dupfinder ingest ./docs # add files into DB (no run)
dupfinder list --max 50 # preview what's inside
dupfinder wipe --yes # irreversibly delete all documents
dupfinder import-csv ./rows.csv --limit 2000
```

### Performance & tuning (advanced)

Good defaults work for most users.

```bash
# Try a faster “auto” perf profile, bump MinHash perms and use a specific embed model
dupfinder run-folder ./docs --perf-preset auto   --minhash-perm 128 --model-name all-MiniLM-L6-v2   --cand-per-doc 3000 --cand-total 5000000   --lsh-threshold 0.65 --shingle-size 3   --simhash-bits 128 --simhash-mode wshingle --simhash-wshingle 3
```

**Key flags** you might care about:
- `--preset {balanced|high|recall}` – profile thresholds.  
- `--calibration` – enable bootstrap calibration (OFF by default in CLI).  
- **Perf**: `--perf-preset {auto|high-end|medium|light|high-throughput|high-recall|custom}`, `--workers`, `--emb-batch`, `--minhash-perm`, `--cand-per-doc`, `--cand-total`, `--model-name`.  
- **SimHash**: `--simhash-bits`, `--simhash-mode {unigram|wshingle|cngram}`, `--simhash-wshingle`, `--simhash-cngram`, `--simhash-posbucket`, `--simhash-minlen`, `--simhash-maxw`, `--simhash-norm-strict` / `--no-simhash-norm-strict`, `--simhash-strip-ids` / `--no-simhash-strip-ids`.  
- **Candidates**: `--lsh-threshold`, `--shingle-size`.  
- **Self-training**: `--sl-epochs`, `--no-self-training`.  

Reports are written as `reports/run_<RUN_ID>_<TIMESTAMP>.html`.


## Reports (what you'll see)
- **Run summary**: total pairs, duplicates (exact/near), uncertain, consensus %, escalations %  
- **Per‑learner snapshot**: N, positive rate, AUC, Brier, threshold  
- **Calibration snapshot**: method, threshold, Brier, reliability points  
- **Charts** per learner: reliability, ROC, PR, threshold sweep, score histogram  
- **Examples**: easy positives, escalated/hard, uncertain  
- **Clusters**: groups of documents connected by duplicate decisions  
- **Config**: full JSON snapshot for reproducibility

Open via Run History -> Open report or by calling `generate_report(...)`.


## How it works

- Ingestion extracts text & metadata, stores them in SQLite (src/storage/sqlite_store.py).

- Learners (SimHash, MinHash, Embeddings) each output a raw similarity which the learner maps to a probability of “duplicate”.

- The Arbiter fuses these probabilities using configured thresholds and a gray zone margin:

  - High confidence -> DUPLICATE or NON_DUPLICATE

  - Ambiguous -> UNCERTAIN (escalation path)

- Calibration (bootstrap): we build a temporary labeled set from exact duplicate text matches found in your corpus:

  - Positives = pairs with identical normalized text

  - Negatives = pairs from different text groups

  - Learners fit simple calibration to convert raw scores to probabilities.

  - We record threshold and Brier score per learner.

- Metrics & Report:

  - Run summary: total pairs, duplicates, uncertain, consensus rate, escalations

  - Per-learner snapshot (N, pos rate, Brier, thresholds)

  - Examples (easy positives / escalated / uncertain) and clusters of duplicates

  - Full config JSON is embedded for reproducibility

## Interpreting scores (Decision Trace)

- raw: the model’s native similarity (cosine for embeddings, Jaccard for MinHash, Hamming based similarity for SimHash).

- prob: calibrated probability of “duplicate” (0–1), derived from the bootstrap calibration or raw probability if calibration is disabled.

- The Arbiter compares per-learner probs to thresholds and uses consensus rules; uncertain cases get escalated.

## Metrics you’ll see

- Total pairs (scored candidate pairs)

- Duplicates / Non-duplicates / Uncertain

- Consensus rate (fraction of pairs where learners agreed confidently)

- Escalations % (fraction needing extra scrutiny)

- Per-learner: N, positive rate, threshold, Brier

- Clusters: groups of documents connected by duplicate decisions

## Project layout (high-level)
src/
  app.py                     # Tkinter application
  gui/                       # GUI widgets & panels
  features/                  # text normalization & tokenization
  learners/                  # SimHash, MinHash, Embedding
  ensemble/                  # Arbiter & escalation logic
  pipelines/                 # orchestration & calibration bootstrap
  metrics/                   # metrics & snapshots
  storage/                   # SQLite document store
  persistence/               # run state & calibration state store
  reporting/                 # HTML/MD report builder

## Troubleshooting
- The DB has 2,029 rows by default with stock data from a CSV. To run the algorithm on a smaller set, delete the documents in the GUI first before running the algorithm

- If more data needs to be inserted from the CSV, uncomment the import_rows_from_csv() function in the main app function, and change the number of rows in import_rows_from_csv() function itself to insert more rows. Run the app to see new documents

- Remember to comment out the function and save after running the app to avoid accidentally inserting more data

- First run is slow: models download; keep network open or pre-cache.

- Torch/CUDA issues: prefer CPU wheels unless you specifically need GPU.

- Tkinter not found: install system Tk.

- DB locks: don’t run multiple writers; close the app before deleting DB files.

- Report doesn’t open: use Run History -> Open report; a fresh report will be generated and opened automatically.

## Data & privacy

- Everything runs locally; SQLite DB and reports stay on your machine.

## Contributing

See CONTRIBUTING.md for branch workflow, code style, testing and PR guidelines.