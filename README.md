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

> If you installed with `pip install -e .[dev]`, a console script `aa660` may be available. Otherwise, use the module form `python -m src.cli_nd` if your environment exposes it.

**Run the pipeline on a folder and emit a report**
```bash
aa660 run --input ./docs --profile balanced --db ./aa660.db --reports ./reports
# Fallback form (if console script not available):
python -m src.cli_nd run --input ./docs --profile balanced --db ./aa660.db --reports ./reports
```

**Generate a fresh HTML report for a past run**
```bash
aa660 report --run-id 42 --reports ./reports
# Fallback:
python -m src.cli_nd report --run-id 42 --reports ./reports
```

**List run history**
```bash
aa660 history --limit 20
# Fallback:
python -m src.cli_nd history --limit 20
```

**Notes**
- `--profile` accepts: `balanced`, `high-precision`, `recall-heavy`.
- `--db` points to (or creates) the SQLite DB file.
- Reports are written as `reports/run_<RUN_ID>_<TIMESTAMP>.html`.
- The CLI mirrors the GUI’s defaults; flags are optional in simple cases.


### Ingesting documents from an HTTP API (CLI)

You can run the pipeline on docs fetched from a REST endpoint, no files required.

**Basic usage**
```bash
# Balanced preset, fetch JSON from an API, write report to ./reports
python -m src.cli_nd --preset balanced   --api-url "https://api.example.com/my-docs"   --api-headers "{\"Authorization\":\"Bearer YOUR_TOKEN\"}"   --report-dir ./reports
```

**Accepted response shapes**

Your endpoint must return either:

- A JSON **list** of items, or  
- A JSON **object** with an `items` list.

Each item may use any of these equivalent keys:
```json
[
  { "doc_id": "A123", "text": "Document text here..." },
  { "id": "A124",    "text": "Another doc..." },
  { "doc_id": "A125","raw_text": "Raw content..." }
]
```

**Header auth example**
```bash
--api-headers "{\"Authorization\":\"Bearer sk_live_...\",\"X-Tenant\":\"acme\"}"
```
> The value must be **valid JSON** (double-quoted keys/values). If it isn’t valid, the CLI logs a warning and ignores it.

**Troubleshooting / notes**
- Provide **at least 2 items**; otherwise the run will exit early.
- For offline runs with a saved payload, you can also point to a file:
  ```bash
  python -m src.cli_nd --preset balanced --json-file ./my_docs.json --report-dir ./reports
  ```
- Reports are written to `--report-dir` as `run_<RUN_ID>_<TIMESTAMP>.html`.


## Profiles

- **Balanced** – Good overall trade‑off.  
- **High Precision** – Stricter thresholds, fewer false positives.  
- **Recall‑Heavy** – More aggressive duplicate finding (expect more uncertain/escalations).

> You can tune the gray‑zone margin, escalation steps and self‑train epochs in the *Main* tab before running.


## Reports (what you'll see)
- **Run summary**: total pairs, duplicates (exact/near), uncertain, consensus %, escalations %  
- **Per‑learner snapshot**: N, positive rate, AUC, Brier, threshold  
- **Calibration snapshot**: method, threshold, Brier, reliability points  
- **Charts** per learner: reliability, ROC, PR, threshold sweep, score histogram  
- **Examples**: easy positives, escalated/hard, uncertain  
- **Clusters**: groups of documents connected by duplicate decisions  
- **Config**: full JSON snapshot for reproducibility

Open via Run History -> Open report or by calling `generate_report(...)`.


## Windows: run the packaged EXE

- Double click AA660-DocAI.exe, or

- From PowerShell:
.\AA660-DocAI.exe

First launch tips:

- Windows SmartScreen may appear. Click More info -> Run anyway.

- The first run can take longer while NLP models and font caches initialize.

- The app will create needed folders in the current working directory.

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

- prob: calibrated probability of “duplicate” (0–1), derived from the bootstrap calibration.

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

- First run is slow: models download; keep network open or pre-cache.

- Torch/CUDA issues: prefer CPU wheels unless you specifically need GPU.

- Tkinter not found: install system Tk.

- DB locks: don’t run multiple writers; close the app before deleting DB files.

- Report doesn’t open: use Run History -> Open report; a fresh report will be generated and opened automatically.

## Data & privacy

- Everything runs locally; SQLite DB and reports stay on your machine.

## Contributing

See CONTRIBUTING.md for branch workflow, code style, testing and PR guidelines.