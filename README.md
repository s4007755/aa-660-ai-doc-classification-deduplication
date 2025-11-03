## GitHub URL: https://github.com/s4007755/aa-660-ai-doc-classification-deduplication

# AA-660 Doc AI

# Document Deduplication README (Scroll down for Classification README)

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

  **Deduplication Dataset**

  - Approximately 30,000 rows of completely randomised duplicate/near-duplicate rows each with varying differences in text for full deduplication analysis.
  - Running import CSV via CLI will yield a random subset of selected rows depending on total number specified, giving variety to deduplication output.
  - If static results need to be shown, delete all current documents first in the DB. Then, uncomment the unit_test function in app.py and call it above main(). Then, change the desired CSV scenario for path and run via python -m src.app 

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
- For deduplication, the default for running the algorithm found at ./dataset/Dataset.csv. 
- Dataset.CSV is formatted into 4 columns for each row with the 'text' column being the column of analysis for deduplication. 
- To ensure compatibility on other different CSV files, format them the same as Dataset.csv.

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
dupfinder import-csv ./rows.csv --limit 2000 #placeholder file path, change to dataset\Dataset.csv for default dataset
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


---
# Document Classification System README

Complete guide for document classification, clustering, and querying using vector embeddings and AI-powered labels.

## Overview

The system provides two interfaces for managing document collections:

- **CLI**: Interactive command-line interface for collection management, ingestion, clustering, and classification
- **API Server**: REST API for programmatic access to collections, clusters, and labels

## Prerequisites

- **Python 3.8+**
- **Docker** (for Qdrant vector database)
- **OpenAI API Key** (for generating embeddings)

## Installation

### 1. Install Dependencies

```bash
pip install qdrant-client openai rich scikit-learn numpy fastapi uvicorn
```

Or from requirements:
```bash
pip install -r config/requirements.txt
```

### 2. Set Up Qdrant

```bash
# Quick start (in-memory)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# With persistent storage
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

Or use Docker Compose:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

### 3. Configure OpenAI API Key

```bash
# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here
```

## Quick Start

### CLI Usage

```bash
# Start CLI
python -m src.pipelines.classification.cli

# Basic workflow
create news_articles --description "News articles"
use news_articles
source /path/to/documents
cluster
classify labels.json
query label:Technology
info
```

### API Server

```bash
# Start API server
python api_server.py

# Access points
# - API: http://localhost:8000
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

## CLI Commands

### Collection Management

| Command | Description |
|---------|-------------|
| `create <name> [model] [--description "..."]` | Create collection (models: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`) |
| `use <collection>` | Select collection |
| `ls` | List all collections |
| `ls-models` | Show available embedding models |
| `rm [collection] [--yes]` | Delete collection + dedup index |
| `show` | Display connection status |

### Data Ingestion

```bash
source <path> [--limit N] [--text-column COL] [--url-column COL]
```

**Supported sources:**
- Directories (`.pdf`, `.docx`, `.txt`)
- CSV files
- URLs
- Single files

**Examples:**
```bash
source /path/to/documents
source data.csv --text-column "content"
source https://example.com/article.html --limit 100
```

### Clustering

```bash
cluster [--num-clusters N] [--debug]
```

**Modes:**
- **Unsupervised** (default): Auto-determines clusters using Agglomerative Clustering
- **K-means**: Specify number with `--num-clusters`

**Features:**
- AI-generated cluster names
- Large datasets (>5000 docs) auto-switch to K-means

### Classification

```bash
classify [labels.json] [--use-collection-labels] [--enrich]
```

**Label file format (`labels.json`):**
```json
{
  "0": {"label": "Technology", "description": "Tech content"},
  "1": {"label": "Sports", "description": "Sports news"}
}
```

**Label management:**
```bash
add-label "Science" [--description "..."] [--enrich]
rm-label <label_id|label_name> [--by id|name] [--yes]
ls-labels
```

### Querying

```bash
query <query_term>
```

**Query types:**
- `cluster:0` - Documents in cluster ID 0
- `cluster:Technology` - Documents in cluster named "Technology"
- `label:Sports` - Documents with label "Sports"
- `<document_id>` - Specific document by ID
- `<url>` - Document from URL

### Statistics

```bash
info  # Collection statistics (clusters, labels, document types)
```

## API Endpoints

### Collection Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections` | GET | List all collections with metadata |
| `/collections/{name}/info` | GET | Get detailed collection information |

### Document Queries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/points` | GET | Get all documents (supports `?limit=N&offset=N`) |
| `/collections/{name}/points/{id}` | GET | Get specific document by ID |

### Cluster Queries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/clusters` | GET | List all clusters with counts |
| `/collections/{name}/clusters/{id}` | GET | Get documents in specific cluster |

### Label Queries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections/{name}/labels` | GET | List all labels with counts |
| `/collections/{name}/labels/{label}` | GET | Get documents with specific label |

**Query Parameters:**
- `limit`: Max results (1-10000, default varies by endpoint)
- `offset`/`page_offset`: Pagination
- `use_cache`: Use cached metadata (default: true)

### Example API Requests

**Python:**
```python
import requests

BASE_URL = "http://localhost:8000"
COLLECTION = "my_collection"

# List collections
collections = requests.get(f"{BASE_URL}/collections").json()

# Get clusters
clusters = requests.get(f"{BASE_URL}/collections/{COLLECTION}/clusters").json()

# Get documents by label
docs = requests.get(
    f"{BASE_URL}/collections/{COLLECTION}/labels/Technology",
    params={"limit": 50}
).json()
```

**cURL:**
```bash
# List collections
curl http://localhost:8000/collections

# Get collection info
curl http://localhost:8000/collections/my_collection/info

# Get documents in cluster
curl "http://localhost:8000/collections/my_collection/clusters/0?limit=50"

# Get documents by label
curl "http://localhost:8000/collections/my_collection/labels/Technology?limit=100"
```

## Complete Workflow

```bash
# 1. Start CLI
python -m src.pipelines.classification.cli

# 2. Create collection
create news_articles --description "News article classification"

# 3. Select and ingest
use news_articles
source /path/to/news/articles

# 4. Cluster documents
cluster

# 5. View clusters
ls-clusters

# 6. Create labels.json
{
  "0": {"label": "Politics", "description": "Political news"},
  "1": {"label": "Sports", "description": "Sports news"},
  "2": {"label": "Technology", "description": "Tech news"}
}

# 7. Classify
classify labels.json

# 8. View results
ls-labels
query label:Politics

# 9. View statistics
info

# 10. Access via API
# Start API server in another terminal
python api_server.py
# Then query via HTTP
curl "http://localhost:8000/collections/news_articles/labels/Politics"
```

## Response Models

### Point Response
```json
{
  "id": 1,
  "source": "/path/to/document.pdf",
  "cluster_id": 0,
  "cluster_name": "Technology",
  "predicted_label": "Tech",
  "confidence": 0.85,
  "text_content": "Document preview..."
}
```

### Collection Info
```json
{
  "name": "my_collection",
  "vector_count": 1500,
  "dimension": 1536,
  "distance_metric": "Cosine",
  "embedding_model": "text-embedding-3-small",
  "created_at": "2024-01-15T10:30:00",
  "description": "My collection",
  "clustering_algorithm": "kmeans",
  "num_clusters": 5
}
```

## Best Practices

### Performance
- Use `--limit` when testing on large datasets
- For collections >5000 docs, K-means clustering is more efficient
- Duplicate detection happens during ingestion

### Classification
- Provide clear, descriptive label names
- Use `--enrich` to generate detailed descriptions automatically
- Store labels in collection for reuse

### Clustering
- Start with unsupervised clustering to discover natural groupings
- Use `--num-clusters` when you know expected categories

### API Usage
- Use pagination (`limit`) for large result sets
- Enable caching (`use_cache=true`) when metadata summaries are available
- Filter early using cluster/label endpoints rather than fetching all points

## Troubleshooting

### Connection Issues
```bash
# Check Qdrant
curl http://localhost:6333/collections

# CLI: Reconnect
retry --host localhost --port 6333

# API: Verify server
curl http://localhost:8000/
```

### Common Issues
- **OpenAI API Errors**: Verify `OPENAI_API_KEY` is set correctly
- **Empty Collections**: Ensure documents were ingested (`info` command)
- **No Clusters/Labels**: Run `cluster` or `classify` first
- **Collection Not Found**: Collections are case-sensitive

### File Formats
- Supported: `.pdf`, `.docx`, `.txt`, CSV files

## Features

- **Automatic Duplicate Detection**: Hashed during ingestion
- **Vector Storage**: Supports millions of documents per collection
- **Metadata Caching**: Fast cluster/label summaries from metadata
- **AI-Powered Naming**: Clusters automatically named using LLM
- **Flexible Querying**: By cluster, label, URL, or document ID

## Environment Variables

- `OPENAI_API_KEY`: Required for generating embeddings

## API Configuration

**Standard startup:**
```bash
python api_server.py
```
Server starts on `http://localhost:8000` by default.

**Start API server with custom settings:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8080 --workers 1
```

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Schema: http://localhost:8000/openapi.json
