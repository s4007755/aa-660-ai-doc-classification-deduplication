# Document Classification System Guide

This guide covers the document classification system including CLI usage and API endpoints.

## Table of Contents

1. [Overview](#overview)
2. [Installation and Setup](#installation-and-setup)
3. [CLI Usage](#cli-usage)
4. [API Endpoints](#api-endpoints)
5. [Examples](#examples)

## Overview

The document classification system provides:
- Document ingestion from directories, CSV files, and URLs
- Vector clustering using K-means and unsupervised methods
- Cosine similarity based classification using predefined labels
- REST API for programmatic access
- CLI interface for interactive use

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Docker (for Qdrant)
- OpenAI API key

### Qdrant Docker Setup

**Official Guide:** https://qdrant.tech/documentation/quickstart/

#### Quick Start
```bash
# Run Qdrant with Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Or with persistent storage
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

#### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped
```

```bash
# Start Qdrant
docker-compose up -d

# Check status
curl http://localhost:6333/collections
```

### Python Environment Setup

#### Install Dependencies
```bash
# Install from requirements
pip install -r config/requirements.txt

# Or install individually
pip install qdrant-client openai fastapi uvicorn rich scikit-learn
```

#### Environment Variables
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Starting the Services

#### CLI
```bash
python -m src.pipelines.classification.cli
```

#### API Server
```bash
python api_server.py
```

**API Documentation:** http://localhost:8000/docs *(start server first)*

## CLI Usage

### Starting the CLI

```bash
python -m src.pipelines.classification.cli
```

### Basic Commands

#### Collection Management
```bash
# Create a collection
create my_collection --description "My document collection"

# List collections
ls

# Select a collection
use my_collection

# Show current status
show
```

#### Data Ingestion
```bash
# Load from directory
source /path/to/documents

# Load from CSV
source data.csv --text-column "content"

# Load from URLs
source urls.csv --url-column "url"

# Limit processing
source /path/to/docs --limit 1000
```

#### Clustering
```bash
# Auto-detect clusters
cluster

# Specify number of clusters
cluster --num-clusters 5

# Debug clustering
cluster --debug
```

#### Classification
```bash
# Classify with labels file
classify labels.json

# Use collection labels
classify --use-collection-labels

# Enrich labels with AI
classify labels.json --enrich-labels
```

#### Querying
```bash
# Query by cluster ID
query cluster:0

# Query by cluster name
query cluster:Technology

# Query by label
query label:Sports

# Query by file path
query "/path/to/document.txt"
```

#### Statistics
```bash
# Show collection stats
stats

# List all labels
list-labels

# Add new label
add-label "New Category"
```

### Advanced CLI Features

#### Query Examples
```bash
# Multiple query types
query cluster:0          # Documents in cluster 0
query cluster:Economy    # Documents in Economy cluster
query label:Sports       # Documents labeled as Sports
query "file path"        # Documents by file path
query "data.csv-123"     # Documents by CSV row
```

#### Help and Exit
```bash
help                    # Show all commands
exit                    # Exit CLI
quit                    # Alternative exit
```

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Collection Management

#### List Collections
```http
GET /collections
```

**Response:**
```json
{
  "collections": [
    {
      "name": "my_collection",
      "vector_count": 1500,
      "dimension": 1536,
      "distance_metric": "Cosine",
      "embedding_model": "text-embedding-3-small",
      "created_at": "2024-01-15T10:30:00",
      "description": "My document collection",
      "clustering_algorithm": "KMeans",
      "num_clusters": 5
    }
  ]
}
```

#### Get Collection Info
```http
GET /collections/{collection_name}/info
```

**Response:**
```json
{
  "name": "my_collection",
  "vector_count": 1500,
  "dimension": 1536,
  "distance_metric": "Cosine",
  "embedding_model": "text-embedding-3-small",
  "created_at": "2024-01-15T10:30:00",
  "description": "My document collection",
  "clustering_algorithm": "KMeans",
  "num_clusters": 5
}
```

### Document Queries

#### Get All Points
```http
GET /collections/{collection_name}/points?limit=100&offset=0
```

**Parameters:**
- `limit` (optional): Maximum points to return (default: all)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "points": [
    {
      "id": 1,
      "payload": {
        "source": "/path/to/doc.txt",
        "text_content": "Document content...",
        "cluster_id": 0,
        "cluster_name": "Technology",
        "predicted_label": "Tech",
        "confidence": 0.85
      },
      "vector": [0.1, 0.2, ...]
    }
  ],
  "total_count": 1500,
  "page_offset": "100",
  "has_more": true
}
```

#### Get Points by Cluster
```http
GET /collections/{collection_name}/clusters/{cluster_id}?limit=100
```

#### Get Points by Label
```http
GET /collections/{collection_name}/labels/{label_name}?limit=100
```

#### Get All Clusters
```http
GET /collections/{collection_name}/clusters
```

**Response:**
```json
{
  "clusters": [
    {
      "id": 0,
      "name": "Technology",
      "count": 300,
      "representative_docs": ["doc1.txt", "doc2.txt"]
    }
  ]
}
```

#### Get All Labels
```http
GET /collections/{collection_name}/labels
```

### API Examples

#### Python Requests
```python
import requests

# List collections
response = requests.get("http://localhost:8000/collections")
collections = response.json()

# Get points with pagination
response = requests.get(
    "http://localhost:8000/collections/my_collection/points",
    params={"limit": 100, "offset": 0}
)
points = response.json()

# Query by cluster
response = requests.get(
    "http://localhost:8000/collections/my_collection/clusters/0"
)
cluster_docs = response.json()
```

#### cURL Examples
```bash
# List collections
curl "http://localhost:8000/collections"

# Get points (first 100)
curl "http://localhost:8000/collections/my_collection/points?limit=100"

# Get points (next 100)
curl "http://localhost:8000/collections/my_collection/points?limit=100&offset=100"

# Get cluster info
curl "http://localhost:8000/collections/my_collection/clusters"
```

## Examples

### Complete Workflow Example

#### 1. CLI Workflow
```bash
# Start CLI
python -m src.pipelines.classification.cli

# Create collection
create news_articles --description "News article classification"

# Load data
source /path/to/news/articles

# Cluster documents
cluster --num-clusters 5

# Classify with labels
classify labels.json

# Query results
query cluster:0
query label:Politics

# Show statistics
stats
```

#### 2. API Workflow
```python
import requests

# Create collection (via CLI or direct Qdrant)
# Load data (via CLI)

# Get collection info
response = requests.get("http://localhost:8000/collections/news_articles/info")
info = response.json()

# Get all clusters
response = requests.get("http://localhost:8000/collections/news_articles/clusters")
clusters = response.json()

# Get documents in specific cluster
response = requests.get("http://localhost:8000/collections/news_articles/clusters/0")
docs = response.json()

# Get documents with specific label
response = requests.get("http://localhost:8000/collections/news_articles/labels/Politics")
politics_docs = response.json()
```

### Classification Labels Format

#### labels.json
```json
{
  "0": {
    "label": "Technology",
    "description": "Technology-related content"
  },
  "1": {
    "label": "Sports",
    "description": "Sports and athletics content"
  },
  "2": {
    "label": "Politics",
    "description": "Political news and analysis"
  }
}
```

---

## Quick Reference

### CLI Commands
| Command | Description |
|---------|-------------|
| `create <name>` | Create collection |
| `use <name>` | Select collection |
| `source <path>` | Load documents |
| `cluster` | Cluster documents |
| `classify <file>` | Classify documents |
| `query <term>` | Query documents |
| `stats` | Show statistics |
| `help` | Show help |
| `exit` | Exit CLI |

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collections` | GET | List collections |
| `/collections/{name}/info` | GET | Collection info |
| `/collections/{name}/points` | GET | Get all points |
| `/collections/{name}/clusters` | GET | Get clusters |
| `/collections/{name}/clusters/{id}` | GET | Get cluster points |
| `/collections/{name}/labels` | GET | Get labels |
| `/collections/{name}/labels/{name}` | GET | Get label points |
