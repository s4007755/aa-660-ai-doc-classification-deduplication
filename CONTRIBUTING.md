# Contributing to AA-660 Doc AI
This project is a document classification and deduplication system built with **FastAPI**, **scikit-learn**, and **NLP embeddings**.

# Getting Started

1. Fork the repo and clone your fork locally.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   .venv\Scripts\activate.ps1
3. Install dependencies:
    pip install -e .[dev]

# Running the App (GUI)
- GUI (Tkinter):
  python -m src.app

# Branching and Workflow
Use GitHub Flow

- Create a branch from main:
  - git switch -c feature/your-feature-name


  - Make your changes.

  - Commit with a clear message.

  - Push and open a Pull Request (PR).

Branch naming:

- `feature/...` for new features

- `develop` for integration

- `main` for production

Commit messages (e.g.):
- feat: add near-duplicate clustering

- fix: handle empty document edge case

- chore: bump scikit-learn

- docs: extend usage examples

- refactor: split into module

This improves readability.

# Code Style & Quality
- Keep modules focused and small.

- Prefer pure functions for text/metrics logic; keep side effects (DB, filesystem) isolated.

- Add docstrings for public functions/classes.

- Handle edge cases, and log clearly.

Architecture hints:
- src/features/*: text normalization & tokenization

- src/learners/*: SimHash, MinHash, Embedding models

- src/ensemble/*: Arbiter & escalation

- src/pipelines/*: orchestration & calibration bootstrap

- src/metrics/*: metrics & snapshots

- src/storage/*, src/persistence/*: SQLite/state management

- src/gui/*: Tkinter widgets and panels

- src/reporting/*: report builders

# Adding/Updating Dependencies
- Prefer minimal, widely used libraries.

- Pin upper bounds to avoid breaking major updates.

- Add runtime deps under [project.dependencies] and dev tools under [project.optional-dependencies].dev in pyproject.toml.

- If a dependency requires system packages, call it out in the README.

# Pull Request Guidelines
Before opening a PR:

- Test GUI (src.app)

- Update docs/README if changes

- Keep PRs focused and < 400 lines when possible

In the PR description:

- What changed & why

- Screenshots

- Migration/compatability impacts

# Documentation
- Keep README.md user-focused.

- Inline docstrings for functions/classes users may call.

GUI screenshots to include in PRs when relevant:
- Documents tab (ingestion & counts)

- Main run panel status & quick summary

- Metrics tab (run summary + per-learner stats)

- Decision Traces (example rows)

- Run History (detail panel + notes)

- Generated HTML report

# Troubleshooting
- Tkinter not found: install python3-tk/tk.

- PyTorch install issues: check CUDA vs CPU wheels and Python version compatibility.

- Long model download times: embeddings models are downloaded on first run; ensure network access or use a local cache.

- SQLite locks: close the GUI before removing DB files; avoid parallel writers.