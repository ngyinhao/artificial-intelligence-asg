# ComplaintCompass

ComplaintCompass is a single-contributor NLP research prototype that routes English
consumer financial complaint narratives into six CFPB product categories. It compares
Naive Bayes, a calibrated linear SVM, and MiniLM sentence embeddings with Logistic
Regression under one reproducible evaluation protocol.

## Intended use

This is an academic prototype. It must not be used to judge a complaint's merit, make
financial or regulatory decisions, or infer facts about consumers or companies.

## Setup

Use Python 3.12 on Windows PowerShell:

```powershell
C:\Users\ngyh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Reproducible workflow

```powershell
.\.venv\Scripts\python.exe -m complaint_compass.data download
.\.venv\Scripts\python.exe -m complaint_compass.data prepare
.\.venv\Scripts\python.exe -m complaint_compass.train --all
.\.venv\Scripts\python.exe -m complaint_compass.evaluate
.\.venv\Scripts\streamlit.exe run app.py
```

The download command first queries the official CFPB API in bounded product/date
windows. If the CFPB edge service denies API access, it automatically builds a bounded,
deterministic pool from byte ranges of the official CSV export. Raw and processed
narratives, trained artifacts, and model weights remain local and are excluded from Git.
CSV range progress is checkpointed under `data/raw/`, so an interrupted command, a
larger candidate target, or a larger `--max-range-chunks` run skips completed ranges.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Main outputs

- `data/raw/download_manifest.json`: data query and source checksum.
- `data/processed/dataset_manifest.json`: label/split counts and dataset checksum.
- `artifacts/registry.json`: available artifacts and default model.
- `reports/validation_metrics.json`: model-selection results.
- `reports/test_metrics.json`: sealed-test results.
- `reports/model_comparison.csv`: compact comparison used by the interface.

See [the implementation plan](docs/nlp-implementation-plan.md),
[dataset card](docs/dataset-card.md), and [model card](docs/model-card.md) for the
frozen methodology and limitations.
