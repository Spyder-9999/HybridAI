# HybridAI

Hybrid (machine learning + symbolic reasoning) crash-risk system for South
Australian signalised intersections. COMP3742/9742 group assignment.

## Repository layout

| Path | Part | What it is |
|---|---|---|
| `crash-risk-ingestion/` | Part 1 | Data ingestion pipeline: fetches, validates and spatially fuses five data.sa.gov.au datasets into model-ready tables and a data contract. See its [README](crash-risk-ingestion/README.md). |
| `hybrid_ai_starter/` | Part 2 | ML models, Python-Prolog interface and crash-risk rules. |
| `requirements.txt` | Both | Shared Python environment (Python 3.11+). |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The Prolog step also needs [SWI-Prolog](https://www.swi-prolog.org/) (`swipl`) on the PATH.

## Running

**Part 1: data ingestion** (from `crash-risk-ingestion/`)

```bash
python data_ingestion.py      # downloads the source data, then processes it (~10 s)
python -m pytest tests
```

Outputs go to `crash-risk-ingestion/data/` (not committed). The schema snapshot
used by Part 2 is `crash-risk-ingestion/docs/data_contract.json`.

**Part 2: hybrid system** (from `hybrid_ai_starter/`)

```bash
python train_models.py                       # trains and compares models -> models/, output/*_metrics.json
python prepare_site.py --site-key signal:TS:002   # or no flag for interactive search
python prolog_interface.py                   # ML prediction + Prolog assessment -> output/hybrid_result.json
```

`train_models.py` selects its features from the data contract and excludes every
crash-outcome column (injury counts, crash types, rates) because they would leak
the `any_fsi` target. `hybrid_ai_starter/processed/intersection_sites.csv` is a
copy of Part 1's `intersection_sites` output.
