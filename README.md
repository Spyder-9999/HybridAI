# HybridAI

Hybrid (machine learning + symbolic reasoning) crash-risk system for South
Australian signalised intersections. COMP3742/9742 group assignment.

## Repository layout

```
HybridAI/
├── requirements.txt                  # combined environment for both artefacts
├── crash-risk-ingestion/             # Artefact 1: data ingestion
│   ├── data_ingestion.py             # entry point: fetch -> validate -> transform
│   ├── eda.py                        # report figures and EDA summary
│   ├── ingestion/                    # config, fetch, validate, transform, contract
│   ├── docs/data_contract.json       # schema snapshot used by Artefact 2
│   ├── tests/test_pipeline.py
│   └── README.md                     # full Artefact 1 documentation
└── hybrid_ai/                # Artefact 2: ML + Prolog
    ├── prolog_interface.py           # entry point: the full hybrid pipeline
    ├── train_models.py               # trains/compares models, saves best_model.joblib
    ├── prepare_site.py               # turns one site row into JSON + Prolog facts
    ├── predict_site.py               # ML prediction for one site -> JSON + Prolog facts
    ├── crash_risk_rules.pl           # Prolog crash-risk rules
    └── processed/intersection_sites.csv   # copy of Artefact 1's output
```

`models/` and `output/` inside `hybrid_ai/` are created when the
pipeline runs and are not committed.

## Setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The reasoning step also needs [SWI-Prolog](https://www.swi-prolog.org/)
(macOS: `brew install swi-prolog`). `prolog_interface.py` looks for `swipl` on
the PATH and in common install locations. You can also point the `SWIPL`
environment variable at the executable.

## Running the hybrid pipeline

Run from `hybrid_ai/`:

```bash
python prolog_interface.py                         # interactive site search
python prolog_interface.py --site-key signal:TS:025  # one site
python prolog_interface.py --all                   # every intersection
python prolog_interface.py --from-a1 --all         # run Artefact 1 ingestion first
python prolog_interface.py --from-a1 --a1-offline  # ingestion from cached downloads
```

Each run does the following:

1. **Ingestion** (only with `--from-a1`). Runs `crash-risk-ingestion/data_ingestion.py`,
   checks its `intersection_sites.csv` has the columns this system needs, and
   copies it into `processed/`. `--a1-dir` sets a different Artefact 1 folder.
2. **Training.** Runs `train_models.py`, which compares logistic regression,
   decision tree, random forest and XGBoost and saves the best one as
   `models/best_model.joblib`.
3. **Inference.** Loads `best_model.joblib`, predicts P(`any_fsi`) for the
   selected site(s) and writes the prediction as Prolog facts.
4. **Reasoning.** SWI-Prolog consults `crash_risk_rules.pl` with the site and
   ML facts. It infers an action, a reason, data-quality issues and contextual
   triggers for each site.

### Outputs (`hybrid_ai/output/`)

| File | Contents |
|---|---|
| `hybrid_result.json` | Single-site result: input, ML prediction and Prolog conclusion |
| `hybrid_results_all.csv`, `hybrid_summary.json` | Results for every site (`--all`) |
| `model_comparison.json`, `*_metrics.json` | Model comparison and per-model metrics |
| `site_facts.pl`, `ml_prediction_facts.pl`, `ml_prediction.json` | Intermediate facts passed to Prolog |
| `pipeline.log` | Run log |

### Running steps individually

```bash
python train_models.py                            # models/ + output/*_metrics.json
python prepare_site.py --site-key signal:TS:002   # or no flag for interactive search
python predict_site.py                            # uses models/best_model.joblib
```

`predict_site.py` reads `output/site_record.json`, which `prepare_site.py`
writes, so run `prepare_site.py` first.

`train_models.py` selects its features from
`crash-risk-ingestion/docs/data_contract.json`. It leaves out every
crash-outcome column (injury counts, crash types, rates), because those would
leak the `any_fsi` target.

## Artefact 1: data ingestion on its own

Run from `crash-risk-ingestion/`:

```bash
python data_ingestion.py      # download (only what changed) and process
python data_ingestion.py --offline --local-dir /path/to/Data
python eda.py                 # report figures
python -m pytest tests
```

Outputs go to `crash-risk-ingestion/data/`, which is not committed. See
[crash-risk-ingestion/README.md](crash-risk-ingestion/README.md) for details.
