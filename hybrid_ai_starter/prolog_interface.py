# Student Name: Aftab Hussaini
# Student FAN:  huss0138
# File:         prolog_interface.py
# Date:         03-11-2026
# Description:  Automated hybrid pipeline: (A1 ingestion) -> train -> ML inference -> Prolog reasoning -> result.
# Usage:        python prolog_interface.py [--site-key signal:TS:025] [--all]
#               python prolog_interface.py --from-a1 [--a1-offline] [--a1-dir PATH] [--all]
# Licence:      MIT Licence
"""Integration layer of the Hybrid AI Prediction and Inference System.

This module connects the team's subsystems into one automated pipeline:

    0. Ingestion phase  (optional, --from-a1) Artefact 1's data_ingestion.py
                        is run to fetch, validate and transform the source
                        data. Its intersection_sites.csv is checked against
                        the columns this system needs and copied into
                        processed/, so Artefact 2 is built on a fresh,
                        validated Artefact 1 output.
    1. Training phase   train_models.py runs on every execution: it trains
                        and compares the candidate classifiers and saves
                        the best one as models/best_model.joblib. Any
                        best_model.joblib left from an earlier run is
                        removed first, so the model used is always the one
                        selected by this run's comparison.
    2. Inference phase  best_model.joblib is loaded, P(any_fsi) is predicted
                        for the selected site(s) and the prediction is
                        written as Prolog facts.
    3. Reasoning phase  SWI-Prolog consults crash_risk_rules.pl together
                        with the site facts and the ML facts and infers an
                        action, a reason, every data-quality issue and every
                        contextual trigger for each site.
    4. Output           input, ML prediction and Prolog conclusion are saved
                        together so each recommendation can be traced.

Two modes are provided:
    * single site (default): interactive search, or --site-key KEY.
      Output: output/hybrid_result.json
    * all sites (--all): every intersection in one Prolog run.
      Output: output/hybrid_results_all.csv and output/hybrid_summary.json

Prolog is run as a separate ``swipl`` process (not embedded with pyswip)
because a subprocess is portable across Windows, macOS and Linux, can be
given a timeout, and fails with a clear error message.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import predict_site
import prepare_site

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RULES_FILE = BASE_DIR / "crash_risk_rules.pl"
TRAINING_SCRIPT = BASE_DIR / "train_models.py"
CSV_FILE = BASE_DIR / "processed" / "intersection_sites.csv"
PREVIOUS_CSV_FILE = BASE_DIR / "processed" / "intersection_sites.previous.csv"

# Artefact 1 (data ingestion). Default location matches the repository
# layout, where crash-risk-ingestion/ sits next to hybrid_ai_starter/.
DEFAULT_A1_DIR = BASE_DIR.parent / "crash-risk-ingestion"
A1_SCRIPT_NAME = "data_ingestion.py"
A1_OUTPUT = Path("data") / "processed" / "intersection_sites.csv"
A1_REPORT = Path("data") / "processed" / "run_report.json"
A1_CONTRACT = Path("docs") / "data_contract.json"
MODEL_DIR = BASE_DIR / "models"
# The model train_models.py saves after comparing all candidates.
BEST_MODEL_FILE = MODEL_DIR / "best_model.joblib"
MODEL_COMPARISON_FILE = BASE_DIR / "output" / "model_comparison.json"
OUTPUT_DIR = BASE_DIR / "output"

SITE_FACTS_FILE = OUTPUT_DIR / "site_facts.pl"
ML_FACTS_FILE = OUTPUT_DIR / "ml_prediction_facts.pl"
ML_PREDICTION_FILE = OUTPUT_DIR / "ml_prediction.json"
RESULT_FILE = OUTPUT_DIR / "hybrid_result.json"
BATCH_RESULTS_FILE = OUTPUT_DIR / "hybrid_results_all.csv"
BATCH_SUMMARY_FILE = OUTPUT_DIR / "hybrid_summary.json"
LOG_FILE = OUTPUT_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
INGESTION_TIMEOUT_S = 1800
TRAINING_TIMEOUT_S = 600
PROLOG_TIMEOUT_S = 120

# Every fact predicate the rules may query. Declaring them dynamic before
# the facts are loaded means a fact that is absent for a site (for example
# traffic_volume/2 at a site with no volume estimate) makes the query fail
# normally instead of raising an "Unknown procedure" error.
FACT_PREDICATES = (
    "site/1",
    "traffic_volume/2",
    "pedestrian_crossings/2",
    "school_crossings/2",
    "volume_stale/1",
    "volume_current/1",
    "has_volume/1",
    "no_volume/1",
    "ml_classification/2",
    "ml_positive_probability/2",
)

# Prefix that marks a result line in Prolog's output, so warnings or other
# text printed by SWI-Prolog can never be mistaken for a result.
RESULT_TAG = "HYBRID_RESULT"

# Fallback install locations used when swipl is not on PATH.
SWIPL_CANDIDATES = (
    "/Applications/SWI-Prolog.app/Contents/MacOS/swipl",
    "/opt/homebrew/bin/swipl",
    "/usr/local/bin/swipl",
    r"C:\Program Files\swipl\bin\swipl.exe",
)

log = logging.getLogger("hybrid_pipeline")


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot complete."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    """Log to the console and append to output/pipeline.log."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Phase 0: Artefact 1 ingestion (optional)
# ---------------------------------------------------------------------------
def required_columns(a1_dir: Path) -> set[str]:
    """Columns intersection_sites.csv must have for Artefact 2 to run.

    The authority is Artefact 1's data contract (docs/data_contract.json),
    the same file train_models.py selects its features from, plus the
    fields prepare_site.py writes as Prolog facts. If the contract is not
    available, only the fact fields and the target are checked.
    """
    columns = {"site_key", "site_desc", "entering_aadt",
               "ped_crossings_200m", "school_crossings_300m",
               "volume_stale", "has_volume", "any_fsi"}
    contract_file = a1_dir / A1_CONTRACT
    if contract_file.is_file():
        contract = json.loads(contract_file.read_text(encoding="utf-8"))
        table = contract.get("tables", {}).get("intersection_sites", {})
        columns |= {column["name"] for column in table.get("columns", [])}
    else:
        log.warning("Data contract not found (%s); checking core columns only",
                    contract_file)
    return columns


def run_artefact1(a1_dir: Path, offline: bool) -> Path:
    """Run Artefact 1's ingestion pipeline and hand its output to A2.

    Steps:
        1. run data_ingestion.py in its own folder (live output shown);
        2. check it succeeded and wrote intersection_sites.csv;
        3. check the CSV has every column A2 needs and at least one row;
        4. keep a copy of the current A2 CSV, then replace it with the
           new A1 output.

    Returns:
        Path of the CSV now used by Artefact 2.

    Raises:
        PipelineError: If A1 is missing, fails, times out, or produces
            output that does not match what A2 expects.
    """
    a1_dir = a1_dir.resolve()
    script = a1_dir / A1_SCRIPT_NAME
    if not script.is_file():
        raise PipelineError(
            f"Artefact 1 not found: {script}\n"
            "Use --a1-dir to point at the crash-risk-ingestion folder.")

    command = [sys.executable, str(script)]
    if offline:
        command.append("--offline")
    log.info("Ingestion phase: running Artefact 1 (%s%s)",
             A1_SCRIPT_NAME, " --offline" if offline else "")
    try:
        # A1 logs its own progress; let it print live (it can take minutes).
        completed = subprocess.run(
            command, cwd=a1_dir, stdin=subprocess.DEVNULL,
            timeout=INGESTION_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as error:
        raise PipelineError("Artefact 1 ingestion timed out.") from error
    if completed.returncode != 0:
        raise PipelineError(
            "Artefact 1 ingestion failed (see its log in "
            f"{a1_dir / 'data' / 'logs'}). Artefact 2 was not run.")

    a1_csv = a1_dir / A1_OUTPUT
    if not a1_csv.is_file():
        raise PipelineError(f"Artefact 1 finished but {a1_csv} is missing.")

    # Schema hand-off check: fail here, with a clear message, rather than
    # partway through training or prediction.
    header = pd.read_csv(a1_csv, nrows=0).columns
    missing = sorted(required_columns(a1_dir) - set(header))
    if missing:
        raise PipelineError(
            "Artefact 1 output is missing columns Artefact 2 needs: "
            + ", ".join(missing))
    rows = sum(1 for _ in a1_csv.open(encoding="utf-8")) - 1
    if rows < 1:
        raise PipelineError("Artefact 1 output has no rows.")

    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CSV_FILE.is_file():
        shutil.copy2(CSV_FILE, PREVIOUS_CSV_FILE)
        log.info("Previous A2 dataset kept as %s", PREVIOUS_CSV_FILE.name)
    shutil.copy2(a1_csv, CSV_FILE)

    report_file = a1_dir / A1_REPORT
    if report_file.is_file():
        report = json.loads(report_file.read_text(encoding="utf-8"))
        log.info("Artefact 1 run report: %s", report_file)
        for name, counts in report.get("rows", {}).items():
            log.info("  %-17s raw %6s  clean %6s  rejected %5s", name,
                     counts.get("raw"), counts.get("clean"),
                     counts.get("rejected"))
    log.info("Ingestion phase complete: %d sites handed to Artefact 2 (%s)",
             rows, CSV_FILE)
    return CSV_FILE


# ---------------------------------------------------------------------------
# Phase 1: training and model selection
# ---------------------------------------------------------------------------
def use_model_file(model_file: Path) -> None:
    """Point predict_site.py at the chosen model file.

    predict_site.py reads its module-level MODEL_FILE when predicting, so
    setting it here makes single-site and --all runs use the same model.

    Args:
        model_file: Saved model to use for inference.
    """
    predict_site.MODEL_FILE = model_file


def train_and_select_model() -> Path:
    """Train and compare all candidate models, then select the best one.

    Training runs on every execution so the model comparison is
    reproduced each time rather than taken from earlier output files.
    Any best_model.joblib from a previous run is deleted first, and the
    new file must be written by this run's training, so a stale model can
    never be used by mistake.

    Returns:
        Path of the freshly saved best model.

    Raises:
        PipelineError: If training fails, times out, or does not save a
            new best_model.joblib.
    """
    BEST_MODEL_FILE.unlink(missing_ok=True)
    log.info("Training phase: running %s to train and compare models",
             TRAINING_SCRIPT.name)
    try:
        completed = subprocess.run(
            [sys.executable, str(TRAINING_SCRIPT)],
            cwd=BASE_DIR,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=TRAINING_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise PipelineError("Model training timed out.") from error

    # Show the training script's output, including its comparison table.
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        raise PipelineError(
            "Model training failed.\n" + completed.stderr.strip())
    if not BEST_MODEL_FILE.is_file():
        raise PipelineError(
            f"{TRAINING_SCRIPT.name} finished but did not save "
            f"{BEST_MODEL_FILE.name}. The training script must save the "
            "selected model to models/best_model.joblib.")

    log.info("Training phase complete: selected model is %s (%s)",
             model_description(joblib.load(BEST_MODEL_FILE)),
             BEST_MODEL_FILE.name)
    if MODEL_COMPARISON_FILE.is_file():
        log.info("Model comparison saved to %s", MODEL_COMPARISON_FILE)
    use_model_file(BEST_MODEL_FILE)
    return BEST_MODEL_FILE


# ---------------------------------------------------------------------------
# Phase 2: preparation and ML inference
# ---------------------------------------------------------------------------
def load_model():
    """Load the saved scikit-learn pipeline.

    Returns:
        The fitted logistic-regression pipeline.

    Raises:
        PipelineError: If the model file is missing.
    """
    model_file = predict_site.MODEL_FILE
    if not Path(model_file).is_file():
        raise PipelineError(f"Trained model not found: {model_file}")
    return joblib.load(model_file)


def model_description(model) -> str:
    """Name the algorithm inside a saved pipeline, e.g. LogisticRegression.

    Args:
        model: Loaded scikit-learn pipeline or estimator.

    Returns:
        The classifier's class name.
    """
    steps = getattr(model, "named_steps", {})
    classifier = steps.get("classifier", model)
    return type(classifier).__name__


def classification_label(predicted_class: int) -> str:
    """Map the model's 0/1 output to the label the Prolog rules expect.

    Args:
        predicted_class: 1 for the any_fsi class, 0 otherwise.

    Returns:
        historical_fsi_present or historical_fsi_absent.
    """
    if predicted_class == 1:
        return "historical_fsi_present"
    return "historical_fsi_absent"


def model_input(data: pd.DataFrame, model) -> pd.DataFrame:
    """Build the model's input table from intersection_sites.csv rows.

    The columns are the ones the saved pipeline was trained on
    (feature_names_in_), so prediction always matches whatever features
    train_models.py selected, even if that list changes. Numeric columns
    are coerced to numbers; text columns keep missing values as NaN for
    the pipeline's own imputers.

    Raises:
        PipelineError: If the CSV lacks a column the model needs.
    """
    names = list(getattr(model, "feature_names_in_", predict_site.FEATURES))
    missing = [name for name in names if name not in data.columns]
    if missing:
        raise PipelineError(
            "Dataset is missing columns the model was trained on: "
            + ", ".join(missing))
    features = data[names].copy()
    for name in names:
        if pd.api.types.is_numeric_dtype(features[name]) or \
                pd.api.types.is_bool_dtype(features[name]):
            features[name] = pd.to_numeric(features[name], errors="coerce")
        else:
            features[name] = features[name].astype("object").where(
                features[name].notna(), np.nan)
    return features


def predict_rows(data: pd.DataFrame, model) -> tuple[list[int], list[float]]:
    """Return predicted class and P(any_fsi) for each row of data."""
    features = model_input(data, model)
    classes = [int(value) for value in model.predict(features)]
    positive_index = list(model.classes_).index(1)
    probabilities = [float(value) for value in
                     model.predict_proba(features)[:, positive_index]]
    return classes, probabilities


def ml_fact_lines(site_id: str, predicted_class: int,
                  probability: float) -> list[str]:
    """Build the ML facts in the same format as predict_site.py.

    Args:
        site_id: Prolog atom identifying the site.
        predicted_class: Model class prediction (0 or 1).
        probability: Model probability of the any_fsi class.

    Returns:
        Prolog clauses, one per list item.

    Raises:
        PipelineError: If the probability is outside [0, 1].
    """
    if not 0.0 <= probability <= 1.0:
        raise PipelineError(
            f"Probability {probability} for {site_id} is outside [0, 1].")
    return [
        f"ml_classification({site_id}, "
        f"{classification_label(predicted_class)}).",
        f"ml_positive_probability({site_id}, {probability:.6f}).",
    ]


def prepare_single_site(site_key: str | None) -> dict | None:
    """Select one site and write its record and site facts.

    Uses prepare_site.py so the facts match the team's agreed format.

    Args:
        site_key: Exact site key, or None to open the interactive search.

    Returns:
        The site record, or None if the user cancelled the search.
    """
    if site_key is None:
        site_key = prepare_site.choose_site_interactively(CSV_FILE)
        if site_key is None:
            return None

    site = prepare_site.load_site(CSV_FILE, site_key)
    prepare_site.write_json(site, predict_site.RECORD_FILE)
    prepare_site.write_prolog_facts(site, SITE_FACTS_FILE)
    log.info("Prepared %s (%s)", site["site_key"], site["description"])
    return site


def predict_single_site(site: dict) -> dict:
    """Run predict_site.py's functions for one site.

    Args:
        site: Record produced by prepare_single_site.

    Returns:
        The ML result dictionary, also saved to output/ml_prediction.json
        together with output/ml_prediction_facts.pl.
    """
    data = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    row = data[data["site_key"].astype(str).str.strip() == site["site_key"]]
    if row.empty:
        raise PipelineError(f"Site {site['site_key']} not found in {CSV_FILE}")
    classes, probabilities = predict_rows(row.head(1), load_model())
    predicted_class, probability = classes[0], probabilities[0]
    # predict_site.py's writer saves ml_prediction.json and the ML facts.
    result = predict_site.save_prediction(site, predicted_class, probability)
    # predict_site.py labels every result "Logistic Regression"; record the
    # model actually loaded so the output stays correct if best_model.joblib
    # holds a different algorithm.
    result["model"] = model_description(load_model())
    result["model_file"] = Path(predict_site.MODEL_FILE).name
    log.info("ML inference: %s, P(any_fsi) = %.3f",
             result["classification"], probability)
    return result


def prepare_and_predict_all() -> tuple[pd.DataFrame, Path, Path]:
    """Predict every site and write one site-facts and one ML-facts file.

    The model is loaded once and all rows are scored together, which is
    much faster than running the single-site scripts once per site.

    Returns:
        (table of site_key, description, ML class and probability,
         path of the site facts file, path of the ML facts file).
    """
    data = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    model = load_model()
    model_name = model_description(model)
    classes, probabilities = predict_rows(data, model)
    log.info("ML inference: scored %d sites", len(data))

    site_lines, ml_lines, rows = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_facts = Path(tmp) / "site.pl"
        for key, desc, cls, prob in zip(data["site_key"], data["site_desc"],
                                        classes, probabilities):
            site = prepare_site.load_site(CSV_FILE, str(key).strip())
            # Reuse the team's fact writer so batch facts are identical
            # to single-site facts.
            prepare_site.write_prolog_facts(site, tmp_facts)
            site_lines.extend(
                line for line in tmp_facts.read_text("utf-8").splitlines()
                if not line.startswith("%"))
            ml_lines.extend(
                ml_fact_lines(site["prolog_site_id"], int(cls), float(prob)))
            rows.append({
                "site_key": site["site_key"],
                "prolog_site_id": site["prolog_site_id"],
                "description": desc,
                "ml_classification": classification_label(int(cls)),
                "ml_probability_any_fsi": round(float(prob), 6),
                "ml_model": model_name,
            })

    site_path = OUTPUT_DIR / "all_site_facts.pl"
    ml_path = OUTPUT_DIR / "all_ml_prediction_facts.pl"
    site_path.write_text(
        "% Generated by prolog_interface.py --all\n"
        + "\n".join(site_lines) + "\n", encoding="utf-8")
    ml_path.write_text(
        "% Generated by prolog_interface.py --all\n"
        + "\n".join(ml_lines) + "\n", encoding="utf-8")
    return pd.DataFrame(rows), site_path, ml_path


# ---------------------------------------------------------------------------
# Phase 3: Prolog reasoning
# ---------------------------------------------------------------------------
def find_swipl() -> str:
    """Locate the SWI-Prolog executable.

    Checks the SWIPL environment variable, then PATH, then common
    install locations.

    Returns:
        Path to the swipl executable.

    Raises:
        PipelineError: If SWI-Prolog cannot be found.
    """
    override = os.environ.get("SWIPL")
    if override and Path(override).is_file():
        return override
    on_path = shutil.which("swipl")
    if on_path:
        return on_path
    for candidate in SWIPL_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise PipelineError(
        "SWI-Prolog (swipl) was not found. Install it from "
        "https://www.swi-prolog.org and add it to PATH, or set the "
        "SWIPL environment variable to the swipl executable.")


def prolog_quoted(path: Path) -> str:
    """Return a file path as a Prolog string literal.

    json.dumps escapes quotes and backslashes the same way Prolog does,
    so Windows paths and paths containing spaces are passed safely.

    Args:
        path: File path to quote.

    Returns:
        A double-quoted literal.
    """
    return json.dumps(str(path))


def build_prolog_goal(fact_files: list[Path]) -> str:
    """Build the goal that loads the knowledge base and reports every site.

    For each site/1 fact the goal prints one tab-separated line:
    tag, site, action, reason, data issues, contextual reasons.
    Each site is wrapped in catch/3 so an error at one site is reported
    for that site and does not stop the run.

    Args:
        fact_files: Facts files to consult after the rules.

    Returns:
        The goal text for ``swipl -g``.
    """
    loads = ", ".join(
        f"consult({prolog_quoted(path)})"
        for path in [RULES_FILE, *fact_files])
    dynamics = ", ".join(FACT_PREDICATES)
    return (
        f"dynamic([{dynamics}]), "
        "style_check(-discontiguous), "
        f"{loads}, "
        "forall(site(S), ("
        "  catch(("
        "      ( site_assessment(S, A, R) -> true "
        "      ; A = no_assessment, R = no_rule_matched ), "
        "      site_data_issues(S, Issues), "
        "      findall(C, contextual_reason(S, C), Cs0), sort(Cs0, Cs)"
        "    ), Error, ("
        "      A = prolog_error, "
        "      term_to_atom(Error, R), Issues = [], Cs = []"
        "    )), "
        "  atomic_list_concat(Issues, ';', IssueText), "
        "  atomic_list_concat(Cs, ';', ContextText), "
        f"  format('{RESULT_TAG}\\t~w\\t~w\\t~w\\t~w\\t~w~n', "
        "         [S, A, R, IssueText, ContextText])"
        "))"
    )


def run_prolog(fact_files: list[Path]) -> dict[str, dict]:
    """Run SWI-Prolog over the given facts and parse the conclusions.

    Args:
        fact_files: Site facts and ML facts files to consult.

    Returns:
        Mapping of Prolog site id to its assessment: action, reason,
        data_issues (list) and contextual_reasons (list).

    Raises:
        PipelineError: If Prolog is missing, fails, times out or returns
            no results.
    """
    for path in [RULES_FILE, *fact_files]:
        if not path.is_file():
            raise PipelineError(f"Prolog input not found: {path}")

    command = [find_swipl(), "-q",
               "-g", build_prolog_goal(fact_files),
               "-t", "halt"]
    log.info("Reasoning phase: consulting %s", RULES_FILE.name)
    try:
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            stdin=subprocess.DEVNULL,  # never wait for keyboard input
            capture_output=True,
            text=True,
            timeout=PROLOG_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise PipelineError(
            f"SWI-Prolog did not finish within {PROLOG_TIMEOUT_S} s."
        ) from error

    if completed.returncode != 0:
        raise PipelineError(
            "SWI-Prolog returned an error.\n" + completed.stderr.strip())

    results = {}
    for line in completed.stdout.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 6 or parts[0] != RESULT_TAG:
            continue
        _, site_id, action, reason, issues, contexts = parts
        results[site_id] = {
            "action": action,
            "reason": reason,
            "data_issues": issues.split(";") if issues else [],
            "contextual_reasons": contexts.split(";") if contexts else [],
        }
    if not results:
        raise PipelineError(
            "SWI-Prolog returned no assessments. Check the facts files.")
    log.info("Reasoning phase: %d assessment(s) inferred", len(results))
    return results


# ---------------------------------------------------------------------------
# Phase 4: output
# ---------------------------------------------------------------------------
def save_single_result(site: dict, ml_result: dict,
                       assessment: dict) -> dict:
    """Save the input, ML prediction and Prolog conclusion together.

    Args:
        site: Site record (the input facts).
        ml_result: Output of predict_single_site.
        assessment: This site's entry from run_prolog.

    Returns:
        The combined result, also written to output/hybrid_result.json.
    """
    result = {
        "site_key": site["site_key"],
        "description": site.get("description", ""),
        "input_facts": {k: v for k, v in site.items()
                        if k not in ("site_key", "description")},
        "machine_learning": ml_result,
        "prolog_assessment": assessment,
        "prototype_note": (
            "The ML model classifies historical crash labels. The Prolog "
            "rules provide illustrative contextual reasoning and are not "
            "validated road-safety standards."),
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    return result


def save_batch_results(table: pd.DataFrame,
                       assessments: dict[str, dict]) -> dict:
    """Merge ML and Prolog results for all sites and save a summary.

    Args:
        table: ML results from prepare_and_predict_all.
        assessments: Prolog results keyed by Prolog site id.

    Returns:
        Summary counts, also written to output/hybrid_summary.json.
    """
    missing = set(table["prolog_site_id"]) - set(assessments)
    if missing:
        log.warning("%d site(s) received no Prolog result", len(missing))

    def field(site_id: str, name: str) -> str:
        value = assessments.get(site_id, {}).get(name, "")
        return ";".join(value) if isinstance(value, list) else value

    for name in ("action", "reason", "data_issues", "contextual_reasons"):
        table[f"prolog_{name}"] = [field(s, name)
                                   for s in table["prolog_site_id"]]
    table.to_csv(BATCH_RESULTS_FILE, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "sites": int(len(table)),
        "ml_classification_counts": dict(
            Counter(table["ml_classification"])),
        "prolog_action_counts": dict(Counter(table["prolog_action"])),
        "action_by_ml_class": (
            table.groupby(["ml_classification", "prolog_action"])
            .size().unstack(fill_value=0).to_dict(orient="index")),
        "ml_positive_but_held_for_data_review": int((
            (table["ml_classification"] == "historical_fsi_present")
            & (table["prolog_action"] == "data_review")).sum()),
        "ml_negative_but_flagged_by_context": int((
            (table["ml_classification"] == "historical_fsi_absent")
            & (table["prolog_action"] == "contextual_review")).sum()),
    }
    BATCH_SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n",
                                  encoding="utf-8")
    return summary


def print_single_result(result: dict) -> None:
    """Print a readable summary of one hybrid assessment."""
    ml = result["machine_learning"]
    pl = result["prolog_assessment"]
    print("\nHybrid AI assessment")
    print("--------------------")
    print("Site:              ", result["site_key"])
    print("Description:       ", result["description"])
    print("ML classification: ", ml["classification"])
    print("P(any_fsi):        ",
          f"{ml['estimated_probability_any_fsi']:.2%}")
    print("Prolog action:     ", pl["action"])
    print("Prolog reason:     ", pl["reason"])
    print("Data issues:       ", ", ".join(pl["data_issues"]) or "none")
    print("Contextual reasons:",
          ", ".join(pl["contextual_reasons"]) or "none")
    print("\nSaved to:", RESULT_FILE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run the hybrid ML + Prolog crash-risk pipeline.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--site-key",
        help="Assess one site, e.g. signal:TS:025 (default: search menu).")
    target.add_argument(
        "--all", action="store_true",
        help="Assess every intersection and write a results table.")
    a1 = parser.add_argument_group("Artefact 1 link")
    a1.add_argument(
        "--from-a1", action="store_true",
        help="Run Artefact 1's data_ingestion.py first and use its output.")
    a1.add_argument(
        "--a1-dir", type=Path, default=DEFAULT_A1_DIR,
        help="Artefact 1 folder (default: ../crash-risk-ingestion).")
    a1.add_argument(
        "--a1-offline", action="store_true",
        help="Run Artefact 1 with --offline (use its cached downloads).")
    args = parser.parse_args(argv)
    if (args.a1_offline or args.a1_dir != DEFAULT_A1_DIR) and not args.from_a1:
        parser.error("--a1-dir and --a1-offline need --from-a1")
    return args


def run_single(site_key: str | None) -> int:
    """Run the pipeline for one site; return a process exit code."""
    site = prepare_single_site(site_key)
    if site is None:
        return 0
    ml_result = predict_single_site(site)
    assessments = run_prolog([SITE_FACTS_FILE, ML_FACTS_FILE])
    assessment = assessments.get(site["prolog_site_id"])
    if assessment is None:
        raise PipelineError("Prolog returned no result for this site.")
    print_single_result(save_single_result(site, ml_result, assessment))
    return 0


def run_all() -> int:
    """Run the pipeline for every site; return a process exit code."""
    table, site_facts, ml_facts = prepare_and_predict_all()
    assessments = run_prolog([site_facts, ml_facts])
    summary = save_batch_results(table, assessments)
    print("\nHybrid AI assessment: all sites")
    print("-------------------------------")
    print(json.dumps(summary, indent=2))
    print("\nResults saved to:", BATCH_RESULTS_FILE)
    print("Summary saved to:", BATCH_SUMMARY_FILE)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the full automated hybrid pipeline."""
    args = parse_args(argv)
    setup_logging()
    try:
        if args.from_a1:
            run_artefact1(args.a1_dir, args.a1_offline)
        if not CSV_FILE.is_file():
            raise PipelineError(f"Dataset not found: {CSV_FILE}")
        train_and_select_model()
        if args.all:
            return run_all()
        return run_single(args.site_key)
    except (PipelineError, FileNotFoundError, ValueError, KeyError,
            OSError) as error:
        log.error("Hybrid pipeline failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
