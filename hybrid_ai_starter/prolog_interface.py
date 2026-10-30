"""
File: prolog_interface.py
Student: dobi0016
Course: COMP3742/9742 - Artificial Intelligence

Purpose:
Coordinate the machine-learning prediction and Prolog assessment.
The ML prediction facts and site facts are loaded into SWI-Prolog
so the symbolic rules can use both sources of information.
"""

import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------
# 1. Define file locations
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RULES_FILE = BASE_DIR / "crash_risk_rules.pl"
PREDICTION_SCRIPT = BASE_DIR / "predict_site.py"

OUTPUT_DIR = BASE_DIR / "output"

FACTS_FILE = OUTPUT_DIR / "site_facts.pl"
ML_FACTS_FILE = OUTPUT_DIR / "ml_prediction_facts.pl"
RECORD_FILE = OUTPUT_DIR / "site_record.json"
RESULT_FILE = OUTPUT_DIR / "hybrid_result.json"


# ---------------------------------------------------------
# 2. Validate required files
# ---------------------------------------------------------

def check_required_files():
    """Check that the source files needed by the pipeline exist."""

    required_files = [
        RULES_FILE,
        PREDICTION_SCRIPT,
        FACTS_FILE,
        RECORD_FILE,
    ]

    missing_files = [
        path for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        missing_names = "\n".join(
            str(path) for path in missing_files
        )

        raise FileNotFoundError(
            "Required file(s) not found:\n"
            + missing_names
            + "\nRun prepare_site.py first."
        )


# ---------------------------------------------------------
# 3. Load and validate the selected site
# ---------------------------------------------------------

def load_site_record():
    """Read the selected site's JSON record."""

    with RECORD_FILE.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    site_id = record.get("prolog_site_id")

    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_"
    )

    if (
        not isinstance(site_id, str)
        or not site_id
        or site_id[0].isdigit()
        or not all(character in allowed for character in site_id)
    ):
        raise ValueError(
            "The site record contains an invalid Prolog identifier."
        )

    return record


# ---------------------------------------------------------
# 4. Run the machine-learning prediction
# ---------------------------------------------------------

def run_ml_prediction():
    """Run predict_site.py to create fresh ML output and facts."""

    print("Running machine-learning prediction...")

    try:
        completed = subprocess.run(
            [sys.executable, str(PREDICTION_SCRIPT)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "The machine-learning prediction timed out."
        ) from error

    if completed.stdout.strip():
        print(completed.stdout.strip())

    if completed.returncode != 0:
        raise RuntimeError(
            "The machine-learning prediction failed.\n"
            + completed.stderr.strip()
        )

    if not ML_FACTS_FILE.is_file():
        raise FileNotFoundError(
            "The prediction script did not create "
            + str(ML_FACTS_FILE)
        )


# ---------------------------------------------------------
# 5. Build the SWI-Prolog goal
# ---------------------------------------------------------

def build_prolog_goal(site_id):
    """Build a query that loads all facts and runs the assessment."""

    return (
        "consult(Rules), "
        "consult(SiteFacts), "
        "consult(MLFacts), "
        "Site = " + site_id + ", "
        "site_assessment(Site, Action, Reason), "
        "write_term(result(Action, Reason), "
        "[quoted(true), ignore_ops(true)]), "
        "nl, "
        "halt"
    )


# ---------------------------------------------------------
# 6. Execute SWI-Prolog
# ---------------------------------------------------------

def run_prolog_assessment(site_id):
    """Run the symbolic assessment using both sets of facts."""

    goal = build_prolog_goal(site_id)

    command = [
        "swipl",
        "-q",
        "-g",
        "Rules = " + prolog_string(str(RULES_FILE)) + ", "
        "SiteFacts = " + prolog_string(str(FACTS_FILE)) + ", "
        "MLFacts = " + prolog_string(str(ML_FACTS_FILE)) + ", "
        + goal,
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    except FileNotFoundError as error:
        raise RuntimeError(
            "Could not find swipl. Install SWI-Prolog and ensure "
            "its installation folder is included in Windows PATH."
        ) from error

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "SWI-Prolog did not finish within 30 seconds."
        ) from error

    if completed.returncode != 0:
        raise RuntimeError(
            "SWI-Prolog returned an error.\n"
            + completed.stderr.strip()
        )

    output_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        raise RuntimeError(
            "SWI-Prolog returned no assessment. "
            "Check the Prolog facts and rules."
        )

    return parse_prolog_result(output_lines[-1])


# ---------------------------------------------------------
# 7. Safely quote strings for Prolog
# ---------------------------------------------------------

def prolog_string(value):
    """Return a quoted Prolog string literal."""

    return json.dumps(str(value))


# ---------------------------------------------------------
# 8. Parse the Prolog result
# ---------------------------------------------------------

def parse_prolog_result(result_text):
    """Parse result(Action, Reason) from Prolog output."""

    prefix = "result("
    suffix = ")"

    if not (
        result_text.startswith(prefix)
        and result_text.endswith(suffix)
    ):
        raise RuntimeError(
            "Unexpected output from Prolog: " + result_text
        )

    contents = result_text[len(prefix):-len(suffix)]
    parts = contents.split(",", 1)

    if len(parts) != 2:
        raise RuntimeError(
            "Could not parse Prolog result: " + result_text
        )

    return {
        "action": parts[0].strip(),
        "reason": parts[1].strip(),
    }


# ---------------------------------------------------------
# 9. Save the hybrid assessment
# ---------------------------------------------------------

def save_hybrid_result(record, assessment):
    """Save the current ML and Prolog results together."""

    with (OUTPUT_DIR / "ml_prediction.json").open(
        "r",
        encoding="utf-8"
    ) as handle:
        ml_result = json.load(handle)

    result = {
        "site_key": record["site_key"],
        "description": record.get("description", ""),
        "machine_learning": ml_result,
        "prolog_assessment": assessment,
        "prototype_note": (
            "The ML model classifies historical crash labels. "
            "The Prolog rules provide illustrative contextual "
            "reasoning and are not validated road-safety standards."
        ),
    }

    with RESULT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    return result


# ---------------------------------------------------------
# 10. Run the hybrid pipeline
# ---------------------------------------------------------

def main():
    """Run ML prediction followed by the Prolog assessment."""

    try:
        check_required_files()
        record = load_site_record()

        # Generate a fresh prediction before consulting Prolog.
        run_ml_prediction()

        assessment = run_prolog_assessment(
            record["prolog_site_id"]
        )

        result = save_hybrid_result(record, assessment)

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        RuntimeError,
        KeyError,
    ) as error:
        print("\nHybrid assessment failed:")
        print(error)
        raise SystemExit(1)

    print("\nHybrid AI prototype assessment")
    print("--------------------------------")
    print("Site:", result["site_key"])
    print("Description:", result["description"])
    print(
        "ML classification:",
        result["machine_learning"]["classification"]
    )
    print(
        "Estimated positive-class probability:",
        f"{result['machine_learning']['estimated_probability_any_fsi']:.2%}"
    )
    print(
        "Prolog action:",
        result["prolog_assessment"]["action"]
    )
    print(
        "Prolog reason:",
        result["prolog_assessment"]["reason"]
    )
    print("\nHybrid result saved to:", RESULT_FILE)


if __name__ == "__main__":
    main()
