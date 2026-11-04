"""

Student Name: Patrick
Student FAN: dobi0016
File: prepare_site.py
Date: 03-11-2026
Description: Convert one row from intersection_sites.csv into a small JSON record
         and a Prolog facts file.
"""

import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "processed" / "intersection_sites.csv"
OUTPUT_DIR = BASE_DIR / "output"


def parse_number(value, field_name):
    """Convert a CSV value to a number, or return None for blank/unusable data."""
    if value is None or str(value).strip() == "":
        return None

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except (TypeError, ValueError):
        print(
            f"Warning: could not parse {field_name}={value!r}; "
            "treating it as missing."
        )
        return None


def parse_flag(value):
    """Interpret common CSV true/false values without assuming blanks are false."""
    if value is None or str(value).strip() == "":
        return None

    return str(value).strip().lower() in {
        "1", "1.0", "true", "yes", "y"
    }


def prolog_atom(text):
    """Create a simple Prolog identifier from a site key."""
    atom = re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        str(text)
    ).strip("_").lower()

    if not atom:
        atom = "unknown_site"

    if atom[0].isdigit():
        atom = "site_" + atom

    return atom


def number_for_prolog(value):
    """Format a Python number for use in a Prolog fact."""
    if value is None:
        return "unknown"

    if isinstance(value, float):
        return repr(value)

    return str(value)


def load_site(csv_path, requested_site_key=None):
    """Read one site from the processed intersection CSV."""

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        reader = csv.DictReader(handle)

        required = {
            "site_key",
            "site_desc",
            "entering_aadt",
            "ped_crossings_200m",
            "school_crossings_300m",
            "volume_stale",
            "has_volume",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            if (
                requested_site_key is None
                or row["site_key"].strip() == requested_site_key
            ):

                site_key = row["site_key"].strip()

                return {
                    "site_key": site_key,
                    "prolog_site_id": prolog_atom(site_key),
                    "description": row.get("site_desc", "").strip(),

                    "traffic_volume": parse_number(
                        row.get("entering_aadt"),
                        "entering_aadt"
                    ),

                    "pedestrian_crossings_200m": parse_number(
                        row.get("ped_crossings_200m"),
                        "ped_crossings_200m"
                    ),

                    "school_crossings_300m": parse_number(
                        row.get("school_crossings_300m"),
                        "school_crossings_300m"
                    ),

                    "volume_stale": parse_flag(
                        row.get("volume_stale")
                    ),

                    "has_volume": parse_flag(
                        row.get("has_volume")
                    ),
                }

    if requested_site_key is None:
        raise ValueError("The CSV contains no data rows.")

    raise ValueError(
        f"Site key {requested_site_key!r} was not found in {csv_path}."
    )



def search_sites(csv_path, search_text, limit=15):
    """Return matching site keys and descriptions, limited for readability."""
    matches = []
    query = search_text.strip().casefold()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("site_key") or "").strip()
            description = (row.get("site_desc") or "").strip()

            if query in key.casefold() or query in description.casefold():
                matches.append({
                    "site_key": key,
                    "description": description,
                })

    return matches[:limit], len(matches)


def choose_site_interactively(csv_path):
    """Let the user search and select a site without listing every row."""
    print("\nHYBRID AI TRAFFIC-RISK SYSTEM")
    print("Search for an intersection by road name, description, or site key.")
    print("Tip: enter a short road name to find matching intersections.\n")

    while True:
        query = input("Search intersections (or type 'q' to quit): ").strip()

        if query.casefold() in {"q", "quit", "exit"}:
            print("Selection cancelled.")
            return None

        if not query:
            print("Please enter a road name, description, or site key.\n")
            continue

        matches, total = search_sites(csv_path, query)

        if not matches:
            print("No matching intersections found. Try a different search.\n")
            continue

        print(f"\nFound {total} matching intersection(s).")
        if total > len(matches):
            print(f"Showing the first {len(matches)} matches. Narrow your search to see fewer results.")

        for index, match in enumerate(matches, start=1):
            print(f"{index}. {match['description'] or '(no description)'}")
            print(f"   Site key: {match['site_key']}")

        choice = input("Enter a number to select, or press Enter to search again: ").strip()
        if not choice:
            print()
            continue

        try:
            selected_index = int(choice)
        except ValueError:
            print("Please enter one of the displayed numbers.\n")
            continue

        if 1 <= selected_index <= len(matches):
            selected_key = matches[selected_index - 1]["site_key"]
            print(f"\nSelected: {matches[selected_index - 1]['description']}")
            print(f"Site key: {selected_key}\n")
            return selected_key

        print(f"Please enter a number between 1 and {len(matches)}.\n")

def write_json(site, output_path):
    """Save the selected site as a JSON record."""

    output_path.write_text(
        json.dumps(site, indent=2) + "\n",
        encoding="utf-8"
    )


def write_prolog_facts(site, output_path):
    """Write facts; missing numeric values are omitted instead of guessed."""

    sid = site["prolog_site_id"]

    lines = [
        "% Generated by prepare_site.py. Do not edit this generated file manually.",
        f"site({sid}).",
    ]

    numeric_fields = [
        ("traffic_volume", "traffic_volume"),
        ("pedestrian_crossings_200m", "pedestrian_crossings"),
        ("school_crossings_300m", "school_crossings"),
    ]

    for key, predicate in numeric_fields:
        value = site[key]

        if value is not None:
            lines.append(
                f"{predicate}({sid}, {number_for_prolog(value)})."
            )

    # Only write true/false facts when the source field was available.
    flag_fields = [
        ("volume_stale", "volume_stale", "volume_current"),
        ("has_volume", "has_volume", "no_volume"),
    ]

    for key, true_predicate, false_predicate in flag_fields:
        value = site[key]

        if value is not None:
            predicate = (
                true_predicate if value else false_predicate
            )
            lines.append(f"{predicate}({sid}).")

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


def main():
    """Run the data preparation process from the command line."""

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prepare one intersection for the Python-Prolog "
            "starter interface."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to intersection_sites.csv"
    )

    parser.add_argument(
        "--site-key",
        help=(
            "Exact site_key, for example signal:TS:002. "
            "If omitted, an interactive search-and-select menu opens."
        )
    )

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Could not find {args.csv}. Put intersection_sites.csv "
            "in the processed folder or use --csv."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    requested_site_key = args.site_key

    if requested_site_key is None:
        requested_site_key = choose_site_interactively(args.csv)

        if requested_site_key is None:
            return

    site = load_site(
        args.csv,
        requested_site_key
    )

    json_path = OUTPUT_DIR / "site_record.json"
    facts_path = OUTPUT_DIR / "site_facts.pl"

    write_json(
        site,
        json_path
    )

    write_prolog_facts(
        site,
        facts_path
    )

    print("Prepared site record:")
    print(json.dumps(site, indent=2))

    print(f"\nJSON written to: {json_path}")
    print(f"Prolog facts written to: {facts_path}")


if __name__ == "__main__":
    main()
