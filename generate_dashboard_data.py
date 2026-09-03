#!/usr/bin/env python3
"""
Generate dashboard data for the interactive Leaflet dashboard.

Writes a compact summary index to data/projections.json for initial map loading,
and one per-transect detail file under data/projections_details/ for on-demand
chart rendering.

Layout of a detail file (data/projections_details/<transect_id>.json):
  {
    "site_id": ...,
    "transect_id": ...,
    "historical": [rows with series_type satellite_observation / loess_historical],
    "scenarios": {"ssp1_2.6": [projection rows only], ...}
  }
The historical series is identical for every scenario (it comes from the same
satellite record), so it is stored once per transect instead of once per
scenario; index.html falls back to per-scenario rows for older files.

Transects are processed in parallel worker processes; floats are rounded to
4 decimals (0.1 mm) and JSON is written compactly to keep payloads small.
"""

import csv
import json
import os
import glob
from pathlib import Path

from tqdm.contrib.concurrent import process_map


DEFAULT_SUMMARY_PATH = Path("data/projections.json")
DEFAULT_DETAILS_DIR = Path("data/projections_details")
FLOAT_DECIMALS = 4  # 0.1 mm for shoreline positions in metres
MAX_WORKERS = os.cpu_count() or 1


def csv_to_records(csv_file):
    """Read CSV file and return list of dictionaries with rounded floats."""
    records = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted_row = {}
            for key, value in row.items():
                try:
                    converted_row[key] = round(float(value), FLOAT_DECIMALS)
                except (ValueError, TypeError):
                    converted_row[key] = value
            records.append(converted_row)
    return records


def compute_normalized_change(projection_rows, baseline_year=2025, target_year=2050):
    """Return the projection mean difference between the baseline and target year."""
    if not projection_rows:
        return None

    baseline_row = next((row for row in projection_rows if int(row.get("year", -1)) == baseline_year), None)
    target_row = next((row for row in projection_rows if int(row.get("year", -1)) == target_year), None)
    if not baseline_row or not target_row:
        return None

    baseline = baseline_row.get("proj_shoreline_mean")
    target = target_row.get("proj_shoreline_mean")
    if baseline is None or target is None:
        return None

    try:
        return round(float(target) - float(baseline), FLOAT_DECIMALS)
    except (TypeError, ValueError):
        return None


def choose_default_scenario(scenarios):
    """Prefer the compound scenario, falling back to the first available scenario."""
    compound_key = next((scenario for scenario in scenarios if "compound" in scenario.lower()), None)
    if compound_key:
        return compound_key
    return next(iter(sorted(scenarios)), None)


def parse_csv_name(csv_file):
    """Split a projection CSV path into (site_id, transect_id, scenario) or None.

    Naming convention: {site_id}_{transect_id}_{scenario}_projection_results.csv
    Example: nzd0137_nzd0137-0001_ssp1_2.6_projection_results.csv
    """
    filename = Path(csv_file).stem
    parts = filename.split("_projection_results")[0].split("_")
    if len(parts) < 3:
        return None
    return parts[0], parts[1], "_".join(parts[2:])


def process_transect(job):
    """Build one transect's detail file and return its summary entry.

    `job` is (transect_id, site_id, [(scenario, csv_path), ...]).
    Runs in a worker process; the detail JSON is written here so the parent
    only has to merge the (small) summary entries.
    """
    transect_id, site_id, scenario_files = job

    historical = []
    scenarios = {}
    summary_scenarios = {}
    errors = []

    for scenario, csv_file in sorted(scenario_files):
        try:
            records = csv_to_records(csv_file)
        except Exception as e:  # keep going; report at the end
            errors.append(f"{csv_file}: {e}")
            continue

        projection_rows = [r for r in records if r.get("series_type") in (None, "", "projection")]
        historical_rows = [r for r in records if r.get("series_type") not in (None, "", "projection")]

        # The historical (satellite + LOESS) series is scenario-independent;
        # keep the longest copy seen in case a scenario file is truncated.
        if len(historical_rows) > len(historical):
            historical = historical_rows

        scenarios[scenario] = projection_rows
        summary_scenarios[scenario] = {
            "normalized_change_2025_2050": compute_normalized_change(projection_rows)
        }

    if not scenarios:
        return {"transect_id": transect_id, "errors": errors, "summary": None}

    detail_payload = {
        "site_id": site_id,
        "transect_id": transect_id,
        "historical": historical,
        "scenarios": scenarios,
    }
    detail_file = DEFAULT_DETAILS_DIR / f"{transect_id}.json"
    with detail_file.open("w", encoding="utf-8") as f:
        json.dump(detail_payload, f, separators=(",", ":"))

    summary_entry = {
        "site_id": site_id,
        "transect_id": transect_id,
        "default_scenario": choose_default_scenario(scenarios.keys()),
        "scenarios": summary_scenarios,
    }
    return {"transect_id": transect_id, "errors": errors, "summary": summary_entry}


def generate_projections_json(output_dir="outputs", output_file=str(DEFAULT_SUMMARY_PATH)):
    """
    Scan all projection CSV files in the outputs directory and consolidate them into:
    - a compact summary index at data/projections.json
    - one full detail file per transect under data/projections_details/
    """
    summary_path = Path(output_file)
    os.makedirs(summary_path.parent, exist_ok=True)
    os.makedirs(DEFAULT_DETAILS_DIR, exist_ok=True)

    csv_files = glob.glob(f"{output_dir}/**/*_projection_results.csv", recursive=True)
    print(f"Found {len(csv_files)} CSV projection files")

    # Group the scenario CSVs by transect so each worker writes one detail file.
    transects = {}
    for csv_file in csv_files:
        parsed = parse_csv_name(csv_file)
        if parsed is None:
            print(f"  ✗ Unrecognised CSV name, skipped: {csv_file}")
            continue
        site_id, transect_id, scenario = parsed
        transects.setdefault(transect_id, (site_id, []))[1].append((scenario, csv_file))

    jobs = [(tid, site_id, files) for tid, (site_id, files) in sorted(transects.items())]
    n_workers = min(MAX_WORKERS, os.cpu_count() or 1, max(1, len(jobs)))
    results = process_map(
        process_transect,
        jobs,
        max_workers=n_workers,
        chunksize=32,
        desc="Building transect detail files",
    )

    summary_data = {}
    n_errors = 0
    for result in results:
        for err in result["errors"]:
            n_errors += 1
            print(f"  ✗ Error processing {err}")
        if result["summary"] is not None:
            summary_data[result["transect_id"]] = result["summary"]

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, separators=(",", ":"))

    print(f"\n✓ Saved {len(summary_data)} transects to {summary_path}")
    print(f"✓ Saved {len(summary_data)} per-transect detail files to {DEFAULT_DETAILS_DIR}")
    if n_errors:
        print(f"✗ {n_errors} CSV file(s) failed to process")
    return summary_data


if __name__ == "__main__":
    generate_projections_json()
