#!/usr/bin/env python3
"""
Generate dashboard data for the interactive Leaflet dashboard.

Writes a compact summary index to data/projections.json for initial map loading,
and one per-transect detail file under data/projections_details/<site_id>/ for
on-demand chart rendering. The data is committed with the repository so GitHub
Pages serves it next to index.html, so it is kept as small as practical.

Layout of a detail file (data/projections_details/<site_id>/<transect_id>.json):
  {
    "site_id": "nzd0004",
    "transect_id": "nzd0004-0034",
    "historical": {
      "satellite_observation": {"year_decimal": [...], "shoreline_position": [...]},
      "loess_historical":      {"year_decimal": [...], "shoreline_position": [...]}
    },
    "scenarios": {
      "ssp1_1.9": {"year": [2025, 2030, ...], "proj_shoreline_mean": [...], ...},
      ...
    }
  }
Columns are arrays (one entry per row) rather than one object per row, empty
CSV cells are not written, and the historical series — identical for every
scenario because it comes from the same satellite record — is stored once per
transect instead of once per scenario. index.html expands this back into the
row layout its chart code uses and still accepts the older row-based files.

Transects are processed in parallel worker processes. Shoreline positions are
rounded to 2 decimals (1 cm) and decimal years to 3 decimals (~0.4 day).
"""

import csv
import json
import os
import glob
from pathlib import Path

from tqdm.contrib.concurrent import process_map


DEFAULT_SUMMARY_PATH = Path("data/projections.json")
DEFAULT_DETAILS_DIR = Path("data/projections_details")
POSITION_DECIMALS = 2  # 1 cm for shoreline positions in metres
YEAR_DECIMALS = 3  # ~0.4 day for decimal years
MAX_WORKERS = os.cpu_count() or 1

HISTORICAL_SERIES = ("satellite_observation", "loess_historical")
HISTORICAL_COLUMNS = ("year_decimal", "shoreline_position")
PROJECTION_COLUMNS = (
    "year",
    "shoreline_position",
    "proj_shoreline_mean",
    "proj_shoreline_q05",
    "proj_shoreline_q95",
    "slr_only_shoreline_mean",
    "slr_only_shoreline_q05",
    "slr_only_shoreline_q95",
    "trend_only_shoreline_mean",
    "trend_only_shoreline_q05",
    "trend_only_shoreline_q95",
)


def parse_value(column, value):
    """Convert one CSV cell to a compact number, or None when empty/non-numeric."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if column == "year":
        return int(number) if number.is_integer() else round(number, YEAR_DECIMALS)
    if column == "year_decimal":
        return round(number, YEAR_DECIMALS)
    return round(number, POSITION_DECIMALS)


def read_projection_csv(csv_file):
    """Read one projection CSV into columnar series.

    Returns (historical, projection) where historical maps series_type ->
    {column: [values]} for the satellite/LOESS rows and projection is
    {column: [values]} for the projection rows.
    """
    historical = {name: {col: [] for col in HISTORICAL_COLUMNS} for name in HISTORICAL_SERIES}
    projection = {col: [] for col in PROJECTION_COLUMNS}

    with open(csv_file, "r", newline="") as f:
        for row in csv.DictReader(f):
            series_type = row.get("series_type") or "projection"
            if series_type in historical:
                target, columns = historical[series_type], HISTORICAL_COLUMNS
            elif series_type == "projection":
                target, columns = projection, PROJECTION_COLUMNS
            else:
                continue
            for col in columns:
                target[col].append(parse_value(col, row.get(col)))

    return historical, projection


def compute_normalized_change(projection, baseline_year=2025, target_year=2050):
    """Return the projection mean difference between the baseline and target year."""
    years = projection.get("year") or []
    means = projection.get("proj_shoreline_mean") or []
    try:
        baseline = means[years.index(baseline_year)]
        target = means[years.index(target_year)]
    except (ValueError, IndexError):
        return None
    if baseline is None or target is None:
        return None
    return round(target - baseline, POSITION_DECIMALS)


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

    historical = None
    scenarios = {}
    summary_scenarios = {}
    errors = []

    for scenario, csv_file in sorted(scenario_files):
        try:
            scenario_historical, projection = read_projection_csv(csv_file)
        except Exception as e:  # keep going; report at the end
            errors.append(f"{csv_file}: {e}")
            continue

        # The historical series is scenario-independent; keep the longest copy
        # seen in case one scenario's file is truncated.
        n_hist = sum(len(cols["year_decimal"]) for cols in scenario_historical.values())
        if historical is None or n_hist > sum(len(c["year_decimal"]) for c in historical.values()):
            historical = scenario_historical

        scenarios[scenario] = projection
        summary_scenarios[scenario] = {
            "normalized_change_2025_2050": compute_normalized_change(projection)
        }

    if not scenarios:
        return {"transect_id": transect_id, "errors": errors, "summary": None}

    detail_payload = {
        "site_id": site_id,
        "transect_id": transect_id,
        "historical": {name: cols for name, cols in historical.items() if cols["year_decimal"]},
        "scenarios": scenarios,
    }
    site_dir = DEFAULT_DETAILS_DIR / site_id
    site_dir.mkdir(parents=True, exist_ok=True)
    with (site_dir / f"{transect_id}.json").open("w", encoding="utf-8") as f:
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
    - one detail file per transect under data/projections_details/<site_id>/
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
    print(f"✓ Saved {len(summary_data)} per-transect detail files under {DEFAULT_DETAILS_DIR}/<site_id>/")
    if n_errors:
        print(f"✗ {n_errors} CSV file(s) failed to process")
    return summary_data


if __name__ == "__main__":
    generate_projections_json()
