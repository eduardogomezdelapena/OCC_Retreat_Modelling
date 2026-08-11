#!/usr/bin/env python3
"""
Generate dashboard data for the interactive Leaflet dashboard.

Writes a compact summary index to data/projections.json for initial map loading,
and one per-transect detail file under data/projections_details/ for on-demand
chart rendering.
"""

import csv
import json
import os
import glob
from pathlib import Path


DEFAULT_SUMMARY_PATH = Path("data/projections.json")
DEFAULT_DETAILS_DIR = Path("data/projections_details")

def csv_to_records(csv_file):
    """Read CSV file and return list of dictionaries."""
    records = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to floats
            converted_row = {}
            for key, value in row.items():
                try:
                    converted_row[key] = float(value)
                except (ValueError, TypeError):
                    converted_row[key] = value
            records.append(converted_row)
    return records


def compute_normalized_change(records, baseline_year=2025, target_year=2050):
    """Return the projection mean difference between the baseline and target year."""
    projection_rows = [row for row in records if row.get("series_type") in (None, "projection")]
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
        return float(target) - float(baseline)
    except (TypeError, ValueError):
        return None


def choose_default_scenario(scenarios):
    """Prefer the compound scenario, falling back to the first available scenario."""
    compound_key = next((scenario for scenario in scenarios if "compound" in scenario.lower()), None)
    if compound_key:
        return compound_key
    return next(iter(sorted(scenarios)), None)

def generate_projections_json(output_dir="outputs", output_file=str(DEFAULT_SUMMARY_PATH)):
    """
    Scan all CSV files in outputs directory and consolidate them into:
    - a compact summary index at data/projections.json
    - one full detail file per transect under data/projections_details/
    
    CSV file naming convention: {site_id}_{transect_id}_{scenario}_projection_results.csv
    Example: nzd0137_nzd0137-0001_ssp1_2.6_projection_results.csv
    """
    
    # Create output directory if needed
    summary_path = Path(output_file)
    details_dir = DEFAULT_DETAILS_DIR
    os.makedirs(summary_path.parent, exist_ok=True)
    os.makedirs(details_dir, exist_ok=True)
    
    summary_data = {}
    detail_data = {}
    
    # Find all CSV files
    csv_files = glob.glob(f"{output_dir}/**/*.csv", recursive=True)
    print(f"Found {len(csv_files)} CSV projection files")
    
    for csv_file in csv_files:
        try:
            # Parse filename to extract transect_id and scenario
            filename = Path(csv_file).stem  # Remove .csv
            parts = filename.split("_projection_results")[0].split("_")
            
            if len(parts) >= 3:
                site_id = parts[0]
                transect_id = parts[1]
                # Scenario is everything after transect_id (e.g., ssp1, 2.6 -> ssp1_2.6)
                scenario = "_".join(parts[2:])
                
                # Read CSV
                records = csv_to_records(csv_file)
                
                # Store full detail data under transect_id
                if transect_id not in detail_data:
                    detail_data[transect_id] = {
                        "site_id": site_id,
                        "transect_id": transect_id,
                        "scenarios": {}
                    }

                detail_data[transect_id]["scenarios"][scenario] = records

                # Store compact summary metadata for the dashboard bootstrap payload.
                if transect_id not in summary_data:
                    summary_data[transect_id] = {
                        "site_id": site_id,
                        "transect_id": transect_id,
                        "detail_file": f"data/projections_details/{transect_id}.json",
                        "default_scenario": None,
                        "scenarios": {}
                    }

                summary_data[transect_id]["scenarios"][scenario] = {
                    "row_count": len(records),
                    "normalized_change_2025_2050": compute_normalized_change(records)
                }

                print(f"  ✓ {transect_id} - {scenario}")
                
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {e}")

    # Fill default scenario fields and write per-transect detail payloads.
    for transect_id, summary_entry in summary_data.items():
        scenario_names = summary_entry["scenarios"].keys()
        summary_entry["default_scenario"] = choose_default_scenario(scenario_names)

        detail_file = details_dir / f"{transect_id}.json"
        with detail_file.open("w", encoding="utf-8") as f:
            json.dump(detail_data[transect_id], f, indent=2)

    # Write summary index for the dashboard bootstrap payload.
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✓ Saved {len(summary_data)} transects to {summary_path}")
    print(f"✓ Saved {len(detail_data)} per-transect detail files to {details_dir}")
    return summary_data

if __name__ == "__main__":
    generate_projections_json()
