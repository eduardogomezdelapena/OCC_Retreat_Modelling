#!/usr/bin/env python3
"""
Generate consolidated projection data for interactive Leaflet dashboard.
Aggregates all CSV projection results into a single JSON file indexed by transect_id.
"""

import csv
import json
import os
import glob
from pathlib import Path

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

def generate_projections_json(output_dir="outputs", output_file="data/projections.json"):
    """
    Scan all CSV files in outputs directory and consolidate into JSON.
    
    CSV file naming convention: {site_id}_{transect_id}_{scenario}_projection_results.csv
    Example: nzd0137_nzd0137-0001_ssp1_2.6_projection_results.csv
    """
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    projections_data = {}
    
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
                
                # Store under transect_id
                if transect_id not in projections_data:
                    projections_data[transect_id] = {
                        "site_id": site_id,
                        "transect_id": transect_id,
                        "scenarios": {}
                    }
                
                projections_data[transect_id]["scenarios"][scenario] = records
                print(f"  ✓ {transect_id} - {scenario}")
                
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {e}")
    
    # Write to JSON
    with open(output_file, "w") as f:
        json.dump(projections_data, f, indent=2)
    
    print(f"\n✓ Saved {len(projections_data)} transects to {output_file}")
    return projections_data

if __name__ == "__main__":
    generate_projections_json()
