"""Write point-based shoreline projections to Oceanum DataMesh.

This script flattens the per-transect detail files under data/projections_details/
into a long table (transect/scenario/year), optionally joins point geometry from
preprocessing/points_ref_shoreline_2025.geojson, and uploads the result as a
DataMesh datasource.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from oceanum.datamesh import Connector


DEFAULT_JSON_PATH = Path("data/projections_details")
DEFAULT_POINTS_PATH = Path("preprocessing/points_ref_shoreline_2025.geojson")


def build_projection_table(json_path: Path) -> pd.DataFrame:
    records: list[dict] = []
    payload_files = [json_path] if json_path.is_file() else sorted(json_path.glob("*.json"))
    if not payload_files:
        raise ValueError(f"No projection JSON files found in {json_path}")

    for payload_file in payload_files:
        with payload_file.open("r", encoding="utf-8") as f:
            projections = json.load(f)

        if isinstance(projections, dict) and "transect_id" in projections and "scenarios" in projections:
            transect_id = projections.get("transect_id") or payload_file.stem
            site_id = projections.get("site_id")
            scenarios = projections.get("scenarios", {})

            for scenario, time_series in scenarios.items():
                if not isinstance(time_series, list):
                    continue
                for row in time_series:
                    records.append(
                        {
                            "transect_id": transect_id,
                            "site_id": site_id,
                            "scenario": scenario,
                            "year": row.get("year"),
                            "proj_shoreline_mean": row.get("proj_shoreline_mean"),
                            "proj_shoreline_q05": row.get("proj_shoreline_q05"),
                            "proj_shoreline_q95": row.get("proj_shoreline_q95"),
                            "slr_only_shoreline_mean": row.get("slr_only_shoreline_mean"),
                            "slr_only_shoreline_q05": row.get("slr_only_shoreline_q05"),
                            "slr_only_shoreline_q95": row.get("slr_only_shoreline_q95"),
                            "trend_only_shoreline_mean": row.get("trend_only_shoreline_mean"),
                            "trend_only_shoreline_q05": row.get("trend_only_shoreline_q05"),
                            "trend_only_shoreline_q95": row.get("trend_only_shoreline_q95"),
                        }
                    )
            continue

        if isinstance(projections, dict):
            for transect_id, transect_payload in projections.items():
                site_id = transect_payload.get("site_id")
                scenarios = transect_payload.get("scenarios", {})

                for scenario, time_series in scenarios.items():
                    if not isinstance(time_series, list):
                        continue
                    for row in time_series:
                        records.append(
                            {
                                "transect_id": transect_id,
                                "site_id": site_id,
                                "scenario": scenario,
                                "year": row.get("year"),
                                "proj_shoreline_mean": row.get("proj_shoreline_mean"),
                                "proj_shoreline_q05": row.get("proj_shoreline_q05"),
                                "proj_shoreline_q95": row.get("proj_shoreline_q95"),
                                "slr_only_shoreline_mean": row.get("slr_only_shoreline_mean"),
                                "slr_only_shoreline_q05": row.get("slr_only_shoreline_q05"),
                                "slr_only_shoreline_q95": row.get("slr_only_shoreline_q95"),
                                "trend_only_shoreline_mean": row.get("trend_only_shoreline_mean"),
                                "trend_only_shoreline_q05": row.get("trend_only_shoreline_q05"),
                                "trend_only_shoreline_q95": row.get("trend_only_shoreline_q95"),
                            }
                        )

    if not records:
        raise ValueError(f"No projection records found in {json_path}")

    df = pd.DataFrame(records)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.sort_values(["transect_id", "scenario", "year"]).reset_index(drop=True)
    return df


def attach_point_geometry(df: pd.DataFrame, points_path: Path) -> gpd.GeoDataFrame:
    gdf_points = gpd.read_file(points_path)
    if "transect_id" not in gdf_points.columns:
        raise KeyError(f"'transect_id' column not found in {points_path}")

    # Keep point attributes that are useful for filtering/metadata in DataMesh.
    keep_cols = [
        col
        for col in ["transect_id", "site_id", "coastline", "geometry"]
        if col in gdf_points.columns
    ]
    gdf_points = gdf_points[keep_cols].drop_duplicates(subset=["transect_id"])

    merged = df.merge(gdf_points, on="transect_id", how="left", suffixes=("", "_point"))
    gdf_merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf_points.crs)
    return gdf_merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write OCC point projections from JSON to Oceanum DataMesh"
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path to projection detail JSON files or directory (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--points-path",
        type=Path,
        default=DEFAULT_POINTS_PATH,
        help=(
            "Path to points GeoJSON used to attach geometry "
            f"(default: {DEFAULT_POINTS_PATH})"
        ),
    )
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="Upload a plain table without point geometry",
    )
    parser.add_argument(
        "--datasource-id",
        default="occ_point_shoreline_projections",
        help="DataMesh datasource_id",
    )
    parser.add_argument(
        "--name",
        default="OCC shoreline projections (point-level)",
        help="DataMesh datasource name",
    )
    parser.add_argument(
        "--description",
        default=(
            "Point-level shoreline projections by transect, scenario, and year, "
            "derived from data/projections_details/."
        ),
        help="DataMesh datasource description",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("DATAMESH_TOKEN"),
        help="Oceanum DataMesh token (or set DATAMESH_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the upload dataset but do not write to DataMesh",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")

    if not args.no_geometry and not args.points_path.exists():
        raise FileNotFoundError(f"Points file not found: {args.points_path}")

    df = build_projection_table(args.json_path)

    if args.no_geometry:
        upload_data: pd.DataFrame | gpd.GeoDataFrame = df
    else:
        upload_data = attach_point_geometry(df, args.points_path)

    print(f"Rows ready for upload: {len(upload_data):,}")
    print(f"Unique transects: {upload_data['transect_id'].nunique():,}")
    print(f"Unique scenarios: {upload_data['scenario'].nunique():,}")

    if args.dry_run:
        print("Dry run complete. No upload performed.")
        return

    if not args.token:
        raise ValueError("No DataMesh token provided. Use --token or DATAMESH_TOKEN env var.")

    datamesh = Connector(token=args.token)
    datasource = datamesh.write_datasource(
        datasource_id=args.datasource_id,
        name=args.name,
        description=args.description,
        data=upload_data,
    )
    print("Upload complete:", datasource)


if __name__ == "__main__":
    main()
