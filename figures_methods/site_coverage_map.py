#!/usr/bin/env python3
"""Map the sandy-beach sites included in the shoreline-projection analysis."""
from pathlib import Path
import os

# Avoid an incompatible system PROJ database overriding the conda environment.
os.environ.pop("PROJ_DATA", None)
os.environ.pop("PROJ_LIB", None)

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
SITE_POINTS_PATH = ROOT / "preprocessing" / "points_ref_shoreline_2025_Nickupdate.geojson"
REGIONS_PATH = ROOT / "Olds" / "postprocessing" / "regions" / "regional-council-2025.gpkg"
COASTLINE_PATH = ROOT / "Olds" / "postprocessing" / "regions" / "nz-coastlines-topo-150k.gpkg"
OUTPUT_DIR = Path(__file__).resolve().parent
SITE_CRS = "EPSG:2193"
EXPECTED_SITE_COUNT = 534
ANALYSIS_REGIONS = {
    "Auckland",
    "Bay of Plenty Region",
    "Canterbury Region",
    "Gisborne Region",
    "Hawke's Bay Region",
    "Manawatū-Whanganui Region",
    "Marlborough Region",
    "Nelson Region",
    "Northland Region",
    "Otago Region",
    "Southland Region",
    "Taranaki Region",
    "Tasman Region",
    "Waikato Region",
    "Wellington Region",
    "West Coast Region",
}



def load_site_centroids():
    """Return one representative point for each included CoastSat site."""
    points = gpd.read_file(SITE_POINTS_PATH)
    points = points.dropna(subset=["site_id", "geometry"])
    points = points[points["site_id"].astype(str).str.startswith("nzd")]

    site_geometries = (
        points.groupby("site_id")["geometry"]
        .agg(lambda geometries: unary_union(list(geometries)).centroid)
        .reset_index()
    )
    sites = gpd.GeoDataFrame(site_geometries, geometry="geometry", crs=points.crs)
    if len(sites) != EXPECTED_SITE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SITE_COUNT} sites, found {len(sites)} in {SITE_POINTS_PATH}"
        )
    return sites



def add_scale_bar(ax, length_km=100):
    """Add a simple scale bar in the projected NZTM coordinate system."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.68 * (xmax - xmin)
    y0 = ymin + 0.06 * (ymax - ymin)
    length = length_km * 1000
    ax.plot([x0, x0 + length], [y0, y0], color="#20252b", linewidth=2.2, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.006 * (ymax - ymin), y0 + 0.006 * (ymax - ymin)], color="#20252b", linewidth=1)
    ax.plot(
        [x0 + length, x0 + length],
        [y0 - 0.006 * (ymax - ymin), y0 + 0.006 * (ymax - ymin)],
        color="#20252b",
        linewidth=1,
    )
    ax.text(x0 + length / 2, y0 + 0.018 * (ymax - ymin), f"{length_km} km", ha="center", va="bottom", fontsize=8)



def add_north_arrow(ax):
    """Add a north arrow in axes coordinates."""
    arrow = FancyArrowPatch(
        (0.93, 0.12),
        (0.93, 0.22),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=0.9,
        color="#20252b",
    )
    ax.add_patch(arrow)
    ax.text(0.93, 0.235, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, fontweight="bold")



def build_map():
    sites = load_site_centroids().to_crs(SITE_CRS)
    regions = gpd.read_file(REGIONS_PATH).to_crs(SITE_CRS)
    regions = regions[regions["REGC2025_V1_00_NAME"].isin(ANALYSIS_REGIONS)]
    coastline = gpd.read_file(COASTLINE_PATH).to_crs(SITE_CRS)

    fig, ax = plt.subplots(figsize=(5.8, 8.8))
    regions.plot(ax=ax, facecolor="#f3f4f1", edgecolor="#9aa0a6", linewidth=0.45, zorder=1)
    coastline.plot(ax=ax, color="#50565c", linewidth=0.7, alpha=0.7, zorder=2)
    sites.plot(
        ax=ax,
        color="#176b87",
        edgecolor="white",
        linewidth=0.25,
        markersize=9,
        alpha=0.9,
        zorder=3,
    )

    xmin, ymin, xmax, ymax = sites.total_bounds
    padx = 0.06 * (xmax - xmin)
    pady = 0.035 * (ymax - ymin)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")
    ax.axis("off")

    add_scale_bar(ax)
    add_north_arrow(ax)
    fig.tight_layout(pad=0.2)
    return fig



def main():
    fig = build_map()
    png_path = OUTPUT_DIR / "site_coverage_map.png"
    pdf_path = OUTPUT_DIR / "site_coverage_map.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
