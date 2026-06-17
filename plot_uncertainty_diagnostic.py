#!/usr/bin/env python3
"""
Create a diagnostic plot of mean uncertainty range for three projection types:
- compound (proj_shoreline)
- slr-only (slr_only_shoreline)
- trend-only (trend_only_shoreline)

Mean uncertainty range is computed as (q95 - q05), aggregated across all transects.

Usage:
  python plot_uncertainty_diagnostic.py
  python plot_uncertainty_diagnostic.py --outputs-dir outputs --scenario ssp1_2.6
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = {
    "compound": ("proj_shoreline_q05", "proj_shoreline_q95"),
    "slr_only": ("slr_only_shoreline_q05", "slr_only_shoreline_q95"),
    "trend_only": ("trend_only_shoreline_q05", "trend_only_shoreline_q95"),
}

COLORS = {
    "compound": "#2ca02c",
    "slr_only": "#ff1493",
    "trend_only": "#000000",
}

LABELS = {
    "compound": "Compound",
    "slr_only": "SLR only",
    "trend_only": "Trend only",
}


def parse_filename_scenario(csv_path: Path) -> str | None:
    """Parse scenario from filename: <site>_<transect>_<scenario>_projection_results.csv."""
    stem = csv_path.stem
    if not stem.endswith("_projection_results"):
        return None

    base = stem[: -len("_projection_results")]
    parts = base.split("_")
    if len(parts) < 3:
        return None

    # Scenario is everything after <site>_<transect>
    return "_".join(parts[2:])


def collect_csv_files(outputs_dir: Path, scenario_filter: str | None) -> list[Path]:
    csv_files = sorted(outputs_dir.rglob("*_projection_results.csv"))
    if scenario_filter is None:
        return csv_files

    selected = []
    for csv_file in csv_files:
        scenario = parse_filename_scenario(csv_file)
        if scenario == scenario_filter:
            selected.append(csv_file)
    return selected


def infer_single_scenario(csv_files: list[Path]) -> str | None:
    scenarios = sorted({parse_filename_scenario(p) for p in csv_files if parse_filename_scenario(p)})
    if len(scenarios) == 1:
        return scenarios[0]
    return None


def aggregate_mean_ranges(
    csv_files: list[Path],
) -> tuple[list[float], dict[str, list[float]], dict[float, dict[str, list[float]]]]:
    # year -> metric -> list of uncertainty widths across transects
    by_year = defaultdict(lambda: {metric: [] for metric in METRICS})

    for csv_file in csv_files:
        with csv_file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    year = float(row["year"])
                except (ValueError, KeyError, TypeError):
                    continue

                for metric, (q05_col, q95_col) in METRICS.items():
                    try:
                        q05 = float(row[q05_col])
                        q95 = float(row[q95_col])
                    except (ValueError, KeyError, TypeError):
                        continue
                    by_year[year][metric].append(q95 - q05)

    years = sorted(by_year.keys())
    mean_ranges = {metric: [] for metric in METRICS}
    for year in years:
        for metric in METRICS:
            vals = by_year[year][metric]
            mean_ranges[metric].append(sum(vals) / len(vals) if vals else float("nan"))

    return years, mean_ranges, by_year


def compute_overall_means(mean_ranges: dict[str, list[float]]) -> dict[str, float]:
    overall = {}
    for metric, vals in mean_ranges.items():
        finite_vals = [v for v in vals if v == v]  # NaN-safe check
        overall[metric] = sum(finite_vals) / len(finite_vals) if finite_vals else float("nan")
    return overall


def make_plot(
    years: list[float],
    mean_ranges: dict[str, list[float]],
    year_2050_ranges: dict[str, list[float]],
    scenario_label: str,
    output_file: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel 1: mean uncertainty range by year
    ax = axes[0]
    for metric in METRICS:
        ax.plot(
            years,
            mean_ranges[metric],
            marker="o",
            linewidth=2,
            markersize=4,
            color=COLORS[metric],
            label=LABELS[metric],
        )
    ax.set_title("Mean uncertainty range by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean uncertainty range (m)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # Panel 2: uncertainty range distribution at 2050
    ax2 = axes[1]
    metrics = list(METRICS.keys())
    box_data = [year_2050_ranges.get(m, []) for m in metrics]

    if any(len(vals) > 0 for vals in box_data):
        bp = ax2.boxplot(
            box_data,
            labels=[LABELS[m] for m in metrics],
            patch_artist=True,
            showfliers=False,
        )

        for patch, metric in zip(bp["boxes"], metrics):
            patch.set_facecolor(COLORS[metric])
            patch.set_alpha(0.55)

        ax2.set_title("Uncertainty range distribution (2050)")
        ax2.set_ylabel("Uncertainty range q95-q05 (m)")
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.set_title("Uncertainty range distribution (2050)")
        ax2.text(0.5, 0.5, "No 2050 data available", ha="center", va="center")
        ax2.set_xticks([])
        ax2.set_yticks([])

    fig.suptitle(f"Uncertainty diagnostic ({scenario_label})", fontsize=13)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean uncertainty diagnostic for projection CSVs")
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing projection CSV files (default: outputs)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Climate scenario key filter (e.g., ssp1_2.6). If omitted, all files are used.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: postprocessing/uncertainty_diagnostic_<scenario|all>.png",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    csv_files = collect_csv_files(outputs_dir, args.scenario)
    if not csv_files:
        raise FileNotFoundError("No projection CSV files found for the requested filter.")

    inferred = infer_single_scenario(csv_files)
    if args.scenario:
        scenario_label = args.scenario
    elif inferred:
        scenario_label = inferred
    else:
        scenario_label = "all_scenarios"

    years, mean_ranges, by_year = aggregate_mean_ranges(csv_files)
    if not years:
        raise RuntimeError("No valid year/uncertainty data found in CSV files.")

    overall_means = compute_overall_means(mean_ranges)
    year_2050_ranges = by_year.get(2050.0, {metric: [] for metric in METRICS})

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path("postprocessing") / f"uncertainty_diagnostic_{scenario_label}.png"

    make_plot(years, mean_ranges, year_2050_ranges, scenario_label, output_file)

    print(f"Processed CSV files: {len(csv_files)}")
    print(f"Scenario label: {scenario_label}")
    print("Overall mean uncertainty range (m):")
    for metric in METRICS:
        print(f"  {LABELS[metric]}: {overall_means[metric]:.3f}")
    print("2050 sample sizes:")
    for metric in METRICS:
        print(f"  {LABELS[metric]}: {len(year_2050_ranges.get(metric, []))}")
    print(f"Saved plot: {output_file}")


if __name__ == "__main__":
    main()
