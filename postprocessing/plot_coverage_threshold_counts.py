#!/usr/bin/env python3
"""Plot transect counts above vs at/below a coverage threshold per site.

Reads site-level summary CSVs produced by loess_trend_coverage.py:
  postprocessing/coverage_results/<site>/<site>_all_transects_loess_trend_coverage.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def collect_site_summary_csvs(coverage_root: Path):
    """Return all site summary CSVs under coverage_results."""
    return sorted(coverage_root.glob("*/**/*_all_transects_loess_trend_coverage.csv"))


def build_counts_table(site_csv_paths, threshold_percent: float):
    """Build a per-site table of transect counts relative to threshold."""
    rows = []
    for csv_path in site_csv_paths:
        df = pd.read_csv(csv_path)
        if "coverage_percent" not in df.columns:
            continue

        site_id = csv_path.stem.replace("_all_transects_loess_trend_coverage", "")
        coverage = pd.to_numeric(df["coverage_percent"], errors="coerce")
        coverage = coverage.dropna()

        above = int((coverage > threshold_percent).sum())
        at_or_below = int((coverage <= threshold_percent).sum())
        total = int(coverage.size)

        rows.append(
            {
                "site_id": site_id,
                "total_transects": total,
                "count_above_threshold": above,
                "count_at_or_below_threshold": at_or_below,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "site_id",
                "total_transects",
                "count_above_threshold",
                "count_at_or_below_threshold",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values("site_id").reset_index(drop=True)


def plot_counts(counts_df: pd.DataFrame, threshold_percent: float, output_path: Path):
    """Create a stacked bar plot of counts by site."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(counts_df)), 6))

    x = range(len(counts_df))
    below = counts_df["count_at_or_below_threshold"].to_numpy()
    above = counts_df["count_above_threshold"].to_numpy()

    ax.bar(x, below, color="tab:red", alpha=0.8, label=f"<= {threshold_percent:.0f}%")
    ax.bar(x, above, bottom=below, color="tab:green", alpha=0.8, label=f"> {threshold_percent:.0f}%")

    ax.set_xticks(list(x))
    ax.set_xticklabels(counts_df["site_id"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Number of transects")
    ax.set_xlabel("Site")
    ax.set_title(
        "Transect counts by LOESS-in-envelope coverage threshold\n"
        f"(above {threshold_percent:.0f}% vs {threshold_percent:.0f}% or below)"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_total_counts(counts_df: pd.DataFrame, threshold_percent: float, output_path: Path):
    """Create a pie-chart diagnostic plot for totals across all sites."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_above = int(counts_df["count_above_threshold"].sum())
    total_at_or_below = int(counts_df["count_at_or_below_threshold"].sum())
    total = total_above + total_at_or_below

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    sizes = [total_above, total_at_or_below]
    labels = [f"> {threshold_percent:.0f}%", f"<= {threshold_percent:.0f}%"]
    colors = ["tab:green", "tab:red"]

    def autopct_fmt(pct):
        absolute = int(round(pct / 100.0 * total))
        return f"{pct:.1f}%\n(n={absolute})"

    wedges, text_labels, text_pcts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=autopct_fmt,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        textprops={"fontsize": 10},
    )
    for txt in text_labels + text_pcts:
        txt.set_color("black")

    ax.set_title(
        "Total transect counts by coverage threshold\n"
        f"(above {threshold_percent:.0f}% vs {threshold_percent:.0f}% or below)"
    )
    ax.axis("equal")

    summary_text = (
        f"Total transects: {total}\n"
        f"> {threshold_percent:.0f}%: {total_above}\n"
        f"<= {threshold_percent:.0f}%: {total_at_or_below}"
    )
    fig.text(
        0.03,
        0.02,
        summary_text,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot counts of transects above vs at/below a coverage threshold per site."
    )
    parser.add_argument(
        "--coverage-root",
        default=str(PROJECT_ROOT / "postprocessing" / "coverage_results"),
        help="Root directory containing per-site coverage results.",
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=50.0,
        help="Threshold in percent for splitting transect counts.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Output PNG path. Defaults to coverage_root/coverage_threshold_counts.png",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path for per-site counts. Defaults to coverage_root/coverage_threshold_counts.csv",
    )
    parser.add_argument(
        "--output-total-plot",
        default=None,
        help="Output PNG path for total (all-sites combined) counts. Defaults to coverage_root/coverage_threshold_counts_total.png",
    )
    args = parser.parse_args()

    coverage_root = Path(args.coverage_root)
    site_csv_paths = collect_site_summary_csvs(coverage_root)
    if not site_csv_paths:
        raise FileNotFoundError(f"No site summary CSVs found under {coverage_root}")

    counts_df = build_counts_table(site_csv_paths, threshold_percent=float(args.threshold_percent))
    if counts_df.empty:
        raise ValueError("No valid coverage_percent values found in site summary CSVs.")

    output_plot = (
        Path(args.output_plot)
        if args.output_plot is not None
        else coverage_root / "coverage_threshold_counts.png"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else coverage_root / "coverage_threshold_counts.csv"
    )
    output_total_plot = (
        Path(args.output_total_plot)
        if args.output_total_plot is not None
        else coverage_root / "coverage_threshold_counts_total.png"
    )

    counts_df.to_csv(output_csv, index=False)
    plot_counts(counts_df, threshold_percent=float(args.threshold_percent), output_path=output_plot)
    plot_total_counts(
        counts_df,
        threshold_percent=float(args.threshold_percent),
        output_path=output_total_plot,
    )

    print(f"Processed {len(counts_df)} sites")
    print(f"Saved counts table to {output_csv}")
    print(f"Saved diagnostic plot to {output_plot}")
    print(f"Saved total diagnostic plot to {output_total_plot}")


if __name__ == "__main__":
    main()
