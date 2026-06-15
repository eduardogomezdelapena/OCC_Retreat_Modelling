#!/usr/bin/env python3
"""Postprocess one transect to measure LOESS coverage inside the trend-only envelope.

The metric is defined on the post-projection-start portion of the LOESS-smoothed
curve only. For each LOESS evaluation point at or after the projection start year,
the script checks whether the smoothed shoreline value falls inside the trend-only
projection envelope (q05 to q95), using linear interpolation of the saved envelope
onto the LOESS evaluation years.

This is intended to be cheap to rerun after projections already exist.
"""
#%%
from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loess.loess_1d import loess_1d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COASTSAT_RAW_URL = (
    "https://raw.githubusercontent.com/UoA-eResearch/CoastSat/main/data/"
    "{site_id}/transect_time_series_tidally_corrected_smoothed.csv"
)


def load_transect_data(site_id: str):
    """Load the CoastSat transect time series and convert timestamps to decimal years."""
    url = COASTSAT_RAW_URL.format(site_id=site_id)
    df = pd.read_csv(url, header=0)
    df["dates"] = pd.to_datetime(df["dates"])
    t_years = df["dates"].dt.year + (df["dates"].dt.dayofyear - 1) / 365.25
    return t_years, df


def build_loess_time_grid(t):
    """Shift time to a local origin so LOESS smoothing matches the existing workflow."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("LOESS time input must be one-dimensional.")
    if t.size == 0:
        return t.copy(), t.copy(), 0.0

    t0 = float(np.min(t))
    t_numeric = t - t0
    t_grid = t_numeric.copy()
    return t_numeric, t_grid, t0


def filter_to_longest_consecutive_year_run(
    t,
    y,
    dates,
    min_span_years,
    max_gap_months,
    site_id,
    transect_id,
):
    """Keep the most recent consecutive record segment that spans the minimum window."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))

    meta = {
        "site_id": site_id,
        "transect_id": transect_id,
        "status": None,
        "kept_years": [],
        "removed_years": [],
        "segment_start": None,
        "segment_end": None,
        "segment_span_years": None,
        "n_long_gaps": 0,
    }

    if y.size == 0:
        meta["status"] = "empty"
        return t[:0], y[:0], meta

    if t.size != y.size or t.size != dates.size:
        raise ValueError("Time, shoreline, and date arrays must have the same length.")

    order = np.argsort(dates)
    t = t[order]
    y = y[order]
    dates = dates[order]

    all_years = sorted(np.unique(dates.year).tolist())

    previous_dates = pd.Series(dates[:-1])
    next_dates = pd.Series(dates[1:])
    gap_breaks = next_dates > (previous_dates + pd.DateOffset(months=int(max_gap_months)))
    break_indices = np.where(gap_breaks.to_numpy())[0] + 1
    meta["n_long_gaps"] = int(break_indices.size)

    segment_starts = np.r_[0, break_indices]
    segment_ends = np.r_[break_indices, t.size]

    selected_segment = None
    for start_idx, end_idx in zip(segment_starts[::-1], segment_ends[::-1]):
        if end_idx - start_idx < 2:
            continue

        span_years = (dates[end_idx - 1] - dates[start_idx]).days / 365.25
        if span_years >= min_span_years:
            selected_segment = (start_idx, end_idx, span_years)
            break

    if selected_segment is None:
        meta["status"] = "insufficient_recent_span"
        meta["removed_years"] = all_years
        return t[:0], y[:0], meta

    start_idx, end_idx, span_years = selected_segment
    mask = np.zeros(t.size, dtype=bool)
    mask[start_idx:end_idx] = True

    kept_dates = dates[mask]
    kept_years = sorted(np.unique(kept_dates.year).tolist())

    meta["status"] = "ok"
    meta["kept_years"] = kept_years
    meta["removed_years"] = sorted(set(all_years) - set(kept_years))
    meta["segment_start"] = kept_dates[0].isoformat()
    meta["segment_end"] = kept_dates[-1].isoformat()
    meta["segment_span_years"] = float(span_years)

    return t[mask], y[mask], meta


def compute_loess_curve(site_id: str, transect_id: str, loess_window: float = 10.0):
    """Compute the same LOESS curve used in the projection plot."""
    t_years, df = load_transect_data(site_id)

    if transect_id not in df.columns:
        raise KeyError(f"Transect column not found in CoastSat data: {transect_id}")

    y = df[transect_id]
    mask = np.isfinite(y)
    t_clean = t_years[mask].values
    y_clean = y[mask].values
    dates_clean = df.loc[mask, "dates"].values

    t_clean, y_clean, meta = filter_to_longest_consecutive_year_run(
        t_clean,
        y_clean,
        dates_clean,
        min_span_years=loess_window,
        max_gap_months=10,
        site_id=site_id,
        transect_id=transect_id,
    )

    if meta["status"] != "ok":
        raise ValueError(
            f"{site_id} {transect_id}: {meta['status']} (best_run_len={meta.get('best_run_len')})"
        )

    t_loess, t_loess_grid, t_loess_origin = build_loess_time_grid(t_clean)

    total_timespan_years = float(t_clean.max() - t_clean.min())
    frac = loess_window / total_timespan_years
    frac = float(np.clip(frac, 0.15, 0.9))

    x_smooth, y_smooth, _ = loess_1d(
        t_loess,
        y_clean,
        xnew=t_loess_grid,
        frac=frac,
        degree=1,
    )

    t_smooth = x_smooth + t_loess_origin
    return {
        "t_obs": t_clean,
        "y_obs": y_clean,
        "t_loess": t_smooth,
        "y_loess": y_smooth,
        "meta": meta,
    }


def compute_loess_curve_from_df(
    site_id: str,
    transect_id: str,
    t_years,
    df: pd.DataFrame,
    loess_window: float = 10.0,
):
    """Compute LOESS for one transect using preloaded site data."""
    if transect_id not in df.columns:
        raise KeyError(f"Transect column not found in CoastSat data: {transect_id}")

    y = df[transect_id]
    mask = np.isfinite(y)
    t_clean = t_years[mask].values
    y_clean = y[mask].values
    dates_clean = df.loc[mask, "dates"].values

    t_clean, y_clean, meta = filter_to_longest_consecutive_year_run(
        t_clean,
        y_clean,
        dates_clean,
        min_span_years=loess_window,
        max_gap_months=10,
        site_id=site_id,
        transect_id=transect_id,
    )

    if meta["status"] != "ok":
        raise ValueError(
            f"{site_id} {transect_id}: {meta['status']} (best_run_len={meta.get('best_run_len')})"
        )

    t_loess, t_loess_grid, t_loess_origin = build_loess_time_grid(t_clean)

    total_timespan_years = float(t_clean.max() - t_clean.min())
    frac = loess_window / total_timespan_years
    frac = float(np.clip(frac, 0.15, 0.9))

    x_smooth, y_smooth, _ = loess_1d(
        t_loess,
        y_clean,
        xnew=t_loess_grid,
        frac=frac,
        degree=1,
    )

    t_smooth = x_smooth + t_loess_origin
    return {
        "t_obs": t_clean,
        "y_obs": y_clean,
        "t_loess": t_smooth,
        "y_loess": y_smooth,
        "meta": meta,
    }


def find_projection_csv(site_id: str, transect_id: str, outputs_dir: Path):
    """Find the saved projection CSV for one site/transect."""
    matches = sorted(outputs_dir.glob(f"{site_id}/{site_id}_{transect_id}_*_projection_results.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No projection CSV found for {site_id} {transect_id} under {outputs_dir}"
        )
    if len(matches) > 1:
        names = "\n".join(str(path) for path in matches)
        raise FileExistsError(
            f"Multiple projection CSVs found for {site_id} {transect_id}; pass --projection-csv explicitly:\n{names}"
        )
    return matches[0]


def list_projection_csvs_for_site(site_id: str, outputs_dir: Path):
    """Return a mapping transect_id -> projection CSV path for all transects in one site."""
    site_dir = outputs_dir / site_id
    if not site_dir.exists():
        raise FileNotFoundError(f"Site outputs directory not found: {site_dir}")

    pattern = re.compile(
        rf"^{re.escape(site_id)}_(?P<transect>[^_]+)_.+_projection_results$"
    )
    transect_to_csv = {}
    for csv_path in sorted(site_dir.glob("*_projection_results.csv")):
        stem = csv_path.stem
        match = pattern.match(stem)
        if not match:
            continue
        transect_id = match.group("transect")
        if transect_id in transect_to_csv:
            raise FileExistsError(
                f"Multiple projection CSVs found for {site_id} {transect_id}. "
                "Use --transect-id or --projection-csv to disambiguate."
            )
        transect_to_csv[transect_id] = csv_path

    if not transect_to_csv:
        raise FileNotFoundError(f"No projection CSV files found under {site_dir}")

    return transect_to_csv


def list_sites_with_projection_outputs(outputs_dir: Path):
    """List site IDs that have at least one projection-results CSV under outputs/."""
    sites = []
    for site_dir in sorted(outputs_dir.iterdir()):
        if not site_dir.is_dir() or not site_dir.name.startswith("nzd"):
            continue
        if any(site_dir.glob("*_projection_results.csv")):
            sites.append(site_dir.name)
    return sites


def process_transect(
    site_id: str,
    transect_id: str,
    projection_csv: Path,
    output_dir: Path,
    loess_window: float,
    evaluation_end_year: float,
    save_plot: bool,
    t_years=None,
    df: pd.DataFrame | None = None,
):
    """Run the coverage workflow for one transect and save outputs."""
    proj_df = pd.read_csv(projection_csv)
    required_columns = {
        "year",
        "trend_only_shoreline_mean",
        "trend_only_shoreline_q05",
        "trend_only_shoreline_q95",
    }
    missing = required_columns - set(proj_df.columns)
    if missing:
        raise KeyError(f"Projection CSV is missing required columns: {sorted(missing)}")

    if df is None or t_years is None:
        loess_result = compute_loess_curve(site_id, transect_id, loess_window=loess_window)
    else:
        loess_result = compute_loess_curve_from_df(
            site_id,
            transect_id,
            t_years,
            df,
            loess_window=loess_window,
        )

    coverage = compute_coverage_ratio(
        loess_result["t_loess"],
        loess_result["y_loess"],
        proj_df,
        evaluation_end_year=evaluation_end_year,
    )

    summary = pd.DataFrame(
        [
            {
                "site_id": site_id,
                "transect_id": transect_id,
                "projection_csv": str(projection_csv),
                "projection_start_year": coverage["projection_start_year"],
                "projection_end_year": coverage["projection_end_year"],
                "evaluation_end_year": coverage["evaluation_end_year"],
                "loess_window_years": float(loess_window),
                "eval_points": coverage["eval_points"],
                "inside_points": coverage["inside_points"],
                "coverage_ratio": coverage["coverage_ratio"],
                "coverage_percent": coverage["coverage_percent"],
                "loess_segment_start": loess_result["meta"]["segment_start"],
                "loess_segment_end": loess_result["meta"]["segment_end"],
                "loess_segment_span_years": loess_result["meta"]["segment_span_years"],
            }
        ]
    )

    summary_csv = output_dir / f"{site_id}_{transect_id}_loess_trend_coverage.csv"
    summary.to_csv(summary_csv, index=False)

    if save_plot:
        plot_fp = output_dir / f"{site_id}_{transect_id}_loess_trend_coverage.png"
        save_diagnostic_plot(site_id, transect_id, proj_df, coverage, plot_fp)
    else:
        plot_fp = None

    return coverage, summary.iloc[0].to_dict(), summary_csv, plot_fp


def compute_coverage_ratio(t_loess, y_loess, proj_df, evaluation_end_year=None):
    """Measure how much of the post-start LOESS curve lies inside the trend-only band."""
    projection_years = proj_df["year"].to_numpy(dtype=float)
    trend_q05 = proj_df["trend_only_shoreline_q05"].to_numpy(dtype=float)
    trend_q95 = proj_df["trend_only_shoreline_q95"].to_numpy(dtype=float)

    projection_start_year = float(projection_years[0])
    projection_end_year = float(projection_years[-1])

    if evaluation_end_year is None:
        effective_end_year = projection_end_year
    else:
        effective_end_year = min(float(evaluation_end_year), projection_end_year)

    eval_mask = (t_loess >= projection_start_year) & (t_loess <= effective_end_year)
    eval_years = t_loess[eval_mask]
    eval_loess = y_loess[eval_mask]

    if eval_years.size == 0:
        return {
            "projection_start_year": projection_start_year,
            "projection_end_year": projection_end_year,
            "evaluation_end_year": effective_end_year,
            "eval_points": 0,
            "inside_points": 0,
            "coverage_ratio": np.nan,
            "coverage_percent": np.nan,
            "eval_years": eval_years,
            "eval_loess": eval_loess,
            "envelope_q05": np.array([], dtype=float),
            "envelope_q95": np.array([], dtype=float),
            "inside_mask": np.array([], dtype=bool),
        }

    envelope_q05 = np.interp(eval_years, projection_years, trend_q05)
    envelope_q95 = np.interp(eval_years, projection_years, trend_q95)
    inside_mask = (eval_loess >= envelope_q05) & (eval_loess <= envelope_q95)

    return {
        "projection_start_year": projection_start_year,
        "projection_end_year": projection_end_year,
        "evaluation_end_year": effective_end_year,
        "eval_points": int(eval_years.size),
        "inside_points": int(np.count_nonzero(inside_mask)),
        "coverage_ratio": float(np.mean(inside_mask)),
        "coverage_percent": float(np.mean(inside_mask) * 100.0),
        "eval_years": eval_years,
        "eval_loess": eval_loess,
        "envelope_q05": envelope_q05,
        "envelope_q95": envelope_q95,
        "inside_mask": inside_mask,
    }


def save_diagnostic_plot(
    site_id: str,
    transect_id: str,
    proj_df: pd.DataFrame,
    result: dict,
    output_path: Path,
):
    """Save a simple plot that shows the post-start LOESS coverage result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projection_years = proj_df["year"].to_numpy(dtype=float)
    trend_q05 = proj_df["trend_only_shoreline_q05"].to_numpy(dtype=float)
    trend_q95 = proj_df["trend_only_shoreline_q95"].to_numpy(dtype=float)
    trend_mean = proj_df["trend_only_shoreline_mean"].to_numpy(dtype=float)

    plot_mask = projection_years <= result["evaluation_end_year"]
    projection_years = projection_years[plot_mask]
    trend_q05 = trend_q05[plot_mask]
    trend_q95 = trend_q95[plot_mask]
    trend_mean = trend_mean[plot_mask]

    eval_years = result["eval_years"]
    eval_loess = result["eval_loess"]
    inside_mask = result["inside_mask"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(
        projection_years,
        trend_q05,
        trend_q95,
        color="0.8",
        alpha=0.6,
        label="Trend-only envelope (5-95%)",
    )
    ax.plot(
        projection_years,
        trend_mean,
        color="black",
        linewidth=1.8,
        alpha=0.7,
        label="Trend-only mean",
    )
    ax.plot(
        eval_years,
        eval_loess,
        color="black",
        linewidth=2.0,
        label="LOESS (post-start)",
    )
    if eval_years.size > 0:
        ax.scatter(
            eval_years[inside_mask],
            eval_loess[inside_mask],
            color="tab:green",
            s=25,
            zorder=3,
            label="Inside envelope",
        )
        ax.scatter(
            eval_years[~inside_mask],
            eval_loess[~inside_mask],
            color="tab:red",
            s=25,
            zorder=3,
            label="Outside envelope",
        )

    ax.axvline(result["projection_start_year"], color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlim(result["projection_start_year"], result["evaluation_end_year"])
    ax.set_title(
        f"LOESS vs trend-only envelope: {site_id} {transect_id}\n"
        f"Coverage ratio = {result['coverage_ratio']:.3f} ({result['inside_points']}/{result['eval_points']})"
    )
    coverage_text = (
        f"Inside envelope: {result['coverage_percent']:.1f}%\n"
        f"({result['inside_points']}/{result['eval_points']} LOESS points)"
    )
    ax.text(
        0.02,
        0.98,
        coverage_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax.set_xlabel("Year (decimal)")
    ax.set_ylabel("Shoreline position [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Measure how much post-start LOESS lies inside the trend-only envelope."
    )
    parser.add_argument("--site-id", default="nzd0144", help="CoastSat site ID.")
    parser.add_argument(
        "--transect-id",
        default="nzd0144-0002",
        help="Transect ID to process.",
    )
    parser.add_argument(
        "--projection-csv",
        default=None,
        help="Explicit projection CSV path. If omitted, the script searches outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "postprocessing" / "coverage_results"),
        help="Directory for summary CSV and optional plot.",
    )
    parser.add_argument(
        "--loess-window",
        type=float,
        default=10.0,
        help="LOESS window length in years.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a diagnostic plot alongside the summary CSV.",
    )
    parser.add_argument(
        "--all-transects",
        action="store_true",
        help="Process all transects for the site based on available projection CSVs.",
    )
    parser.add_argument(
        "--evaluation-end-year",
        type=float,
        default=2030.0,
        help="Last year included in the LOESS-vs-envelope coverage metric and diagnostic plot.",
    )
    parser.add_argument(
        "--random-sites-count",
        type=int,
        default=0,
        help="Number of additional random sites (with available projections) to process.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for selecting additional sites.",
    )
    args = parser.parse_args()

    outputs_dir = PROJECT_ROOT / "outputs"
    output_dir = Path(args.output_dir)

    if args.random_sites_count < 0:
        raise ValueError("--random-sites-count must be >= 0")
    if args.random_sites_count > 0 and not args.all_transects:
        raise ValueError("--random-sites-count requires --all-transects")

    site_ids_to_process = [args.site_id]
    if args.random_sites_count > 0:
        available_sites = list_sites_with_projection_outputs(outputs_dir)
        candidate_sites = [s for s in available_sites if s != args.site_id]
        if args.random_sites_count > len(candidate_sites):
            raise ValueError(
                f"Requested {args.random_sites_count} random sites, but only "
                f"{len(candidate_sites)} are available (excluding {args.site_id})."
            )
        rng = random.Random(args.random_seed)
        sampled_sites = sorted(rng.sample(candidate_sites, k=args.random_sites_count))
        site_ids_to_process.extend(sampled_sites)
        print(
            f"Randomly selected {args.random_sites_count} additional sites "
            f"(seed={args.random_seed}): {', '.join(sampled_sites)}"
        )

    for current_site_id in site_ids_to_process:
        site_start = time.perf_counter()
        site_output_dir = output_dir / current_site_id
        site_output_dir.mkdir(parents=True, exist_ok=True)

        if args.all_transects:
            transect_to_csv = list_projection_csvs_for_site(current_site_id, outputs_dir)
            t_years, df = load_transect_data(current_site_id)
            all_rows = []
            failures = []
            total_transects = len(transect_to_csv)

            print(f"Processing site {current_site_id}: {total_transects} transects")

            for idx, (transect_id, projection_csv) in enumerate(transect_to_csv.items(), start=1):
                try:
                    coverage, row, summary_csv, plot_fp = process_transect(
                        site_id=current_site_id,
                        transect_id=transect_id,
                        projection_csv=projection_csv,
                        output_dir=site_output_dir,
                        loess_window=args.loess_window,
                        evaluation_end_year=args.evaluation_end_year,
                        save_plot=args.save_plot,
                        t_years=t_years,
                        df=df,
                    )
                    all_rows.append(row)
                except Exception as exc:
                    failures.append({
                        "site_id": current_site_id,
                        "transect_id": transect_id,
                        "projection_csv": str(projection_csv),
                        "error": str(exc),
                    })

                success_count = len(all_rows)
                failure_count = len(failures)
                print(
                    f"\r[{current_site_id}] Progress: {idx}/{total_transects} transects "
                    f"(ok={success_count}, failed={failure_count})",
                    end="",
                    flush=True,
                )

            print()

            site_summary_csv = site_output_dir / f"{current_site_id}_all_transects_loess_trend_coverage.csv"
            pd.DataFrame(all_rows).to_csv(site_summary_csv, index=False)
            print(f"Saved site summary to {site_summary_csv}")

            if failures:
                failures_csv = site_output_dir / f"{current_site_id}_all_transects_loess_trend_coverage_failures.csv"
                pd.DataFrame(failures).to_csv(failures_csv, index=False)
                print(f"Saved failures to {failures_csv}")

            elapsed_seconds = time.perf_counter() - site_start
            print(
                f"Site {current_site_id} processed in {elapsed_seconds:.2f} s "
                f"({elapsed_seconds / 60.0:.2f} min)"
            )

        else:
            if args.projection_csv is None:
                projection_csv = find_projection_csv(current_site_id, args.transect_id, outputs_dir)
            else:
                projection_csv = Path(args.projection_csv)

            coverage, _, summary_csv, plot_fp = process_transect(
                site_id=current_site_id,
                transect_id=args.transect_id,
                projection_csv=projection_csv,
                output_dir=site_output_dir,
                loess_window=args.loess_window,
                evaluation_end_year=args.evaluation_end_year,
                save_plot=args.save_plot,
            )

            print(
                f"{current_site_id} {args.transect_id}: {coverage['inside_points']}/{coverage['eval_points']} "
                f"post-start LOESS points inside trend-only envelope through {coverage['evaluation_end_year']:.0f} "
                f"({coverage['coverage_percent']:.1f}%)"
            )
            print(f"Saved summary to {summary_csv}")
            if plot_fp is not None:
                print(f"Saved diagnostic plot to {plot_fp}")

            elapsed_seconds = time.perf_counter() - site_start
            print(
                f"Site {current_site_id} processed in {elapsed_seconds:.2f} s "
                f"({elapsed_seconds / 60.0:.2f} min)"
            )


if __name__ == "__main__":
    main()
# %%
