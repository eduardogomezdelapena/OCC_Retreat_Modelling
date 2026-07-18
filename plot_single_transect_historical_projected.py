#!/usr/bin/env python3
"""Create a historical + projected trend-only shoreline plot for one transect.

Design goals for the output figure:
- Historical satellite observations as semi-transparent grey points.
- Historical LOESS-smoothed trend line.
- Projected trend-only time series split into colored segments:
  2025-2030, 2030-2040, 2040-2050.
- Segment uncertainty bands using trend-only q05/q95.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def decimal_year(dates: pd.Series) -> np.ndarray:
    """Convert datetimes to decimal years."""
    return (
        dates.dt.year.to_numpy(dtype=float)
        + (dates.dt.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    )


def loess_smooth_time_series(
    t_years: np.ndarray,
    y: np.ndarray,
    loess_window_years: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth shoreline observations against time using LOESS."""
    t = np.asarray(t_years, dtype=float)
    y = np.asarray(y, dtype=float)

    t0 = float(np.min(t))
    x = t - t0

    span = float(np.max(x) - np.min(x))
    if span <= 0.0:
        return t.copy(), y.copy()

    frac = np.clip(loess_window_years / span, 0.05, 1.0)

    try:
        from loess.loess_1d import loess_1d

        x_smooth, y_smooth, _ = loess_1d(
            x,
            y,
            frac=frac,
            xnew=x,
            degree=1,
        )
        return x_smooth + t0, y_smooth
    except Exception:
        # Fallback for environments without loess package.
        from statsmodels.nonparametric.smoothers_lowess import lowess

        y_smooth = lowess(y, x, frac=frac, return_sorted=False)
        return x + t0, y_smooth


def load_historical_series(site_id: str, transect_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one transect historical series from CoastSat data."""
    url = (
        "https://raw.githubusercontent.com/UoA-eResearch/CoastSat/main/data/"
        f"{site_id}/transect_time_series_tidally_corrected_smoothed.csv"
    )
    df = pd.read_csv(url)

    if transect_id not in df.columns:
        raise KeyError(f"Transect '{transect_id}' was not found in source CSV columns.")

    df["dates"] = pd.to_datetime(df["dates"], errors="coerce")
    y_all = pd.to_numeric(df[transect_id], errors="coerce")

    valid = df["dates"].notna() & y_all.notna()
    dates = df.loc[valid, "dates"]
    y = y_all.loc[valid].to_numpy(dtype=float)
    t = decimal_year(dates)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    t_loess, y_loess = loess_smooth_time_series(t, y, loess_window_years=10.0)
    return t, y, t_loess, y_loess


def load_trend_only_projection(
    projections_json_path: Path,
    transect_id: str,
    scenario: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load trend-only projection mean and uncertainty for one transect."""
    with projections_json_path.open("r", encoding="utf-8") as f:
        projections = json.load(f)

    if transect_id not in projections:
        raise KeyError(f"Transect '{transect_id}' is missing from {projections_json_path}.")

    scenario_dict = projections[transect_id].get("scenarios", {})
    if not scenario_dict:
        raise ValueError(f"No scenarios available for transect '{transect_id}'.")

    if scenario is None:
        selected_scenario = sorted(scenario_dict.keys())[0]
    else:
        if scenario not in scenario_dict:
            available = ", ".join(sorted(scenario_dict.keys()))
            raise KeyError(
                f"Scenario '{scenario}' not available for '{transect_id}'. Available: {available}"
            )
        selected_scenario = scenario

    records = scenario_dict[selected_scenario]
    if not records:
        raise ValueError(f"Scenario '{selected_scenario}' for '{transect_id}' has no records.")

    years = np.array([float(r["year"]) for r in records], dtype=float)
    mean = np.array([float(r["trend_only_shoreline_mean"]) for r in records], dtype=float)
    q05 = np.array([float(r["trend_only_shoreline_q05"]) for r in records], dtype=float)
    q95 = np.array([float(r["trend_only_shoreline_q95"]) for r in records], dtype=float)

    order = np.argsort(years)
    return years[order], mean[order], q05[order], q95[order], selected_scenario


def make_plot(
    site_id: str,
    transect_id: str,
    scenario: str,
    t_hist: np.ndarray,
    y_hist: np.ndarray,
    t_loess: np.ndarray,
    y_loess: np.ndarray,
    proj_years: np.ndarray,
    trend_mean: np.ndarray,
    trend_q05: np.ndarray,
    trend_q95: np.ndarray,
    output_path: Path,
) -> None:
    """Render and save the requested combined time-series figure."""
    fig, ax = plt.subplots(figsize=(14, 6.8))

    # Historical data and smoothed trend.
    ax.scatter(
        t_hist,
        y_hist,
        s=18,
        color="#6f6f6f",
        alpha=0.5,
        edgecolors="none",
        label="Satellite shoreline observations",
        zorder=2,
    )
    ax.plot(
        t_loess,
        y_loess,
        color="#1f1f1f",
        linewidth=2.0,
        label="Historical LOESS smoothed trend",
        zorder=3,
    )

    # Connect historical trend to the first projection node.
    if t_loess.size > 0 and proj_years.size > 0:
        ax.plot(
            [t_loess[-1], proj_years[0]],
            [y_loess[-1], trend_mean[0]],
            color="#7f7f7f",
            linewidth=1.1,
            linestyle=(0, (2, 2)),
            alpha=0.8,
            zorder=3,
        )

    seg_colors = ["#1f5fbf", "#1a7f37", "#d95f02"]
    seg_labels = ["Segment 1", "Segment 2", "Segment 3"]

    for i in range(len(proj_years) - 1):
        x_seg = proj_years[i : i + 2]
        y_seg = trend_mean[i : i + 2]
        y_low = trend_q05[i : i + 2]
        y_high = trend_q95[i : i + 2]

        color = seg_colors[i] if i < len(seg_colors) else "#4f4f4f"
        label = seg_labels[i] if i < len(seg_labels) else f"Segment {i + 1}"

        ax.fill_between(
            x_seg,
            y_low,
            y_high,
            color=color,
            alpha=0.14,
            zorder=1,
        )
        ax.plot(
            x_seg,
            y_seg,
            color=color,
            linewidth=2.4,
            linestyle=(0, (5, 3)),
            label=f"{label} ({int(x_seg[0])}-{int(x_seg[1])})",
            zorder=4,
        )
        ax.scatter(
            x_seg,
            y_seg,
            s=42,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=5,
        )

        x_mid = float(np.mean(x_seg))
        y_top = float(max(y_high))
        ax.text(
            x_mid,
            y_top + 0.6,
            f"{label}\n{int(x_seg[0])}-{int(x_seg[1])}",
            color=color,
            fontsize=12,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Section guides inspired by reference design.
    boundaries = [float(proj_years[0]), 2030.0, 2040.0, 2050.0]
    for xb in boundaries:
        ax.axvline(xb, color="#b8b8b8", linestyle=(0, (3, 3)), linewidth=1.0, zorder=0)

    ax.axhline(0.0, color="#9c9c9c", linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)

    y_text = float(np.nanmax(y_hist)) + 1.8
    ax.text(2009.5, y_text, "Historical", color="#4a4a4a", fontsize=15, fontweight="bold", ha="center")
    ax.text(2040.0, y_text, "Projected", color="#5a3e99", fontsize=15, fontweight="bold", ha="center")

    y_min = float(min(np.nanmin(y_hist), np.nanmin(trend_q05))) - 2.0
    y_max = float(max(np.nanmax(y_hist), np.nanmax(trend_q95))) + 2.8

    ax.set_xlim(float(np.floor(np.nanmin(t_hist))), 2051.0)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Shoreline position (m)")
    ax.set_xlabel("Year")
    ax.set_title(f"{site_id} {transect_id} | Trend-only projection with uncertainty ({scenario})")
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one transect historical + projected trend-only time series")
    parser.add_argument("--site", default="nzd0135", help="Site ID, e.g. nzd0135")
    parser.add_argument("--transect", default="nzd0135-0004", help="Transect ID, e.g. nzd0135-0004")
    parser.add_argument(
        "--projections-json",
        default="data/projections.json",
        help="Path to projections JSON generated for dashboard",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario key, e.g. ssp1_2.6. If omitted, first available scenario is used.",
    )
    parser.add_argument(
        "--output",
        default="outputs/nzd0135/nzd0135-0004_historical_projected_trend_only.png",
        help="Output PNG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t_hist, y_hist, t_loess, y_loess = load_historical_series(args.site, args.transect)
    proj_years, trend_mean, trend_q05, trend_q95, selected_scenario = load_trend_only_projection(
        Path(args.projections_json),
        args.transect,
        args.scenario,
    )

    required_years = np.array([2025.0, 2030.0, 2040.0, 2050.0], dtype=float)
    if not np.all(np.isin(required_years, proj_years)):
        missing = required_years[~np.isin(required_years, proj_years)]
        raise ValueError(f"Missing required projection years for segment plot: {missing.tolist()}")

    # Keep only the requested segment nodes (2025, 2030, 2040, 2050).
    mask = np.isin(proj_years, required_years)
    proj_years = proj_years[mask]
    trend_mean = trend_mean[mask]
    trend_q05 = trend_q05[mask]
    trend_q95 = trend_q95[mask]

    order = np.argsort(proj_years)
    proj_years = proj_years[order]
    trend_mean = trend_mean[order]
    trend_q05 = trend_q05[order]
    trend_q95 = trend_q95[order]

    output_path = Path(args.output)
    make_plot(
        site_id=args.site,
        transect_id=args.transect,
        scenario=selected_scenario,
        t_hist=t_hist,
        y_hist=y_hist,
        t_loess=t_loess,
        y_loess=y_loess,
        proj_years=proj_years,
        trend_mean=trend_mean,
        trend_q05=trend_q05,
        trend_q95=trend_q95,
        output_path=output_path,
    )

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
