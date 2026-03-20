#!/usr/bin/env python3
"""Visualization helpers for the trend uncertainty workflow."""

import matplotlib.pyplot as plt
import numpy as np


def plot_delta_s_debug_histogram(
    sampled_delta_s,
    summary,
    q17,
    q50,
    q83,
    site_id,
    transect_id,
    ssp,
    scenario,
    year,
    out_fp,
):
    """Save a debug histogram of sampled SLR for one transect."""
    sampled_delta_s = np.asarray(sampled_delta_s, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sampled_delta_s, bins=50, density=True, alpha=0.55, color="steelblue")

    mu = float(summary.get("delta_S_mu_m", np.nan))
    sigma = float(summary.get("delta_S_sigma_m", np.nan))

    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        x_low = float(np.quantile(sampled_delta_s, 0.001))
        x_high = float(np.quantile(sampled_delta_s, 0.999))
        x = np.linspace(x_low, x_high, 400)
        pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, pdf, color="black", linewidth=2.0, label="Gaussian fit")

    ax.axvline(q17, color="tab:orange", linestyle="--", linewidth=1.5, label="q17 target")
    ax.axvline(q50, color="tab:green", linestyle="--", linewidth=1.5, label="q50 target")
    ax.axvline(q83, color="tab:red", linestyle="--", linewidth=1.5, label="q83 target")

    ax.set_xlabel("Sampled delta_S [m]")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{site_id} {transect_id} delta_S samples ({ssp}-{scenario}, year={year})"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=180)
    plt.close(fig)


def plot_single_run_segments(
    dy_segments,
    dt,
    segment_years,
    site_id,
    transect_id,
    out_fp,
):
    """Save per-segment and cumulative change for one simulated path."""
    dy_segments = np.asarray(dy_segments, dtype=float).ravel()
    if dy_segments.size == 0:
        raise ValueError("dy_segments is empty; expected one simulated path.")

    seg_durations = np.full(dy_segments.size, float(segment_years), dtype=float)
    remainder = dt % segment_years
    if remainder > 0:
        seg_durations[-1] = float(remainder)

    seg_ends = np.cumsum(seg_durations)
    cumulative = np.concatenate([[0.0], np.cumsum(dy_segments)])
    cumulative_time = np.concatenate([[0.0], seg_ends])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axes[0].bar(seg_ends, dy_segments, width=np.minimum(seg_durations * 0.9, 4.5), color="tab:blue", alpha=0.8)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("5-year dy [m]")
    axes[0].set_title(f"Single run segments: {site_id} {transect_id}")
    axes[0].set_xticks(seg_ends)
    axes[0].set_xlabel("Years from baseline")

    axes[1].step(cumulative_time, cumulative, where="post", color="tab:green", linewidth=2.0)
    axes[1].scatter(cumulative_time, cumulative, color="tab:green", s=18)
    axes[1].set_xlabel("Years from baseline")
    axes[1].set_ylabel("Cumulative dy [m]")
    axes[1].set_title(f"Cumulative path (total = {cumulative[-1]:.2f} m)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_fp, dpi=180)
    plt.close(fig)


def plot_observed_and_projected_single_run(
    t_obs,
    y_obs,
    projection_start_year,
    dy_segments,
    r_segments,
    slr_years,
    slr_q17_values,
    slr_q50_values,
    slr_q83_values,
    dt,
    segment_years,
    site_id,
    transect_id,
    out_fp,
):
    """Plot observed time series and single-run projection with dashed 5-year trend lines."""
    t_obs = np.asarray(t_obs, dtype=float).ravel()
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    dy_segments = np.asarray(dy_segments, dtype=float).ravel()
    r_segments = np.asarray(r_segments, dtype=float).ravel()
    slr_years = np.asarray(slr_years, dtype=float).ravel()
    slr_q17_values = np.asarray(slr_q17_values, dtype=float).ravel()
    slr_q50_values = np.asarray(slr_q50_values, dtype=float).ravel()
    slr_q83_values = np.asarray(slr_q83_values, dtype=float).ravel()

    if t_obs.size == 0 or y_obs.size == 0:
        raise ValueError("Observed time series is empty.")
    if t_obs.size != y_obs.size:
        raise ValueError("Observed time and shoreline arrays have different lengths.")
    if dy_segments.size == 0:
        raise ValueError("dy_segments is empty; expected projected segments.")
    if r_segments.size != dy_segments.size:
        raise ValueError("r_segments and dy_segments must have the same length.")
    if slr_years.size != slr_q17_values.size:
        raise ValueError("slr_years and slr_q17_values must have the same length.")
    if slr_years.size != slr_q50_values.size:
        raise ValueError("slr_years and slr_q50_values must have the same length.")
    if slr_years.size != slr_q83_values.size:
        raise ValueError("slr_years and slr_q83_values must have the same length.")

    order = np.argsort(t_obs)
    t_obs = t_obs[order]
    y_obs = y_obs[order]

    seg_durations = np.full(dy_segments.size, float(segment_years), dtype=float)
    remainder = dt % segment_years
    if remainder > 0:
        seg_durations[-1] = float(remainder)

    seg_starts = projection_start_year + np.concatenate(([0.0], np.cumsum(seg_durations)[:-1]))
    seg_ends = seg_starts + seg_durations

    baseline_idx = int(np.argmin(np.abs(t_obs - projection_start_year)))
    baseline_y = float(y_obs[baseline_idx])

    cum_dy = np.cumsum(dy_segments)
    proj_years = np.concatenate(([projection_start_year], seg_ends))
    proj_shoreline = baseline_y + np.concatenate(([0.0], cum_dy))
    seg_start_shoreline = baseline_y + np.concatenate(([0.0], cum_dy[:-1]))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_obs, y_obs, color="tab:blue", marker="o", markersize=2.5, linewidth=1.2, label="Observed")
    ax.plot(
        proj_years,
        proj_shoreline,
        color="tab:green",
        marker="o",
        markersize=3.0,
        linewidth=2.0,
        label="Projected single run",
    )

    first_label = True
    for i in range(dy_segments.size):
        x0 = seg_starts[i]
        x1 = seg_ends[i]
        y0 = seg_start_shoreline[i]
        y1 = y0 + r_segments[i] * seg_durations[i]

        ax.plot(
            [x0, x1],
            [y0, y1],
            linestyle="--",
            color="tab:orange",
            linewidth=1.1,
            alpha=0.85,
            label="Sampled 5-year trend" if first_label else None,
        )
        first_label = False

    ax_slr = None
    if slr_years.size > 0:
        slr_order = np.argsort(slr_years)
        slr_years = slr_years[slr_order]
        slr_q17_values = slr_q17_values[slr_order]
        slr_q50_values = slr_q50_values[slr_order]
        slr_q83_values = slr_q83_values[slr_order]

        ax_slr = ax.twinx()
        ax_slr.plot(
            slr_years,
            slr_q50_values,
            color="deeppink",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            alpha=0.9,
            label="SLR projection",
        )
        ax_slr.plot(
            slr_years,
            slr_q17_values,
            color="lightpink",
            linewidth=1.2,
            linestyle="--",
            alpha=0.95,
            label="SLR bounds (q17/q83)",
        )
        ax_slr.plot(
            slr_years,
            slr_q83_values,
            color="lightpink",
            linewidth=1.2,
            linestyle="--",
            alpha=0.95,
        )
        ax_slr.set_ylabel("SLR change from 2025 [m]", color="deeppink")
        ax_slr.tick_params(axis="y", colors="deeppink")
        ax_slr.spines["right"].set_color("deeppink")

    ax.axvline(projection_start_year, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Year (decimal)")
    ax.set_ylabel("Shoreline position [m]")
    ax.set_title(f"Observed + projected path with 5-year trend segments: {site_id} {transect_id}")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if ax_slr is not None:
        handles_slr, labels_slr = ax_slr.get_legend_handles_labels()
        handles.extend(handles_slr)
        labels.extend(labels_slr)
    ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=180)
    plt.close(fig)


def plot_observed_only_time_series(
    t_obs,
    y_obs,
    site_id,
    transect_id,
    out_fp,
):
    """Plot only the observed shoreline time series for one transect."""
    t_obs = np.asarray(t_obs, dtype=float).ravel()
    y_obs = np.asarray(y_obs, dtype=float).ravel()

    if t_obs.size == 0 or y_obs.size == 0:
        raise ValueError("Observed time series is empty.")
    if t_obs.size != y_obs.size:
        raise ValueError("Observed time and shoreline arrays have different lengths.")

    order = np.argsort(t_obs)
    t_obs = t_obs[order]
    y_obs = y_obs[order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        t_obs,
        y_obs,
        color="tab:blue",
        marker="o",
        markersize=2.5,
        linewidth=1.3,
        label="Observed",
    )
    ax.set_xlabel("Year (decimal)")
    ax.set_ylabel("Shoreline position [m]")
    ax.set_title(f"Observed shoreline time series: {site_id} {transect_id}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=180)
    plt.close(fig)


def plot_original_time_series_before_filter(
    t_raw,
    y_raw,
    site_id,
    transect_id,
    out_fp,
):
    """Plot the original transect time series before any filtering rules are applied."""
    t_raw = np.asarray(t_raw, dtype=float).ravel()
    y_raw = np.asarray(y_raw, dtype=float).ravel()

    if t_raw.size == 0 or y_raw.size == 0:
        raise ValueError("Original time series is empty.")
    if t_raw.size != y_raw.size:
        raise ValueError("Original time and shoreline arrays have different lengths.")

    order = np.argsort(t_raw)
    t_raw = t_raw[order]
    y_raw = y_raw[order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        t_raw,
        y_raw,
        color="tab:purple",
        marker="o",
        markersize=2.5,
        linewidth=1.1,
        alpha=0.9,
        label="Original series (pre-filter)",
    )
    ax.set_xlabel("Time (decimal years)")
    ax.set_ylabel("Shoreline position [m]")
    ax.set_title(f"Original shoreline series before filtering: {site_id} {transect_id}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=180)
    plt.close(fig)


def plot_uncertainty_summary(dy_df):
    """Show summary plots for shoreline-change uncertainty results."""
    if dy_df.empty:
        raise ValueError("dy_df is empty; expected Monte Carlo results to plot.")

    dy_plot_df = dy_df.copy()
    dy_plot_df["dy_ci90_len_m"] = dy_plot_df["dy_p95_m"] - dy_plot_df["dy_p05_m"]

    fig, ax = plt.subplots()
    ax.hist(dy_plot_df["dy_ci90_len_m"].dropna(), bins=40)
    ax.set_xlabel("CI length (dy_p95 - dy_p05) [m]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of dy 5-95% CI length (all transects)")
    fig.tight_layout()
    plt.show()

    labels = dy_plot_df["site_id"].unique()
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (_, group) in enumerate(dy_plot_df.groupby("site_id"), start=1):
        values = group["dy_ci90_len_m"].values
        x_values = np.full_like(values, i, dtype=float)
        ax.plot(x_values, values, "o", alpha=0.4)

    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=90)
    ax.set_ylabel("dy 5-95% CI length [m]")
    ax.set_title("Transect-level CI length spread per site")
    fig.tight_layout()
    plt.show()

    groups = [
        group["dy_ci90_len_m"].values
        for _, group in dy_plot_df.groupby("site_id")
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(groups, showfliers=True)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=90)
    ax.set_ylabel("dy 5-95% CI length [m]")
    ax.set_title("Spread of shoreline-change uncertainty per site")

    for y_value, label in [(100, "100 m"), (200, "200 m"), (500, "500 m")]:
        ax.axhline(y_value, color="red", linestyle="--", linewidth=1)
        ax.text(0.5, y_value, label, color="red", fontsize=9, va="bottom")

    fig.tight_layout()
    plt.show()

    groups_median = [
        group["dy_median_m"].values
        for _, group in dy_plot_df.groupby("site_id")
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(groups_median, showfliers=True)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=90)
    ax.set_ylabel("Median shoreline change (m)")
    ax.set_title("Median shoreline change per site")
    fig.tight_layout()
    plt.show()