"""Visualization helpers for the trend uncertainty workflow."""

import matplotlib.pyplot as plt
import numpy as np

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
