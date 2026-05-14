"""Visualization helpers for the trend uncertainty workflow."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

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
    t_loess,
    y_loess,
    trend_pool,
    slr_only_dy_segments,
):
    """Plot observed shoreline and projected trajectories.

    Supports either:
    - single realization: dy_segments shape (n_segments,)
    - ensemble realizations: dy_segments shape (n_realizations, n_segments)

    In ensemble mode, plots the mean projected shoreline with a 5-95% envelope.
    """
    t_obs = np.asarray(t_obs, dtype=float).ravel()
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    if t_loess is not None and y_loess is not None:
        t_loess = np.asarray(t_loess, dtype=float).ravel()
        y_loess = np.asarray(y_loess, dtype=float).ravel()
    else:
        t_loess = None
        y_loess = None
    dy_segments = np.asarray(dy_segments, dtype=float)
    r_segments = np.asarray(r_segments, dtype=float)
    slr_years = np.asarray(slr_years, dtype=float).ravel()
    slr_q17_values = np.asarray(slr_q17_values, dtype=float).ravel()
    slr_q50_values = np.asarray(slr_q50_values, dtype=float).ravel()
    slr_q83_values = np.asarray(slr_q83_values, dtype=float).ravel()
    if trend_pool is not None:
        trend_pool = np.asarray(trend_pool, dtype=float).ravel()
        trend_pool = trend_pool[np.isfinite(trend_pool)]
    if slr_only_dy_segments is not None:
        slr_only_dy_segments = np.asarray(slr_only_dy_segments, dtype=float)

    if t_obs.size == 0 or y_obs.size == 0:
        raise ValueError("Observed time series is empty.")
    if t_obs.size != y_obs.size:
        raise ValueError("Observed time and shoreline arrays have different lengths.")
    if t_loess is not None and t_loess.size != y_loess.size:
        raise ValueError("LOESS time and shoreline arrays have different lengths.")
    if dy_segments.size == 0:
        raise ValueError("dy_segments is empty; expected projected segments.")
    if dy_segments.ndim == 1:
        dy_matrix = dy_segments[np.newaxis, :]
    elif dy_segments.ndim == 2:
        dy_matrix = dy_segments
    else:
        raise ValueError("dy_segments must be a 1D or 2D array.")

    if r_segments.size == 0:
        raise ValueError("r_segments is empty; expected projected trend segments.")
    if r_segments.ndim == 1:
        r_matrix = r_segments[np.newaxis, :]
    elif r_segments.ndim == 2:
        r_matrix = r_segments
    else:
        raise ValueError("r_segments must be a 1D or 2D array.")

    if r_matrix.shape[1] != dy_matrix.shape[1]:
        raise ValueError("r_segments and dy_segments must have the same number of segments.")
    if slr_only_dy_segments is not None:
        if slr_only_dy_segments.ndim == 1:
            slr_only_matrix = slr_only_dy_segments[np.newaxis, :]
        elif slr_only_dy_segments.ndim == 2:
            slr_only_matrix = slr_only_dy_segments
        else:
            raise ValueError("slr_only_dy_segments must be a 1D or 2D array.")
        if slr_only_matrix.shape != dy_matrix.shape:
            raise ValueError("slr_only_dy_segments must have the same shape as dy_segments.")
    else:
        slr_only_matrix = None
    if slr_years.size != slr_q17_values.size:
        raise ValueError("slr_years and slr_q17_values must have the same length.")
    if slr_years.size != slr_q50_values.size:
        raise ValueError("slr_years and slr_q50_values must have the same length.")
    if slr_years.size != slr_q83_values.size:
        raise ValueError("slr_years and slr_q83_values must have the same length.")

    order = np.argsort(t_obs)
    t_obs = t_obs[order]
    y_obs = y_obs[order]
    if t_loess is not None:
        loess_order = np.argsort(t_loess)
        t_loess = t_loess[loess_order]
        y_loess = y_loess[loess_order]

    n_segments = dy_matrix.shape[1]
    seg_durations = np.full(n_segments, float(segment_years), dtype=float)
    remainder = dt % segment_years
    if remainder > 0:
        seg_durations[-1] = float(remainder)

    seg_starts = projection_start_year + np.concatenate(([0.0], np.cumsum(seg_durations)[:-1]))
    seg_ends = seg_starts + seg_durations

    # Anchor projections at the trailing 5-year mean of the LOESS curve.
    # If LOESS is unavailable in that window, fall back to observed values.
    baseline_window_years = 5.0
    window_start = projection_start_year - baseline_window_years

    baseline_source_t = t_loess if (t_loess is not None and t_loess.size > 0) else t_obs
    baseline_source_y = y_loess if (y_loess is not None and y_loess.size > 0) else y_obs

    baseline_mask = (
        (baseline_source_t >= window_start)
        & (baseline_source_t <= projection_start_year)
    )
    if np.any(baseline_mask):
        baseline_y = float(np.mean(baseline_source_y[baseline_mask]))
    else:
        baseline_idx = int(np.argmin(np.abs(baseline_source_t - projection_start_year)))
        baseline_y = float(baseline_source_y[baseline_idx])

    cum_dy = np.cumsum(dy_matrix, axis=1)
    proj_years = np.concatenate(([projection_start_year], seg_ends))
    proj_shoreline_all = baseline_y + np.concatenate(
        [np.zeros((dy_matrix.shape[0], 1)), cum_dy],
        axis=1,
    )
    proj_shoreline_mean = np.mean(proj_shoreline_all, axis=0)
    proj_shoreline_q05 = np.quantile(proj_shoreline_all, 0.05, axis=0)
    proj_shoreline_q95 = np.quantile(proj_shoreline_all, 0.95, axis=0)

    if slr_only_matrix is not None:
        slr_only_cum = np.cumsum(slr_only_matrix, axis=1)
        slr_only_shoreline_all = baseline_y + np.concatenate(
            [np.zeros((slr_only_matrix.shape[0], 1)), slr_only_cum],
            axis=1,
        )
        slr_only_shoreline_mean = np.mean(slr_only_shoreline_all, axis=0)

    dy_mean = np.mean(dy_matrix, axis=0)
    seg_start_shoreline = baseline_y + np.concatenate(([0.0], np.cumsum(dy_mean)[:-1]))
    r_mean = np.mean(r_matrix, axis=0)

    fig = plt.figure(figsize=(15, 6.5))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[3.6, 1.4],
        height_ratios=[1.0, 1.0],
    )
    ax = fig.add_subplot(gs[:, 0])
    ax_trend = fig.add_subplot(gs[0, 1])
    ax_slr = fig.add_subplot(gs[1, 1])
    ax.plot(t_obs, y_obs, color="tab:blue", marker="o", markersize=2.5, linewidth=1.2, label="Observed")
    if t_loess is not None and t_loess.size > 0:
        ax.plot(
            t_loess,
            y_loess,
            color="black",
            linewidth=1.6,
            alpha=0.9,
            label="LOESS smoothed",
        )
    if dy_matrix.shape[0] > 1:
        ax.fill_between(
            proj_years,
            proj_shoreline_q05,
            proj_shoreline_q95,
            color="tab:green",
            alpha=0.22,
            label="Projected envelope (5-95%)",
        )
        ax.plot(
            proj_years,
            proj_shoreline_mean,
            color="tab:green",
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label="Projected mean",
        )
    else:
        ax.plot(
            proj_years,
            proj_shoreline_mean,
            color="tab:green",
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label="Projected single run",
        )

    if slr_only_matrix is not None:
        ax.plot(
            proj_years,
            slr_only_shoreline_mean,
            color="deeppink",
            linewidth=2.0,
            alpha=0.7,
            label="Projected SLR-only",
        )

    first_label = True
    for i in range(n_segments):
        x0 = seg_starts[i]
        x1 = seg_ends[i]
        y0 = seg_start_shoreline[i]
        y1 = y0 + r_mean[i] * seg_durations[i]

        ax.plot(
            [x0, x1],
            [y0, y1],
            linestyle="--",
            color="tab:orange",
            linewidth=1.1,
            alpha=0.85,
            label="Mean 5-year trend" if first_label else None,
        )
        first_label = False

    if slr_years.size > 0:
        slr_order = np.argsort(slr_years)
        slr_years = slr_years[slr_order]
        slr_q17_values = slr_q17_values[slr_order]
        slr_q50_values = slr_q50_values[slr_order]
        slr_q83_values = slr_q83_values[slr_order]

        ax_slr.plot(
            slr_years,
            slr_q50_values,
            color="deeppink",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            alpha=0.9,
            label="SLR q50",
        )
        ax_slr.plot(
            slr_years,
            slr_q17_values,
            color="lightpink",
            linewidth=1.2,
            linestyle="--",
            alpha=0.95,
            label="SLR q17/q83",
        )
        ax_slr.plot(
            slr_years,
            slr_q83_values,
            color="lightpink",
            linewidth=1.2,
            linestyle="--",
            alpha=0.95,
        )
        ax_slr.set_xlim(float(np.min(slr_years)), float(np.max(slr_years)))
        ax_slr.grid(alpha=0.25)
    else:
        ax_slr.text(
            0.5,
            0.5,
            "No SLR series available",
            ha="center",
            va="center",
            transform=ax_slr.transAxes,
            color="gray",
        )
        ax_slr.grid(alpha=0.25)

    ax.axvline(projection_start_year, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Year (decimal)")
    ax.set_ylabel("Shoreline position [m]")
    ax.set_title(f"Observed + projected shoreline with uncertainty: {site_id} {transect_id}")
    ax.grid(alpha=0.25)

    # Right subplot: trend distribution from bootstrap, with sampled realization trends highlighted.
    used_trends = r_matrix.ravel()
    used_trends = used_trends[np.isfinite(used_trends)]
    all_trends = used_trends if trend_pool is None else trend_pool

    if all_trends.size > 0:
        if all_trends.size == 1 or np.allclose(np.std(all_trends), 0.0):
            ax_trend.axvline(
                float(np.mean(all_trends)),
                color="gray",
                linewidth=2.0,
                label="Extracted trends (degenerate)",
            )
        else:
            # Gaussian KDE with Silverman's rule-of-thumb bandwidth.
            n_all = all_trends.size
            sigma_all = float(np.std(all_trends, ddof=1))
            bw = 1.06 * sigma_all * (n_all ** (-1.0 / 5.0))
            bw = max(bw, 1e-6)
            x_lo = float(np.min(all_trends) - 3.0 * bw)
            x_hi = float(np.max(all_trends) + 3.0 * bw)
            x_grid = np.linspace(x_lo, x_hi, 300)
            z = (x_grid[:, np.newaxis] - all_trends[np.newaxis, :]) / bw
            kde = np.exp(-0.5 * z * z).sum(axis=1) / (n_all * bw * np.sqrt(2.0 * np.pi))
            ax_trend.plot(
                x_grid,
                kde,
                color="dimgray",
                linewidth=2.0,
                label="Extracted trends (KDE)",
            )
    if used_trends.size > 0:
        # Rug ticks for sampled trends used in realizations.
        y_min, y_max = ax_trend.get_ylim()
        rug_bottom = y_min
        rug_top = y_min + 0.08 * (y_max - y_min if y_max > y_min else 1.0)
        ax_trend.vlines(
            used_trends,
            rug_bottom,
            rug_top,
            color="tab:green",
            alpha=0.8,
            linewidth=1.2,
            label="Trends used in realizations (rug)",
        )

    ax_trend.set_xlabel("Trend [m/yr]")
    ax_trend.set_ylabel("Probability density [1/(m/yr)]")
    ax_trend.set_title("Trend distribution")
    ax_trend.grid(alpha=0.25)
    ax_trend.plot([], [], " ", label="Note: KDE area = 1")
    ax_trend.legend(loc="best", fontsize=8)

    ax_slr.set_xlabel("Year")
    ax_slr.set_ylabel("SLR change from 2025 [m]")
    ax_slr.set_title("Sea-level projection")
    if slr_years.size > 0:
        ax_slr.legend(loc="best", fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=300)
    plt.close(fig)
