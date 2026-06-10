"""Visualization helpers for the trend uncertainty workflow."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def build_projection_segment_durations(projection_start_year, dt, segment_years):
    """Build decade-aligned segment durations for projection plotting.

    The first segment runs to the next decade boundary, then regular
    ``segment_years`` steps are used.
    """
    total_years = float(dt)
    if total_years <= 0.0:
        raise ValueError("dt must be positive.")
    if segment_years <= 0:
        raise ValueError("segment_years must be positive.")

    start_year = float(projection_start_year)
    next_decade = float(np.ceil(start_year / 10.0) * 10.0)
    first_step = next_decade - start_year

    durations = []
    remaining = total_years
    tol = 1e-9

    if first_step > tol and first_step < (remaining - tol):
        durations.append(first_step)
        remaining -= first_step

    step_years = float(segment_years)
    while remaining > tol:
        step = min(step_years, remaining)
        durations.append(step)
        remaining -= step

    seg_durations = np.asarray(durations, dtype=float)
    if seg_durations.size == 0:
        raise ValueError("No projection segment durations were generated.")
    return seg_durations

#%%
def prepare_observed_and_projected_single_run_data(
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
    t_loess,
    y_loess,
    trend_pool,
    recent_trend_pool,
    sampled_historic_trends,
    sampled_recent_trends,
    slr_only_dy_segments,
    trend_only_dy_segments,
):
    """Validate and prepare arrays used by the shoreline projection plot.

    Parameters
    ----------
    t_obs : array-like
        Observation timestamps in decimal years.
    y_obs : array-like
        Observed shoreline position [m], same length as ``t_obs``.
    projection_start_year : float
        Year where projection begins and where projected shoreline is anchored.
    dy_segments : array-like
        Projected shoreline change per segment [m].
        Shape can be ``(n_segments,)`` for a single run or
        ``(n_realizations, n_segments)`` for an ensemble.
    r_segments : array-like
        Trend values used per segment [m/yr], with the same segment layout as
        ``dy_segments``.
    slr_years : array-like
        Years associated with sea-level projection quantiles.
    slr_q17_values : array-like
        Sea-level change [m] at the 17th percentile for each year in
        ``slr_years``.
    slr_q50_values : array-like
        Sea-level change [m] at the 50th percentile for each year in
        ``slr_years``.
    slr_q83_values : array-like
        Sea-level change [m] at the 83rd percentile for each year in
        ``slr_years``.
    dt : int
        Total projection duration in years.
    segment_years : int
        Duration of each projection segment in years.
    t_loess : array-like or None
        Optional LOESS timestamps in decimal years.
    y_loess : array-like or None
        Optional LOESS shoreline values [m], same length as ``t_loess``.
    trend_pool : array-like or None
        Optional full pool of extracted trends [m/yr] for the trend
        distribution panel. If ``None``, trends used in realizations are used.
    slr_only_dy_segments : array-like or None
        Optional SLR-only shoreline change per segment [m] with the same shape
        as ``dy_segments``.
    trend_only_dy_segments : array-like or None
        Optional trend-only shoreline change per segment [m] (no SLR
        contribution) with the same shape as ``dy_segments``.

    Returns
    -------
    dict
        Dictionary containing normalized arrays and derived quantities used by
        plotting, including sorted time series, projected shoreline statistics,
        trend density precomputations, and metadata needed for labels.
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
    if recent_trend_pool is not None:
        recent_trend_pool = np.asarray(recent_trend_pool, dtype=float).ravel()
        recent_trend_pool = recent_trend_pool[np.isfinite(recent_trend_pool)]
    if sampled_historic_trends is not None:
        sampled_historic_trends = np.asarray(sampled_historic_trends, dtype=float).ravel()
        sampled_historic_trends = sampled_historic_trends[np.isfinite(sampled_historic_trends)]
    if sampled_recent_trends is not None:
        sampled_recent_trends = np.asarray(sampled_recent_trends, dtype=float).ravel()
        sampled_recent_trends = sampled_recent_trends[np.isfinite(sampled_recent_trends)]
    if slr_only_dy_segments is not None:
        slr_only_dy_segments = np.asarray(slr_only_dy_segments, dtype=float)
    if trend_only_dy_segments is not None:
        trend_only_dy_segments = np.asarray(trend_only_dy_segments, dtype=float)

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
    if trend_only_dy_segments is not None:
        if trend_only_dy_segments.ndim == 1:
            trend_only_matrix = trend_only_dy_segments[np.newaxis, :]
        elif trend_only_dy_segments.ndim == 2:
            trend_only_matrix = trend_only_dy_segments
        else:
            raise ValueError("trend_only_dy_segments must be a 1D or 2D array.")
        if trend_only_matrix.shape != dy_matrix.shape:
            raise ValueError("trend_only_dy_segments must have the same shape as dy_segments.")
    else:
        trend_only_matrix = None
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
    seg_durations = build_projection_segment_durations(
        projection_start_year=projection_start_year,
        dt=dt,
        segment_years=segment_years,
    )
    if n_segments != int(seg_durations.size):
        raise ValueError(
            "dy_segments shape is inconsistent with decade-aligned timeline for projection_start_year/dt/segment_years."
        )

    seg_starts = projection_start_year + np.concatenate(([0.0], np.cumsum(seg_durations)[:-1]))
    seg_ends = seg_starts + seg_durations

    # Anchor projections at the last point of the LOESS curve.
    # If LOESS is unavailable, fall back to observed values.
    baseline_source_t = t_loess if (t_loess is not None and t_loess.size > 0) else t_obs
    baseline_source_y = y_loess if (y_loess is not None and y_loess.size > 0) else y_obs
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

    slr_only_shoreline_mean = None
    slr_only_shoreline_q05 = None
    slr_only_shoreline_q95 = None
    if slr_only_matrix is not None:
        slr_only_cum = np.cumsum(slr_only_matrix, axis=1)
        slr_only_shoreline_all = baseline_y + np.concatenate(
            [np.zeros((slr_only_matrix.shape[0], 1)), slr_only_cum],
            axis=1,
        )
        slr_only_shoreline_mean = np.mean(slr_only_shoreline_all, axis=0)
        slr_only_shoreline_q05 = np.quantile(slr_only_shoreline_all, 0.05, axis=0)
        slr_only_shoreline_q95 = np.quantile(slr_only_shoreline_all, 0.95, axis=0)

    trend_only_shoreline_mean = None
    trend_only_shoreline_q05 = None
    trend_only_shoreline_q95 = None
    if trend_only_matrix is not None:
        trend_only_cum = np.cumsum(trend_only_matrix, axis=1)
        trend_only_shoreline_all = baseline_y + np.concatenate(
            [np.zeros((trend_only_matrix.shape[0], 1)), trend_only_cum],
            axis=1,
        )
        trend_only_shoreline_mean = np.mean(trend_only_shoreline_all, axis=0)
        trend_only_shoreline_q05 = np.quantile(trend_only_shoreline_all, 0.05, axis=0)
        trend_only_shoreline_q95 = np.quantile(trend_only_shoreline_all, 0.95, axis=0)

    if slr_years.size > 0:
        slr_order = np.argsort(slr_years)
        slr_years = slr_years[slr_order]
        slr_q17_values = slr_q17_values[slr_order]
        slr_q50_values = slr_q50_values[slr_order]
        slr_q83_values = slr_q83_values[slr_order]

    used_trends = r_matrix.ravel()
    used_trends = used_trends[np.isfinite(used_trends)]
    if sampled_historic_trends is not None:
        historic_rug_trends = sampled_historic_trends
    else:
        historic_rug_trends = used_trends
    if sampled_recent_trends is not None:
        recent_rug_trends = sampled_recent_trends
    else:
        recent_rug_trends = np.array([], dtype=float)
    all_trends = trend_pool if trend_pool is not None else historic_rug_trends

    trend_line_x = None
    trend_line_y = None
    trend_degenerate_mean = None
    if all_trends.size > 0:
        if all_trends.size == 1 or np.allclose(np.std(all_trends), 0.0):
            trend_degenerate_mean = float(np.mean(all_trends))
        else:
            # Gaussian KDE with Silverman's rule-of-thumb bandwidth.
            n_all = all_trends.size
            sigma_all = float(np.std(all_trends, ddof=1))
            bw = 1.06 * sigma_all * (n_all ** (-1.0 / 5.0))
            bw = max(bw, 1e-6)
            x_lo = float(np.min(all_trends) - 3.0 * bw)
            x_hi = float(np.max(all_trends) + 3.0 * bw)
            trend_line_x = np.linspace(x_lo, x_hi, 300)
            z = (trend_line_x[:, np.newaxis] - all_trends[np.newaxis, :]) / bw
            trend_line_y = np.exp(-0.5 * z * z).sum(axis=1) / (n_all * bw * np.sqrt(2.0 * np.pi))

    return {
        "t_obs": t_obs,
        "y_obs": y_obs,
        "projection_start_year": float(projection_start_year),
        "proj_years": proj_years,
        "proj_shoreline_mean": proj_shoreline_mean,
        "proj_shoreline_q05": proj_shoreline_q05,
        "proj_shoreline_q95": proj_shoreline_q95,
        "slr_only_shoreline_mean": slr_only_shoreline_mean,
        "slr_only_shoreline_q05": slr_only_shoreline_q05,
        "slr_only_shoreline_q95": slr_only_shoreline_q95,
        "trend_only_shoreline_mean": trend_only_shoreline_mean,
        "trend_only_shoreline_q05": trend_only_shoreline_q05,
        "trend_only_shoreline_q95": trend_only_shoreline_q95,
        "t_loess": t_loess,
        "y_loess": y_loess,
        "is_ensemble": bool(dy_matrix.shape[0] > 1),
        "slr_years": slr_years,
        "slr_q17_values": slr_q17_values,
        "slr_q50_values": slr_q50_values,
        "slr_q83_values": slr_q83_values,
        "used_trends": used_trends,
        "historic_rug_trends": historic_rug_trends,
        "recent_rug_trends": recent_rug_trends,
        "trend_line_x": trend_line_x,
        "trend_line_y": trend_line_y,
        "trend_degenerate_mean": trend_degenerate_mean,
    }

#%%
def plot_observed_and_projected_single_run_from_processed(
    prepared,
    site_id,
    transect_id,
    out_fp,
):
    """Render the shoreline projection figure from precomputed data only."""
    t_obs = prepared["t_obs"]
    y_obs = prepared["y_obs"]
    projection_start_year = prepared["projection_start_year"]
    proj_years = prepared["proj_years"]
    proj_shoreline_mean = prepared["proj_shoreline_mean"]
    proj_shoreline_q05 = prepared["proj_shoreline_q05"]
    proj_shoreline_q95 = prepared["proj_shoreline_q95"]
    slr_only_shoreline_mean = prepared["slr_only_shoreline_mean"]
    slr_only_shoreline_q05 = prepared["slr_only_shoreline_q05"]
    slr_only_shoreline_q95 = prepared["slr_only_shoreline_q95"]
    trend_only_shoreline_mean = prepared["trend_only_shoreline_mean"]
    trend_only_shoreline_q05 = prepared["trend_only_shoreline_q05"]
    trend_only_shoreline_q95 = prepared["trend_only_shoreline_q95"]
    t_loess = prepared["t_loess"]
    y_loess = prepared["y_loess"]
    is_ensemble = prepared["is_ensemble"]
    slr_years = prepared["slr_years"]
    slr_q17_values = prepared["slr_q17_values"]
    slr_q50_values = prepared["slr_q50_values"]
    slr_q83_values = prepared["slr_q83_values"]
    used_trends = prepared["used_trends"]
    historic_rug_trends = prepared["historic_rug_trends"]
    recent_rug_trends = prepared["recent_rug_trends"]
    trend_line_x = prepared["trend_line_x"]
    trend_line_y = prepared["trend_line_y"]
    trend_degenerate_mean = prepared["trend_degenerate_mean"]

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

    if is_ensemble:
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

    if slr_only_shoreline_mean is not None:
        ax.plot(
            proj_years,
            slr_only_shoreline_mean,
            color="deeppink",
            linewidth=2.0,
            alpha=0.7,
            label="Projected SLR-only",
        )
        ax.fill_between(
            proj_years,
            slr_only_shoreline_q05,
            slr_only_shoreline_q95,
            color="deeppink",
            alpha=0.15,
            label="SLR-only envelope (5-95%)",
        )

    if trend_only_shoreline_mean is not None:
        ax.plot(
            proj_years,
            trend_only_shoreline_mean,
            color="black",
            linewidth=2.0,
            alpha=0.7,
            label="Projected trend-only",
        )
        ax.fill_between(
            proj_years,
            trend_only_shoreline_q05,
            trend_only_shoreline_q95,
            color="black",
            alpha=0.15,
            label="Trend-only envelope (5-95%)",
        )

    if slr_years.size > 0:
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

    if trend_degenerate_mean is not None:
        ax_trend.axvline(
            trend_degenerate_mean,
            color="gray",
            linewidth=2.0,
            label="Extracted trends (degenerate)",
        )
    elif trend_line_x is not None and trend_line_y is not None:
        ax_trend.plot(
            trend_line_x,
            trend_line_y,
            color="dimgray",
            linewidth=2.0,
            label="Extracted trends (KDE)",
        )

    if historic_rug_trends.size > 0 or recent_rug_trends.size > 0:
        # Rug ticks for historic and recent trend pools.
        y_min, y_max = ax_trend.get_ylim()
        rug_bottom = y_min
        rug_top = y_min + 0.08 * (y_max - y_min if y_max > y_min else 1.0)
        if historic_rug_trends.size > 0:
            ax_trend.vlines(
                historic_rug_trends,
                rug_bottom,
                rug_top,
                color="tab:green",
                alpha=0.8,
                linewidth=1.2,
                label="Historic sampled trends used in projection (rug)",
            )
        if recent_rug_trends.size > 0:
            ax_trend.vlines(
                recent_rug_trends,
                rug_bottom,
                rug_top,
                color="darkorange",
                alpha=0.85,
                linewidth=1.2,
                label="Recent sampled trends used in projection (rug)",
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

#%%
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
    recent_trend_pool,
    sampled_historic_trends=None,
    sampled_recent_trends=None,
    slr_only_dy_segments=None,
    trend_only_dy_segments=None,
):
    """Plot observed shoreline and projected trajectories.

    Supports either:
    - single realization: dy_segments shape (n_segments,)
    - ensemble realizations: dy_segments shape (n_realizations, n_segments)

    In ensemble mode, plots the mean projected shoreline with a 5-95% envelope.
    """
    prepared = prepare_observed_and_projected_single_run_data(
        t_obs=t_obs,
        y_obs=y_obs,
        projection_start_year=projection_start_year,
        dy_segments=dy_segments,
        r_segments=r_segments,
        slr_years=slr_years,
        slr_q17_values=slr_q17_values,
        slr_q50_values=slr_q50_values,
        slr_q83_values=slr_q83_values,
        dt=dt,
        segment_years=segment_years,
        t_loess=t_loess,
        y_loess=y_loess,
        trend_pool=trend_pool,
        recent_trend_pool=recent_trend_pool,
        sampled_historic_trends=sampled_historic_trends,
        sampled_recent_trends=sampled_recent_trends,
        slr_only_dy_segments=slr_only_dy_segments,
        trend_only_dy_segments=trend_only_dy_segments,
    )
    plot_observed_and_projected_single_run_from_processed(
        prepared=prepared,
        site_id=site_id,
        transect_id=transect_id,
        out_fp=out_fp,
    )
    
    return prepared