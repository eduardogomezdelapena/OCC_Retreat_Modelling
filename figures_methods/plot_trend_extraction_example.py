#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone methods-figure: satellite shoreline observations, LOESS smoothing,
and the resulting empirical (block-bootstrap) trend-rate distribution, for a
single example transect.

This is a self-contained script (does not import trend_uncertainty.py, whose
module-level code runs the full multi-site pipeline). The data-processing
steps mirror block_bootstrap_slopes / filter_to_longest_consecutive_year_run
in trend_uncertainty.py.
"""
#%%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from loess.loess_1d import loess_1d

# ---- Example transect to plot (matches the reference screenshot) ----------
SITE_ID = "nzd0135"
TRANSECT_ID = "nzd0135-0004"
CUSTOM_REF_YEAR = 2025
LOESS_WINDOW_YEARS = 10  # also used as the block-bootstrap window length
RECENT_TREND_WINDOW_YEARS = 10.0  # look-back window for the "recent" trend pool
N_BOOT = 1000
SEED = 42
OUT_FP = "figures_methods/trend_extraction_example.png"


def load_transect_data(site_id):
    url = (
        "https://raw.githubusercontent.com/UoA-eResearch/CoastSat/main/data/"
        f"{site_id}/transect_time_series_tidally_corrected_smoothed.csv"
    )
    df = pd.read_csv(url, header=0)
    df["dates"] = pd.to_datetime(df["dates"])
    t_years = df["dates"].dt.year + (df["dates"].dt.dayofyear - 1) / 365.25
    return t_years, df


def filter_to_longest_consecutive_year_run(
    t, y, dates, min_span_years, max_gap_months, custom_ref_year,
):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))

    order = np.argsort(dates)
    t, y, dates = t[order], y[order], dates[order]

    previous_dates = pd.Series(dates[:-1])
    next_dates = pd.Series(dates[1:])
    gap_breaks = next_dates > (previous_dates + pd.DateOffset(months=int(max_gap_months)))
    break_indices = np.where(gap_breaks.to_numpy())[0] + 1

    segment_starts = np.r_[0, break_indices]
    segment_ends = np.r_[break_indices, t.size]

    must_bepresent_year = custom_ref_year - 1
    selected_segment = None
    for start_idx, end_idx in zip(segment_starts[::-1], segment_ends[::-1]):
        if end_idx - start_idx < 2:
            continue
        segment_dates = dates[start_idx:end_idx]
        if not np.any(segment_dates.year == must_bepresent_year):
            continue
        span_years = (dates[end_idx - 1] - dates[start_idx]).days / 365.25
        if span_years >= min_span_years:
            selected_segment = (start_idx, end_idx)
            break

    if selected_segment is None:
        raise ValueError(
            f"No segment with >= {min_span_years} years spanning {must_bepresent_year} found "
            f"for this transect."
        )

    start_idx, end_idx = selected_segment
    mask = np.zeros(t.size, dtype=bool)
    mask[start_idx:end_idx] = True
    return t[mask], y[mask]


def build_loess_time_grid(t):
    t = np.asarray(t, dtype=float)
    t0 = float(np.min(t))
    return t - t0, t0


def block_bootstrap_slopes(t, y, block_years, n_boot, random_state):
    rng = np.random.default_rng(random_state)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]

    n = len(y)
    window_ends = t + block_years
    ends = np.searchsorted(t, window_ends, side="right")
    valid_starts = np.where((window_ends <= t[-1]) & ((ends - np.arange(n)) >= 2))[0]
    if valid_starts.size == 0:
        raise ValueError(f"No {block_years}-year windows contain >=2 observations.")

    slopes = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        start = int(rng.choice(valid_starts))
        window_start = float(t[start])
        window_end = window_start + block_years
        end = int(np.searchsorted(t, window_end, side="right"))
        idx = np.arange(start, end)
        t_fit, y_fit = t[idx], y[idx]
        if t_fit[-1] < window_end:
            y_end = float(np.interp(window_end, t, y))
            t_fit = np.append(t_fit, window_end)
            y_fit = np.append(y_fit, y_end)
        slopes[i] = Polynomial.fit(t_fit, y_fit, 1).convert().coef[1]
    return slopes


def gaussian_kde_silverman(samples, n_points=300):
    """Gaussian KDE with Silverman's rule-of-thumb bandwidth (matches viz.py)."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    sigma = float(np.std(samples, ddof=1))
    bw = max(1.06 * sigma * (n ** (-1.0 / 5.0)), 1e-6)
    x_lo, x_hi = np.min(samples) - 3.0 * bw, np.max(samples) + 3.0 * bw
    x = np.linspace(x_lo, x_hi, n_points)
    z = (x[:, np.newaxis] - samples[np.newaxis, :]) / bw
    density = np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))
    return x, density


#%% Load, clean, and smooth the example transect.
t_years, df = load_transect_data(SITE_ID)
y = df[TRANSECT_ID]

mask = np.isfinite(y)
t_clean = t_years[mask].values
y_clean = y[mask].values
dates_clean = df.loc[mask, "dates"].values

t_clean, y_clean = filter_to_longest_consecutive_year_run(
    t_clean, y_clean, dates_clean,
    min_span_years=LOESS_WINDOW_YEARS,
    max_gap_months=10,
    custom_ref_year=CUSTOM_REF_YEAR,
)

t_loess, t_origin = build_loess_time_grid(t_clean)
total_timespan_years = t_clean.max() - t_clean.min()
frac = np.clip(LOESS_WINDOW_YEARS / total_timespan_years, 0.15, 0.9)

x_smooth, y_smooth, _ = loess_1d(
    t_loess, y_clean, xnew=t_loess, frac=frac, degree=1,
)
t_smooth = x_smooth + t_origin

# Use the LOESS shoreline position at 2025 as the zero-reference datum.
if not (t_smooth.min() <= CUSTOM_REF_YEAR <= t_smooth.max()):
    raise ValueError("The LOESS time series does not span the 2025 reference year.")
ref_position = float(np.interp(CUSTOM_REF_YEAR, t_smooth, y_smooth))
y_clean = y_clean - ref_position
y_smooth = y_smooth - ref_position

#%% Block-bootstrap the historical trend-rate ensemble.
boot_slopes = block_bootstrap_slopes(
    t_smooth, y_smooth,
    block_years=LOESS_WINDOW_YEARS,
    n_boot=N_BOOT,
    random_state=SEED,
)
kde_x, kde_y = gaussian_kde_silverman(boot_slopes)

#%% Block-bootstrap the "recent" trend-rate ensemble (most recent ~10 years),
# mirroring the p_recent(r) mixture-weighting logic in trend_uncertainty.py.
recent_boot_slopes = None
recent_mask = t_smooth >= (float(np.max(t_smooth)) - RECENT_TREND_WINDOW_YEARS)
t_recent = t_smooth[recent_mask]
y_recent = y_smooth[recent_mask]

if t_recent.size >= 2:
    recent_span_years = float(np.max(t_recent) - np.min(t_recent))
    if recent_span_years > 0.0:
        recent_block_years = min(RECENT_TREND_WINDOW_YEARS, recent_span_years * 0.8)
        if recent_block_years > 0.0:
            try:
                recent_boot_slopes = block_bootstrap_slopes(
                    t_recent, y_recent,
                    block_years=recent_block_years,
                    n_boot=N_BOOT,
                    random_state=SEED + 1,
                )
            except ValueError:
                recent_boot_slopes = None

#%% Figure: observations + LOESS (left), trend-rate distribution (right).
fig, (ax_ts, ax_trend) = plt.subplots(1, 2, figsize=(12, 4.5))

ax_ts.plot(
    t_clean, y_clean, "o",
    color="0.6", markersize=3.5, alpha=0.7, label="Satellite observations",
)
ax_ts.plot(t_smooth, y_smooth, "-", color="black", linewidth=2.0, label="LOESS smoothed")
ax_ts.axhline(0, color="0.6", linewidth=0.8, linestyle="--")
ax_ts.axvline(CUSTOM_REF_YEAR, color="0.4", linewidth=0.8, linestyle=":")
ax_ts.set_title("Historical")
ax_ts.set_xlabel("")
ax_ts.set_ylabel("Shoreline\nposition (m)")
ax_ts.legend(loc="lower left", fontsize=8, frameon=False)

ax_trend.hist(
    boot_slopes, bins=25, density=True,
    color="mediumseagreen", alpha=0.5, edgecolor="none",
    label="Historic trend samples",
)
ax_trend.plot(kde_x, kde_y, color="black", linewidth=2.0, label="Historic KDE (Silverman)")

if recent_boot_slopes is not None:
    y_min, y_max = ax_trend.get_ylim()
    rug_height = 0.08 * (y_max - y_min)
    ax_trend.vlines(
        recent_boot_slopes,
        y_min,
        y_min + rug_height,
        color="darkorange",
        alpha=0.45,
        linewidth=0.8,
        label=f"Recent trends (last {RECENT_TREND_WINDOW_YEARS:.0f} yr)",
    )

ax_trend.set_title(f"Historic trend distribution: {SITE_ID} {TRANSECT_ID}")
ax_trend.set_xlabel("Trend [m/yr]")
ax_trend.set_ylabel("Density")
ax_trend.legend(loc="upper right", fontsize=8, frameon=True)

fig.tight_layout()
fig.savefig(OUT_FP, dpi=300)
print(f"Saved figure to {OUT_FP}")
plt.close(fig)
