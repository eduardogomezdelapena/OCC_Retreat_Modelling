#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for satellite trend bootsrapping, using Orewa as study case"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
from viz import (
    plot_observed_and_projected_single_run
)
#%% Read data, declare sites ID

#Which directories exist under data/
import requests

owner = "UoA-eResearch"
repo = "CoastSat"
path = "data"

url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
resp = requests.get(url)
resp.raise_for_status()

nzd_sites = sorted(
    item["name"]
    for item in resp.json()
    if item["type"] == "dir" and "nzd" in item["name"]
)

print(f"{len(nzd_sites)} NZD sites found")

# Define random seed for reproducibility
seed = 42 
single_preview_random_state = seed
n_mc_realizations_per_transect = 50
loess_window = 10  # years, used for both LOESS smoothing 
#and block bootstrap window to match the timescale 
# of variability captured by the smoothed trend.


#%% Download data for a given site, and convert to decimal years. Function DEFINITION
def load_transect_data(site_id):
    url = (
        "https://raw.githubusercontent.com/UoA-eResearch/CoastSat/main/data/"
        f"{site_id}/transect_time_series_tidally_corrected_smoothed.csv"
    )
    df = pd.read_csv(url, header=0)
    df["dates"] = pd.to_datetime(df["dates"])
    # Convert to decimal years (for linear fit)
    t_years = (
        df["dates"].dt.year
        + (df["dates"].dt.dayofyear - 1) / 365.25
    )
    # Return time in decimal years, and the full dataframe
    return t_years, df

def build_loess_time_grid(t, n_grid=None):
    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError("LOESS time input must be one-dimensional.")

    if t.size == 0:
        return t.copy(), t.copy(), 0.0

    t0 = float(np.min(t))
    t_numeric = t - t0

    if n_grid is None:
        t_grid = t_numeric.copy()
    else:
        t_grid = np.linspace(t_numeric.min(), t_numeric.max(), num=int(n_grid))

    return t_numeric, t_grid, t0

#%% Filter to most recent continuous segment separated by long gaps. Function DEFINITION
def filter_to_longest_consecutive_year_run(
    t, y, dates,
    min_span_years,
    max_gap_months,
    site_id,
    transect_id,
):
    # Convert to numpy arrays for easier processing
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))

    # Initialize metadata
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

    # Basic checks
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

    # Metadata reporting
    meta["status"] = "ok"
    meta["kept_years"] = kept_years
    meta["removed_years"] = sorted(set(all_years) - set(kept_years))
    meta["segment_start"] = kept_dates[0].isoformat()
    meta["segment_end"] = kept_dates[-1].isoformat()
    meta["segment_span_years"] = float(span_years)

    # Return filtered (decimal) time, shoreline, and metadata
    return t[mask], y[mask], meta

# %% Bootstrap the trend, function DEFINITION
def block_bootstrap_slopes(
    t, y,
    block_years,
    n_boot,
    random_state,
    return_table=False,
):
    """
    Block bootstrap for linear trend slopes.

    Parameters
    ----------
    t : array
        Time in decimal years
    y : array
        Shoreline position
    block_years : float
        Window length in years. Each bootstrap sample is a linear 
        fit to a randomly sampled block-year window from the record.
    n_boot : int
        Number of bootstrap realizations

    Returns
    -------
    slopes : array
        Bootstrapped slope estimates.
    table : pandas.DataFrame, optional
        Returned only when return_table=True. Contains four columns:
        realization, slope_m_per_yr, date_range, window_length_years.
    """

    # Set up random generator
    rng = np.random.default_rng(random_state)

    # Convert to numpy arrays and ensure they are 1D
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort by time (just in case)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # Basic checks
    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 observations total.")

    # Valid start indices are those whose block-year window stays inside the record.
    window_ends = t + block_years
    ends = np.searchsorted(t, window_ends, side="right")

    # Require at least two observed points before interpolation at the window end.
    valid_starts = np.where((window_ends <= t[-1]) & ((ends - np.arange(n)) >= 2))[0]
    if valid_starts.size == 0:
        raise ValueError(
            f"No {block_years}-year windows contain >=2 observations. "
            "Check data density."
        )
    
    slopes = np.empty(n_boot, dtype=float)
    date_ranges = []
    window_lengths_years = []

    for i in range(n_boot):
        # Sample one block-year window per realization and fit its local slope.
        start = int(rng.choice(valid_starts))
        window_start = float(t[start])
        window_end = window_start + block_years
        end = int(np.searchsorted(t, window_end, side="right"))  # exclusive
        idx = np.arange(start, end)

        t_fit = t[idx]
        y_fit = y[idx]
        if t_fit[-1] < window_end:
            # Interpolate shoreline at exactly +block_years to enforce a true block-year fit span.
            y_end = float(np.interp(window_end, t, y))
            t_fit = np.append(t_fit, window_end)
            y_fit = np.append(y_fit, y_end)

        slopes[i] = np.polyfit(t_fit, y_fit, 1)[0]  # linear fit
        date_ranges.append(f"{window_start:.2f} to {window_end:.2f}")
        window_lengths_years.append(block_years)

    if return_table:
        table = pd.DataFrame(
            {
                "realization": np.arange(1, n_boot + 1, dtype=int),
                "slope_m_per_yr": slopes,
                "date_range": date_ranges,
                "window_length_years": window_lengths_years,
            }
        )
        return slopes, table

    return slopes


def build_projection_segment_durations(
    projection_start_year,
    dt,
    segment_years,
    align_to_decades,
):
    """Build segment durations from a start year and total horizon.

    If align_to_decades is True, the first segment ends at the next decade
    boundary (e.g., 2026->2030 gives 4 years), then regular segment_years
    segments are used for the remaining horizon.
    """
    total_years = float(dt)
    if total_years <= 0.0:
        raise ValueError("dt must be positive.")
    if segment_years <= 0:
        raise ValueError("segment_years must be positive.")

    durations = []
    remaining = total_years

    if align_to_decades:
        if projection_start_year is None:
            raise ValueError("projection_start_year is required when align_to_decades=True.")
        start_year = float(projection_start_year)
        next_decade = float(np.ceil(start_year / 10.0) * 10.0)
        first_step = next_decade - start_year
        tol = 1e-9
        if first_step > tol and first_step < (remaining - tol):
            durations.append(first_step)
            remaining -= first_step

    step_years = float(segment_years)
    tol = 1e-9
    while remaining > tol:
        step = min(step_years, remaining)
        durations.append(step)
        remaining -= step

    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        raise ValueError("No segment durations were generated.")
    return durations

# %% Monte Carlo simulations. Function DEFINITION
def mc_shoreline_change(
    c, tan_beta, delta_S,
    r_samples,
    segment_years,
    random_state,   
    delta_S_q17,
    delta_S_q50,
    delta_S_q83,
    projection_start_year=None,
    align_to_decades=False,
    dt=25,                      # 2025 as baseline year, projections to 2100
    n=200_000,
    return_delta_s_samples=True,
    return_r_segment_samples=True,
    return_slr_segment_samples=False,
    return_trend_segment_samples=False,
):
    """
    Monte Carlo propagation for shoreline change using an
    empirical (bootstrap) distribution of shoreline trend rates.

    For each simulation, dt is split into block-year segments and an independent
    trend rate is sampled per segment:

    dy_total = sum_k [ -(c/tanβ) * ΔS * (Δt_k/dt) + (r_k * Δt_k) ]

    where Δt_k is usually segment_years. If align_to_decades is enabled, the
    first segment can be shorter so projections land on decade years.

    Parameters:
    - c: adjustment factor (fixed)
    - tan_beta: nominal beach slope (m/m); uncertainty added as ±20%
        - delta_S: sea level rise (m), used directly if quantiles are not supplied
        - delta_S_q17, delta_S_q50, delta_S_q83: optional shifted SLR quantiles
            used to fit a Gaussian SLR distribution for sampling
    - r_samples: empirical trend rates (sampled from r_samples)
    - dt: time horizon (years)
        - return_r_segment_samples: if True, also return sampled r values for each
            time segment (useful for single-run diagnostics)
        - return_slr_segment_samples: if True, also return the SLR-only shoreline
            change component per segment for each realization
        - return_trend_segment_samples: if True, also return the trend-only shoreline
            change component per segment for each realization (i.e., without SLR contribution)

    Uncertainty sources:
    - Trend rate r_samples (from bootstrap)
    - Beach slope tan_beta (±20% uniform around nominal)
    - SLR delta_S (deterministic or Gaussian from 17/50/83 quantiles)

    """

    rng = np.random.default_rng(random_state)

    # --- empirical sampling of trend rate ---
    r_samples = np.asarray(r_samples)
    r_samples = r_samples[np.isfinite(r_samples)]
    if r_samples.size == 0:
        raise ValueError("r_samples is empty after removing non-finite values.")

    trend_low = float(np.min(r_samples))
    trend_high = float(np.max(r_samples))
    if not np.isfinite(trend_low) or not np.isfinite(trend_high):
        raise ValueError("Trend range is invalid after filtering non-finite values.")
    if trend_low == trend_high:
        raise ValueError("Trend range has zero width; cannot sample uniformly.")

    # If quantiles are provided, fit a Gaussian to SLR quantiles and sample delta_S.
    # For a Normal distribution: q_p = mu + sigma*z_p.
    use_slr_gaussian = all(v is not None for v in (delta_S_q17, delta_S_q50, delta_S_q83))
    z83 = NormalDist().inv_cdf(0.83)

    if use_slr_gaussian:
        q17 = float(delta_S_q17)
        q50 = float(delta_S_q50)
        q83 = float(delta_S_q83)

        if not (q17 <= q50 <= q83):
            raise ValueError("Expected SLR quantiles to satisfy q17 <= q50 <= q83.")

        sigma_low = (q50 - q17) / z83
        sigma_high = (q83 - q50) / z83
        slr_sigma = float(np.mean([sigma_low, sigma_high]))
        slr_mu = q50

        if not np.isfinite(slr_sigma) or slr_sigma < 0.0:
            raise ValueError("Derived SLR sigma is invalid from q17/q50/q83.")

        sampled_delta_s = rng.normal(loc=slr_mu, scale=slr_sigma, size=n)

    else:
        sampled_delta_s = np.full(n, float(delta_S), dtype=float)
        slr_mu = float(delta_S)
        slr_sigma = 0.0

    # Vectorized Monte Carlo sampling for speed and reproducibility.
    sampled_tan_beta = rng.uniform(low=tan_beta * 0.8, high=tan_beta * 1.2, size=n)

    # The base SLR-driven term is negative (shoreline retreat)
    #  and is scaled by the sampled tan_beta per realization.
    bases = -(c / sampled_tan_beta) * sampled_delta_s         

    seg_durations = build_projection_segment_durations(
        projection_start_year=projection_start_year,
        dt=dt,
        segment_years=segment_years,
        align_to_decades=align_to_decades,
    )
    n_segments = int(seg_durations.size)
    duration_weights = seg_durations / float(dt)

    # Sample an independent r per simulation per segment uniformly across the
    # full extracted trend range: shape (n, n_segments)
    sampled_r_all = rng.uniform(low=trend_low, high=trend_high, size=(n, n_segments))

    # dy has shape (n, n_segments): change during each segment simulation.
    dy = (
        bases[:, np.newaxis] * duration_weights[np.newaxis, :]
        + sampled_r_all * seg_durations[np.newaxis, :]
    )

    # SLR-only shoreline change per segment, shape-matched with dy.
    slr_dy = bases[:, np.newaxis] * duration_weights[np.newaxis, :]

    # Trend-only shoreline change per segment (no SLR contribution).
    trend_dy = sampled_r_all * seg_durations[np.newaxis, :]

    # Total 75-year change used for summary statistics
    dy_total = dy.sum(axis=1)                                  # (n,)

    summary = {
        "dt_years": dt,
        "base_term_m": float(np.mean(bases)),  # mean of sampled bases
        "segment_durations_years": seg_durations.tolist(),
        "align_to_decades": bool(align_to_decades),

        "slr_sampling": "gaussian_q17_q50_q83" if use_slr_gaussian else "deterministic",
        "delta_S_mean_m": float(np.mean(sampled_delta_s)),
        "delta_S_p05_m": float(np.quantile(sampled_delta_s, 0.05)),
        "delta_S_p95_m": float(np.quantile(sampled_delta_s, 0.95)),
        "delta_S_mu_m": float(slr_mu),
        "delta_S_sigma_m": float(slr_sigma),

        "trend_rate_mean": float(np.mean(r_samples)),  # population mean
        "trend_rate_p05": float(np.quantile(r_samples, 0.05)),
        "trend_rate_p95": float(np.quantile(r_samples, 0.95)),
        "trend_rate_min": trend_low,
        "trend_rate_max": trend_high,

        "dy_median_m": float(np.median(dy_total)),
        "dy_p05_m": float(np.quantile(dy_total, 0.05)),
        "dy_p95_m": float(np.quantile(dy_total, 0.95)),


    }

    outputs = [dy, summary]
    if return_delta_s_samples:
        outputs.append(sampled_delta_s)
    if return_r_segment_samples:
        outputs.append(sampled_r_all)
    if return_slr_segment_samples:
        outputs.append(slr_dy)
    if return_trend_segment_samples:
        outputs.append(trend_dy)

    return tuple(outputs)


#%%

import pandas as pd
import geopandas as gpd
from utils import load_metadata_nzrise, load_slrdata_nzrise, load_and_merge_coastsat_data, nearest_points
#% Define constants, load data, and merge datasets (similar to merge_slr_sat.py)

# Constants
CRS_WGS84 = 4326 # Lat/Lon
custom_ref_year = 2025


meta_data_fp = "NZ_VLM_final_May24.csv"
slr_fp = f"NZ_Searise_noVLM-2005_{custom_ref_year}adjusted.csv"
meta_data = load_metadata_nzrise(meta_data_fp, crs_str= CRS_WGS84 ) # gpd.DataFrame
slr_data  = load_slrdata_nzrise(slr_fp) #pd.DataFrame

merged= pd.merge(meta_data,slr_data,
                        on='nzrise_site_id', how='outer')
nzrise_merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{CRS_WGS84}", geometry= 'geom_nzrise')

coastsat_merged = load_and_merge_coastsat_data(
    "transects_reindexed_Nickupdate.geojson",
    f"points_ref_shoreline_{custom_ref_year}_Nickupdate.geojson",
    CRS_WGS84
)

#Calculate distances to nearest NZRise points
distances, nearest_indices = nearest_points(meta_data,
                                             coastsat_merged)

#Merge nzrise_merged & coastsat_merged, based on nearest_indices
#to each coastsat_transect_id there is a matching nzrise_site_id
#nzrise_site_id is repeated, repeat also coastsat_transect_id as many times needed
#add column of nzrise_site_id to coastsat_merged

coastsat_merged["nzrise_site_id"]=meta_data["nzrise_site_id"].iloc[nearest_indices].values

#Now merge based on nzrise_site_id
all_merged = pd.merge( coastsat_merged, nzrise_merged,
                      on='nzrise_site_id', how='left')
#Print all columns, there should be 3 geometry columns (points ref , coastsat transects
# and nzrise points).

print(all_merged.columns)
#Transform into geopandas? A geometry column needs to be picked, points ref .

all_merged = gpd.GeoDataFrame(all_merged,crs=f"EPSG:{CRS_WGS84}",
                               geometry= 'geom_points_ref')

# Alias CoastSat IDs to names used in the transect processing loop.
all_merged["site_id"] = all_merged["coastsat_site_id"]
all_merged["transect_id"] = all_merged["coastsat_transect_id"]

# Keep only the requested climate pathway for SLR extraction.
scenario_target = "2.6"
ssp_target = "ssp1"
target_year = custom_ref_year + 75  # 2025 baseline to 2100

all_merged = all_merged[
    (all_merged["scenario"].astype(str) == scenario_target)
    & (all_merged["SSP"].astype(str).str.lower() == ssp_target)
].copy()
all_merged["year"] = pd.to_numeric(all_merged["year"], errors="coerce")


# %% Main loop: load data, apply filters, bootstrap, Monte Carlo

dy_rows = []
meta_rows = []

from pathlib import Path
out_dir = Path("original_plots_ts")
debug_slr_histograms = True
debug_variance_plots = True

# nzd_sites_trial = nzd_sites[0:5]
# nzd_sites_trial = ["nzd0001"]
nzd_sites_trial = ["nzd0137"]

#Try only 2 sites first
for site_id in nzd_sites_trial:

    # Create output directory for this sites plots
    site_dir = out_dir / site_id
    site_dir.mkdir(parents=True, exist_ok=True)

    # Load data for the site (all transects), and convert to decimal years.
    # The function returns time in decimal years, and the full dataframe.
    t_years, df = load_transect_data(site_id)

    # Define transect columns (those that start with site_id + "-")
    transect_cols = [
        c for c in df.columns
        if c.startswith(site_id + "-")
    ]
    
    transect_trials=transect_cols[1:2] #Try only 1 transect first

    for transect_id in transect_trials:
    # for transect_id in transect_cols:

        # Extract shoreline position for this transect
        y = df[transect_id]


        fig = plt.figure(figsize=(10, 4))
        plt.plot(t_years, y, "o-", label="Original data")
        plt.xlabel("Time (decimal years)")
        plt.ylabel("Shoreline position (m)")
        plt.title(f"{site_id} {transect_id} – Original shoreline time series")
        plt.legend()
        plt.xlim(t_years.min() , t_years.max() )
        plt.tight_layout()
        plt.savefig(site_dir / f"{site_id}_{transect_id}_original_timeseries.png", dpi=300)    
        plt.close()


        # Filter out NaN values (and corresponding time and year arrays)
        mask = np.isfinite(y)
        t_clean = t_years[mask].values
        y_clean = y[mask].values
        # Extract non-decimal years for the after NaN observations
        dates_clean = df.loc[mask, "dates"].values

        # Keep the most recent segment without a gap longer than 9 months.
        t_clean, y_clean, meta = filter_to_longest_consecutive_year_run(
            t_clean, y_clean, dates_clean,
            min_span_years=5,
            max_gap_months=10,
            site_id = site_id,
            transect_id = transect_id,
        )

        meta_rows.append({
            "site_id": site_id,
            "transect_id": transect_id,
            **meta
        })

        # If not enough consecutive years, skip bootstrap for this transect
        if meta["status"] != "ok":
            print(f"SKIP {site_id} {transect_id}: {meta['status']} (best_run_len={meta.get('best_run_len')})")
            continue

#%
        # Apply LOESS smoothing to the cleaned time series.
        
        from loess.loess_1d import loess_1d

        t_loess, t_loess_grid, t_loess_origin = build_loess_time_grid(
            t_clean,
            n_grid=None,
        )
#%

        # Dynamic frac based on 5-year window
        total_timespan_years = t_clean.max() - t_clean.min()
        frac = loess_window / total_timespan_years
        frac = np.clip(frac, 0.15, 0.9)  # Keep within reasonable bounds

        x_smooth, y_smooth, _ = loess_1d(
            t_loess,
            y_clean,
            xnew=t_loess_grid,
            frac=frac,  # Dynamically set for ~5 year window
            degree=1,  # Linear local fit is more stable at series boundaries.
        )

        t_smooth = x_smooth + t_loess_origin

        fig = plt.figure()
        plt.plot(t_clean, y_clean, "*-", label="Original data") 
        plt.plot(t_smooth, y_smooth, "r-", label="LOESS smoothed")
        plt.xlabel("Time (decimal years)")
        plt.ylabel("Shoreline position (m)")
        plt.title(f"{site_id} {transect_id} – LOESS smoothed shoreline")
        plt.legend()
        plt.tight_layout()
        loess_plot_fp = site_dir / f"{site_id}_{transect_id}_loess_smoothed.png"
        fig.savefig(loess_plot_fp, dpi=200, bbox_inches="tight")
        plt.close(fig)

        ###############################
        # Apply bootstrap and Monte Carlo functions

        boot_slopes, boot_slopes_table = block_bootstrap_slopes(
            t_smooth, y_smooth,
            block_years= loess_window,  # Match the LOESS window for consistency
            n_boot=1000,  #1000
            random_state=seed,
            return_table=True,
        )


        # Extract tan_beta (beach slope) from the coastsat data for this transect
        if "beach_slope" not in coastsat_merged.columns:
            raise KeyError("Expected 'beach_slope' column in coastsat_merged")

        mask_tb = (
            (coastsat_merged["coastsat_site_id"] == site_id)
            & (coastsat_merged["coastsat_transect_id"] == transect_id)
        )
        if not mask_tb.any():
            print(f"Skipping {site_id} {transect_id}: no matching tan_beta in coastsat_merged")
            continue

        tan_vals = coastsat_merged.loc[mask_tb, "beach_slope"]
        if len(tan_vals) > 1:
            # In case of duplicates, average them
            tan_beta = float(tan_vals.mean())
        else:
            tan_beta = float(tan_vals.iloc[0])

        # Extract shifted SLR quantiles for this transect.
        mask_slr = (
            (all_merged["site_id"] == site_id)
            & (all_merged["transect_id"] == transect_id)
            & (all_merged["year"] == target_year)
            & (all_merged["scenario"].astype(str) == scenario_target)
            & (all_merged["SSP"].astype(str).str.lower() == ssp_target)
        )
  
        if not mask_slr.any():
            print(f"Skipping {site_id} {transect_id}: no matching SLR data in all_merged")
            continue

        slr_q17_vals = all_merged.loc[mask_slr, "17_shifted"].dropna()
        slr_q50_vals = all_merged.loc[mask_slr, "50_shifted"].dropna()
        slr_q83_vals = all_merged.loc[mask_slr, "83_shifted"].dropna()

        if slr_q17_vals.empty or slr_q50_vals.empty or slr_q83_vals.empty:
            print(
                f"Skipping {site_id} {transect_id}: missing one or more shifted SLR quantiles"
            )
            continue

        delta_s_q17 = float(slr_q17_vals.mean())
        delta_s_q50 = float(slr_q50_vals.mean())
        delta_s_q83 = float(slr_q83_vals.mean())
##########################
        slr_series = (
            all_merged.loc[
                (all_merged["site_id"] == site_id)
                & (all_merged["transect_id"] == transect_id),
                ["year", "17_shifted", "50_shifted", "83_shifted"],
            ]
            .dropna()
            .groupby("year", as_index=False)[["17_shifted", "50_shifted", "83_shifted"]]
            .mean()
            .sort_values("year")
        )

        slr_projection_years = np.array([], dtype=float)
        slr_projection_q17 = np.array([], dtype=float)
        slr_projection_q50 = np.array([], dtype=float)
        slr_projection_q83 = np.array([], dtype=float)

        slr_series_plot = slr_series.loc[slr_series["year"] <= target_year].copy()
        if not slr_series_plot.empty:
            year_values = slr_series_plot["year"].to_numpy(dtype=float)
            if year_values.min() <= custom_ref_year <= year_values.max():
                has_baseline_year = bool(np.isclose(year_values, custom_ref_year).any())

                for col in ["17_shifted", "50_shifted", "83_shifted"]:
                    col_values = slr_series_plot[col].to_numpy(dtype=float)
                    baseline_value = float(np.interp(custom_ref_year, year_values, col_values))
                    slr_series_plot[col] = col_values - baseline_value

                if not has_baseline_year:
                    slr_series_plot = pd.concat(
                        [
                            pd.DataFrame(
                                [{
                                    "year": float(custom_ref_year),
                                    "17_shifted": 0.0,
                                    "50_shifted": 0.0,
                                    "83_shifted": 0.0,
                                }]
                            ),
                            slr_series_plot,
                        ],
                        ignore_index=True,
                    )

                slr_series_plot = (
                    slr_series_plot.loc[slr_series_plot["year"] >= custom_ref_year]
                    .sort_values("year")
                    .reset_index(drop=True)
                )

                slr_projection_years = slr_series_plot["year"].to_numpy(dtype=float)
                slr_projection_q17 = slr_series_plot["17_shifted"].to_numpy(dtype=float)
                slr_projection_q50 = slr_series_plot["50_shifted"].to_numpy(dtype=float)
                slr_projection_q83 = slr_series_plot["83_shifted"].to_numpy(dtype=float)
########################

        print(
            f"{site_id} {transect_id}: tan_beta = {tan_beta}, "
            f"delta_S_shifted(q17/q50/q83)=({delta_s_q17:.3f}, {delta_s_q50:.3f}, {delta_s_q83:.3f}) m "
            f"for {ssp_target}-{scenario_target}, year={target_year}"
        )

 
        mc_result = mc_shoreline_change(
            c=1.0,
            tan_beta=tan_beta ,
            delta_S=delta_s_q50,
            r_samples = boot_slopes,
            segment_years= loess_window,  # Match the LOESS window for consistency
            random_state=single_preview_random_state,
            delta_S_q17=delta_s_q17,
            delta_S_q50=delta_s_q50,
            delta_S_q83=delta_s_q83,
            projection_start_year=float(custom_ref_year),
            align_to_decades=True,
            n=n_mc_realizations_per_transect,
            return_delta_s_samples=True,
            return_r_segment_samples=True,
            return_slr_segment_samples=True,
            return_trend_segment_samples=True,
            )
        
        dy, summ, sampled_delta_s, sampled_r_all, sampled_slr_dy, sampled_trend_dy = mc_result

        # Store results
        dy_rows.append({
                "site_id": site_id,
                "transect_id": transect_id,
                "n_obs": int(y_clean.size),
                "n_boot": 1000,
            "n_mc": n_mc_realizations_per_transect,
                "dt_years": float(summ["dt_years"]),
                "tan_beta": float(tan_beta),
                "delta_S_q17_m": float(delta_s_q17),
                "delta_S_q50_m": float(delta_s_q50),
                "delta_S_q83_m": float(delta_s_q83),
                "delta_S_sigma_m": float(summ["delta_S_sigma_m"]),
                "slr_year": int(target_year),
                "scenario": scenario_target,
                "SSP": ssp_target,
                "base_term_m": float(summ["base_term_m"]),
                "dy_p05_m": float(summ["dy_p05_m"]),
                "dy_median_m": float(summ["dy_median_m"]),
                "dy_p95_m": float(summ["dy_p95_m"])

            })

        # Pass all sampled realizations so the plot can show mean + uncertainty envelope.
        preview_dy = np.asarray(dy, dtype=float)
        preview_r_samples = np.asarray(sampled_r_all, dtype=float)
        preview_slr_only_dy = np.asarray(sampled_slr_dy, dtype=float)
        preview_trend_only_dy = np.asarray(sampled_trend_dy, dtype=float)

        preview_segment_years = loess_window  # Match the Monte Carlo 
        # segment length to the LOESS window for interpretability

        # The overall projection time horizon is the same as
        #  used in the Monte Carlo function, which is determined by 
        # the "dt" parameter.
        preview_dt = int(summ["dt_years"])

        if preview_r_samples.shape != preview_dy.shape:
            raise ValueError("Preview r samples and dy segments must have matching shapes.")
        if preview_slr_only_dy.shape != preview_dy.shape:
            raise ValueError("Preview SLR-only segments and dy segments must have matching shapes.")
        if preview_trend_only_dy.shape != preview_dy.shape:
            raise ValueError("Preview trend-only segments and dy segments must have matching shapes.")

        preview_ts_plot_fp = site_dir / f"{site_id}_{transect_id}_single_run_observed_projected.png"
        
        
        prepared = plot_observed_and_projected_single_run(
            t_obs=t_clean,
            y_obs=y_clean,
            t_loess=t_smooth,
            y_loess=y_smooth,
            projection_start_year=float(custom_ref_year),
            dy_segments=preview_dy,
            r_segments=preview_r_samples,
            slr_years=slr_projection_years,
            slr_q17_values=slr_projection_q17,
            slr_q50_values=slr_projection_q50,
            slr_q83_values=slr_projection_q83,
            dt=preview_dt,
            segment_years=preview_segment_years,
            site_id=site_id,
            transect_id=transect_id,
            out_fp=preview_ts_plot_fp,
            trend_pool=boot_slopes,
            slr_only_dy_segments=preview_slr_only_dy,
            trend_only_dy_segments=preview_trend_only_dy,
        )

        print(site_id, transect_id)

#%% Export results to CSV for further plotting in a webdashboard.

def export_results_to_csv(prepared, meta_str):
    """Export the prepared data for a single site/transect to a CSV file."""


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


    # Export projected  YEAR, mean, q05, q95 to CSV
    df_proj = pd.DataFrame({
        "year": proj_years,
        "proj_shoreline_mean": proj_shoreline_mean,
        "proj_shoreline_q05": proj_shoreline_q05,
        "proj_shoreline_q95": proj_shoreline_q95,
        "slr_only_shoreline_mean": slr_only_shoreline_mean,
        "slr_only_shoreline_q05": slr_only_shoreline_q05,
        "slr_only_shoreline_q95": slr_only_shoreline_q95,
        "trend_only_shoreline_mean": trend_only_shoreline_mean,
        "trend_only_shoreline_q05": trend_only_shoreline_q05,
        "trend_only_shoreline_q95": trend_only_shoreline_q95,
    })
    # Save to csv, but include ssp, site_id and transect_id in the filename
    # # for easy identification
    csv_fp = f"{meta_str}_projection_results.csv"
    df_proj.to_csv(csv_fp, index=False)
    print(f"Exported projection results to {csv_fp}")

#create outputs directory if it doesn't exist
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
#create site-specific directory if it doesn't exist
site_dir = output_dir / site_id
site_dir.mkdir(parents=True, exist_ok=True)

meta_str =site_dir / f"{site_id}_{transect_id}_{ssp_target}_{scenario_target}"


#Then save the CSV in the site-specific directory
export_results_to_csv(prepared, meta_str)




#%% Plot projections

#What is in preview_dy? 
# #It should be the full array of shape (n_mc_realizations_per_transect, n_segments)
# containing the projected shoreline change 
# for each Monte Carlo realization and each time segment. 
# Each row corresponds to one realization's trajectory of shoreline change
#  over the projection period, segmented into 5-year intervals 
# (or however the segments were defined). 
# The values in preview_dy represent the change in shoreline
#  position (in meters) for each segment of each realization,
#  which can be summed across segments to get the total projected 
# change at the end of the time horizon.

#Plotting average
plt_dy = np.asarray(preview_dy, dtype=float)

plt.plot(plt_dy.T, alpha=0.2)  # Plot all individual trajectories with low opacity
plt.plot(np.cumsum(np.average(plt_dy, axis=0)), "o-", label ='average cumulative change')  # Plot average cumulative change
plt.plot(np.mean(plt_dy, axis=0), "o-", label ='average change per segment')
plt.legend(loc="lower right")

#How to visualize uncertainty in the projected trajectories?
# One common approach is to plot the mean trajectory along with 
# a shaded area representing the range of trajectories
#  (e.g., min-max or confidence intervals).


cum_dy = np.cumsum(plt_dy, axis=1)                         # cumulative per realization
cum_mean = np.mean(cum_dy, axis=0)
cum_q05, cum_q25, cum_q75, cum_q95 = np.quantile(cum_dy, [0.05, 0.25, 0.75, 0.95], axis=0)


plt.plot(cum_mean, "o-", label="Mean cumulative change")
plt.fill_between(
    np.arange(cum_dy.shape[1]),
    cum_q05,
    cum_q95,
    color="blue",
    alpha=0.3,
    label="90% CI",
)
plt.xlabel("Segment index")
plt.ylabel("Cumulative shoreline change (m)")

#%%
#What about preview_slr_only_dy?
# preview_slr_only_dy should be the array of shape (n_mc_realizations_per_transect, 
# n_segments) containing the sampled SLR-only shoreline change (dy_slr_k) for each Monte Carlo
#  realization and each time segment.

plt_slr_only_dy = np.asarray(preview_slr_only_dy, dtype=float)

plt.plot(plt_slr_only_dy.T, alpha=0.2)  # Plot all individual trajectories with low opacity
plt.plot(np.cumsum(np.average(plt_slr_only_dy, axis=0)), "o-", label ='average cumulative change')  # Plot average cumulative change
plt.plot(np.mean(plt_slr_only_dy, axis=0), "o-", label ='average change per segment')
plt.legend(loc="lower right")


cum_dy_slr = np.cumsum(plt_slr_only_dy, axis=1)                         # cumulative per realization
cum_mean = np.mean(cum_dy_slr, axis=0)
cum_q05, cum_q25, cum_q75, cum_q95 = np.quantile(cum_dy_slr, [0.05, 0.25, 0.75, 0.95], axis=0)


plt.plot(cum_mean, "o-", label="Mean cumulative change")
plt.fill_between(
    np.arange(cum_dy_slr.shape[1]),
    cum_q05,
    cum_q95,
    color="blue",
    alpha=0.3,
    label="90% CI",
)
plt.xlabel("Segment index")
plt.ylabel("Cumulative shoreline change (m)")

#%%

plt.plot(year_average_end_obs + np.cumsum(np.average(plt_dy, axis=0)), "o-")
#%%
projection_years = np.arange(float(custom_ref_year) + preview_segment_years,
                             float(custom_ref_year) + preview_dt + 1,
                             preview_segment_years)

proj_mean= year_average_end_obs +np.cumsum(np.average(plt_dy, axis=0))
proj_min = proj_mean - np.cumsum(np.min(plt_dy, axis=0))
proj_max = proj_mean + np.cumsum(np.max(plt_dy, axis=0))    

fig = plt.figure(figsize=(10, 5))
plt.plot(t_smooth, y_smooth, "r-", label="LOESS smoothed observed")
plt.plot(projection_years, proj_mean, "o-", label="Mean projected change")
plt.fill_between(
    projection_years,
    proj_min,
    proj_max,
    color="blue",
    alpha=0.3,
    label="Min-Max range",
)
plt.xlabel("Segment index")
plt.ylabel("Cumulative shoreline change (m)")
plt.title(f"{site_id} {transect_id} – Projected shoreline change (min-max range)")
plt.legend()
plt.tight_layout()
plt.show()
#%%
# Use the same baseline and trajectory construction as
#  plot_observed_and_projected_single_run.
baseline_mask = (
    (t_smooth >= (custom_ref_year - 5.0))
    & (t_smooth <= custom_ref_year)
)
# If there are points in the 5-year window before the reference year,
#  average them for the baseline.
if np.any(baseline_mask): 
    year_average_end_obs = float(np.mean(y_smooth[baseline_mask]))

# If no points in the 5-year window, use the closest point to the reference year.
else:
    baseline_idx = int(np.argmin(np.abs(t_smooth - custom_ref_year)))
    year_average_end_obs = float(y_smooth[baseline_idx])


# Construct projection years based on segment lengths.
#  This allows for non-uniform segments if the last one is shorter.
n_segments = plt_dy.shape[1]
# By default, segments are 5 years, but the last segment can be shorter 
# if dt is not divisible by 5.
seg_durations = np.full(n_segments, float(preview_segment_years), dtype=float)
remainder = preview_dt % preview_segment_years
if remainder > 0:
    seg_durations[-1] = float(remainder)

# The projection years are the cumulative sum of segment durations
#  added to the reference year.
projection_years = np.concatenate(
    ([float(custom_ref_year)], float(custom_ref_year) + np.cumsum(seg_durations))
)

# Construct cumulative change for each realization at each projection year.
cum_dy = np.cumsum(plt_dy, axis=1)
proj_shoreline_all = year_average_end_obs + np.concatenate(
    [np.zeros((plt_dy.shape[0], 1)), cum_dy],
    axis=1,
)

plt.plot(proj_shoreline_all)
proj_shoreline_mean = np.mean(proj_shoreline_all, axis=0)
proj_shoreline_q05 = np.quantile(proj_shoreline_all, 0.05, axis=0)
proj_shoreline_q95 = np.quantile(proj_shoreline_all, 0.95, axis=0)

# Plot projections from Monte Carlo simulations
plt.figure(figsize=(10, 5))
plt.plot(t_smooth, y_smooth, "r-", label="LOESS smoothed observed")
plt.plot(projection_years, proj_shoreline_mean, "b-", label="Mean projected change")
plt.fill_between(
    projection_years,
    proj_shoreline_q05,
    proj_shoreline_q95,
    color="blue",
    alpha=0.3,
    label="90% CI"
)
plt.xlabel("Year")
plt.ylabel("Shoreline position (m)")
plt.title(f"{site_id} {transect_id} – Projected shoreline change with uncertainty")
plt.legend()
plt.tight_layout()
plt.show()  







#%%
#Loop ends
dy_df = (
        pd.DataFrame(dy_rows)
        .sort_values(["site_id", "transect_id"])
        .reset_index(drop=True)
    )

print(dy_df.head())
meta_df = pd.DataFrame(meta_rows).sort_values(["site_id", "transect_id"]).reset_index(drop=True)
print(meta_df.head())

# %% Quick plot

dy_df["dy_ci90_len_m"] = dy_df["dy_p95_m"] - dy_df["dy_p05_m"]

plt.figure()
plt.hist(dy_df["dy_ci90_len_m"].dropna(), bins=40)
plt.xlabel("CI length (dy_p95 - dy_p05) [m]")
plt.ylabel("Count")
plt.title("Distribution of dy 5–95% CI length (all transects)")
plt.show()
# %% Plot by site (median CI length per site)

plt.figure(figsize=(12, 5))
labels = dy_df["site_id"].unique()

for i, (site_id, g) in enumerate(dy_df.groupby("site_id"), start=1):
    y = g["dy_ci90_len_m"].values
    x = np.full_like(y, i, dtype=float)
    plt.plot(x, y, "o", alpha=0.4)

plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
plt.ylabel("dy 5–95% CI length [m]")
plt.title("Transect-level CI length spread per site")
plt.tight_layout()
plt.show()
# %%

# CI length (if not already computed)
dy_df["dy_ci90_len_m"] = dy_df["dy_p95_m"] - dy_df["dy_p05_m"]

# Group CI lengths by site
groups = [
    g["dy_ci90_len_m"].values
    for _, g in dy_df.groupby("site_id")
]

labels = dy_df["site_id"].unique()

plt.figure(figsize=(12, 5))
plt.boxplot(groups, showfliers=True)
for i, vals in enumerate(groups, start=1):
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    x = np.full(len(vals), i, dtype=float) + jitter
    plt.plot(x, vals, "o", alpha=0.6, color="tab:blue", markersize=4)
plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
plt.ylim(0, 100)
plt.ylabel("dy 5–95% CI length [m]")
plt.title("Spread of shoreline-change uncertainty per site")

# # --- reference lines ---
# for y, txt in [(100, "100 m"), (200, "200 m"), (500, "500 m")]:
#     plt.axhline(
#         y,
#         color="red",
#         linestyle="--",
#         linewidth=1
#     )
#     plt.text(
#         0.5, y,
#         txt,
#         color="red",
#         fontsize=9,
#         va="bottom"
#     )

plt.tight_layout()
plt.show()

# %%
# Plot dy_median_m by site (boxplot)
plt.figure(figsize=(12, 5))
groups_median = [
    g["dy_median_m"].values
    for _, g in dy_df.groupby("site_id")
]

labels = dy_df["site_id"].unique()

plt.boxplot(groups_median, showfliers=True)
for i, vals in enumerate(groups_median, start=1):
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    x = np.full(len(vals), i, dtype=float) + jitter
    plt.plot(x, vals, "o", alpha=0.6, color="tab:blue", markersize=4)
plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
plt.ylabel("Median shoreline change (m)")
plt.title("Median shoreline change per site")
# plt.ylim(-0, 20)
plt.tight_layout()
plt.show()


# %%
