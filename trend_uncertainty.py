#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for satellite trend bootsrapping, using Orewa as study case"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist
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

#%%  Filter to longest run of consecutive years with min obs. Function DEFINITION
def filter_to_longest_consecutive_year_run(
    t, y, years,
    min_consecutive, # min no. of consecutive years required to keep any data
    min_obs_per_year, # min no. of observations per year to consider that year eligible
    site_id,
    transect_id,
):
    # Convert to numpy arrays for easier processing
    t, y, years = map(np.asarray, (t, y, years))

    # Initialize metadata
    meta = {
        "site_id": site_id,
        "transect_id": transect_id,
        "status": None,
        "kept_years": [],
        "removed_years": [],
    }

    # Basic checks
    if y.size == 0:
        meta["status"] = "empty"
        return t[:0], y[:0], meta

    # Count observations per year and find eligible years
    yr, ct = np.unique(years.astype(int), return_counts=True)
    eligible = np.sort(yr[ct >= min_obs_per_year])

    # If no years meet the minimum obs/year, return empty with metadata
    if eligible.size == 0:
        meta["status"] = "no_years_meet_min_obs"
        meta["removed_years"] = yr.tolist()
        return t[:0], y[:0], meta

    # Find longest run of consecutive eligible years
    blocks = np.split(eligible, np.where(np.diff(eligible) != 1)[0] + 1)
    # Select the longest block (if multiple have same length, the first is chosen)
    best = max(blocks, key=len)

    # If the longest run is shorter than min_consecutive, return empty with metadata
    if len(best) < min_consecutive:
        # Report the reason for skipping in metadata
        meta["status"] = "insufficient_consecutive_years"
        meta["removed_years"] = yr.tolist()
        return t[:0], y[:0], meta
    
    # Otherwise, keep only observations from the best run of consecutive years  
    keep = np.arange(best[0], best[-1] + 1, dtype=int)
    # Mask to keep only observations from the selected years
    mask = np.isin(years, keep)

    # Metadata reporting
    meta["status"] = "ok"
    meta["kept_years"] = keep.tolist()
    meta["removed_years"] = sorted(set(yr.tolist()) - set(keep.tolist()))

    # Return filtered (decimal) time, shoreline, and metadata 
    return t[mask], y[mask], meta

# %% Bootstrap the trend, function DEFINITION
def block_bootstrap_slopes(
    t, y,
    block_years,
    n_boot,
    random_state,
):
    """
    Block bootstrap for linear trend slopes.

    Parameters
    ----------
    t : array
        Time in decimal years
    y : array
        Shoreline position
    block_size : int
        Number of consecutive samples per block
    n_boot : int
        Number of bootstrap realizations

    Returns
    -------
    slopes : array
        Bootstrapped slope estimates
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

    # Find block start indices and corresponding end indices
    ends = np.searchsorted(t, t + block_years, side="left")

    # Valid starts must yield at least 2 points in the window to fit a slope
    valid_starts = np.where((ends - np.arange(n)) >= 2)[0]
    if valid_starts.size == 0:
        raise ValueError(
            f"No 5-year windows contain >=2 observations. "
            f"Try shorter block_years or check data density."
        )
    
    slopes = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.choice(valid_starts)
            end = ends[start]          # exclusive
            idx.extend(range(start, end))  # ALL obs in [t[start], t[start]+5)

        idx = np.asarray(idx[:n])
        slopes[i] = np.polyfit(t[idx], y[idx], 1)[0]

    return slopes

# %% Monte Carlo simulations. Function DEFINITION
def mc_shoreline_change(
    c, tan_beta, delta_S,
    r_samples,
    p_low, p_high,
    random_state,   
    delta_S_q17,
    delta_S_q50,
    delta_S_q83,
    dt=75,                      # 2025 as baseline year, projections to 2100
    n=200_000,
    return_delta_s_samples=False,
):
    """
    Monte Carlo propagation for shoreline change using an
    empirical (bootstrap) distribution of shoreline trend rates.

    Δy = (c/tanβ) * ΔS + (p * r_sat * dt)

    Parameters:
    - c: adjustment factor (fixed)
    - tan_beta: nominal beach slope (m/m); uncertainty added as ±20%
        - delta_S: sea level rise (m), used directly if quantiles are not supplied
        - delta_S_q17, delta_S_q50, delta_S_q83: optional shifted SLR quantiles
            used to fit a Gaussian SLR distribution for sampling
    - p: persistence factor (sampled uniformly between p_low and p_high)
    - r_sat: empirical trend rates (sampled from r_samples)
    - dt: time horizon (years)

    Uncertainty sources:
    - Trend rate r_sat (from bootstrap)
    - Persistence factor p (uniform)
    - Beach slope tan_beta (±20% uniform around nominal)
    - SLR delta_S (deterministic or Gaussian from 17/50/83 quantiles)

    """

    rng = np.random.default_rng(random_state)

    # --- empirical sampling of trend rate ---
    r_samples = np.asarray(r_samples)
    r_samples = r_samples[np.isfinite(r_samples)]
    if r_samples.size == 0:
        raise ValueError("r_samples is empty after removing non-finite values.")

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

    bases = (c / sampled_tan_beta) * sampled_delta_s
    sampled_r = rng.choice(r_samples, size=n)
    sampled_p = rng.uniform(low=p_low, high=p_high, size=n)
    trend_terms = sampled_p * sampled_r * dt
    dy = bases + trend_terms

    # Empirical variance decomposition: Var(A+B)=Var(A)+Var(B)+2Cov(A,B).
    base_var = float(np.var(bases, ddof=1))
    trend_var = float(np.var(trend_terms, ddof=1))
    cov_bt = float(np.cov(bases, trend_terms, ddof=1)[0, 1])
    cross_var = float(2.0 * cov_bt)
    dy_var = float(np.var(dy, ddof=1))

    if dy_var > 0:
        base_var_frac_pct = float(100.0 * base_var / dy_var)
        trend_var_frac_pct = float(100.0 * trend_var / dy_var)
        cross_var_frac_pct = float(100.0 * cross_var / dy_var)
    else:
        base_var_frac_pct = np.nan
        trend_var_frac_pct = np.nan
        cross_var_frac_pct = np.nan

    summary = {
        "dt_years": dt,
        "base_term_m": float(np.mean(bases)),  # mean of sampled bases

        "p_mean": float(np.mean(sampled_p)),
        "p_p05": p_low,  # since uniform
        "p_p95": p_high,

        "slr_sampling": "gaussian_q17_q50_q83" if use_slr_gaussian else "deterministic",
        "delta_S_mean_m": float(np.mean(sampled_delta_s)),
        "delta_S_p05_m": float(np.quantile(sampled_delta_s, 0.05)),
        "delta_S_p95_m": float(np.quantile(sampled_delta_s, 0.95)),
        "delta_S_mu_m": float(slr_mu),
        "delta_S_sigma_m": float(slr_sigma),

        "trend_rate_mean": float(np.mean(r_samples)),  # population mean
        "trend_rate_p05": float(np.quantile(r_samples, 0.05)),
        "trend_rate_p95": float(np.quantile(r_samples, 0.95)),

        "dy_median_m": float(np.median(dy)),
        "dy_p05_m": float(np.quantile(dy, 0.05)),
        "dy_p95_m": float(np.quantile(dy, 0.95)),

        "base_var_m2": base_var,
        "trend_var_m2": trend_var,
        "cross_var_m2": cross_var,
        "dy_var_m2": dy_var,
        "base_sd_m": float(np.sqrt(max(base_var, 0.0))),
        "trend_sd_m": float(np.sqrt(max(trend_var, 0.0))),
        "dy_sd_m": float(np.sqrt(max(dy_var, 0.0))),
        "base_var_frac_pct": base_var_frac_pct,
        "trend_var_frac_pct": trend_var_frac_pct,
        "cross_var_frac_pct": cross_var_frac_pct,
    }

    if return_delta_s_samples:
        return dy, summary, sampled_delta_s

    return dy, summary


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


def plot_variance_decomposition(
    summary,
    site_id,
    transect_id,
    ssp,
    scenario,
    year,
    out_fp,
):
    """Save per-transect variance decomposition bars for dy components."""
    base_var = float(summary.get("base_var_m2", np.nan))
    trend_var = float(summary.get("trend_var_m2", np.nan))
    cross_var = float(summary.get("cross_var_m2", np.nan))
    total_var = float(summary.get("dy_var_m2", np.nan))

    labels = ["Base", "Trend", "2*Cov"]
    values = np.array([base_var, trend_var, cross_var], dtype=float)
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1.0)

    if np.isfinite(total_var):
        ax.axhline(
            total_var,
            color="crimson",
            linestyle="--",
            linewidth=1.6,
            label=f"Total var = {total_var:.2f} m^2",
        )

    pct_vals = [
        float(summary.get("base_var_frac_pct", np.nan)),
        float(summary.get("trend_var_frac_pct", np.nan)),
        float(summary.get("cross_var_frac_pct", np.nan)),
    ]

    finite_abs = np.abs(values[np.isfinite(values)])
    scale = float(np.max(finite_abs)) if finite_abs.size else 1.0
    if scale == 0.0:
        scale = 1.0
    y_off = 0.04 * scale

    for i, (val, pct) in enumerate(zip(values, pct_vals)):
        if not np.isfinite(val):
            continue
        y_text = val + y_off if val >= 0 else val - y_off
        va = "bottom" if val >= 0 else "top"
        pct_txt = f"{pct:.1f}%" if np.isfinite(pct) else "nan"
        ax.text(i, y_text, pct_txt, ha="center", va=va, fontsize=9)

    ax.set_ylabel("Variance contribution [m^2]")
    ax.set_title(
        f"{site_id} {transect_id} variance decomposition ({ssp}-{scenario}, year={year})"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=180)
    plt.close(fig)


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

nzd_sites_trial = nzd_sites[0:5]
# nzd_sites_trial = ["nzd0003"]
# nzd_sites_trial = ["nzd0161"]

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
    # transect_cols=['nzd0161-0187']

    #Transect loop 
    for transect_id in transect_cols:

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
        years_clean = df.loc[mask, "dates"].dt.year.values  

        # Filter to consecutive years (min 5) with min observations/per year
        t_clean, y_clean, meta = filter_to_longest_consecutive_year_run(
            t_clean, y_clean, years_clean,
            min_consecutive=7,
            min_obs_per_year=3,
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

        ###############################
        # Apply bootstrap and Monte Carlo functions

        boot_slopes = block_bootstrap_slopes(
            t_clean, y_clean,
            block_years= 3.0,
            n_boot=1000,
            random_state=seed
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
            p_low=0.5,  # persistance factor range
            p_high=1.5,
            random_state= seed,
            delta_S_q17=delta_s_q17,
            delta_S_q50=delta_s_q50,
            delta_S_q83=delta_s_q83,
            return_delta_s_samples=debug_slr_histograms,
        )

        if debug_slr_histograms:
            dy, summ, sampled_delta_s = mc_result
            debug_hist_fp = site_dir / f"{site_id}_{transect_id}_deltaS_hist.png"
            plot_delta_s_debug_histogram(
                sampled_delta_s=sampled_delta_s,
                summary=summ,
                q17=delta_s_q17,
                q50=delta_s_q50,
                q83=delta_s_q83,
                site_id=site_id,
                transect_id=transect_id,
                ssp=ssp_target,
                scenario=scenario_target,
                year=target_year,
                out_fp=debug_hist_fp,
            )
        else:
            dy, summ = mc_result

        if debug_variance_plots:
            var_plot_fp = site_dir / f"{site_id}_{transect_id}_variance_decomp.png"
            plot_variance_decomposition(
                summary=summ,
                site_id=site_id,
                transect_id=transect_id,
                ssp=ssp_target,
                scenario=scenario_target,
                year=target_year,
                out_fp=var_plot_fp,
            )

        print(
            f"{site_id} {transect_id} var_decomp [%] -> "
            f"base={summ['base_var_frac_pct']:.1f}, "
            f"trend={summ['trend_var_frac_pct']:.1f}, "
            f"2cov={summ['cross_var_frac_pct']:.1f}"
        )

        # Store results
        dy_rows.append({
                "site_id": site_id,
                "transect_id": transect_id,
                "n_obs": int(y_clean.size),
                "n_boot": 1000,
                "n_mc": 200_000,
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
                "base_var_m2": float(summ["base_var_m2"]),
                "trend_var_m2": float(summ["trend_var_m2"]),
                "cross_var_m2": float(summ["cross_var_m2"]),
                "dy_var_m2": float(summ["dy_var_m2"]),
                "base_sd_m": float(summ["base_sd_m"]),
                "trend_sd_m": float(summ["trend_sd_m"]),
                "dy_sd_m": float(summ["dy_sd_m"]),
                "base_var_frac_pct": float(summ["base_var_frac_pct"]),
                "trend_var_frac_pct": float(summ["trend_var_frac_pct"]),
                "cross_var_frac_pct": float(summ["cross_var_frac_pct"]),
                "dy_p05_m": float(summ["dy_p05_m"]),
                "dy_median_m": float(summ["dy_median_m"]),
                "dy_p95_m": float(summ["dy_p95_m"]),
            })
        
        print(site_id, transect_id)

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
plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
plt.ylim(0, 1000)
plt.ylabel("dy 5–95% CI length [m]")
plt.title("Spread of shoreline-change uncertainty per site")

# --- reference lines ---
for y, txt in [(100, "100 m"), (200, "200 m"), (500, "500 m")]:
    plt.axhline(
        y,
        color="red",
        linestyle="--",
        linewidth=1
    )
    plt.text(
        0.5, y,
        txt,
        color="red",
        fontsize=9,
        va="bottom"
    )

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
plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
plt.ylabel("Median shoreline change (m)")
plt.title("Median shoreline change per site")
plt.ylim(-50, 100)
plt.tight_layout()
plt.show()


# %%
