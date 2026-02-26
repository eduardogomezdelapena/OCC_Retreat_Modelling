#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for satellite trend bootsrapping, using Orewa as study case"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    dt=75,                      # 2025 as baseline year, projections to 2100
    n=200_000,
):
    """
    Monte Carlo propagation for shoreline change using an
    empirical (bootstrap) distribution of shoreline trend rates:

    Δy = (c/tanβ) * ΔS + (p * r_sat * dt)

    Parameters:
    - c, tan_beta, delta_S: deterministic terms
    - p is persistance factor
    - r_sat are rates (e.g., m/yr).
    - dt in years.

    Arrays:
    ----------
    r_samples : array-like
        Empirical bootstrap samples of shoreline trend rate r_sat (m/yr).
        Sampling is performed with replacement.

    """

    rng = np.random.default_rng(random_state)

    # --- deterministic component ---
    base = (c / tan_beta) * delta_S

    # --- empirical sampling of trend rate ---
    r_samples = np.asarray(r_samples)
    r_samples = r_samples[np.isfinite(r_samples)]
    if r_samples.size == 0:
        raise ValueError("r_samples is empty after removing non-finite values.")

    r = rng.choice(r_samples, size=n, replace=True)

    # Sample persistence factor p
    p = rng.uniform(low=p_low, high=p_high, size=n)

    # Propagate
    dy = base + (p * r * dt)

    summary = {
        "dt_years": dt,
        "base_term_m": float(base),

        "p_mean": float(np.mean(p)),
        "p_p05": float(np.quantile(p, 0.05)),
        "p_p95": float(np.quantile(p, 0.95)),

        "trend_rate_mean": float(np.mean(r)),
        "trend_rate_p05": float(np.quantile(r, 0.05)),
        "trend_rate_p95": float(np.quantile(r, 0.95)),

        "dy_median_m": float(np.median(dy)),
        "dy_p05_m": float(np.quantile(dy, 0.05)),
        "dy_p95_m": float(np.quantile(dy, 0.95)),
    }
    return dy, summary

# %% Main loop: load data, apply filters, bootstrap, Monte Carlo

dy_rows = []
meta_rows = []

from pathlib import Path
out_dir = Path("original_plots_ts")

# nzd_sites_trial = nzd_sites[0:11]
nzd_sites_trial = ["nzd0010"]
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
            min_consecutive=5,
            min_obs_per_year=1,
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
        #Apply bootstrap and Monte Carlo functions

        boot_slopes = block_bootstrap_slopes(
            t_clean, y_clean,
            block_years= 3.0,
            n_boot=1000,
            random_state=seed
        )

        dy, summ = mc_shoreline_change(
            c=1.0,
            tan_beta=0.0075,
            delta_S=0.55,      # meters of SLR term in 2100 (SSP2-4.5)
            r_samples = boot_slopes,
            p_low=0.5,  # persistance factor range
            p_high=1.5,
            random_state= seed,
        )

        # Store results
        dy_rows.append({
                "site_id": site_id,
                "transect_id": transect_id,
                "n_obs": int(y_clean.size),
                "n_boot": 1000,
                "n_mc": 200_000,
                "dt_years": float(summ["dt_years"]),
                "base_term_m": float(summ["base_term_m"]),
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

plt.ylim(0,650)
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(6, 4))
plt.hist(boot_slopes, bins=40, density=True, alpha=0.7)

plt.xlabel("Trend slope (m/year)")
plt.ylabel("Density")
plt.title(f"{transect_id} – bootstrapped trend uncertainty")
plt.legend()
plt.tight_layout()
plt.show()
# %%
