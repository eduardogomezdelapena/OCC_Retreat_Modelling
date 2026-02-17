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
print(nzd_sites[:10])

seed = 42 

# %% Bootstrap the trend, function DEFINITION

def block_bootstrap_slopes(
    t, y,
    block_years: float = 5.0,
    n_boot: int = 1000,
    random_state=0,
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
    rng = np.random.default_rng(random_state)

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 observations total.")

    # Precompute end index for each start: end = first index with t >= t[start] + block_years
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

# %%
#Monte Carlo simulations. Function DEFINITION

def mc_shoreline_change(
    c, tan_beta, delta_S,
    r_samples,
    dt=75,                      # 2025 as baseline year, projections to 2100
    n=200_000,
    dist="normal",              # "normal" or "triangular"
    p_low=0.5,  p_high=1.5,     # persistance factor range
    random_state = seed         # Seeding randomness
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
#%%
def passes_seasonal_guardrail(
    dates,
    min_obs_per_season,
    max_failed_years,
    verbose: bool = False,
):
    """
    Check that each calendar year has at least `min_obs_per_season`
    observations in each season (NZ seasons).

    Parameters
    ----------
    dates : pandas Series (datetime64)
        Observation timestamps (NaNs already removed).
    min_obs_per_season : int
        Minimum number of observations required per season per year.
    max_failed_years : int
        Allow up to this many years to fail the rule.
        (0 = strict Option B)
    verbose : bool
        If True, prints diagnostics when failing.

    Returns
    -------
    bool
        True if transect passes the rule, False otherwise.
    """

    if len(dates) == 0:
        return False

    def nz_season_from_month(month: int) -> str:
        if month in (12, 1, 2):
            return "summer"
        if month in (3, 4, 5):
            return "autumn"
        if month in (6, 7, 8):
            return "winter"
        return "spring"

    years = dates.dt.year.values
    seasons = dates.dt.month.map(nz_season_from_month).values

    all_years = np.arange(years.min(), years.max() + 1)
    all_seasons = np.array(["summer", "autumn", "winter", "spring"], dtype=object)

    # Count observations per (year, season)
    ys = pd.DataFrame({"year": years, "season": seasons})
    counts = (
        ys.value_counts()
          .rename("n")
          .reset_index()
    )

    # Complete grid to include missing combinations
    grid = pd.MultiIndex.from_product(
        [all_years, all_seasons],
        names=["year", "season"]
    ).to_frame(index=False)

    counts_full = (
        grid.merge(counts, on=["year", "season"], how="left")
            .fillna({"n": 0})
    )

    # Identify failing year-season combinations
    counts_full["fail"] = counts_full["n"] < min_obs_per_season

    # Count failing years (if ANY season fails in that year)
    failed_years = (
        counts_full.groupby("year")["fail"]
        .any()
        .sum()
    )

    if failed_years > max_failed_years:
        if verbose:
            bad = counts_full[counts_full["fail"]]
            print(
                f"Seasonal guardrail failed: "
                f"{failed_years} failing years "
                f"(allowed {max_failed_years}). "
                f"Examples: {bad.head(6).to_dict(orient='records')}"
            )
        return False

    return True

# %% Reproducing linear fit as displayed in e-research coastsat dashboard

dy_rows = []

# nzd_sites_trial = nzd_sites[0:11]

nzd_sites_trial = ["nzd0161"]
#Try only 2 sites first
for site_id in nzd_sites_trial:

    # URL to the raw CSV file on GitHub
    url = (
        "https://raw.githubusercontent.com/UoA-eResearch/CoastSat/main/data/"
        f"{site_id}/transect_time_series_tidally_corrected_smoothed.csv"
    )
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(url, header=0)

    # Ensure datetime and pick site
    df["dates"] = pd.to_datetime(df["dates"])

    #Conver to decimal years (for linear fit)
    t = df["dates"]
    #y = df[transect_id]

    t_years = (
        t.dt.year
        + (t.dt.dayofyear - 1) / 365.25
    )

    transect_cols = [
        c for c in df.columns
        if c.startswith(site_id + "-")
    ]

    #Transect loop 
    for transect_id in transect_cols:

        y = df[transect_id]
        # Remove NaNs
        mask = np.isfinite(y)

        #% Bootstrap function APPLIED

        #Apply NaN removal mask
        t_clean = t_years[mask].values
        y_clean = y[mask].values

        dates_clean = df["dates"][mask]

        if not passes_seasonal_guardrail(
            dates_clean,
            min_obs_per_season=1,
            max_failed_years=10,   # strict Option B
            verbose=False
        ):
            print(f"Skipping {site_id} {transect_id}: seasonal guardrail")
            continue

        ###############################
        #Guardrails 
        block_years = 5.0

        boot_slopes = block_bootstrap_slopes(
            t_clean, y_clean,
            block_years=block_years,
            n_boot=1000,
            random_state=seed
        )

        dy, summ = mc_shoreline_change(
            c=1.0,
            tan_beta=0.0075,
            delta_S=0.55,      # meters of SLR term in 2100 (SSP2-4.5)
            r_samples = boot_slopes,
            n=200_000,
            p_low=0.5,
            p_high=1.5
        )

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
