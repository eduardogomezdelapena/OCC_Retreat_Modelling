#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for satellite trend bootsrapping, using Orewa as study case"""
#%%
#0138 is Orewa, download the smoothed, tidally corrected time series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

# Declare site t ID
site_id = "nzd0161"
transect_id = site_id + "-0187"

# URL to the raw CSV file on GitHub
url = (
    "https://raw.githubusercontent.com/UoA-eResearch/CoastSat/main/data/"
    f"{site_id}/transect_time_series_tidally_corrected_smoothed.csv"
)
# Load the CSV into a pandas DataFrame
df = pd.read_csv(url, header=0)

# Quick look at the data
print(df.columns)

# %% Reproducing linear fit as displayed in e-research coastsat dashboard

# Ensure datetime and pick site
df["dates"] = pd.to_datetime(df["dates"])



#Conver to decimal years (for linear fit)
t = df["dates"]
y = df[transect_id]

t_years = (
    t.dt.year
    + (t.dt.dayofyear - 1) / 365.25
)

# Remove NaNs
mask = np.isfinite(t_years) & np.isfinite(y)

# Linear regression: y = a*t + b
coef = np.polyfit(t_years[mask], y[mask], 1)
slope, intercept = coef
y_fit = slope * t_years + intercept
print(f"Linear trend for {transect_id}:")
print(f"Trend slope: {slope:.3f} m/year")
print(f"  Intercept = {intercept:.2f} m")

# ------------------------
# Plot linear fit
# ------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, y, ".", alpha=0.6, label="Observed")
plt.plot(t, y_fit, "-", linewidth=2, label="Linear trend")

plt.xlabel("Time")
plt.ylabel("Shoreline position (m)")
plt.title(f"{transect_id} shoreline time series")
plt.legend()
plt.tight_layout()
plt.show()

# %% Bootstrap the trend, function DEFINITION

def block_bootstrap_slopes(t, y, block_size, n_boot,
                            random_state = None   
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

    n = len(y)
    slopes = np.zeros(n_boot)

    for i in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n - block_size + 1)
            idx.extend(range(start, start + block_size))

        idx = np.array(idx[:n])  # trim to length n
        coef = np.polyfit(t[idx], y[idx], 1) # Calculate linear fit
        slopes[i] = coef[0]

    return slopes

#%% Bootstrap function APPLIED

#Apply NaN removal mask
t_clean = t_years[mask].values
y_clean = y[mask].values

boot_slopes = block_bootstrap_slopes(
    t_clean,
    y_clean,
    block_size= 24*5 , #Every 5 years, fortnightly data
    n_boot=1000
)

# Confidence intervals
ci_5, ci_50, ci_95 = np.percentile(boot_slopes, [5, 50, 95])

# Probability trend is positive
p_positive = np.mean(boot_slopes > 0)

print("Bootstrap trend uncertainty:")
print(f"Median slope: {ci_50:.3f} m/yr")
print(f"5–95% CI: [{ci_5:.3f}, {ci_95:.3f}] m/yr")
print(f"P(slope > 0) = {p_positive:.2f}")

# --------------------------------------------------------------
# Plot Bootstrapped trend uncertainty
# --------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(boot_slopes, bins=40, density=True, alpha=0.7)
plt.axvline(slope, color="k", linestyle="--", label="OLS slope") #Ordinary Least Squares Slope
plt.axvline(ci_5, color="r", linestyle=":")
plt.axvline(ci_50, color="k", linestyle=":", label = "Median slope")
plt.axvline(ci_95, color="r", linestyle=":", label="5–95% CI")
plt.xlabel("Trend slope (m/year)")
plt.ylabel("Density")
plt.title(f"{transect_id} – bootstrapped trend uncertainty")
plt.legend()
plt.tight_layout()
plt.show()

# %% Save routine

from pathlib import Path
import numpy as np

out_dir = Path("bootstrap_trend_distributions")
out_dir.mkdir(parents=True, exist_ok=True)

# after you compute boot_slopes ...
np.savez_compressed(
    out_dir / f"{site_id}__{transect_id}__boot_slopes.npz",
    boot_slopes=boot_slopes,
    site_id=site_id,
    transect_id=transect_id,
    block_size=24*5,
    n_boot=1000,
    ols_slope=slope,
    t_start=float(np.nanmin(t_clean)),
    t_end=float(np.nanmax(t_clean)),
)
#%% CSV saving
# import pandas as pd
# from pathlib import Path

# out_dir = Path("bootstrap_trend_distributions")
# out_dir.mkdir(parents=True, exist_ok=True)

# pd.DataFrame({
#     "site_id": site_id,
#     "transect_id": transect_id,
#     "boot_slope_m_per_yr": boot_slopes
# }).to_csv(out_dir / f"{site_id}__{transect_id}__boot_slopes.csv", index=False)

# %%
#Monte Carlo simulations. Function DEFINITION

def mc_shoreline_change(
    c, tan_beta, delta_S,
    r_samples,
    dt=75,                      # 2025 as baseline year, projections to 2100
    n=200_000,
    dist="normal",              # "normal" or "triangular"
    p_low=0.5,  p_high=1.5,     # persistance factor range
    random_state = None         # Seeding randomness
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

#%%  #Monte Carlo simulations. Function APPLIED

data = np.load(out_dir / f"{site_id}__{transect_id}__boot_slopes.npz", allow_pickle=True)
boot_slopes_test = data["boot_slopes"]


dy, summ = mc_shoreline_change(
    c=1.0,
    tan_beta=0.0075,
    delta_S=0.55,      # meters of SLR term in 2100 (SSP2-4.5)
    r_samples = boot_slopes_test,
    n=200_000,
    p_low=0.5,
    p_high=1.5
)

summ

# --------------------------------------------------------------
# Plot Monte Carlo shoreline projections
# --------------------------------------------------------------

plt.figure()
plt.hist(dy, bins=100, density=True)
plt.axvline(np.percentile(dy, 5))
plt.axvline(np.percentile(dy, 50))
plt.axvline(np.percentile(dy, 95))
plt.xlabel("Δy (m)")
plt.ylabel("Probability density")
plt.title("Monte Carlo shoreline change projection")
plt.show()

# --------------------------------------------------------------
# Plot Monte Carlo BOXPLOT shoreline projections
# ---------
# Statistics
median = np.median(dy)
q1 = np.quantile(dy, 0.25)
q3 = np.quantile(dy, 0.75)

plt.figure()
plt.boxplot(dy, vert=True, whis=[5, 95], showfliers=False)
plt.ylabel("Δy (m)")
plt.title("Monte Carlo distribution of shoreline change Δy, Orewa")

# Annotations (placed to the right)
x_text = 1.1

plt.text(x_text, median,
         f"Median = {median:.2f} m",
         va="center")

plt.text(x_text, q1,
         f"Q1 = {q1:.2f} m",
         va="center")

plt.text(x_text, q3,
         f"Q3 = {q3:.2f} m",
         va="center")


plt.show()
# %%
