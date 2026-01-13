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

# URL to the raw CSV file on GitHub
url = (
    "https://raw.githubusercontent.com/UoA-eResearch/CoastSat/main/data/"
    f"{site_id}/transect_time_series_tidally_corrected_smoothed.csv"
)
# Load the CSV into a pandas DataFrame
df = pd.read_csv(url, header=0)

# Quick look at the data
print(df.columns)

# %%
# Ensure datetime
df["dates"] = pd.to_datetime(df["dates"])

transect_id = site_id + "-0187"

#%%
#Conver to decimal years (for linear fit)
t = df["dates"]
y = df[transect_id]

t_years = (
    t.dt.year
    + (t.dt.dayofyear - 1) / 365.25
)

#%%
# Remove NaNs
mask = np.isfinite(t_years) & np.isfinite(y)

coef = np.polyfit(t_years[mask], y[mask], 1)

# Linear regression: y = a*t + b

slope, intercept = coef
y_fit = slope * t_years + intercept
print(f"Linear trend for {transect_id}:")
print(f"Trend slope: {slope:.3f} m/year")
print(f"  Intercept = {intercept:.2f} m")
#%%
# ------------------------
# Plot
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

# %% Bootstrap

def block_bootstrap_slopes(t, y, block_size, n_boot):
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
    n = len(y)
    slopes = np.zeros(n_boot)

    for i in range(n_boot):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, n - block_size + 1)
            idx.extend(range(start, start + block_size))

        idx = np.array(idx[:n])  # trim to length n
        coef = np.polyfit(t[idx], y[idx], 1)
        slopes[i] = coef[0]

    return slopes

#%%

t_clean = t_years[mask].values
y_clean = y[mask].values

boot_slopes = block_bootstrap_slopes(
    t_clean,
    y_clean,
    block_size= 24*3 , #Every 3 years
    n_boot=1000
)
#%%
# Confidence intervals
ci_5, ci_50, ci_95 = np.percentile(boot_slopes, [5, 50, 95])

# Probability trend is positive
p_positive = np.mean(boot_slopes < 0)

print("Bootstrap trend uncertainty:")
print(f"Median slope: {ci_50:.3f} m/yr")
print(f"5–95% CI: [{ci_5:.3f}, {ci_95:.3f}] m/yr")
print(f"P(slope > 0) = {p_positive:.2f}")
# %%
plt.figure(figsize=(6, 4))
plt.hist(boot_slopes, bins=40, density=True, alpha=0.7)
plt.axvline(slope, color="k", linestyle="--", label="OLS slope")
plt.axvline(ci_5, color="r", linestyle=":")
plt.axvline(ci_95, color="r", linestyle=":", label="5–95% CI")
plt.xlabel("Trend slope (m/year)")
plt.ylabel("Density")
plt.title(f"{transect_id} – bootstrapped trend uncertainty")
plt.legend()
plt.tight_layout()
plt.show()
# %%
