#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for satellite trend bootsrapping, using Orewa as study case"""
#%%
#0138 is Orewa, download the smoothed, tidally corrected time series
import pandas as pd
import numpy as np
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

print(f"Linear trend for {transect_id}:")
print(f"Trend slope: {slope:.3f} m/year")
print(f"  Intercept = {intercept:.2f} m")
#%%

