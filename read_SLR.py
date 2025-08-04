#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:08:04 2025

@author: eegp
"""

import pandas as pd

#First read lat lon coordinates, only in VLM file, version 3 of zenodo rep

url_latlon= "https://zenodo.org/records/11398538/files/NZ_VLM_final_May24.csv"
df_latlon= pd.read_csv(url_latlon)
print(df_latlon.head())

df_latlon.iloc[0]


#%% NZSea rise data, with VLM
url= "https://zenodo.org/records/14722058/files/NZ_Searise_VLM-2005.csv"

df = pd.read_csv(url)

print(df.head())

#extract years without repeating
year= set(df.year)

df.iloc[0]

# #%%
# #NZSea rise data, no VLM
# url= "https://zenodo.org/records/14722058/files/NZ_Searise_noVLM-2005.csv"

# df = pd.read_csv(url)

# print(df.head())

# #extract years without repeating
# year= set(df.year)

# df.iloc[0]





