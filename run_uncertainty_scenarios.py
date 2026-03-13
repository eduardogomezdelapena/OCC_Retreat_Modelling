  """ Script that takes all_merged from utils.py and slopes from 
  trend_uncertainty.py and runs a MC simulation with one SSP scenario"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def load_all_merged(path: str = ALL_MERGED_FP) -> gpd.GeoDataFrame:
    """Read previously saved *all_merged* from disk."""
    return gpd.read_file(path, layer="all")
