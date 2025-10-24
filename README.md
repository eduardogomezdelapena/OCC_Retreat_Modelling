# 1. Shoreline change estimation

There are two categories of shoreline change scenarios in the dataset. The first accounts only for shoreline change due to sea-level rise, while the second includes both sea-level rise and the observed beach trend.

## 1.1 Bruun rule: shoreline response to sea level rise

We use the Bruun rule to account for shoreline change $\Delta y_{SLR}$, which is computed per site  following:

$$
\Delta y_{SLR} = \frac{c}{\tan \beta} \Delta  S
$$

where $\tan \beta$ is the active profile (depth of closure to dune crest) slope, the constant $c$ is set to 1, and $\Delta  S$ corresponds to sea level rise. 

### Assumptions of the Bruun rule: 

- The beach profile is in cross-shore equilibrium
- The active beach profile starts in the depth of closure and ends in the dune crest or berm
- The beach system is in sediment balance 
- Negligible alongshore sediment transport
- Lack of accretionary component to SLR (i.e. only erosive response is possible)

## 1.2 Accounting for historic beach trend

To account for the shoreline's historic trend in addition to sea-level rise at each site, the shoreline change ($\Delta y_{SLR + Trend}$) is calculated as:

$$
\Delta y_{SLR + Trend} = \frac{c}{\tan \beta} \Delta  S + T_{satellite}
$$

where the historic beach trend $T_{satellite}$ is derived by fitting a linear regression to satellite observations over approximately the past 20 years at each location.

# 2. Data sources and processing

## 2.1 SLR scenarios ($\Delta S$)

[SLR scenarios](https://searise.nz/maps/) are sourced from the NZ SeaRise: Te Tai Pari O Aotearoa programme, which provides location-specific sea-level projections every 2 km along the coast of Aotearoa New Zealand, along with an open-access [data repository](https://zenodo.org/records/11398538). 

The SLR dataset has 3 percentiles for each sea level rise projection: 

- 17th percentile (lower bound) — represents a likely lower estimate of SLR, meaning there’s a 17% chance the actual SLR will be lower than this value.
- 50th percentile (median) — the central or “best estimate” of SLR, where half the modelled outcomes are higher and half are lower.
- 83rd percentile (upper bound) — represents a likely upper estimate of SLR, meaning there’s a 17% chance the actual SLR will exceed this value.

This convention (17th–50th–83rd) expresses the likely range (≈66% confidence interval) of SLR outcomes under a given emissions scenario.

### Reference year and adjustments

The reference year is 2025 for both the sea-level rise and shoreline change projections.
The reference shoreline position is calculated by averaging positions extracted from satellite images from 2024–2025.

The SLR projections from the NZ SeaRise programme begin in 2005, meaning sea-level rise is set to zero in that year. To align the projections with the reference year 2025, a correction factor is applied. This factor is calculated separately for each scenario, quartile, and location in the SLR dataset.

## 2.2 Beach slope ($\tan \beta$) & Beach historic trend ($T_{satellite}$)

[CoastSat repository](https://uoa-eresearch.github.io/CoastSat/) for beach historic trends ($T_{satellite}$) and beachface slopes ($\tan \beta _{beachface}$) **

** To account for the beachface being steeper than the active profile slope, an adjustment is applied when using the Bruun rule. Specifically, the beachface slope is multiplied by a factor:  

$$
\tan \beta = \tan \beta _{beachface} * b_{adjust}
$$

where $b_{adjust}$ has a value of 0.5, which is in line with some LiDAR measurements.

### 2.2.1 Processing of satellite data

To improve the spatial continuity of the satellite-derived beach slopes ($\tan \beta$), missing values were filled using each beach’s slope mean value, and the data were subsequently smoothed with a [low-pass signal filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html). Small random perturbations were then applied to avoid stretches of coastline having identical slope values.

To improve the spatial continuity and ensure smooth longshore variability within sites for the satellite-derived trends ($T_{satellite}$), a [spline smoother](https://github.com/SanParraguez/smoothn) was applied to each beach's historical trend spatial series. 


## 3. Uncertainty bands

The uncertainty bands in the shoreline change projections correspond to the results obtained under each percentile of the sea-level rise projections when applying the Bruun rule. In this sense, for each site, it is applied as follows:

$$
\Delta y_{Lower Bound} = \frac{c}{\tan \beta} \Delta  S _{Q17}
$$

$$
\Delta y_{Median} = \frac{c}{\tan \beta} \Delta  S _{Q50}
$$

$$
\Delta y_{UpperBound} = \frac{c}{\tan \beta} \Delta  S _{Q83}
$$

where $y_{Lower Bound},y_{Median},y_{UpperBound}$ correspond to the shoreline change given by its corresponding sea level rise percentile  $S _{Q17},S _{Q50},S _{Q83} $.



