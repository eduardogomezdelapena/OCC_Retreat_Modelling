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

[SLR scenarios](https://searise.nz/maps/) are sourced from the NZ SeaRise: Te Tai Pari O Aotearoa programme, which provides location-specific sea-level projections every 2 km along the coast of Aotearoa New Zealand, along with an open-access [data repository](https://zenodo.org/records/11398538). The SLR scenarios correspond to five Shared Socioeconomic Pathways (SSPs) developed by the Intergovernmental Panel on Climate Change (IPCC): SSP1–1.9, SSP1–2.6, SSP2–4.5, SSP3–7.0, and SSP5–8.5.

For each scenario, the SLR dataset has 3 percentiles: 

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
\Delta y_{Lower Bound} = \frac{c}{\tan \beta} \Delta  S _{P17}
$$

$$
\Delta y_{Median} = \frac{c}{\tan \beta} \Delta  S _{P50}
$$

$$
\Delta y_{UpperBound} = \frac{c}{\tan \beta} \Delta  S _{P83}
$$

where $y_{Lower Bound},y_{Median},y_{UpperBound}$ correspond to the shoreline change given by its corresponding sea level rise percentile  $S _{P17},S _{P50},S _{P83} $.




# 4. Runtime and resource usage of `trend_uncertainty.py` (all regions)

Measured on 2026-08-26 by running the script as committed (`ca78c29`, `max_tasks_per_child` trial) with the only change being `RUN_CONFIG["target_region_name"]` set to all 16 regional councils. Everything else was left at the committed values: `max_workers=16`, `n_mc_realizations_per_transect=2000`, `n_boot=1000`, baseline 2025 → target 2050, SSP5-8.5.

## 4.1 Test machine and software

| | |
|---|---|
| Machine | CeR VM, 32-core AMD EPYC-Milan (1 thread/core), 125 GB RAM, 2 GB swap, repo on a network-mounted 21 TB volume |
| Python | 3.12.3 (system), numpy 1.26.4 (OpenBLAS), pandas 2.3.0, geopandas 1.1.1, matplotlib 3.10.3, tqdm 4.67.1, loess 2.1.2 |
| Launch | `PYTHONPATH=. python3 trend_uncertainty.py` (non-interactive; stdout/stderr to a log file) |
| Monitoring | `psutil` sampler every 5 s: system CPU %, system memory, RSS/PSS/USS of the parent and every worker, load average (`htop` was used to cross-check) |

`reqs/OCC_environment.yml` (`clean_OCC`: Python 3.13.5, numpy 2.3.1, pandas 2.3.1) pins the same tqdm 4.67.1, so the tqdm behaviour described in 4.5 applies to that environment as well.

## 4.2 Workload

| | |
|---|---|
| Regions | 16 (all regional councils; "Area Outside Region" excluded) |
| Sites (unique `nzd*` IDs inside the region polygons) | **534** |
| Transects read from the CoastSat CSVs | 31,609 |
| Transects passing the "longest consecutive run ≥ 10 yr" filter (LOESS + bootstrap run) | 30,730 |
| Transects fully processed (Monte Carlo + projection CSV + projection plot) | **24,023** |
| Sites per region | Northland 105 · Auckland 45 · Waikato 61 · Bay of Plenty 22 · Gisborne 23 · Hawke's Bay 24 · Taranaki 7 · Manawatū-Whanganui 9 · Wellington 17 · West Coast 60 · Canterbury 43 · Otago 57 · Southland 28 · Tasman 18 · Nelson 2 · Marlborough 13 |

## 4.3 Runtime

| | |
|---|---|
| Wall-clock, whole run (16 workers) | **73 min 56 s** (10:21:52 → 11:35:48) |
| Sum of per-site processing times | 1,121.7 min (18.7 h of single-core-equivalent work) |
| Effective parallelism | 1,121.7 / 73.9 = 15.2 of 16 workers busy on average (load balance is good; the tail is one 19-minute site) |
| Per-site time | min 5 s · median 59 s · mean 126 s · p90 340 s · max 18.9 min (`nzd0451`; then `nzd0454` 15.9, `nzd0161` 12.9, `nzd0207` 12.6, `nzd0447` 12.6 min) |
| Per-transect time (fully processed) | ≈ 2.8 s of worker time on average (1,121.7 min / 24,023), i.e. ≈ 5.4 transects/s across the pool |
| Data download | negligible: one ≈ 60 KB CSV per site from GitHub raw (≈ 0.03 s) |

Site-time per region (sum of per-site times, i.e. what a *serial* run of that region alone would cost):
Northland 159 min · Auckland 59 · Waikato 64 · Bay of Plenty 81 · Gisborne 34 · Hawke's Bay 77 · Taranaki 10 · Manawatū-Whanganui 53 · Wellington 45 · West Coast 153 · Canterbury 191 · Otago 72 · Southland 71 · Tasman 15 · Nelson 3 · Marlborough 37.
Divide by the number of workers (≈ 15 effective) for the expected wall-clock time of a single-region run.

## 4.4 CPU and memory

| Metric | Value |
|---|---|
| Processes | 1 parent + 16 forked workers (17 Python processes for the whole run; workers were *not* recycled — see 4.5) |
| Threads per worker | 32 (OpenBLAS creates one thread per core in every worker; `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` are unset) → 16 × 32 = 512 compute threads on 32 cores |
| System CPU | mean 72 %, frequently 76–81 %, with each worker at 145–215 % CPU; 1-min load average ≈ 30 (peak 37.6) on 32 cores → the machine is oversubscribed by the BLAS threads, not by the 16 workers |
| Parent process | 1.33 GB RSS after loading the tables (`all_merged` alone is 1.12 M rows / 716 MB in memory; only the SSP5-8.5 rows are kept) |
| Each worker | 1.25–1.30 GB RSS, of which only **≈ 0.31 GB is private (USS)**; the remaining ≈ 0.95 GB is the parent's data inherited copy-on-write via `fork` |
| Whole process tree | summed RSS ≈ 21.3 GB (this is what adding up `htop`'s RES column gives) but real footprint (PSS) **5.9 GB at start → 6.8 GB at the end**, peak 7.0 GB |
| System memory used | 7.8 GB before the run (idle host) → **peak 15.7 GB** during the run, i.e. the run needs ≈ 8 GB of RAM on this machine; 119 GB stayed available throughout, swap unused |
| Memory growth over 74 min | +0.9 GB PSS in total (≈ +60 MB per worker, from copy-on-write pages of `all_merged` being touched by the per-transect boolean masks); no leak-like growth |

So on this machine the all-regions run does **not** approach a memory limit. The sum-of-RSS figure (≈ 21 GB, ≈ 3× the true footprint) is the number that `htop` shows per process and is the one that looks alarming.

## 4.5 Behaviour of the parallel section as committed (things to be aware of)

1. **`max_tasks_per_child=1` is silently not applied.** `tqdm.contrib.concurrent.process_map` (tqdm 4.67.1) only forwards `max_workers` and `chunksize` to `ProcessPoolExecutor`; every other keyword goes to the `tqdm` progress-bar constructor. `tqdm` raises `TqdmKeyError: "Unknown argument(s): {'max_tasks_per_child': 1}"`, but only *after* all 534 sites have already been submitted to the pool, and leaving the `with` block waits for them. Net effect: the full run executes with the default `fork` start method and 16 long-lived workers, no progress bar is ever drawn, per-site "Elapsed time" lines only appear when the workers exit (buffered stdout), and the script ends with a traceback and exit code 1.
2. Because `process_site` no longer returns its results and `create_log()` is never called, `outputs/site_processing.log` and `outputs/zero_spread_trend_sites.log` are not written, so the skip reasons are lost. (The fixed run in 4.7 restores them: of the 31,609 transects, 827 are skipped as `insufficient_recent_span_with_2024`, 52 as `no_segment_with_2024`, and **6,707 as `no matching tan_beta in coastsat_merged`** — i.e. 21 % of the transects in the CoastSat CSVs have no row in `preprocessing/transects_reindexed_Nickupdate.geojson`, which is a data-join gap worth looking at separately.)
3. `site_dir` is reassigned to `outputs/<site>` after the first exported transect, so from the second transect on, the "original time series" and "LOESS" PNGs are written into `outputs/<site>/` rather than `original_plots_ts/<site>/`.
4. If `max_tasks_per_child` *were* honoured — `ProcessPoolExecutor` switches to the `spawn` start method when it is set, and `spawn` is also the default on macOS/Windows — every worker would re-execute the whole module top level (GitHub API call, GeoJSON/CSV loads, ≈ 1.3 GB of tables) once per site, because the parallel call is not under `if __name__ == "__main__":`. That is the configuration in which memory really does grow with the number of sites, so please keep the `fork` start method (Linux) when changing the pool implementation.

## 4.6 Disk output of the all-regions run

| Files written | Count | Size |
|---|---|---|
| `*_original_timeseries.png` (dpi 300) | 31,609 | 7.5 GB |
| `*_loess_smoothed.png` (dpi 200) | 30,730 | 5.0 GB |
| `*_single_run_observed_projected.png` (dpi 300, 15 × 6.5 in) | 24,023 | 17.6 GB |
| `*_projection_results.csv` | 24,023 | 1.4 GB |
| **Total** | **110,385** | **31.4 GB** |

Roughly 85 % of the disk output is diagnostic PNGs; the CSVs the dashboard needs are 1.4 GB.

## 4.7 Same run with the parallel section fixed

To check what the issues in 4.5 cost, the run was repeated with four changes (nothing in the science/statistics was touched; the 24,023 output CSVs/plots are produced for the same transects):

1. one BLAS thread per worker (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1, set before `numpy` is imported);
2. `tqdm.contrib.concurrent.process_map(..., max_tasks_per_child=1)` replaced by `multiprocessing.get_context("fork").Pool(processes=n_workers, maxtasksperchild=1)` + `pool.imap_unordered` wrapped in `tqdm` — this actually recycles each worker after one site and draws the progress bar;
3. the `return {...}` of `process_site` restored and the pool call + `create_log(site_results)` placed under `if __name__ == "__main__":`;
4. inside `process_site`, `all_merged` is sliced once per site (`site_all_merged = all_merged.loc[all_merged["site_id"] == site_id]`) and the per-transect `mask_slr` is built on that ≈ 2,000-row slice instead of on the full 1.12 M-row table.

| | As committed | Fixed | Change |
|---|---|---|---|
| Wall-clock (16 workers) | 73 min 56 s | **54 min 21 s** | −27 % |
| Sum of per-site times | 1,121.7 min | 824.5 min | −27 % (median per-site ratio 0.72) |
| Per-site median / mean / p90 / max | 59 s / 126 s / 340 s / 18.9 min | 44 s / 93 s / 255 s / 14.0 min | |
| System CPU (mean) | 72 % with load average ≈ 30 | 48 % with load average ≈ 16 (= 16 of 32 cores, nothing else) | 8 cores freed *and* faster |
| Threads per worker | 32 | 1 | |
| Worker RSS (mean / max) | 1.25 / 1.30 GB | 1.39 / 1.57 GB (fresh fork per site; still ≈ 1 GB shared) | |
| Process-tree PSS (real footprint), start → end, peak | 5.9 → 6.8 GB, peak 7.0 GB | 4.6 → 4.4 GB, peak 5.9 GB | flat in both |
| System memory used, peak | 15.7 GB | 14.0 GB | |
| Progress bar / exit code / summary logs | none / 1 (`TqdmKeyError`) / not written | yes / 0 / `outputs/site_processing.log`, `outputs/zero_spread_trend_sites.log` written | |

Take-aways:

- **Memory is not the bottleneck on this machine in either configuration** (≈ 5–8 GB real footprint, no growth over 534 sites). A machine running out of memory with this script most likely (a) has far less RAM, (b) uses `max_workers=None` → `os.cpu_count()` workers on a many-core box, (c) runs with the `spawn` start method (see 4.5 item 4: every worker then loads its own ≈ 1.3 GB copy of the tables and, with `maxtasksperchild=1`, reloads them for every site), or (d) is being judged from the per-process RES column of `htop`, which triple-counts the copy-on-write pages. A rough budget for planning: **≈ 1.4 GB for the parent + ≈ 0.35 GB private per forked worker + ≈ 1–2 GB headroom for matplotlib rendering** — e.g. ≈ 8 GB for 16 workers, ≈ 4.5 GB for 8 workers.
- The BLAS oversubscription is the single biggest inefficiency: capping threads to 1 makes the run 27 % faster while using half the CPU.
- With the pool fixed the script terminates cleanly and the per-transect skip reasons are available again.

Minimal replacement for the parallel call (fork context; keeps the already-loaded tables shared with the workers, unlike `spawn`):

```python
# at the very top of trend_uncertainty.py, before `import numpy`
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ... process_site() must `return {...}` again ...

def run_all_sites():
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        return list(tqdm(pool.imap_unordered(process_site, nzd_sites_trial),
                         total=len(nzd_sites_trial), desc="Processing sites"))

if __name__ == "__main__":
    site_results = run_all_sites()
    create_log(site_results)
```

## 4.8 Confirmed: the memory leak is numpy 2.3.0 / 2.3.1 (`clean_OCC` pins numpy 2.3.1)

Tested 2026-08-28 after Eduardo suggested `numpy=2.3.1` as the culprit. Result: **confirmed.**

**Mechanism.** numpy 2.3.0 and 2.3.1 leak the output buffer of any reduction/accumulation called with an `out=` argument ([numpy issue #29355](https://github.com/numpy/numpy/issues/29355), fixed by PR #29414 in [numpy 2.3.2](https://numpy.org/devdocs/release/2.3.2-notes.html), "fix reference leakage for output arrays in reduction functions"). `np.vander` builds its result with `multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)`, so **every `np.vander` call — and therefore every `np.polyfit` call — leaks its whole Vandermonde array**. `block_bootstrap_slopes()` calls `np.polyfit` once per bootstrap realisation: 1,000 for the historic pool and 1,000 for the recent pool, i.e. **2,000 leaked arrays per transect**, and the leak lives in whichever process runs the bootstrap (the workers), for as long as that process lives. With the committed `process_map` the workers live for the whole run (4.5 item 1), so the leaked memory accumulates across all ~1,500 transects each worker handles.

**Micro-benchmarks** (single process, `OPENBLAS_NUM_THREADS=1`, RSS measured with psutil after `gc.collect()`):

| Call (60-point window) | numpy 1.26.4 | numpy 2.3.1 (py 3.12) | numpy 2.3.1 (py 3.13.5) |
|---|---|---|---|
| `np.polyfit(t, y, 1)` | 0 B/call | **+1,265 B/call** (+350 MB per 300 k calls) | **+1,265 B/call** (+350 MB per 300 k calls) |
| `np.vander(t, 2)` | 0 | +1,265 B/call | – |
| `np.multiply.accumulate(a, out=view)` | 0 | +144 B/call | – |
| `np.linalg.lstsq`, `solve`, `inv`, `svd`, `np.interp`, `np.append`, `accumulate` without `out=` | 0 | 0 | – |

`tracemalloc` attributes the retained blocks to `numpy/lib/_twodim_base_impl.py` (`vander`: `empty(...)` and `multiply.accumulate(..., out=tmp[:, 1:])`), called from `_polynomial_impl.py:656` (`polyfit`). The leak is independent of the Python version (3.12.3 and 3.13.5 identical) and of the BLAS thread count.

**numpy version sweep** (same test, ephemeral `uv` environments, Python 3.12):

| numpy | 1.26.4 | 2.0.2 | 2.1.3 | 2.2.6 | **2.3.0** | **2.3.1** | 2.3.2 | 2.3.3 | 2.3.4 | 2.3.5 | 2.4.0 | 2.4.1 | 2.4.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `polyfit` leak per call | 0 | 0 | 0 | 0 | **1,203 B** | **1,202 B** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Fix:** move `clean_OCC` to any numpy ≥ 2.3.2 (e.g. `conda install "numpy>=2.3.2"` or pin `numpy=2.3.5`; numpy 2.4.x also works with the rest of the pins). No code change is needed for this particular problem. The 4.5/4.7 issues (tqdm dropping `max_tasks_per_child`, BLAS oversubscription) are independent of it — but note that, once `maxtasksperchild=1` really recycles workers, even a leaking numpy could only accumulate one site's worth of leak (≤ 447 transects × ~7 MB ≈ 3 GB for the largest site) before the process is discarded.

Why it was not seen in section 4: this VM's system Python has numpy 1.26.4, which does not leak.

**End-to-end measurement with the real script.** The committed script (single worker, `max_workers=1`, `site_ids_override` = 10 mid-size sites: `nzd0004, 0062, 0156, 0239, 0286, 0332, 0369, 0409, 0449, 0489` → 1,105 transects, 1,094 bootstrapped, ≈ 2.19 M `polyfit` calls) was run to completion in three environments while sampling the worker's private memory (USS) every 5 s:

| Environment | Runtime | Worker private memory (USS) start → end | Growth | Per transect | Per `polyfit` call |
|---|---|---|---|---|---|
| Python 3.12.3, **numpy 1.26.4**, pandas 2.3.0 (this VM) | 40.6 min | 0.29 → 0.34 GB (max 0.38) | +0.09 GB | 0.1 MB | ≈ 40 B (noise / allocator fragmentation) |
| Python 3.12.3, **numpy 2.3.1**, pandas 2.3.1, mpl 3.10.0 | 43.6 min | 0.30 → **8.12 GB** | **+7.9 GB** | **7.2 MB** | ≈ 3.6 KB |
| Python 3.13.5, **numpy 2.3.1**, pandas 2.3.1, mpl 3.10.0 (= `clean_OCC` pins) | 41.0 min | 0.34 → **8.19 GB** | **+7.9 GB** | **7.2 MB** | ≈ 3.6 KB |

The growth is linear in the number of transects processed by the worker (≈ 190 MB/min at this workload) and never plateaus. Scaled to the all-regions run (30,730 bootstrapped transects × 7.2 MB ≈ **220 GB leaked in total**, ≈ 14 GB per worker with 16 long-lived workers, on top of the ≈ 8 GB baseline of section 4.4), a `clean_OCC` run with numpy 2.3.1 cannot finish all regions on any machine we have; a single region fits only while its transect count × 7.2 MB / `max_workers` stays below the free RAM (e.g. Auckland: 1,440 bootstrapped transects → ≈ 10 GB extra spread over the workers; Northland: 4,056 → ≈ 29 GB; Canterbury: 5,030 → ≈ 36 GB), which is exactly the "single region works, all regions eventually run out of memory" symptom.

## 4.9 Committed optimised script: all regions × all 5 scenarios in 45 minutes

Measured 2026-09-03 with the optimisations committed to `trend_uncertainty.py` (`1c547ce`, `935505d`), which keep `process_map` and its progress bar. What changed relative to the merged 5-scenario version: BLAS capped at 1 thread per worker; the scenario loop moved inside `process_site` so the gap filter, LOESS, both 1000-draw bootstraps and the two diagnostic plots run **once per transect instead of 5×** (only SLR extraction, Monte Carlo and exports are per-scenario); the projection figure rendered once instead of 5×; per-site slicing of the merged tables; the leftover duplicate `process_map` call after the scenario loop removed (it re-ran all 534 sites a sixth time and discarded the results); `max_workers=None` → all cores. Equivalence was verified on 2 sites × 5 scenarios: all 65 projection CSVs byte-identical to the pre-optimisation code, same log content (and 228 s → 39 s on that smoke test).

| | Merged 5-scenario version (est.) | Committed optimised version (measured) |
|---|---|---|
| Wall-clock, 534 sites × 5 scenarios | ≈ 7.4 h (5 × ≈ 74 min per scenario + the duplicate 6th pass, 16 workers) | **45 min 6 s** (32 workers; script-internal 44 min 47 s) |
| Sum of per-site times | ≈ 5,600 min | 1,180 min (median site 65 s · mean 133 s · p90 375 s · max 19.2 min, `nzd0451`) |
| Effective parallelism | ≈ 15 of 16 | 26.4 of 32 (the 19-min site dominates the tail) |
| System CPU / load | ≈ 72 % but load ≈ 30 from 512 BLAS threads | mean 78 %, peaks 100 % (exactly 33 single-threaded processes), load ≈ 27–39 |
| Memory (real footprint, PSS) | ≈ 6–7 GB | 10.2–10.7 GB (parent 1.9 GB with the full 5-scenario table; workers 1.64 GB RSS / ≈ 0.3 GB private each); system peak 24.5 GB |
| Exit | traceback + exit 1, no logs | exit 0, progress bar, all 10 per-scenario log files written |

Outputs of the full run: **119,745 projection CSVs** (23,735–24,023 per scenario — a few hundred transects lack SLR-year coverage in some scenarios), 24,023 projection PNGs (one per transect), 31,609 + 30,730 diagnostic PNGs now all in `original_plots_ts/`, ≈ 206 k files / 37 GB in total.

# 5. Dashboard data pipeline (`generate_dashboard_data.py`)

The CSV → JSON → lazy-loading design is the right architecture for this dashboard, and it has been kept: a compact `data/projections.json` summary powers the initial map, and one detail file per transect under `data/projections_details/` is fetched only when a transect is clicked. What changed (2026-09-03) is how the JSON is produced and how much of it there is:

1. **Parallel generation** with `tqdm.contrib.concurrent.process_map` (all CPU cores; each worker writes its transects' detail files directly instead of the parent accumulating every payload in memory before writing).
2. **Historical series stored once per transect.** Each projection CSV repeats the identical satellite + LOESS history (hundreds of rows) for every scenario, while the projections themselves are only ~4–6 rows per scenario. Detail files now hold one top-level `historical` array plus projection-only rows per scenario; `index.html` reads it with a backward-compatible fallback for old files.
3. **Compact JSON** (no `indent=2`), floats rounded to 4 decimals (0.1 mm), only `*_projection_results.csv` globbed, and the unused `row_count`/`detail_file` summary fields dropped (`index.html` derives the detail path).

Measured on the full 5-scenario outputs (119,745 CSVs → 24,023 detail files), same machine as section 4:

| | Old generator | New generator |
|---|---|---|
| Runtime | 2 h 10 min (single-threaded) | **5 min 30 s** (32 workers) |
| Peak memory | **64 GB** (whole payload held in RAM) | 0.13 GB |
| `data/projections_details/` size | 51.9 GB | **7.4 GB** (≈ 60–300 KB per transect) |
| `data/projections.json` size | 18.6 MB | 8.7 MB |

On dashboard loading itself: the summary + per-transect lazy fetch already is best practice at this scale — 24 k transects cannot ship their full series up front, and a per-click ~100 KB fetch (a few tens of KB gzipped; GitHub Pages and most static hosts gzip text automatically) renders instantly. The remaining load-time cost is the 8.7 MB summary index (~1–2 MB gzipped), which is fetched once at startup; if that ever feels slow, the next steps would be splitting the summary per region and/or dropping per-scenario deltas the map does not colour by — not a different architecture.
