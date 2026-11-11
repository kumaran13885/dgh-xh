# DGH-XH

Code and result tables for the paper

> K. Thirugnanasambandamoorthy and P. Rangasamy, *A Graph-Adaptive Regression
> Architecture with Metaheuristic Optimization and Tail-Aware Residual
> Correction*. Under review at Discover Computing.

DGH-XH forecasts a target at every node of a sensor network, 1 to 24 hours ahead.
It has two stages. A Huber regressor works on graph-aggregated features, and a
tail-weighted XGBoost model then corrects its residuals. The graph is a k-nearest-neighbour
graph whose edge weights are modulated by a wind-alignment gate. The gate has no
trainable parameters; its sharpness α and floor ε are tuned together with the
regressors by a Jellyfish Search optimizer (JSO).

The repository covers all six datasets in the paper: Chennai, Delhi and Beijing
PM2.5, Beijing temperature, and significant wave height from two NDBC buoy
networks (Gulf of Mexico and the US West Coast).

## Layout

```
notebooks/      one notebook per dataset, plus the Chennai controls and flux variant
scripts/        post-processing: statistics, intervals, exceedance, figures
results/        CSV outputs behind the tables and figures (see results/README.md)
```

Large files (fitted models, prediction arrays, windowed tensors) are not in this
repository. They are archived on Zenodo together with a snapshot of this code:
DOI [to be added].

## Setup

The Chennai experiments ran on a Windows 11 workstation with Python 3.10. The
other datasets ran on Kaggle CPU notebooks. Everything is CPU only. No GPU is
needed.

With pip:

```
python -m venv .venv
.venv\Scripts\activate          # on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

With conda:

```
conda env create -f environment.yml
conda activate dgh-xh
```

Tree-based results can move by a few per cent between library versions, so pin
the versions if you want to match the paper closely. The versions we used were:

| Runs | Python | NumPy | pandas | scikit-learn | XGBoost |
|---|---|---|---|---|---|
| Chennai (local) | 3.10.19 | 2.2.6 | 2.3.3 | 1.7.2 | 3.2.0 |
| Delhi, Beijing, Gulf of Mexico (Kaggle) | 3.12.12 | 2.0.2 | 2.3.3 | 1.6.1 | 3.2.0 |
| US West Coast, five-permutation controls (Kaggle) | 3.13.18 | 2.1.3 | 2.3.3 | 1.6.1 | 3.4.1 |

Each run also writes its versions to `run_manifest.json` in its output folder.

## Data

No raw data is redistributed here. All of it is public.

- **Chennai and Delhi PM2.5**: hourly station data from the CPCB National Air
  Quality Index portal, https://airquality.cpcb.gov.in (2023 to 2025).
- **Meteorology for Chennai and Delhi**: ERA5 hourly single levels from the
  Copernicus Climate Data Store. We use 2 m temperature, 2 m dewpoint, surface
  pressure, and the 10 m wind components u10 and v10. `scripts/download_era5.py`
  fetches them (you need a free CDS account and an API key in `~/.cdsapirc`).
- **Beijing PM2.5 and temperature**: UCI Beijing Multi-Site Air Quality data,
  https://doi.org/10.24432/C5RK5G
- **Wave height**: NOAA NDBC standard meteorological files,
  https://www.ndbc.noaa.gov/historical_data.shtml (2019 to 2021). The two buoy
  notebooks download these themselves, so turn on internet access if you run them
  on Kaggle.

Put the CPCB and ERA5 files where the first cell of each notebook expects them.
The paths there are Kaggle input paths; change them for a local run.

## Running

Each notebook runs top to bottom and writes its tables to an output folder.

| Notebook | Dataset | Notes |
|---|---|---|
| `01_chennai_pm25.ipynb` | Chennai | main experiment, all baselines and the tuned model |
| `02_chennai_controls.ipynb` | Chennai | tuned controls, tail-weighted XGBoost, JSO seeds, random search, k sensitivity, timing |
| `03_delhi_pm25.ipynb` | Delhi | |
| `03b_delhi_reuse_windows.ipynb` | Delhi | refits only the gate-dependent models from saved windows |
| `04_beijing_pm25.ipynb` | Beijing | |
| `05_beijing_temperature.ipynb` | Beijing | |
| `06_gulf_of_mexico_swh.ipynb` | Gulf of Mexico | downloads NDBC data |
| `07_us_west_coast_swh.ipynb` | US West Coast | downloads NDBC data |
| `08_chennai_flux_variant.ipynb` | Chennai | wind-speed and gradient-scaled gate (Discussion); exceedance diagnostics |

`GATE_MODE` near the top of each notebook selects the adjacency:

- `"fixed"` is the gate described in the paper. The wind direction at the donor
  station is compared with the donor-to-receiver bearing, both measured clockwise
  from north.
- `"shuffled"` applies the same gate to randomly permuted edge bearings. This is
  the permuted-bearing control. `SHUFFLE_SEED` sets the permutation; the paper
  uses 12345 and, on the buoy networks, also 1001 to 1004.
- `"legacy"` reproduces the gate as written in the first version of the
  manuscript. It is kept only so the earlier numbers can be checked.

Seeds: Chennai uses seed 1 for JSO and for the boosters. On the other datasets
JSO runs once with seed 42, and the chosen configuration is refitted with booster
seeds 42, 43 and 44.

Run time is dominated by JSO. On Chennai, 192 fitness evaluations take about
3.5 hours per horizon. The other datasets use 54 evaluations.

`scripts/regen_cell.py` is run as an extra cell at the end of
`02_chennai_controls.ipynb`. It refits the Table 7 models once from their saved
settings and writes the per-sample predictions in `results/chennai/regenerated/`.
The other scripts run from the command line; `python scripts/<name>.py --help`
lists their options.

The Chennai statistics can be rerun from the saved predictions:

```
P=results/chennai/regenerated
python scripts/block_bootstrap.py out_bs $P/H12/predictions.npz $P/H1/predictions.npz \
    $P/H24/predictions.npz $P/H3/predictions.npz $P/H6/predictions.npz \
    --ref XGB --ref "CONTROL Tuned Static-Graph Hybrid" \
    --ref "CONTROL Tuned Flat Hybrid" --ref "CONTROL Shuffled-Bearing Graph"
python scripts/exceedance_eval.py out_ex "$P/H*/predictions.npz"
python scripts/conformal_uq.py out_cf "$P/H*/predictions.npz"
python scripts/objective_tradeoff.py out_ot "$P/H*/predictions.npz"
```

The bootstrap draws from one random stream across horizons, so the file order
matters; the order above reproduces `results/chennai/bootstrap/block_bootstrap.csv`
exactly. Adding `--pcts 80,85,90,95,97.5,99,99.5` with only the first two `--ref`
options gives `block_bootstrap_pcts.csv`.

| Paper item | Where it comes from |
|---|---|
| Tables 3 to 6 | `01_chennai_pm25.ipynb` |
| Table 2 and the latency figures | last cell of `02_chennai_controls.ipynb` |
| Table 7 (ablation) | `01` and `02` notebooks |
| Table 8 (gate controls) | dataset notebooks with `GATE_MODE` set to `fixed` and `shuffled`, then `scripts/ndbc_gate_verdict.py` for the buoy networks |
| Table 10 | notebooks `03` to `07` |
| Block bootstrap | `scripts/block_bootstrap.py` |
| Exceedance verification | `scripts/exceedance_eval.py` |
| Prediction intervals | `scripts/conformal_uq.py` |
| Accuracy and tail-bias trade-off | `scripts/objective_tradeoff.py` |
| Neighbour headroom | `scripts/network_redundancy_diagnostic.py` |
| Gate ablation over saved predictions | `scripts/graph_ablation_tails.py` |
| Naive baselines from saved tensors | `scripts/naive_from_tensors.py` |
| Shared-station check | `scripts/common_station_check.py` |
| West Coast buoy screening | `scripts/ndbc_replication_screen.py` |
| Figures 2 to 5 | `scripts/make_figures.py` |

## Results

`results/` holds the CSVs the paper was written from, one folder per dataset and
gate mode, and one subfolder per seed where there are several. `results_main.csv`
has overall MAE, RMSE and R² per model and horizon. `results_tail.csv` has MAE
above the 90th, 95th and 99th percentiles. `hyperparams.csv` holds the selected
JSO settings.

## Licence

The code is released under the MIT licence (see `LICENSE`). The result tables in
`results/` are released under CC BY 4.0. The source datasets keep their own terms.

## Citation

If you use this code, please cite the paper above. A `CITATION.cff` file is
included, and the Zenodo DOI will be added once the archive is published.

## Contact

Parthiban Rangasamy, parthibanr@rvscet.ac.in
