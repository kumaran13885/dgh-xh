# Results

One folder per dataset. The urban and temperature datasets and the Gulf of Mexico have
one subfolder per gate mode (`fixed`, `legacy`, `shuffled`); `fixed` is the gate
used in the paper. Beijing temperature and the Gulf of Mexico have one further
subfolder per booster seed (`seed42` to `seed44`), and so does the US West Coast.

Folder names here are tidied; the notebooks write to their own output folders
(for example `OUT_DIR = "Timing_threads"` for what is now `chennai/deployment/`).

Common files:

- `results_main.csv`: MAE, RMSE and R² per model and horizon
- `results_tail.csv`: MAE and RMSE above the 90th, 95th and 99th test percentiles
- `tau_search.csv`: validation MAE for each candidate lag
- `hyperparams.csv`: JSO-selected settings
- `significance_all.csv`, `significance_pooled.csv`: paired tests between models
- `run_manifest.json`: seed, library versions and run time

## Chennai

| Folder | Contents | Paper |
|---|---|---|
| `main/` | main run, `GATE_MODE = "fixed"` | Tables 4 to 6, Table 7 (untuned rows) |
| `controls/` | tuned controls and tail-weighted XGBoost, merged from the Kaggle jobs | |
| `regenerated/` | all ten Table 7 models refitted in one environment from their saved settings | Table 7 (tuned and tail-weighted rows) |
| `bootstrap/` | day-level block bootstrap on the refitted predictions (`_pcts`: seven tail percentiles; `_gate2`: wind gate against distance-only and permuted bearings at the tuned settings) | Table 7, bootstrap tests |
| `exceedance/` | exceedance verification at the 90th, 95th and 99th percentiles | Exceedance warning head |
| `conformal/` | split-conformal coverage | Prediction intervals |
| `tradeoff/` | MAE against tail bias, re-ranking over the bias weight | Ablation and gate controls |
| `jso_seeds/` | four extra JSO seeds at H = 1 and 24 h | Implementation details |
| `random_search/` | random search with the same budget | Implementation details |
| `k_sensitivity/` | k = 2, 3, 6, 8 | Dynamic spatial graph construction |
| `deployment/` | latency, model size, machine details | Table 2, Edge deployability profile |

Two things to know:

1. `main/all_results_main.csv` and `main/all_results_tail.csv` still hold the
   `CONTROL Tail-Weighted XGB` rows from the first run, in which that booster
   diverged (MAE above 1000 at H = 1 h). Ignore those rows. The corrected model
   is in `controls/` and `regenerated/`.
2. The tuned controls were first run as separate Kaggle jobs (`controls/`). The
   paper uses the single-environment refits instead (`regenerated/`, point
   estimates in `bootstrap/block_bootstrap.csv`). They differ from the merged
   values by 1.2% at most; `regenerated/regen_verification.csv` lists both.

## Buoy networks

`gulf_of_mexico/ndbc_gate_verdict_mexico.csv` and
`us_west_coast/ndbc_gate_verdict_westcoast.csv` hold the per-cell tail MAE of the
wind gate (W), the distance-only graph (D) and the five permuted-bearing runs
(P12345, P1001 to P1004) behind Table 8.

`network_redundancy.csv` is the neighbour-headroom measure in the ablation section.
