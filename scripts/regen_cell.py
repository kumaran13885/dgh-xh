import os, json
import numpy as np
import pandas as pd

REGEN_H_LIST = [1, 3, 6, 12, 24]
REGEN_OUT    = os.path.join(globals().get("OUT_DIR", "."), "boot")

HP_MAIN_CSV = "/kaggle/input/dghxh-hparams/all_results_main.csv"
HP_CTRL_CSV = "/kaggle/input/dghxh-hparams/controls_results_main.csv"

REGEN_VERIFY_TOL_PCT = 1.0
REGEN_CACHE_DIR = globals().get("CTRL_CACHE_DIR", ".")
RANDOM_SEED     = globals().get("RANDOM_SEED", 1)

os.makedirs(REGEN_OUT, exist_ok=True)
print("=" * 78)
print("REGENERATE PER-SAMPLE PREDICTIONS")
print("=" * 78)
for p, lab in ((HP_MAIN_CSV, "HP_MAIN_CSV"), (HP_CTRL_CSV, "HP_CTRL_CSV"),
               (REGEN_CACHE_DIR, "cache dir")):
    print(f"  {'OK  ' if os.path.exists(p) else 'MISSING'} {lab:12s} {p}")
assert os.path.exists(HP_MAIN_CSV)
assert os.path.exists(HP_CTRL_CSV)

_hp_main = pd.read_csv(HP_MAIN_CSV)
_hp_ctrl = pd.read_csv(HP_CTRL_CSV)
print(f"  rows: {len(_hp_main)} main, {len(_hp_ctrl)} controls")


def _row(df, H, model):
    r = df[(df.H == H) & (df.model == model)]
    assert len(r) == 1
    return r.iloc[0]


def _f(r, k, default=None):
    v = r.get(k, None)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        assert default is not None
        return default
    return float(v)


def _xgbp(r):
    return {"objective": "reg:pseudohubererror",
            "max_depth": int(round(_f(r, "max_depth"))),
            "eta": _f(r, "eta"), "subsample": _f(r, "subsample"),
            "colsample_bytree": _f(r, "colsample_bytree"),
            "lambda": _f(r, "lambda"), "tree_method": "hist",
            "seed": RANDOM_SEED, "verbosity": 0}


def _tw(r):
    return dict(w_mid=_f(r, "w_mid"), w_danger=_f(r, "w_danger"),
                w_tail=_f(r, "w_tail"))


def _load(H):
    d = os.path.join(REGEN_CACHE_DIR, f"H{H}")
    need = ["X_train_imp", "Y_train", "M_train",
            "X_test_imp", "Y_test", "M_test"]
    miss = [k for k in need if not os.path.exists(os.path.join(d, k + ".npy"))]
    assert not miss
    return {k: np.load(os.path.join(d, k + ".npy")) for k in need}


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, Ridge
from xgboost import XGBRegressor

_summary = []

for H in REGEN_H_LIST:
    print("\n" + "=" * 78)
    print(f"H = {H}")
    print("=" * 78)
    p = _load(H)
    Xtr, Ytr, Mtr = p["X_train_imp"], p["Y_train"], p["M_train"]
    Xte, Yte, Mte = p["X_test_imp"],  p["Y_test"],  p["M_test"]
    S, N = Yte.shape
    mte = Mte.reshape(-1) > 0.5
    y_true = Yte.reshape(-1)[mte].astype(np.float32)
    si, ni = np.divmod(np.arange(S * N), N)
    si, ni = si[mte], ni[mte]
    print(f"  test: {S} windows x {N} stations, {mte.sum()} station-hours")

    tau_dyn = int(_row(_hp_main, H, "Huber Graph").tau)
    tau_sta = int(_row(_hp_main, H, "Static-Graph Huber").tau)
    print(f"  tau_dyn={tau_dyn}  tau_static={tau_sta}")

    preds, reported = {}, {}

    def keep(name, yp, rep):
        yp = np.asarray(yp, dtype=np.float32)
        assert yp.shape == y_true.shape
        assert np.isfinite(yp).all()
        preds[name] = yp
        reported[name] = float(rep)
        got = float(np.mean(np.abs(yp.astype(np.float64)
                                   - y_true.astype(np.float64))))
        dpct = 100 * (got - rep) / rep
        flag = "OK " if abs(dpct) <= REGEN_VERIFY_TOL_PCT else "DRIFT"
        print(f"    [{flag}] {name:42s} refit {got:9.5f}  reported "
              f"{rep:9.5f}  {dpct:+7.3f}%")
        _summary.append(dict(H=H, model=name, refit_MAE=got,
                             reported_MAE=rep, drift_pct=dpct))

    r = _row(_hp_main, H, "XGB")
    m = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.07,
                     subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                     n_jobs=-1, random_state=RANDOM_SEED)
    keep("XGB", fit_flat_regressor(m, Xtr, Ytr, Mtr, Xte, Yte, Mte)["y_pred"],
         r.MAE)

    r = _row(_hp_main, H, "Huber")
    m = Pipeline([("sc", StandardScaler()),
                  ("h", HuberRegressor(epsilon=1.35, alpha=1e-4,
                                       max_iter=500))])
    keep("Huber", fit_flat_regressor(m, Xtr, Ytr, Mtr, Xte, Yte,
                                     Mte)["y_pred"], r.MAE)

    r = _row(_hp_main, H, "Huber Graph")
    keep("Huber Graph",
         fit_huber_graph(Xtr, Ytr, Mtr, Xte, Yte, Mte, ctx=graph_ctx,
                         tau=tau_dyn)["y_pred"], r.MAE)

    r = _row(_hp_main, H, "CONTROL Shuffled-Bearing Graph")
    keep("CONTROL Shuffled-Bearing Graph",
         fit_huber_graph(Xtr, Ytr, Mtr, Xte, Yte, Mte, ctx=graph_ctx,
                         tau=tau_dyn, alpha=4.0, eps=0.05, huber_alpha=1e-4,
                         gate_mode="shuffled")["y_pred"], r.MAE)

    r = _row(_hp_main, H, "Static-Graph Huber")
    m = Pipeline([("sc", StandardScaler()),
                  ("h", HuberRegressor(epsilon=1.35, alpha=1e-4,
                                       max_iter=500))])
    keep("Static-Graph Huber",
         fit_static_graph_regressor(m, Xtr, Ytr, Mtr, Xte, Yte, Mte,
                                    A_static=A_static,
                                    tau=tau_sta)["y_pred"], r.MAE)

    r = _row(_hp_main, H, "Hybrid (Graph + XGB residual)")
    keep("Hybrid",
         fit_huber_hybrid(Xtr, Ytr, Mtr, Xte, Yte, Mte, ctx=graph_ctx,
                          tau=tau_dyn, num_boost_round=400,
                          seed=RANDOM_SEED)["y_pred_final"], r.MAE)

    r = _row(_hp_main, H, "Tuned Hybrid (Graph + XGB residual)")
    keep("Tuned Hybrid",
         fit_huber_hybrid(Xtr, Ytr, Mtr, Xte, Yte, Mte, ctx=graph_ctx,
                          tau=int(r.tau), alpha=_f(r, "alpha"),
                          eps=_f(r, "eps"), huber_alpha=_f(r, "huber_alpha"),
                          xgb_params=_xgbp(r),
                          num_boost_round=int(round(_f(r, "num_boost_round"))),
                          seed=RANDOM_SEED, **_tw(r))["y_pred_final"], r.MAE)

    r = _row(_hp_ctrl, H, "CONTROL Tuned Flat Hybrid (no graph)")
    base = Pipeline([("sc", StandardScaler()),
                     ("h", HuberRegressor(epsilon=1.35,
                                          alpha=_f(r, "huber_alpha"),
                                          max_iter=500))])
    keep("CONTROL Tuned Flat Hybrid",
         fit_flat_hybrid(base, Xtr, Ytr, Mtr, Xte, Yte, Mte,
                         xgb_params=_xgbp(r),
                         num_boost_round=int(round(_f(r, "num_boost_round"))),
                         seed=RANDOM_SEED, **_tw(r))["y_pred_final"], r.MAE)

    r = _row(_hp_ctrl, H, "CONTROL Tuned Static-Graph Hybrid")
    base = Pipeline([("sc", StandardScaler()),
                     ("h", HuberRegressor(epsilon=1.35,
                                          alpha=_f(r, "huber_alpha"),
                                          max_iter=500))])
    keep("CONTROL Tuned Static-Graph Hybrid",
         fit_static_graph_hybrid(base, Xtr, Ytr, Mtr, Xte, Yte, Mte,
                                 A_static=A_static, tau=int(r.tau),
                                 xgb_params=_xgbp(r),
                                 num_boost_round=int(round(_f(r, "num_boost_round"))),
                                 seed=RANDOM_SEED, **_tw(r))["y_pred_final"],
         r.MAE)

    r = _row(_hp_ctrl, H, "CONTROL Tail-Weighted XGB")
    t = fit_tailweighted_xgb(Xtr, Ytr, Mtr, Xte, Yte, Mte,
                             objective="published", num_boost_round=400,
                             seed=RANDOM_SEED, w_mid=2.0, w_danger=5.0,
                             w_tail=10.0)
    assert not t["diverged"]
    keep("CONTROL Tail-Weighted XGB", t["y_pred"], r.MAE)

    d = os.path.join(REGEN_OUT, f"H{H}")
    os.makedirs(d, exist_ok=True)
    store = {"y_true": y_true, "sample_idx": si.astype(np.int64),
             "station_idx": ni.astype(np.int64),
             "target_time": si.astype(np.int64)}
    store.update({f"pred__{k}": v for k, v in preds.items()})
    np.savez_compressed(os.path.join(d, "predictions.npz"), **store)
    json.dump({k: reported[k] for k in reported},
              open(os.path.join(d, "reported_mae.json"), "w"), indent=1)
    print(f"  saved {os.path.join(d, 'predictions.npz')}  "
          f"({len(preds)} models, {len(y_true)} station-hours)")

sm = pd.DataFrame(_summary)
sm.to_csv(os.path.join(REGEN_OUT, "regen_verification.csv"), index=False)
print("\n" + "=" * 78)
print("REFIT vs REPORTED MAE (drift %)")
print("=" * 78)
print(sm.pivot_table(index="model", columns="H",
                     values="drift_pct").round(3).to_string())
bad = sm[sm.drift_pct.abs() > REGEN_VERIFY_TOL_PCT]
if len(bad):
    print(f"\n  {len(bad)} cells drift more than {REGEN_VERIFY_TOL_PCT}%:")
    print(bad.to_string(index=False))
else:
    print(f"\n  all refits within {REGEN_VERIFY_TOL_PCT}%")
print(f"\n  saved {os.path.join(REGEN_OUT, 'regen_verification.csv')}")
