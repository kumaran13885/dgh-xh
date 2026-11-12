import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

GATE_H_LIST = [1, 3, 6, 12, 24]
HP_MAIN_CSV = "/kaggle/input/dghxh-hparams/all_results_main.csv"
REGEN_DIR = os.path.join(globals().get("OUT_DIR", "."), "boot")
GATE_OUT = os.path.join(globals().get("OUT_DIR", "."), "boot_gate")
GATE_CACHE_DIR = globals().get("CTRL_CACHE_DIR", ".")
GATE_SEED = globals().get("RANDOM_SEED", 1)

NAME_STATIC = "Tuned Hybrid (static adjacency, matched HP)"
NAME_SHUF = "Tuned Hybrid (shuffled bearings)"

os.makedirs(GATE_OUT, exist_ok=True)
for p in (HP_MAIN_CSV, REGEN_DIR, GATE_CACHE_DIR):
    print(f"  {'OK  ' if os.path.exists(p) else 'MISSING'} {p}")
_hp = pd.read_csv(HP_MAIN_CSV)


def _hp_row(H):
    r = _hp[(_hp.H == H) & (_hp.model == "Tuned Hybrid (Graph + XGB residual)")]
    assert len(r) == 1
    r = r.iloc[0]
    xp = {"objective": "reg:pseudohubererror",
          "max_depth": int(round(r.max_depth)), "eta": float(r.eta),
          "subsample": float(r.subsample),
          "colsample_bytree": float(r.colsample_bytree),
          "lambda": float(r["lambda"]), "tree_method": "hist",
          "seed": GATE_SEED, "verbosity": 0}
    return dict(tau=int(r.tau), alpha=float(r.alpha), eps=float(r.eps),
                huber_alpha=float(r.huber_alpha), xgb_params=xp,
                rounds=int(round(r.num_boost_round)),
                w_mid=float(r.w_mid), w_danger=float(r.w_danger),
                w_tail=float(r.w_tail))


def _stage2(ytr_all, mtr, mte, pred_tr_all, pred_te_all, Xtr, Xte, hp):
    res = ytr_all - pred_tr_all
    y = ytr_all[mtr]
    w = np.ones_like(y, dtype=np.float32)
    w[y >= np.percentile(y, 90)] = hp["w_mid"]
    w[y >= np.percentile(y, 95)] = hp["w_danger"]
    w[y >= np.percentile(y, 99)] = hp["w_tail"]
    gtr = np.asarray(flatten_window_per_node(Xtr), dtype=np.float32)
    gte = np.asarray(flatten_window_per_node(Xte), dtype=np.float32)
    bst = xgb.train(hp["xgb_params"], xgb.DMatrix(gtr[mtr], label=res[mtr], weight=w),
                    num_boost_round=hp["rounds"])
    out = pred_te_all.copy()
    out[mte] = out[mte] + bst.predict(xgb.DMatrix(gte))[mte]
    return out[mte]


def _dynamic(Xtr, Ytr, Mtr, Xte, Yte, Mte, hp, gate_mode):
    b = fit_huber_graph(Xtr, Ytr, Mtr, Xte, Yte, Mte, ctx=graph_ctx, tau=hp["tau"],
                        alpha=hp["alpha"], eps=hp["eps"],
                        huber_alpha=hp["huber_alpha"], gate_mode=gate_mode)
    return _stage2(b["y_train_all"], b["mask_train"], b["mask_test"],
                   b["pred_train_all"], b["pred_test_all"], Xtr, Xte, hp)


def _static(Xtr, Ytr, Mtr, Xte, Yte, Mte, hp):
    Ztr = make_graph_features_static(Xtr, A_static, tau=hp["tau"])
    Zte = make_graph_features_static(Xte, A_static, tau=hp["tau"])
    Ztr, Zte = Ztr.reshape(-1, Ztr.shape[-1]), Zte.reshape(-1, Zte.shape[-1])
    ytr = Ytr.reshape(-1)
    mtr, mte = Mtr.reshape(-1) > 0.5, Mte.reshape(-1) > 0.5
    m = Pipeline([("scaler", StandardScaler()),
                  ("huber", HuberRegressor(epsilon=1.35, alpha=hp["huber_alpha"],
                                           max_iter=500))])
    m.fit(Ztr[mtr], ytr[mtr])
    return _stage2(ytr, mtr, mte, m.predict(Ztr), m.predict(Zte), Xtr, Xte, hp)


_rows = []
for H in GATE_H_LIST:
    d = os.path.join(GATE_CACHE_DIR, f"H{H}")
    Xtr, Ytr, Mtr, Xte, Yte, Mte = (np.load(os.path.join(d, k + ".npy")) for k in
                                    ("X_train_imp", "Y_train", "M_train",
                                     "X_test_imp", "Y_test", "M_test"))
    hp = _hp_row(H)
    z = dict(np.load(os.path.join(REGEN_DIR, f"H{H}", "predictions.npz")))
    y = z["y_true"].astype(np.float64)

    p_fix = _dynamic(Xtr, Ytr, Mtr, Xte, Yte, Mte, hp, "fixed")
    p_sta = _static(Xtr, Ytr, Mtr, Xte, Yte, Mte, hp)
    p_shu = _dynamic(Xtr, Ytr, Mtr, Xte, Yte, Mte, hp, "shuffled")

    check = float(np.max(np.abs(p_fix - z["pred__Tuned Hybrid"])))
    z["pred__" + NAME_STATIC] = p_sta.astype(np.float32)
    z["pred__" + NAME_SHUF] = p_shu.astype(np.float32)
    od = os.path.join(GATE_OUT, f"H{H}")
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, "predictions.npz"), **z)

    def mae(p):
        return float(np.mean(np.abs(p.astype(np.float64) - y)))

    _rows.append(dict(H=H, tuned_hybrid=mae(z["pred__Tuned Hybrid"]),
                      static_matched=mae(p_sta), shuffled_matched=mae(p_shu),
                      refit_max_abs_diff=check))
    print(f"H={H:2d}  TH {_rows[-1]['tuned_hybrid']:.4f}  static {mae(p_sta):.4f}  "
          f"shuffled {mae(p_shu):.4f}  refit diff {check:.2e}")

pd.DataFrame(_rows).to_csv(os.path.join(GATE_OUT, "gate_matched_mae.csv"), index=False)
print(f"saved {GATE_OUT}")
