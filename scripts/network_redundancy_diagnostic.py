import argparse, os, sys
import numpy as np
NRD_NAME = globals().get("NRD_NAME", "Chennai")
NRD_H = int(globals().get("NRD_H", 1))
NRD_LAGS = tuple(globals().get("NRD_LAGS", (1, 3, 6, 12, 24)))
NRD_OUT = globals().get("NRD_OUT", "network_redundancy.csv")
_CACHE = None

if __name__ == "__main__" and len(sys.argv) > 1:
    _ap = argparse.ArgumentParser()
    _ap.add_argument("out")
    _ap.add_argument("--name", required=True)
    _ap.add_argument("--cache", required=True)
    _ap.add_argument("--h", type=int, default=1)
    _ap.add_argument("--lags", default="1,3,6,12,24")
    _a = _ap.parse_args()
    NRD_OUT, NRD_NAME, NRD_H = _a.out, _a.name, _a.h
    NRD_LAGS = tuple(int(x) for x in _a.lags.split(",") if x.strip())
    _CACHE = _a.cache
    assert os.path.isdir(_CACHE)

print("=" * 78)
print(f"NETWORK REDUNDANCY DIAGNOSTIC - {NRD_NAME}")
print("=" * 78)


def _find_targets():
    if not _CACHE:
        for nm in ("Y_full", "Y_all"):
            if nm in globals() and np.asarray(globals()[nm]).ndim == 2:
                return np.asarray(globals()[nm], float), nm
    parts, names = [], []
    for nm in ("Y_train", "Y_test"):
        v = None if _CACHE else globals().get(nm)
        if v is None:
            for d in ([_CACHE] if _CACHE else
                      [globals()[k] for k in ("CTRL_CACHE_DIR", "REGEN_CACHE_DIR")
                       if k in globals()]):
                p = os.path.join(d, f"H{NRD_H}", nm + ".npy")
                if os.path.exists(p):
                    v = np.load(p)
                    break
                p2 = os.path.join(d, nm + ".npy")
                if os.path.exists(p2):
                    v = np.load(p2)
                    break
        if v is not None:
            v = np.asarray(v, float)
            assert v.ndim == 2
            parts.append(v)
            names.append(nm)
    assert parts
    return np.concatenate(parts, 0), "+".join(names)


def _find_mask(n_rows, n_st):
    parts = []
    for nm in ("M_train", "M_test"):
        v = None if _CACHE else globals().get(nm)
        if v is None:
            for d in ([_CACHE] if _CACHE else
                      [globals()[k] for k in ("CTRL_CACHE_DIR", "REGEN_CACHE_DIR")
                       if k in globals()]):
                for p in (os.path.join(d, f"H{NRD_H}", nm + ".npy"),
                          os.path.join(d, nm + ".npy")):
                    if os.path.exists(p):
                        v = np.load(p)
                        break
                if v is not None:
                    break
        if v is not None:
            parts.append(np.asarray(v, float) > 0.5)
    if not parts:
        return np.ones((n_rows, n_st), bool)
    M = np.concatenate(parts, 0)
    return M if M.shape == (n_rows, n_st) else np.ones((n_rows, n_st), bool)


Y, _src = _find_targets()
T, N = Y.shape
M = _find_mask(T, N)
Y = np.where(M, Y, np.nan)
print(f"  targets {Y.shape} from {_src}   valid fraction {np.isfinite(Y).mean():.3f}")
assert N >= 2


def pairwise_corr(Y):
    out = []
    for i in range(Y.shape[1]):
        for j in range(i + 1, Y.shape[1]):
            m = np.isfinite(Y[:, i]) & np.isfinite(Y[:, j])
            if m.sum() > 30:
                a, b = Y[m, i], Y[m, j]
                if a.std() > 0 and b.std() > 0:
                    out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)


def _nanmean_rows(A):
    ok = np.isfinite(A)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, A, 0.0).sum(axis=1)
    out = np.full(A.shape[0], np.nan)
    nz = cnt > 0
    out[nz] = tot[nz] / cnt[nz]
    return out, int((~nz).sum())


def neighbour_headroom(Y, lag, min_rows=200):
    gains, used, empty = [], [], 0
    for s in range(Y.shape[1]):
        others = [k for k in range(Y.shape[1]) if k != s]
        y = Y[lag:, s]
        own = Y[:-lag, s]
        nb, n_empty = _nanmean_rows(Y[:-lag][:, others])
        empty += n_empty
        m = np.isfinite(y) & np.isfinite(own) & np.isfinite(nb)
        used.append(int(m.sum()))
        if m.sum() < min_rows:
            continue
        y, own, nb = y[m], own[m], nb[m]
        X1 = np.column_stack([np.ones_like(own), own])
        X2 = np.column_stack([np.ones_like(own), own, nb])
        r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
        r2 = y - X2 @ np.linalg.lstsq(X2, y, rcond=None)[0]
        v1 = float(r1.var())
        if v1 > 0:
            gains.append(1.0 - float(r2.var()) / v1)
    return (float(np.median(gains)) if gains else np.nan,
            int(np.median(used)) if used else 0,
            int(len(gains)),
            empty / max(1, Y.shape[1] * (Y.shape[0] - lag)))


c = pairwise_corr(Y)
row = dict(dataset=NRD_NAME, n_stations=int(N), n_rows=int(T),
           n_pairs=int(c.size),
           corr_median=float(np.median(c)) if c.size else np.nan,
           corr_p10=float(np.percentile(c, 10)) if c.size else np.nan,
           corr_p90=float(np.percentile(c, 90)) if c.size else np.nan)
for lag in NRD_LAGS:
    if T > lag + 250:
        h, n_used, n_st_ok, frac_empty = neighbour_headroom(Y, lag)
        row[f"headroom_lag{lag}"] = h
        row[f"rows_used_lag{lag}"] = n_used
        row[f"stations_scored_lag{lag}"] = n_st_ok
        if lag == NRD_LAGS[0]:
            row["neighbour_undefined_frac"] = float(frac_empty)
    else:
        row[f"headroom_lag{lag}"] = np.nan

print("\n" + "-" * 78)
for k, v in row.items():
    print(f"  {k:24s} {v}")
print("-" * 78)

import pandas as pd
_new = pd.DataFrame([row])
if os.path.exists(NRD_OUT):
    _old = pd.read_csv(NRD_OUT)
    _old = _old[_old.dataset != NRD_NAME]
    _new = pd.concat([_old, _new], ignore_index=True)
_new.to_csv(NRD_OUT, index=False)
print(f"\nsaved {os.path.abspath(NRD_OUT)}  ({len(_new)} rows)")
