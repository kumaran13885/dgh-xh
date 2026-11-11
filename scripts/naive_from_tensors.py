import argparse, glob, os, sys
import numpy as np
import pandas as pd

H_LIST = [1, 3, 6, 12, 24]


def find_dir(root, H):
    for name in (f"H{H}", f"h{H}", str(H), f"H={H}"):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    sys.exit(f"no folder for H={H} under {root}")


def load(d):
    out = {}
    for key in ("X_test", "Y_test", "M_test"):
        hits = [f for f in glob.glob(os.path.join(d, "*.npy")) if key in os.path.basename(f)]
        hits.sort(key=lambda f: ("imp" not in f, len(f)))
        if hits:
            out[key] = np.load(hits[0]); out[key + "_file"] = os.path.basename(hits[0])
    if len([k for k in out if not k.endswith("_file")]) < 3:
        z = glob.glob(os.path.join(d, "*.npz"))
        if not z:
            sys.exit(f"{d}: X_test / Y_test / M_test not found")
        f = np.load(z[0], allow_pickle=True)
        for key in ("X_test", "Y_test", "M_test"):
            k = [n for n in f.files if key in n]
            if not k:
                sys.exit(f"{z[0]}: no key containing {key}; keys {f.files}")
            out[key] = f[sorted(k, key=lambda n: ("imp" not in n, len(n)))[0]]
            out[key + "_file"] = os.path.basename(z[0]) + ":" + k[0]
    return out


def tail(y, p, q):
    thr = np.percentile(y, q); m = y >= thr
    return float(np.mean(np.abs(p[m] - y[m]))), float(thr), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tail", required=True)
    ap.add_argument("--target-idx", type=int, default=0)
    a = ap.parse_args()
    T = pd.read_csv(a.tail)
    if "split" in T.columns:
        T = T[T["split"] == "test"]
    th_name = next((n for n in ("Tuned-Hybrid", "Tuned Hybrid (Graph + XGB residual)", "Tuned Hybrid")
                    if n in set(T["model"])), None)
    if th_name is None:
        sys.exit(f"Tuned Hybrid not in {a.tail}")
    for H in H_LIST:
        t = load(find_dir(a.root, H))
        X, Y, M = t["X_test"], t["Y_test"], t["M_test"]
        print(f"\nH={H}  X {X.shape} ({t['X_test_file']})  Y {Y.shape}  M {M.shape}")
        m = M.reshape(-1) > 0.5
        y = Y.reshape(-1)[m]
        last = X[:, -1]
        corr = [np.corrcoef(last[..., f].reshape(-1)[m], y)[0, 1] for f in range(X.shape[-1])]
        best = int(np.nanargmax(corr))
        print("  corr(last hour, target) per channel: " + ", ".join(f"{c:.3f}" for c in corr)
              + ("" if best == a.target_idx else f"   WARNING: channel {best} fits best, not {a.target_idx}"))
        k = a.target_idx
        pers = last[..., k].reshape(-1)[m]
        seas = (X[:, H, :, k] if H < X.shape[1] else X[:, -1, :, k]).reshape(-1)[m]
        sub = T[T["H"] == H]
        for q in (90, 95, 99):
            tp, thr, n = tail(y, pers, q)
            ts, _, _ = tail(y, seas, q)
            row = sub[(sub["model"] == th_name) & (sub["percentile"] == q)].iloc[0]
            xg = sub[(sub["model"] == "XGB") & (sub["percentile"] == q)].iloc[0]
            ok = abs(thr - row["threshold"]) < 1e-4 and n == int(row["n_tail"])
            flag = []
            if tp < row["tail_MAE"]: flag.append("PERSISTENCE BEATS TH")
            if ts < row["tail_MAE"]: flag.append("SEASONAL BEATS TH")
            print(f"  p{q}: Persistence {tp:9.3f}  Seasonal {ts:9.3f}  TH {row['tail_MAE']:9.3f}  "
                  f"XGB {xg['tail_MAE']:9.3f}   thr {thr:.3f}/{row['threshold']:.3f} n {n}/{int(row['n_tail'])} "
                  f"{'match' if ok else 'wrong tensors'}  {' '.join(flag)}")
        print(f"  overall MAE: Persistence {np.mean(np.abs(pers - y)):.3f}  Seasonal {np.mean(np.abs(seas - y)):.3f}")


if __name__ == "__main__":
    main()
