import argparse, glob, os, sys
import numpy as np

_NB_MFB = globals().get("compute_mfb_nmse")
_MFB_FORMULA = ("notebook compute_mfb_nmse()" if _NB_MFB is not None else
                "MFB = mean( 2*(P-O)/(P+O) ) over the tail, pairs with P+O<=0 "
                "dropped")


def mfb(obs, pred):
    if _NB_MFB is not None:
        v = _NB_MFB(obs, pred)
        return float(v[0] if isinstance(v, (tuple, list)) else v)
    o = np.asarray(obs, np.float64)
    p = np.asarray(pred, np.float64)
    s = o + p
    m = s > 0
    if not m.any():
        return float("nan")
    return float(np.mean(2.0 * (p[m] - o[m]) / s[m]))


def pareto_front(points):
    pts = np.asarray(points, np.float64)
    keep = []
    for i, a in enumerate(pts):
        dominated = False
        for j, b in enumerate(pts):
            if j == i:
                continue
            if (b <= a).all() and (b < a).any():
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: pts[i][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("pattern")
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--lam", default="0,1,2,5,10,20,50,100")
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit(f"no files matched {a.pattern!r}")
    lams = [float(x) for x in a.lam.split(",") if x.strip()]
    os.makedirs(a.outdir, exist_ok=True)

    print("=" * 84)
    print("MAE vs |MFB| (tail)")
    print("=" * 84)
    print(f"  MFB convention : {_MFB_FORMULA}")
    print(f"  tail           : y >= p{a.pct:g} of the full test sample")

    rows = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        y = np.asarray(z["y_true"], np.float64).reshape(-1)
        H = os.path.basename(os.path.dirname(f))
        thr = float(np.percentile(y, a.pct))
        tail = y >= thr
        for k in (k for k in z.files if k.startswith("pred__")):
            p = np.asarray(z[k], np.float64).reshape(-1)
            if p.size != y.size:
                raise SystemExit(f"{f}: {k} length mismatch")
            rows.append(dict(
                horizon=H, model=k[6:],
                MAE=float(np.mean(np.abs(y - p))),
                tail_MAE=float(np.mean(np.abs(y[tail] - p[tail]))),
                MFB_tail=mfb(y[tail], p[tail]),
                abs_MFB_tail=abs(mfb(y[tail], p[tail])),
                threshold=thr, n_tail=int(tail.sum())))

    import pandas as pd
    df = pd.DataFrame(rows)
    df["pareto"] = False
    for H, g in df.groupby("horizon"):
        idx = g.index.to_numpy()
        front = pareto_front(g[["MAE", "abs_MFB_tail"]].to_numpy())
        df.loc[idx[front], "pareto"] = True
    for lam in lams:
        col = f"obj_lam{lam:g}"
        df[col] = df["MAE"] + lam * df["abs_MFB_tail"]
        df[f"rank_lam{lam:g}"] = (df.groupby("horizon")[col]
                                  .rank(method="min").astype(int))

    p = os.path.join(a.outdir, "objective_tradeoff.csv")
    df.to_csv(p, index=False)

    for H, g in df.groupby("horizon", sort=False):
        print(f"\n{'-' * 84}\n{H}   tail threshold {g.threshold.iloc[0]:.4f}, "
              f"n_tail {g.n_tail.iloc[0]}\n{'-' * 84}")
        print(f"  {'model':34s} {'MAE':>8s} {'|MFB99|':>8s} {'front':>6s} "
              f"{'rank@0':>7s} {'rank@10':>8s} {'rank@100':>9s}")
        for r in g.sort_values("MAE").itertuples(index=False):
            print(f"  {r.model:34s} {r.MAE:8.4f} {r.abs_MFB_tail:8.4f} "
                  f"{'  *' if r.pareto else '   ':>6s} "
                  f"{getattr(r, 'rank_lam0'):7d} {getattr(r, 'rank_lam10'):8d} "
                  f"{getattr(r, 'rank_lam100'):9d}")

    print("\n" + "=" * 84)
    print("RANKING vs lambda")
    print("=" * 84)
    base = "rank_lam10"
    for lam in lams:
        col = f"rank_lam{lam:g}"
        if col == base:
            continue
        moved = int((df[col] != df[base]).sum())
        win_changed = int(sum(
            g.loc[g[col].idxmin(), "model"] != g.loc[g[base].idxmin(), "model"]
            for _, g in df.groupby("horizon")))
        print(f"  lambda={lam:<6g} models whose rank differs from lambda=10: "
              f"{moved:3d} of {len(df)}   horizons whose BEST model changes: "
              f"{win_changed} of {df.horizon.nunique()}")

    print(f"\nsaved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
