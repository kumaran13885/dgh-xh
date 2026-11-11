import argparse, glob, json, os, sys
import numpy as np


def conformal_q(scores, alpha):
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1.0) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(s[k - 1])


def _strata_edges(v, n_strata):
    qs = np.linspace(0, 1, n_strata + 1)[1:-1]
    e = np.unique(np.quantile(np.asarray(v, dtype=np.float64), qs))
    return e


def _assign(v, edges):
    return np.searchsorted(edges, np.asarray(v, dtype=np.float64), side="right")

def run_cell(y_cal, p_cal, y_ev, p_ev, alpha, n_strata):
    out = {}
    res_cal = np.abs(y_cal - p_cal)


    q = conformal_q(res_cal, alpha)
    out["marginal"] = dict(lo=p_ev - q, hi=p_ev + q,
                           q_note=f"global q={q:.4f}")

    edges = _strata_edges(p_cal, n_strata)
    sc = _assign(p_cal, edges)
    se = _assign(p_ev, edges)
    qs = {}
    for g in range(len(edges) + 1):
        m = sc == g
        qs[g] = conformal_q(res_cal[m], alpha) if m.sum() else np.inf
    qv = np.array([qs[g] for g in se], dtype=np.float64)
    out["mondrian"] = dict(lo=p_ev - qv, hi=p_ev + qv,
                           q_note="per-stratum q: " +
                                  ",".join(f"{qs[g]:.3f}" for g in sorted(qs)))

    smap = {}
    for g in range(len(edges) + 1):
        m = sc == g
        smap[g] = float(np.median(res_cal[m])) if m.sum() else np.nan
    finite = [v for v in smap.values() if np.isfinite(v) and v > 0]
    floor = (min(finite) if finite else 1.0) * 1e-3 + 1e-9
    s_cal = np.array([max(smap[g], floor) if np.isfinite(smap[g]) else floor
                      for g in sc], dtype=np.float64)
    s_ev = np.array([max(smap[g], floor) if np.isfinite(smap[g]) else floor
                     for g in se], dtype=np.float64)
    qn = conformal_q(res_cal / s_cal, alpha)
    out["normalized"] = dict(lo=p_ev - qn * s_ev, hi=p_ev + qn * s_ev,
                             q_note=f"ratio q={qn:.4f}")
    return out


def cover_stats(y, lo, hi, thr):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    d = dict(n_eval=int(y.size),
             coverage=float(inside.mean()),
             mean_width=float(np.mean(width)),
             median_width=float(np.median(width)))
    for name, t in thr.items():
        m = y >= t
        d[f"n_{name}"] = int(m.sum())
        d[f"cov_{name}"] = float(inside[m].mean()) if m.sum() else np.nan
        d[f"width_{name}"] = float(np.mean(width[m])) if m.sum() else np.nan
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("pattern")
    ap.add_argument("--alpha", default="0.1,0.05")
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--strata", type=int, default=5)
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit(f"no files matched {a.pattern!r}")
    alphas = [float(x) for x in a.alpha.split(",")]
    os.makedirs(a.outdir, exist_ok=True)
    rows, meta = [], []

    for f in files:
        z = np.load(f, allow_pickle=False)
        need = ("y_true", "target_time")
        miss = [k for k in need if k not in z.files]
        if miss:
            raise SystemExit(f"{f}: missing {miss}.")
        y = np.asarray(z["y_true"], dtype=np.float64).reshape(-1)
        t = np.asarray(z["target_time"]).reshape(-1)
        models = [k[len("pred__"):] for k in z.files if k.startswith("pred__")]
        if not models:
            raise SystemExit(f"{f}: no pred__* arrays.")
        H = os.path.basename(os.path.dirname(f))
        ut = np.unique(t)
        n_cal_t = int(np.floor(len(ut) * a.cal_frac))
        if n_cal_t < 1 or n_cal_t >= len(ut):
            raise SystemExit(f"{f}: bad cal_frac {a.cal_frac}")
        cut = ut[n_cal_t - 1]
        cal = t <= cut
        ev = ~cal
        if cal.sum() == 0 or ev.sum() == 0:
            raise SystemExit(f"{f}: empty calibration or evaluation split.")
        thr = {f"p{p}": float(np.percentile(y, p)) for p in (90, 95, 99)}
        meta.append(dict(horizon=H, file=f, n_total=int(y.size),
                         n_cal=int(cal.sum()), n_eval=int(ev.sum()),
                         n_hours_total=int(len(ut)), n_hours_cal=int(n_cal_t),
                         cal_cut_hour=float(cut), cal_frac=a.cal_frac,
                         n_strata=a.strata, **{f"thr_{k}": v
                                               for k, v in thr.items()}))
        print(f"\n{H}: {y.size} rows, {len(ut)} hours: "
              f"cal {cal.sum()} / eval {ev.sum()} (split at hour {cut:g})")

        for m in models:
            p = np.asarray(z["pred__" + m], dtype=np.float64).reshape(-1)
            if p.size != y.size:
                raise SystemExit(f"{f}: pred__{m} length {p.size} != {y.size}")
            for al in alphas:
                cells = run_cell(y[cal], p[cal], y[ev], p[ev], al, a.strata)
                for est, r in cells.items():
                    st = cover_stats(y[ev], r["lo"], r["hi"], thr)
                    rows.append(dict(horizon=H, model=m, estimator=est,
                                     alpha=al, nominal=1 - al,
                                     q_note=r["q_note"], **st))
            r90 = [x for x in rows if x["horizon"] == H and x["model"] == m
                   and x["alpha"] == alphas[0]]
            s = "  ".join(f"{x['estimator'][:4]} {x['coverage']:.3f}"
                          f"/{x['cov_p99']:.3f}" for x in r90)
            print(f"    {m:36s} cov(all/p99) @ {1-alphas[0]:.0%}:  {s}")

    import pandas as pd
    df = pd.DataFrame(rows)
    cp = os.path.join(a.outdir, "conformal_coverage.csv")
    df.to_csv(cp, index=False)
    json.dump(meta, open(os.path.join(a.outdir, "conformal_meta.json"), "w"),
              indent=1)

    print("\n" + "=" * 78)
    print("COVERAGE (median over horizons)")
    print("=" * 78)
    for al in alphas:
        print(f"\n  nominal {1-al:.0%}")
        g = (df[df.alpha == al]
             .groupby(["estimator", "model"])
             [["coverage", "cov_p90", "cov_p95", "cov_p99", "mean_width"]]
             .median().reset_index())
        print(f"    {'estimator':11s} {'model':34s} {'all':>7s} {'p90':>7s} "
              f"{'p95':>7s} {'p99':>7s} {'width':>9s}")
        for r in g.itertuples(index=False):
            print(f"    {r.estimator:11s} {r.model:34s} {r.coverage:7.3f} "
                  f"{r.cov_p90:7.3f} {r.cov_p95:7.3f} {r.cov_p99:7.3f} "
                  f"{r.mean_width:9.3f}")
    print("\nsaved " + cp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
