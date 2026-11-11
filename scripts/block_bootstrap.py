import argparse, glob, json, os, sys
import numpy as np


def mae(y, p, idx):
    return float(np.mean(np.abs(p[idx] - y[idx])))


def tail_mae(y, p, idx, thr):
    m = y[idx] >= thr
    if not m.any():
        return np.nan
    return float(np.mean(np.abs(p[idx][m] - y[idx][m])))


def build_blocks(z, block_hours, horizon):
    n = len(z["y_true"])
    tt = np.asarray(z["target_time"]) if "target_time" in z.files else None

    if tt is not None and tt.dtype.kind in "iu" and tt.max() > 10 ** 8:
        day = (tt // 86400).astype(np.int64)
        return day, "target_time (epoch seconds)", len(np.unique(day)), ""

    si = np.asarray(z["sample_idx"]).astype(np.int64)
    u = np.unique(si)
    span = int(u.max() - u.min() + 1)
    density = len(u) / span
    if density < 0.5:
        raise SystemExit(f"H={horizon}: sample_idx too sparse ({density:.1%})")
    day = si // int(block_hours)
    note = (f"{len(u)} distinct windows over a span of {span} "
            f"({density:.1%} dense) -> {span/block_hours:.1f} days")
    if density < 0.98:
        note += " (gaps)"
    return day, f"sample_idx // {block_hours}", len(np.unique(day)), note


def run_horizon(z, horizon, models, refs, pcts, B, block_hours, rng):
    y = np.asarray(z["y_true"], dtype=np.float64)
    preds = {k[6:]: np.asarray(z[k], dtype=np.float64)
             for k in z.files if k.startswith("pred__")}
    missing = [m for m in models if m not in preds]
    if missing:
        raise SystemExit(f"H={horizon}: predictions missing for {missing}. "
                         f"available: {sorted(preds)}")
    for ref in refs:
        if ref not in preds:
            raise SystemExit(
                f"H={horizon}: reference {ref!r} not in {sorted(preds)}")

    day, how, n_blocks, note = build_blocks(z, block_hours, horizon)
    groups = [np.flatnonzero(day == d) for d in np.unique(day)]
    n_obs = len(y)

    thr = {p: float(np.percentile(y, p)) for p in pcts}
    n_tail = {p: int((y >= thr[p]).sum()) for p in pcts}

    metrics = [("MAE", None)] + [(f"tail_MAE_p{p:g}", p) for p in pcts]
    point = {m: {} for m in models}
    for mdl in models:
        for name, p in metrics:
            full = np.arange(n_obs)
            point[mdl][name] = (mae(y, preds[mdl], full) if p is None
                                else tail_mae(y, preds[mdl], full, thr[p]))

    boot = {m: {name: np.empty(B) for name, _ in metrics} for m in models}
    nb = len(groups)
    for b in range(B):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([groups[i] for i in pick])
        for name, p in metrics:
            for mdl in models:
                boot[mdl][name][b] = (
                    mae(y, preds[mdl], idx) if p is None
                    else tail_mae(y, preds[mdl], idx, thr[p]))

    rows = []
    for mdl in models:
        for name, p in metrics:
            bs = boot[mdl][name]
            bs = bs[np.isfinite(bs)]
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs)
                      else (np.nan, np.nan))
            base = dict(H=horizon, model=mdl, metric=name,
                        threshold=(thr[p] if p else ""),
                        n_tail=(n_tail[p] if p else n_obs),
                        estimate=point[mdl][name], ci_lo=float(lo),
                        ci_hi=float(hi), n_blocks=n_blocks, B=B,
                        block_source=how)
            emitted = False
            for ref in refs:
                if mdl == ref:
                    continue

                d = boot[mdl][name] - boot[ref][name]
                d = d[np.isfinite(d)]
                if not len(d):
                    continue
                dlo, dhi = np.percentile(d, [2.5, 97.5])

                frac = min((d <= 0).mean(), (d >= 0).mean())
                row = dict(base)
                row.update(vs_ref=ref,
                           diff=point[mdl][name] - point[ref][name],
                           diff_ci_lo=float(dlo), diff_ci_hi=float(dhi),
                           diff_pct=100 * (point[mdl][name] - point[ref][name])
                           / point[ref][name],
                           p_boot=float(min(1.0, 2 * frac)))
                rows.append(row)
                emitted = True
            if not emitted:
                rows.append(base)
    return rows, note, n_blocks


def resolve_names(given, avail, flag):
    out, bad = [], []
    for raw in given:
        raw = raw.strip()
        if not raw:
            continue
        if raw in avail:
            out.append(raw)
            continue
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        if len(parts) > 1 and all(x in avail for x in parts):
            out.extend(parts)
            continue
        bad.append(raw)
    if bad or not out:
        raise SystemExit(f"{flag}: cannot resolve {bad or given!r}; available: {avail}")
    return out


def holm(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    run = 1.0
    for r in range(n - 1, -1, -1):
        i = o[r]
        run = min(run, n * p[i] / (r + 1))
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--ref", action="append", default=None)
    ap.add_argument("--models", default="")
    ap.add_argument("--pcts", default="90,95,99")
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--block-hours", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    paths = []
    for p in a.npz:
        paths += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("no predictions.npz found")

    pcts = [float(x) for x in a.pcts.split(",") if x.strip()]
    assert all(0 < x < 100 for x in pcts)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.outdir, exist_ok=True)


    print("=" * 78)
    print("DAY-LEVEL BLOCK BOOTSTRAP")
    print("=" * 78)
    print(f"  resamples B        : {a.b}")
    print(f"  block length       : {a.block_hours} h")
    _probe = np.load(paths[0], allow_pickle=False)
    avail = sorted(k[6:] for k in _probe.files if k.startswith("pred__"))
    refs = resolve_names(a.ref or ["XGB"], avail, "--ref")
    assert len(set(refs)) == len(refs)
    print(f"  reference model(s) : {' | '.join(refs)}")
    print(f"  tail percentiles   : {pcts}  (full-sample thresholds)")

    all_rows = []
    for path in paths:
        H = os.path.basename(os.path.dirname(path)).lstrip("Hh")
        try:
            H = int(H)
        except ValueError:
            H = os.path.dirname(path)
        z = np.load(path)
        here = sorted(k[6:] for k in z.files if k.startswith("pred__"))
        models = (resolve_names(a.models.split("\u0000"), here, "--models")
                  if a.models else here)
        rows, note, nb = run_horizon(z, H, models, refs, pcts, a.b,
                                     a.block_hours, rng)
        print(f"\n  H={H}: {len(models)} models, {nb} day blocks"
              f"{' - ' + note if note else ''}")
        all_rows += rows

    comp = [r for r in all_rows if "p_boot" in r]
    if comp:
        for ref in refs:
            fam = [r for r in comp if r["vs_ref"] == ref]
            if not fam:
                continue
            ps = [r["p_boot"] for r in fam]
            for r, h, b_ in zip(fam, holm(ps), bh(ps)):
                r["p_holm"] = float(h)
                r["p_bh"] = float(b_)
            print(f"\n  vs {ref}: Holm/BH correction vs {len(fam)} paired "
                  f"comparisons")

    cols = ["H", "model", "metric", "threshold", "n_tail", "estimate",
            "ci_lo", "ci_hi", "vs_ref", "diff", "diff_pct", "diff_ci_lo",
            "diff_ci_hi", "p_boot", "p_holm", "p_bh", "n_blocks", "B",
            "block_source"]
    out = os.path.join(a.outdir, "block_bootstrap.csv")
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(
                ("" if r.get(c, "") == "" else
                 (f"{r[c]:.10g}" if isinstance(r.get(c), float) else str(r[c])))
                for c in cols) + "\n")
    print(f"\nsaved {out}  ({len(all_rows)} rows)")

    sig = [r for r in comp if r.get("p_holm", 1) < 0.05]
    print(f"\n  {len(sig)} of {len(comp)} paired comparisons remain significant "
          f"at Holm-adjusted alpha = 0.05")
    print("\n  significant, by effect size:")
    for r in sorted(sig, key=lambda r: r["diff_pct"])[:20]:
        print(f"    H={r['H']:<3} {r['metric']:<14} {r['model'][:38]:<38} "
              f"{r['diff_pct']:+7.2f}%  "
              f"[{100*r['diff_ci_lo']/r['estimate']:+6.2f},"
              f"{100*r['diff_ci_hi']/r['estimate']:+6.2f}]%  "
              f"p_holm={r['p_holm']:.1e}")
    ns = [r for r in comp if r.get("p_holm", 1) >= 0.05]
    if ns:
        print(f"\n  NOT significant after correction ({len(ns)}):")
        for r in sorted(ns, key=lambda r: abs(r["diff_pct"]))[:20]:
            print(f"    H={r['H']:<3} {r['metric']:<14} {r['model'][:38]:<38} "
                  f"{r['diff_pct']:+7.2f}%  p_holm={r['p_holm']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
