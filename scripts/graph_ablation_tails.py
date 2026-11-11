from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

ROOT = r"results"

PERCENTILES = [90, 95, 99]
INCLUDE_ALL_SAMPLES = True
OUT_CSV = "graph_ablation_tails.csv"
ALPHA = 0.05


try:
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon
except ImportError as _exc:
    sys.exit(f"Missing dependency: {_exc.name}")

HORIZONS = [1, 3, 6, 12, 24]

FAMILIES: dict[str, list[tuple[str, str, str]]] = {
    "graph": [
        ("Huber-Graph",       "SG-Huber", "wind-bearing gate, untuned"),
        ("Tuned-Huber-Graph", "SG-Huber", "wind-bearing gate, tuned"),
        ("Huber-Graph",       "Huber",    "any graph smoothing at all"),
        ("SG-Huber",          "Huber",    "static graph smoothing at all"),
    ],
    "stages": [
        ("Hybrid",            "Huber-Graph",       "residual corrector"),
        ("Tuned-Hybrid",      "Tuned-Huber-Graph", "residual corrector, tuned"),
        ("Tuned-Hybrid",      "Hybrid",            "JSO tuning on the hybrid"),
        ("Tuned-Huber-Graph", "Huber-Graph",       "JSO tuning on the graph"),
    ],
    "headline": [
        ("Tuned-Hybrid", "XGB",           "vs XGB"),
        ("Tuned-Hybrid", "RF",            "vs random forest"),
        ("Tuned-Hybrid", "Persistence",   "vs persistence"),
        ("Tuned-Hybrid", "SeasonalNaive", "vs seasonal naive"),
    ],
}

EXTRA_ALIASES = {
    "SG-Huber":          ["static_graph_huber", "huber_static", "static_huber"],
    "SG-Ridge":          ["static_graph_ridge"],
    "Huber-Graph":       ["graph_huber", "dynamic_graph_huber", "graph"],
    "Tuned-Huber-Graph": ["tuned_graph", "tuned_graph_huber", "tuned_huber"],
    "Hybrid":            ["hybrid_graph", "hybrid_resid"],
    "Tuned-Hybrid":      ["tuned_hybrid_graph", "tuned_hybrid_resid"],
    "Persistence":       ["persist", "naive"],
    "SeasonalNaive":     ["snaive", "seasonal_naive"],
    "XGB":               ["xgboost", "xgb_model"],
}

SEED_IN_PARENT = re.compile(r"^(?P<base>.+?)[ _-]*seed[ _-]*(?P<seed>\d+)$",
                            re.IGNORECASE)
SEED_DIR = re.compile(r"^[ _-]*seed[ _-]*(?P<seed>\d+)$", re.IGNORECASE)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def has_horizons(d: str) -> bool:
    return any(os.path.isfile(os.path.join(d, f"H{H}", "predictions.npz"))
               for H in HORIZONS)


def discover(root: str) -> dict[str, list[tuple[object, str]]]:
    out: dict[str, list[tuple[object, str]]] = defaultdict(list)
    try:
        entries = sorted(os.listdir(root))
    except OSError as exc:
        sys.exit(f"Cannot list ROOT: {exc}")

    for name in entries:
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue

        m = SEED_IN_PARENT.match(name)
        if m and has_horizons(p):
            out[m.group("base").rstrip("_- ")].append((int(m.group("seed")), p))
            continue

        subs = []
        try:
            for sub in sorted(os.listdir(p)):
                sp = os.path.join(p, sub)
                if not os.path.isdir(sp):
                    continue
                ms = SEED_DIR.match(sub)
                if ms and has_horizons(sp):
                    subs.append((int(ms.group("seed")), sp))
        except OSError:
            pass
        if subs:
            out[name].extend(subs)
            continue

        if has_horizons(p):
            out[name].append(("single", p))

    if not out and has_horizons(root):
        out[os.path.basename(os.path.abspath(root)) or "dataset"] = [("single", root)]
    return dict(out)


def diagnose(root: str) -> None:
    print("nothing found under ROOT:")
    shown = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 3:
            dirnames[:] = []
            continue
        npz = [f for f in filenames if f.endswith(".npz")]
        pkl = [f for f in filenames if f.endswith(".pkl")]
        extra = (f"   [{len(npz)} npz]" if npz else "") + \
                (f"   [pkl: {pkl[:2]}]" if pkl else "")
        print(f"  {'  ' * depth}{os.path.relpath(dirpath, root)}{extra}")
        shown += 1
        if shown > 120:
            print("  ...")
            break


def resolve(npz_keys, canonical: str) -> str | None:
    preds = {k[len("pred__"):]: k for k in npz_keys if k.startswith("pred__")}
    by_norm = {norm(k): v for k, v in preds.items()}
    for cand in [canonical] + EXTRA_ALIASES.get(canonical, []):
        hit = by_norm.get(norm(cand))
        if hit:
            return hit
    return None


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    return float((r[d < 0].sum() - r[d > 0].sum()) / r.sum())


def tail_errors(y_true, p_var, p_ref, pct: float):
    y_true = np.asarray(y_true, float).ravel()
    p_var = np.asarray(p_var, float).ravel()
    p_ref = np.asarray(p_ref, float).ravel()
    n = min(y_true.size, p_var.size, p_ref.size)
    y_true, p_var, p_ref = y_true[:n], p_var[:n], p_ref[:n]
    ok = np.isfinite(y_true) & np.isfinite(p_var) & np.isfinite(p_ref)
    y_true, p_var, p_ref = y_true[ok], p_var[ok], p_ref[ok]
    if y_true.size < 20:
        return None
    mask = (np.ones_like(y_true, bool) if pct <= 0
            else y_true >= np.percentile(y_true, pct))
    if mask.sum() < 10:
        return None
    return (np.abs(p_var[mask] - y_true[mask]),
            np.abs(p_ref[mask] - y_true[mask]))


def signed_wilcoxon(d: np.ndarray) -> float:
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="less", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm_bh(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = p.size
    order = np.argsort(p)
    ps = p[order]
    holm = np.clip(np.maximum.accumulate((m - np.arange(m)) * ps), 0, 1)
    bh = ps * m / (np.arange(m) + 1)
    bh = np.clip(np.minimum.accumulate(bh[::-1])[::-1], 0, 1)
    out_h = np.empty(m)
    out_b = np.empty(m)
    out_h[order] = holm
    out_b[order] = bh
    return out_h, out_b

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--percentiles", nargs="+", type=float, default=PERCENTILES)
    ap.add_argument("--all-samples", dest="all_samples", action="store_true",
                    default=INCLUDE_ALL_SAMPLES)
    ap.add_argument("--no-all-samples", dest="all_samples", action="store_false")
    ap.add_argument("--only-families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        print(f"not a directory: {root}")
        return 2

    pcts = ([0.0] if args.all_samples else []) + list(args.percentiles)
    comparisons = [(f, v, r, w) for f in args.only_families
                   for v, r, w in FAMILIES[f]]

    datasets = discover(root)
    if not datasets:
        diagnose(root)
        return 2

    print("=" * 80)
    print("TAIL ABLATION")
    print("=" * 80)
    print(f"root : {root}")
    for ds, runs in sorted(datasets.items()):
        seeds = [s for s, _ in runs]
        print(f"  {ds:34s} {'single run' if seeds == ['single'] else f'seeds {sorted(seeds)}'}")
    print(f"tails: {['all' if p <= 0 else f'p{int(p)}' for p in pcts]}")
    print(f"families: {args.only_families}")
    print()

    store: dict[tuple, list] = defaultdict(list)
    missing: set[tuple[str, str]] = set()
    n_files = 0

    for ds, runs in sorted(datasets.items()):
        for seed, run_dir in sorted(runs, key=lambda t: str(t[0])):
            for H in HORIZONS:
                path = os.path.join(run_dir, f"H{H}", "predictions.npz")
                if not os.path.isfile(path):
                    continue
                try:
                    z = np.load(path, allow_pickle=False)
                except Exception as exc:
                    print(f"  unreadable {path}: {exc}")
                    continue
                if "y_true" not in z.files:
                    print(f"  no y_true in {path}")
                    continue
                n_files += 1
                y = z["y_true"]
                for fam, var, ref, _why in comparisons:
                    kv, kr = resolve(z.files, var), resolve(z.files, ref)
                    if kv is None:
                        missing.add((ds, var))
                    if kr is None:
                        missing.add((ds, ref))
                    if kv is None or kr is None:
                        continue
                    for pct in pcts:
                        res = tail_errors(y, z[kv], z[kr], pct)
                        if res is None:
                            continue
                        store[(fam, f"{var} vs {ref}", ds, H, pct)].append(
                            (seed, res[0], res[1]))

    print(f"read {n_files} predictions.npz files")
    if not store:
        print("\nno model pairs found")
        for d, m in sorted(missing):
            print(f"  {d}: {m} not present")
        return 3

    rows = []
    for (fam, label, ds, H, pct), chunks in store.items():
        n_seeds = len(chunks)
        per_seed_delta = [ev.mean() - er.mean() for _s, ev, er in chunks]
        frac_better = float(np.mean([x < 0 for x in per_seed_delta]))
        seen: set[tuple[bytes, bytes]] = set()
        uniq = []
        for s, ev, er in chunks:
            sig_key = (ev.tobytes(), er.tobytes())
            if sig_key in seen:
                continue
            seen.add(sig_key)
            uniq.append((s, ev, er))
        n_unique = len(uniq)
        identical = n_seeds > 1 and n_unique == 1

        e_var = np.concatenate([c[1] for c in uniq])
        e_ref = np.concatenate([c[2] for c in uniq])
        d = e_var - e_ref
        mae_var, mae_ref = float(e_var.mean()), float(e_ref.mean())

        rows.append(dict(
            family=fam, comparison=label, dataset=ds, H=H,
            tail=("all" if pct <= 0 else f"p{int(pct)}"),
            MAE_ref=round(mae_ref, 4), MAE_var=round(mae_var, 4),
            delta_pct=round(100.0 * (mae_var - mae_ref) / mae_ref, 2)
            if mae_ref > 0 else float("nan"),
            win_fraction=round(float((d < 0).mean()), 3),
            rbc=round(rank_biserial(d), 3),
            p_raw=signed_wilcoxon(d),
            n=int(d.size), n_seeds=n_seeds, n_unique_seeds=n_unique,
            frac_seeds_better=round(frac_better, 2),
            seed_identical=bool(identical),
        ))

    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    df["q_bh"] = np.nan
    df["bonf_thresh"] = np.nan
    for label, idx in df.groupby("comparison").groups.items():
        idx = list(idx)
        p = df.loc[idx, "p_raw"].to_numpy()
        holm, bh = holm_bh(p)
        df.loc[idx, "p_holm"] = holm
        df.loc[idx, "q_bh"] = bh
        df.loc[idx, "bonf_thresh"] = args.alpha / len(idx)
    df["sig"] = df["p_raw"] < df["bonf_thresh"]
    df["sig_holm"] = df["p_holm"] < args.alpha

    df = df.sort_values(["family", "comparison", "dataset", "H", "tail"])
    out_path = os.path.abspath(args.out)
    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 3000)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    cols = ["dataset", "H", "tail", "MAE_ref", "MAE_var", "delta_pct",
            "win_fraction", "rbc", "p_raw", "sig", "frac_seeds_better",
            "n", "n_seeds", "n_unique_seeds"]

    for fam in args.only_families:
        for var, ref, why in FAMILIES[fam]:
            label = f"{var} vs {ref}"
            sub = df[df["comparison"] == label]
            if sub.empty:
                continue
            print()
            print("=" * 80)
            print(f"[{fam}] {label}    -- {why}")
            print("=" * 80)
            print(sub[cols].to_string(index=False))
            b = int(((sub["delta_pct"] < 0) & sub["sig"]).sum())
            w = int(((sub["delta_pct"] > 0) & sub["sig"]).sum())
            t = sub[sub["tail"] != "all"]
            bt = int(((t["delta_pct"] < 0) & t["sig"]).sum())
            print(f"  ALL ROWS : better {b}/{len(sub)}, worse {w}")
            print(f"  TAILS    : better {bt}/{len(t)}, "
                  f"median delta {t['delta_pct'].median():+.2f}%, "
                  f"mean rbc {t['rbc'].mean():+.3f}")

    print()
    print("=" * 80)
    print("SUMMARY (p90/p95/p99)")
    print("=" * 80)
    print(f"{'comparison':40s} {'better':>7s} {'worse':>6s} {'n':>5s} "
          f"{'med%':>8s} {'meanRBC':>8s} {'allseeds':>9s}")
    for fam in args.only_families:
        for var, ref, _w in FAMILIES[fam]:
            label = f"{var} vs {ref}"
            t = df[(df["comparison"] == label) & (df["tail"] != "all")]
            if t.empty:
                continue
            b = int(((t["delta_pct"] < 0) & t["sig"]).sum())
            w = int(((t["delta_pct"] > 0) & t["sig"]).sum())
            allseeds = int((t["frac_seeds_better"] == 1.0).sum())
            print(f"{label:40s} {b:7d} {w:6d} {len(t):5d} "
                  f"{t['delta_pct'].median():+8.2f} {t['rbc'].mean():+8.3f} "
                  f"{allseeds:5d}/{len(t):<3d}")

    ident = df[df["seed_identical"]]
    if len(ident):
        print()
        print("-" * 80)
        print("ZERO SEED VARIANCE")
        print("-" * 80)
        for label, grp in ident.groupby("comparison"):
            print(f"  {label:40s} {len(grp)} cells, datasets: "
                  f"{sorted(grp['dataset'].unique())}")

        multi = set(df[df["n_seeds"] > 1]["dataset"].unique())
        for label in sorted(ident["comparison"].unique()):
            same = sorted(ident[ident["comparison"] == label]["dataset"].unique())
            allds = sorted(d for d in df[df["comparison"] == label]["dataset"].unique()
                           if d in multi)
            diff = [d for d in allds if d not in same]
            if diff:
                print(f"    {label}: identical on {same}, varies on {diff}")

    if missing:
        print()
        print("models not found:")
        for d, m in sorted(missing):
            print(f"  {d}: {m}")

    print()
    print(f"wrote {out_path}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
