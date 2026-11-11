import argparse, glob, json, os, sys
import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception as e:
    raise SystemExit(f"scikit-learn is required: {e}")

def contingency(y_true_bin, y_hat_bin):
    tp = int(np.sum(y_true_bin & y_hat_bin))
    fp = int(np.sum(~y_true_bin & y_hat_bin))
    fn = int(np.sum(y_true_bin & ~y_hat_bin))
    tn = int(np.sum(~y_true_bin & ~y_hat_bin))
    return tp, fp, fn, tn


def _safe(num, den):
    return float(num) / float(den) if den else float("nan")


def scores_from_counts(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    pod = _safe(tp, tp + fn)
    far = _safe(fp, tp + fp)
    pofd = _safe(fp, fp + tn)
    prec = _safe(tp, tp + fp)
    spec = _safe(tn, tn + fp)
    csi = _safe(tp, tp + fp + fn)
    f1 = _safe(2 * tp, 2 * tp + fp + fn)
    f2 = _safe(5 * tp, 5 * tp + fp + 4 * fn)
    bias = _safe(tp + fp, tp + fn)

    exp_c = (_safe((tp + fn) * (tp + fp), n) + _safe((tn + fp) * (tn + fn), n)
             if n else float("nan"))
    hss = _safe((tp + tn) - exp_c, n - exp_c) if n and not np.isnan(exp_c) \
        else float("nan")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, POD_recall=pod, precision=prec,
                FAR_ratio=far, POFD_rate=pofd, specificity=spec, CSI=csi,
                F1=f1, F2=f2, HSS=hss, freq_bias=bias)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))

def pick_threshold(score_cal, pos_cal, criterion, target_recall):

    u = np.unique(score_cal)
    if u.size == 0:
        return float("nan")
    cands = np.concatenate(([u[0] - 1.0], (u[:-1] + u[1:]) / 2.0,
                            [u[-1] + 1.0])) if u.size > 1 \
        else np.array([u[0] - 1.0, u[0] + 1.0])
    best, best_t = -np.inf, float("nan")
    for t in cands:
        tp, fp, fn, tn = contingency(pos_cal, score_cal >= t)
        s = scores_from_counts(tp, fp, fn, tn)
        if criterion == "F1":
            v = s["F1"]
        elif criterion == "F2":
            v = s["F2"]
        elif criterion == "CSI":
            v = s["CSI"]
        elif criterion == "recall_target":
            v = (-s["POFD_rate"] if s["POD_recall"] >= target_recall
                 else -1e6 + s["POD_recall"])
        else:
            raise ValueError(criterion)
        if np.isnan(v):
            continue
        if v > best or (v == best and t > best_t):
            best, best_t = v, t
    return float(best_t)


def calibrate(score_cal, pos_cal, score_ev):
    if pos_cal.sum() == 0 or pos_cal.sum() == pos_cal.size:
        return np.full(score_ev.shape, float(pos_cal.mean()))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(score_cal, float), pos_cal.astype(float))
    return np.clip(iso.predict(np.asarray(score_ev, float)), 0.0, 1.0)


def reliability(p, pos, n_bins=10):
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    edges = np.array_split(order, min(n_bins, n))
    out = []
    for i, ix in enumerate(edges):
        if ix.size == 0:
            continue
        lo, hi = wilson(int(pos[ix].sum()), int(ix.size))
        out.append(dict(bin=i, n=int(ix.size),
                        mean_pred=float(p[ix].mean()),
                        observed=float(pos[ix].mean()),
                        obs_ci_lo=lo, obs_ci_hi=hi,
                        p_lo=float(p[ix].min()), p_hi=float(p[ix].max())))
    return out


def pr_curve(score, pos, max_pts=200):
    o = np.argsort(-score, kind="mergesort")
    pos_s = pos[o].astype(np.int64)
    tp = np.cumsum(pos_s)
    fp = np.cumsum(1 - pos_s)
    P = int(pos.sum())
    if P == 0:
        return []
    rec = tp / P
    prec = tp / np.maximum(tp + fp, 1)
    keep = np.unique(np.linspace(0, len(rec) - 1, min(max_pts, len(rec))
                                 ).astype(int))
    return [dict(recall=float(rec[i]), precision=float(prec[i]),
                 score=float(score[o][i]), n_alarms=int(i + 1))
            for i in keep]

def run_cell(y_cal, s_cal, y_ev, s_ev, thr, criteria, target_recall, n_bins):
    pos_cal = y_cal >= thr
    pos_ev = y_ev >= thr
    out = {"n_cal": int(y_cal.size), "n_ev": int(y_ev.size),
           "n_pos_cal": int(pos_cal.sum()), "n_pos_ev": int(pos_ev.sum()),
           "base_rate_cal": float(pos_cal.mean()),
           "base_rate_ev": float(pos_ev.mean())}

    if 0 < pos_ev.sum() < pos_ev.size:
        out["ROC_AUC"] = float(roc_auc_score(pos_ev, s_ev))
        out["avg_precision"] = float(average_precision_score(pos_ev, s_ev))
        out["AP_skill"] = float((out["avg_precision"] - out["base_rate_ev"])
                                / (1 - out["base_rate_ev"]))
    else:
        out["ROC_AUC"] = out["avg_precision"] = out["AP_skill"] = float("nan")

    p_ev = calibrate(s_cal, pos_cal, s_ev)
    brier = float(np.mean((p_ev - pos_ev.astype(float)) ** 2))

    ref = float(np.mean((pos_cal.mean() - pos_ev.astype(float)) ** 2))
    out["Brier"] = brier
    out["Brier_ref_climatology"] = ref
    out["Brier_skill"] = float(1 - brier / ref) if ref > 0 else float("nan")

    rng = s_cal.max() - s_cal.min()
    raw_p = (np.clip((s_ev - s_cal.min()) / rng, 0, 1) if rng > 0
             else np.zeros_like(s_ev))
    out["Brier_uncalibrated"] = float(np.mean((raw_p - pos_ev.astype(float))**2))

    ops = []
    for crit in criteria:
        t = pick_threshold(s_cal, pos_cal, crit, target_recall)
        tp, fp, fn, tn = contingency(pos_ev, s_ev >= t)
        m = scores_from_counts(tp, fp, fn, tn)
        rlo, rhi = wilson(tp, tp + fn)
        plo, phi = wilson(tp, tp + fp)
        m.update(criterion=crit, decision_threshold=t,
                 recall_ci_lo=rlo, recall_ci_hi=rhi,
                 precision_ci_lo=plo, precision_ci_hi=phi)
        ops.append(m)
    return out, ops, p_ev, pos_ev

def rolling_mean_by_station(v, station, tt, window):

    v = np.asarray(v, np.float64)
    station = np.asarray(station)
    tt = np.asarray(tt, np.int64)
    out = np.full(v.shape, np.nan)
    ok = np.zeros(v.shape, bool)
    for st in np.unique(station):
        ix = np.flatnonzero(station == st)
        ix = ix[np.argsort(tt[ix], kind="mergesort")]
        t, x = tt[ix], v[ix]
        if len(t) < window:
            continue
        csum = np.concatenate(([0.0], np.cumsum(x)))

        span_ok = (t[window - 1:] - t[:len(t) - window + 1]) == (window - 1)
        means = (csum[window:] - csum[:-window]) / window
        pos = ix[window - 1:]
        out[pos[span_ok]] = means[span_ok]
        ok[pos[span_ok]] = True
    return out, ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("pattern")
    ap.add_argument("--pcts", default="90,95,99")
    ap.add_argument("--abs", default="")
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--target-recall", type=float, default=0.9)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--window", type=int, default=1)
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit(f"no files matched {a.pattern!r}")
    pcts = [float(x) for x in a.pcts.split(",") if x.strip()]
    absl = [float(x) for x in a.abs.split(",") if x.strip()]
    crits = ["F1", "F2", "CSI", "recall_target"]
    os.makedirs(a.outdir, exist_ok=True)
    main_rows, ops_rows, rel_rows, pr_rows, meta = [], [], [], [], []

    for f in files:
        z = np.load(f, allow_pickle=False)
        for k in ("y_true", "target_time"):
            if k not in z.files:
                raise SystemExit(f"{f}: missing {k}.")
        y = np.asarray(z["y_true"], np.float64).reshape(-1)
        t = np.asarray(z["target_time"]).reshape(-1)
        models = [k[6:] for k in z.files if k.startswith("pred__")]
        if not models:
            raise SystemExit(f"{f}: no pred__* arrays.")
        H = os.path.basename(os.path.dirname(f))

        preds = {m: np.asarray(z["pred__" + m], np.float64).reshape(-1)
                 for m in models}
        for m, v in preds.items():
            if v.size != y.size:
                raise SystemExit(f"{f}: pred__{m} length mismatch")
        if a.window > 1:
            if "station_idx" not in z.files:
                raise SystemExit(f"{f}: --window needs station_idx. "
                                 f"Re-run regen_cell.py.")
            st = np.asarray(z["station_idx"]).reshape(-1)
            ya, ok = rolling_mean_by_station(y, st, t, a.window)
            keep = ok
            for m in models:
                pa, ok_m = rolling_mean_by_station(preds[m], st, t, a.window)
                preds[m] = pa
                keep = keep & ok_m
            print(f"\n  --window {a.window}: {int(keep.sum())} of {y.size} "
                  f"positions have a complete {a.window}-h window")
            if keep.sum() < 100:
                raise SystemExit(f"{f}: only {int(keep.sum())} complete "
                                 f"{a.window}-h windows; refusing to report.")
            y, t = ya[keep], t[keep]
            preds = {m: v[keep] for m, v in preds.items()}

        ut = np.unique(t)
        n_cal_t = int(np.floor(len(ut) * a.cal_frac))
        if not 0 < n_cal_t < len(ut):
            raise SystemExit(f"{f}: bad cal_frac")
        cut = ut[n_cal_t - 1]
        cal, ev = t <= cut, t > cut
        thrs = [(f"p{p:g}", float(np.percentile(y, p))) for p in pcts] + \
               [(f"abs{v:g}", float(v)) for v in absl]
        meta.append(dict(horizon=H, file=f, window_hours=a.window,
                         n_total=int(y.size),
                         n_cal=int(cal.sum()), n_eval=int(ev.sum()),
                         n_hours=int(len(ut)), cal_cut_hour=float(cut),
                         cal_frac=a.cal_frac, target_recall=a.target_recall,
                         thresholds={k: v for k, v in thrs}))
        print(f"\n{H}: cal {cal.sum()} / eval {ev.sum()} hours-split at {cut:g}")

        for tname, thr in thrs:
            print(f"  --- event: y >= {thr:.4f} ({tname}), "
                  f"{int((y[ev] >= thr).sum())} positives in eval ---")
            print(f"    {'model':34s} {'AUC':>6s} {'AP':>6s} {'Brier':>7s} "
                  f"{'BSS':>6s} | F2-point: {'recall':>7s} {'prec':>6s} "
                  f"{'POFD':>6s} {'CSI':>6s}")
            for m in models:
                s = preds[m]
                out, ops, p_ev, pos_ev = run_cell(
                    y[cal], s[cal], y[ev], s[ev], thr, crits,
                    a.target_recall, a.bins)
                base = dict(horizon=H, model=m, threshold_name=tname,
                            threshold=thr)
                main_rows.append(dict(base, **out))
                for o in ops:
                    ops_rows.append(dict(base, **o))
                for b in reliability(p_ev, pos_ev, a.bins):
                    rel_rows.append(dict(base, **b))
                for c in pr_curve(s[ev], pos_ev):
                    pr_rows.append(dict(base, **c))
                f2 = [o for o in ops if o["criterion"] == "F2"][0]
                print(f"    {m:34s} {out['ROC_AUC']:6.3f} "
                      f"{out['avg_precision']:6.3f} {out['Brier']:7.5f} "
                      f"{out['Brier_skill']:+6.3f} | "
                      f"{f2['POD_recall']:7.3f} {f2['precision']:6.3f} "
                      f"{f2['POFD_rate']:6.4f} {f2['CSI']:6.3f}")

    import pandas as pd
    paths = {}
    for name, rows in (("exceedance_summary", main_rows),
                       ("exceedance_operating_points", ops_rows),
                       ("exceedance_reliability", rel_rows),
                       ("exceedance_pr_curves", pr_rows)):
        p = os.path.join(a.outdir, name + ".csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        paths[name] = p
    json.dump(meta, open(os.path.join(a.outdir, "exceedance_meta.json"), "w"),
              indent=1, default=str)

    df = pd.DataFrame(main_rows)
    print("\n" + "=" * 86)
    print("median over horizons")
    print("=" * 86)
    for tname in df.threshold_name.unique():
        g = (df[df.threshold_name == tname]
             .groupby("model")[["ROC_AUC", "avg_precision", "AP_skill",
                                "Brier", "Brier_skill", "Brier_uncalibrated"]]
             .median().sort_values("avg_precision", ascending=False))
        print(f"\n  event {tname}  (base rate "
              f"{df[df.threshold_name==tname].base_rate_ev.median():.4f})")
        print(f"    {'model':34s} {'AUC':>6s} {'AP':>7s} {'APskill':>8s} "
              f"{'Brier':>8s} {'BSS':>7s} {'Brier_raw':>10s}")
        for m, r in g.iterrows():
            print(f"    {m:34s} {r.ROC_AUC:6.3f} {r.avg_precision:7.4f} "
                  f"{r.AP_skill:+8.3f} {r.Brier:8.5f} {r.Brier_skill:+7.3f} "
                  f"{r.Brier_uncalibrated:10.5f}")

    print("\nsaved " + "\nsaved ".join(paths.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
