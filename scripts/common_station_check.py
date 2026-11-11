import argparse, os, sys
import numpy as np

H_LIST = [1, 3, 6, 12, 24]
PAIRS = [("Tuned Hybrid", "XGB"), ("Huber Graph", "Static-Graph Huber"),
         ("Huber Graph", "CONTROL Shuffled-Bearing Graph")]


def tail_mae(y, p, q):
    thr = np.percentile(y, q)
    m = y >= thr
    return float(np.mean(np.abs(p[m] - y[m]))), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--exclude", type=int, default=3)
    a = ap.parse_args()
    for H in H_LIST:
        f = os.path.join(a.root, f"H{H}", "predictions.npz")
        if not os.path.exists(f):
            sys.exit(f"missing {f}")
        d = np.load(f, allow_pickle=True)
        y, st = d["y_true"], d["station_idx"]
        counts = {int(s): int((st == s).sum()) for s in np.unique(st)}
        keep = st != a.exclude
        print(f"\nH={H}  stations in test {counts}  excluded slot {a.exclude} "
              f"({(~keep).sum()} samples)  kept n={keep.sum()}")
        for scope, k in (("full", np.ones_like(keep)), ("common6", keep)):
            yy = y[k]
            for new, ref in PAIRS:
                pn, pr = d[f"pred__{new}"][k], d[f"pred__{ref}"][k]
                out = [f"MAE {100*(np.mean(np.abs(pn-yy))-np.mean(np.abs(pr-yy)))/np.mean(np.abs(pr-yy)):+6.2f}%"]
                for q in (90, 95, 99):
                    tn, n = tail_mae(yy, pn, q); tr, _ = tail_mae(yy, pr, q)
                    out.append(f"p{q} {100*(tn-tr)/tr:+6.2f}%")
                print(f"  {scope:8s} {new} vs {ref}: " + "  ".join(out))
            tn, n = tail_mae(yy, d["pred__Tuned Hybrid"][k], 99)
            tx, _ = tail_mae(yy, d["pred__XGB"][k], 99)
            print(f"  {scope:8s} p99 abs: Tuned Hybrid {tn:.2f}  XGB {tx:.2f}  (n99={n}, thr={np.percentile(yy,99):.2f})")


if __name__ == "__main__":
    main()
