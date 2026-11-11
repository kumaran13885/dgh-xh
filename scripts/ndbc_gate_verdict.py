import argparse, os, sys
import numpy as np
import pandas as pd

H_LIST = [1, 3, 6, 12, 24]
PCTS = [90, 95, 99]


def load_tail(d, name, model):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    t = pd.read_csv(p)
    if "split" in t.columns:
        t = t[t["split"] == "test"]
    t = t[t["model"] == model]
    if t.empty:
        sys.exit(f"{p}: no rows for model {model!r}. Models present: "
                 f"{sorted(pd.read_csv(p)['model'].unique())}")
    s = t.set_index(["H", "percentile"])["tail_MAE"]
    missing = [(h, q) for h in H_LIST for q in PCTS if (h, q) not in s.index]
    if missing:
        sys.exit(f"{p}: missing cells {missing}")
    return s.loc[[(h, q) for h in H_LIST for q in PCTS]].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--shuffled", nargs=5, required=True)
    ap.add_argument("--old-shuffled", default=None)
    ap.add_argument("--mexico", action="store_true")
    ap.add_argument("--tail-name", default="results_tail.csv")
    ap.add_argument("--w-model", default="Huber-Graph")
    ap.add_argument("--d-model", default="SG-Huber")
    a = ap.parse_args()

    W = load_tail(a.fixed, a.tail_name, a.w_model)
    D = load_tail(a.fixed, a.tail_name, a.d_model)
    P = np.vstack([load_tail(d, a.tail_name, a.w_model) for d in a.shuffled])
    seeds = [12345, 1001, 1002, 1003, 1004]

    if a.old_shuffled:
        old = load_tail(a.old_shuffled, a.tail_name, a.w_model)
        dmax = float(np.max(np.abs(old - P[0])))
        print(f"seed 12345 new vs old, max |diff| = {dmax:.2e}  "
              f"{'PASS' if dmax < 1e-6 else 'FAIL'}")

    for k, s in enumerate(seeds):
        if np.allclose(P[k], W, rtol=0, atol=1e-9):
            sys.exit(f"seed {s}: shuffled run equals W")
    if len({tuple(np.round(r, 9)) for r in P}) < 5:
        print("WARNING: two permutation runs are identical")

    cells = [f"H{h}/p{q}" for h in H_LIST for q in PCTS]
    rel_D = (W - D) / D
    Pmed = np.median(P, axis=0)
    rel_P = (W - Pmed) / Pmed
    tab = pd.DataFrame({"cell": cells, "W": W, "D": D, "P_median": Pmed,
                        "W_vs_D_%": 100 * rel_D, "W_vs_Pmed_%": 100 * rel_P})
    for k, s in enumerate(seeds):
        tab[f"P{s}"] = P[k]
    pd.set_option("display.width", 200)
    print(f"\n=== {a.label}: tail MAE per cell ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    nD, medD = int((W < D).sum()), float(np.median(rel_D))
    nP, medP = int((W < Pmed).sum()), float(np.median(rel_P))
    per_seed = [int((W < P[k]).sum()) for k in range(5)]
    print(f"\nW beats D            : {nD}/15, median {100*medD:+.2f}%")
    print(f"W beats median(P1-5) : {nP}/15, median {100*medP:+.2f}%")
    print(f"W beats each P_k     : " + ", ".join(f"{s}:{n}/15" for s, n in zip(seeds, per_seed)))

    p2 = nP >= 12
    if a.mexico:
        print(f"\nW beats median P >= 12/15: "
              f"{'MET' if p2 else 'NOT MET'}  ({nP}/15)")
    else:
        p1 = nD >= 12 and medD <= -0.003
        null = nD <= 9 or abs(medD) < 0.001
        verdict = "REPLICATED" if (p1 and p2) else ("NULL" if null else "PARTIAL")
        print(f"\nP1 (W beats D >= 12/15 and median <= -0.3%): {'met' if p1 else 'not met'}")
        print(f"P2 (W beats median P >= 12/15)              : {'met' if p2 else 'not met'}")
        print(f"VERDICT: {verdict}")

    out = f"ndbc_gate_verdict_{a.label}.csv"
    tab.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
