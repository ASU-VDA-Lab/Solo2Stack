"""Task 1 (paired Wilcoxon between configurations) and Task 2 (accuracy vs N).

Reads ONLY the saved per-fold metrics of the final microbump+hybrid campaign:
    <checkpoints>/{A0,A1,A2,A3}_fold{0..9}/result.json
Nothing is retrained and no PACT is run.
"""
# --- repo bootstrap -------------------------------------------------------
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d):
    if _os.path.isdir(_os.path.join(_d, "solo2stack")):
        _sys.path.insert(0, _d)
        break
    _d = _os.path.dirname(_d)
from solo2stack.paths import P  # noqa: E402
from solo2stack.bootstrap import add_repo  # noqa: E402
# --------------------------------------------------------------------------
import json
import os
import csv
import numpy as np
from scipy.stats import wilcoxon

CV = P["cv_dir"]
OUT = P["results"]
ARMS = ["A0", "A1", "A2", "A3"]
LABEL = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "Solo2Stack"}

# metric key -> (display name, higher_is_better)
METRICS = [
    ("mae_pattern_k",        "Pattern MAE (K)",   False),
    ("r2_pattern",           "Pattern R2",        True),
    ("mae_k",                "Absolute MAE (K)",  False),
    ("mae_offset_k",         "Offset MAE (K)",    False),
    ("p99_k",                "p99 error (K)",     False),
    ("max_k",                "Max error (K)",     False),
    ("hotspot_top1%_overlap", "Hotspot overlap",  True),
]

PAIRS = [
    ("A0", "A1", "residual target"),
    ("A2", "A3", "residual target"),
    ("A0", "A2", "solo input"),
    ("A1", "A3", "solo input"),
]


def load():
    """pooled[arm][metric] = list of 10 fold values; depth[arm][N] = 10 values."""
    pooled = {a: {k: [] for k, _, _ in METRICS} for a in ARMS}
    depth = {a: {n: [] for n in (2, 3, 4)} for a in ARMS}
    counts = {}
    for a in ARMS:
        for f in range(10):
            r = json.load(open(os.path.join(CV, "runs", "%s_fold%d" % (a, f),
                                            "result.json")))
            assert r["config"]["base_dir"] == P["data_root"], r["config"]
            assert r["groups"]["test"] == ["group_%02d" % (f + 1)]
            t = r["test"]
            for k, _, _ in METRICS:
                pooled[a][k].append(float(t["pooled"][k]))
            for n in (2, 3, 4):
                depth[a][n].append(float(t["per_depth"][str(n)]["r2_pattern"]))
            counts[(a, f)] = t["counts"]
    return pooled, depth, counts


def ms(v):
    v = np.asarray(v, dtype=float)
    # sample std (ddof=1), matching the paper's "mean +- std over 10 folds"
    return v.mean(), v.std(ddof=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    pooled, depth, counts = load()

    # ---------- sanity: ten-fold means the user quoted ----------
    print("=== ten-fold pooled means (check against expected) ===")
    exp = {("A0", "mae_k"): 0.743, ("A0", "mae_offset_k"): 0.564,
           ("A2", "mae_k"): 0.465, ("A2", "mae_offset_k"): 0.243,
           ("A1", "mae_offset_k"): 0.091, ("A3", "mae_offset_k"): 0.134}
    for (a, k), want in sorted(exp.items()):
        got = np.mean(pooled[a][k])
        print("  %-10s %-16s got %.4f   expected %.3f   %s"
              % (LABEL[a], k, got, want,
                 "OK" if abs(got - want) < 0.0006 else "MISMATCH"))

    # ---------- Task 1 ----------
    rows = []
    print("\n=== Task 1: two-sided Wilcoxon signed-rank, n=10 folds ===")
    for a, b, effect in PAIRS:
        for k, name, hib in METRICS:
            x = np.array(pooled[a][k], dtype=float)
            y = np.array(pooled[b][k], dtype=float)
            d = y - x                      # second minus first
            md = d.mean()
            # all-zero differences would make wilcoxon undefined; never the case here
            st, p = wilcoxon(x, y, alternative="two-sided")
            same = int((np.sign(d) == np.sign(md)).sum())
            better = LABEL[b] if ((md > 0) == hib) else LABEL[a]
            rows.append(dict(
                effect=effect, pair="%s vs %s" % (LABEL[a], LABEL[b]),
                metric=name, mean_a=x.mean(), mean_b=y.mean(),
                mean_diff=md, wilcoxon_stat=st, p_value=p,
                folds_same_direction=same, favours=better,
                significant_at_0p05=bool(p <= 0.05)))
            print("  %-26s %-16s mean_diff %+8.4f  p=%.6f  %2d/10  favours %s%s"
                  % (rows[-1]["pair"], name, md, p, same, better,
                     "" if p <= 0.05 else "   <-- NOT SIGNIFICANT"))

    with open(os.path.join(OUT, "task1_wilcoxon.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- Task 2 ----------
    print("\n=== Task 2: pattern R2 vs number of dies (10-fold mean +- std) ===")
    drows = []
    for a in ARMS:
        m2, s2 = ms(depth[a][2])
        m3, s3 = ms(depth[a][3])
        m4, s4 = ms(depth[a][4])
        # per-fold drop, so its std is the spread of the drop itself
        dr = np.array(depth[a][2]) - np.array(depth[a][4])
        dm, ds = ms(dr)
        st, p = wilcoxon(depth[a][2], depth[a][4], alternative="two-sided")
        drows.append(dict(arm=LABEL[a], r2_N2=m2, sd_N2=s2, r2_N3=m3, sd_N3=s3,
                          r2_N4=m4, sd_N4=s4, drop_N2_to_N4=dm, sd_drop=ds,
                          folds_with_drop=int((dr > 0).sum()), p_value=p))
        print("  %-10s N2 %.4f+-%.4f  N3 %.4f+-%.4f  N4 %.4f+-%.4f   drop %.4f+-%.4f (%d/10, p=%.6f)"
              % (LABEL[a], m2, s2, m3, s3, m4, s4, dm, ds,
                 drows[-1]["folds_with_drop"], p))
    with open(os.path.join(OUT, "task2_depth_r2.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(drows[0].keys()))
        w.writeheader()
        w.writerows(drows)

    # sample counts per depth, for the record
    print("\ncounts per depth (fold 0, A3):", counts[("A3", 0)])
    print("\nwrote %s/task1_wilcoxon.csv and task2_depth_r2.csv" % OUT)


if __name__ == "__main__":
    main()
