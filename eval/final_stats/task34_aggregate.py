#!/usr/bin/env python3
"""Aggregate Tasks 3 and 4 into markdown tables and CSVs.

Inputs (all produced earlier, nothing recomputed here):
    final_stats/task3_solo_only.json        solo-only baseline, per fold
    final_stats/pass/<arm>_fold<k>.json     die-layer-only model metrics
    final_stats/pass/<arm>_fold<k>_peaks.csv  per-sample peak temperatures
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
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

FS = P["results"]
PASS = os.path.join(FS, "pass")
ARMS = ["A0", "A1", "A2", "A3"]
LABEL = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "Solo2Stack",
         "solo_only": "Solo-only (no model)"}
MET = ["r2_pattern", "mae_pattern_k", "hotspot_top1%_overlap"]


def ms(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fmt(v, nd=4):
    return ("%." + str(nd) + "f") % v


def task3():
    solo = json.load(open(os.path.join(FS, "task3_solo_only.json")))
    scopes = ["overall", "N=2", "N=3", "N=4", "tech=microbump", "tech=hybrid"]

    # per-fold series, per method, per scope, per metric
    series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in range(10):
        for sc in scopes:
            r = solo["per_fold"][str(f)][sc]
            for m in MET:
                series["solo_only"][sc][m].append(r[m])
    for arm in ARMS:
        for f in range(10):
            p = json.load(open(os.path.join(PASS, "%s_fold%d.json" % (arm, f))))
            for sc in scopes:
                r = p["die_layers_only"][sc]
                for m in MET:
                    series[arm][sc][m].append(r[m])

    rows = []
    for meth in ["solo_only", "A1", "A3"] + ["A0", "A2"]:
        for sc in scopes:
            d = dict(method=LABEL[meth], scope=sc)
            for m in MET:
                mu, sd = ms(series[meth][sc][m])
                d[m + "_mean"], d[m + "_std"] = mu, sd
            rows.append(d)
    with open(os.path.join(FS, "task3_die_layers.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-fold raw table for the solo-only baseline
    with open(os.path.join(FS, "task3_solo_per_fold.csv"), "w") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "scope"] + MET)
        for f in range(10):
            for sc in scopes:
                r = solo["per_fold"][str(f)][sc]
                w.writerow([f, sc] + [r[m] for m in MET])

    print("\n## Task 3 - die layers only (10-fold mean +- sample std)\n")
    print("| Method | Scope | Pattern R2 | Pattern MAE (K) | Hotspot top-1% |")
    print("|---|---|---|---|---|")
    for d in rows:
        print("| %s | %s | %s +- %s | %s +- %s | %s +- %s |"
              % (d["method"], d["scope"],
                 fmt(d["r2_pattern_mean"]), fmt(d["r2_pattern_std"]),
                 fmt(d["mae_pattern_k_mean"]), fmt(d["mae_pattern_k_std"]),
                 fmt(d["hotspot_top1%_overlap_mean"]),
                 fmt(d["hotspot_top1%_overlap_std"])))
    return series


def task4():
    out_rows, grp_rows = [], []
    print("\n## Task 4a - peak absolute temperature (per fold, then 10-fold "
          "mean +- std)\n")
    print("| Arm | Peak |dT| (K) | Same peak layer | n samples/fold |")
    print("|---|---|---|---|---|")
    peak_err = {}
    same_lay = {}
    sp_mean = {}
    for arm in ARMS:
        pe, sl, sp, ngrp, nper = [], [], [], [], []
        for f in range(10):
            with open(os.path.join(PASS, "%s_fold%d_peaks.csv"
                                   % (arm, f))) as fh:
                rec = list(csv.DictReader(fh))
            pe.append(np.mean([float(r["abs_peak_err_k"]) for r in rec]))
            sl.append(np.mean([int(r["same_peak_layer"]) for r in rec]))

            # Spearman within each (N, bonding, h) group
            g = defaultdict(list)
            for r in rec:
                g[(r["depth"], r["tech"], r["htc"])].append(
                    (float(r["true_peak_k"]), float(r["pred_peak_k"])))
            rhos, sds, rngs, sizes = [], [], [], []
            for key, v in sorted(g.items()):
                t = np.array([a for a, _ in v])
                p = np.array([b for _, b in v])
                rho = spearmanr(t, p).statistic
                rhos.append(rho)
                sds.append(t.std(ddof=1))
                rngs.append(t.max() - t.min())
                sizes.append(len(v))
                grp_rows.append(dict(arm=LABEL[arm], fold=f, N=key[0],
                                     tech=key[1], htc=key[2], n_stacks=len(v),
                                     spearman=rho, true_peak_std_k=t.std(ddof=1),
                                     true_peak_range_k=t.max() - t.min()))
            sp.append(float(np.mean(rhos)))
            ngrp.append(len(rhos))
            nper.append(float(np.mean(sizes)))
            out_rows.append(dict(arm=LABEL[arm], fold=f,
                                 peak_abs_err_k=pe[-1], same_peak_layer=sl[-1],
                                 mean_spearman=sp[-1], n_groups=len(rhos),
                                 mean_stacks_per_group=float(np.mean(sizes)),
                                 mean_true_peak_std_k=float(np.mean(sds)),
                                 mean_true_peak_range_k=float(np.mean(rngs))))
            n_samp = len(rec)
        peak_err[arm], same_lay[arm], sp_mean[arm] = pe, sl, sp
        print("| %s | %s +- %s | %s +- %s | %d |"
              % (LABEL[arm], fmt(np.mean(pe)), fmt(np.std(pe, ddof=1)),
                 fmt(np.mean(sl)), fmt(np.std(sl, ddof=1)), n_samp))

    print("\n## Task 4b - Spearman of peak temperature within (N, bonding, h)\n")
    print("| Arm | Mean Spearman | Groups/fold | Stacks/group |")
    print("|---|---|---|---|")
    for arm in ARMS:
        sub = [r for r in out_rows if r["arm"] == LABEL[arm]]
        print("| %s | %s +- %s | %d | %.0f |"
              % (LABEL[arm], fmt(np.mean(sp_mean[arm])),
                 fmt(np.std(sp_mean[arm], ddof=1)),
                 sub[0]["n_groups"], sub[0]["mean_stacks_per_group"]))

    print("\n## Task 4b - spread of TRUE peak temperature inside a group\n")
    print("| N | Bonding | h (W/m2K) | true peak std (K) | true peak range (K) |")
    print("|---|---|---|---|---|")
    agg = defaultdict(list)
    for r in grp_rows:
        if r["arm"] != LABEL["A3"]:
            continue      # the true spread does not depend on the arm
        agg[(r["N"], r["tech"], r["htc"])].append(
            (r["true_peak_std_k"], r["true_peak_range_k"]))
    for key in sorted(agg, key=lambda k: (int(k[0]), k[1], float(k[2]))):
        v = np.array(agg[key])
        print("| %s | %s | %.0f | %.3f | %.3f |"
              % (key[0], key[1], float(key[2]), v[:, 0].mean(), v[:, 1].mean()))

    with open(os.path.join(FS, "task4_per_fold.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(os.path.join(FS, "task4_groups.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grp_rows[0].keys()))
        w.writeheader()
        w.writerows(grp_rows)


if __name__ == "__main__":
    task3()
    task4()
    print("\nwrote task3_die_layers.csv task3_solo_per_fold.csv "
          "task4_per_fold.csv task4_groups.csv in %s" % FS)
