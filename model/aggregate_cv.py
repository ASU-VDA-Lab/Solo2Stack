#!/usr/bin/env python3
"""Aggregate the 40-run (4 arms x 10 folds) frozen decomp+aux factorial.

Every number here comes from each run's single evaluation of its selected
checkpoint on that fold's held-out test group. Folds are the unit of analysis,
so contrasts are PAIRED within fold and the reported spread is fold-to-fold
variation across held-out chiplet populations.

Sign convention: for r2_pattern and hotspot overlap, higher is better, so a
positive contrast favours the first arm. For every error metric (Pattern MAE,
Abs MAE, Offset MAE, p99, max) lower is better, so contrasts on those are
reported as (lo - hi), i.e. positive still means the first arm is better.
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

import argparse
import glob
import json
import os
import statistics as st

from scipy import stats

ARMS = ["A0", "A1", "A2", "A3"]
DESC = {"A0": "no solo, absolute", "A1": "no solo, residual",
        "A2": "solo, absolute", "A3": "solo, residual"}
HIGHER_BETTER = {"r2_pattern", "hotspot_top1%_overlap", "r2_field"}
METRICS = [("mae_pattern_k", "PatMAE"), ("r2_pattern", "PatR2"),
           ("mae_k", "AbsMAE"), ("mae_offset_k", "OffMAE"),
           ("p99_k", "p99"), ("max_k", "Max"),
           ("hotspot_top1%_overlap", "Hotspot")]


def load(runs_dir):
    R = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        r = json.load(open(p))
        R.setdefault(r["arm"], {})[r["fold"]] = r
    return R


def ms(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def block(R, folds, sel, title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    print("%-4s %-19s %-16s %-16s %-14s %-14s %-13s %-13s %s"
          % ("Arm", "config", "PatMAE (K)", "PatR2", "AbsMAE (K)",
             "OffMAE (K)", "p99 (K)", "Max (K)", "Hotspot"))
    for arm in ARMS:
        cells = []
        for key, _ in METRICS:
            v = [sel(R[arm][f]) [key] for f in folds]
            m, s = ms(v)
            cells.append("%.4f±%.4f" % (m, s))
        print("%-4s %-19s %-16s %-16s %-14s %-14s %-13s %-13s %s"
              % (arm, DESC[arm], cells[0], cells[1], cells[2], cells[3],
                 cells[4], cells[5], cells[6]))


def contrasts(R, folds, sel, label):
    pairs = [("A2", "A0", "Solo effect | absolute target  (A2-A0)"),
             ("A3", "A1", "Solo effect | residual target  (A3-A1)"),
             ("A1", "A0", "Residual-target effect | no solo (A1-A0)"),
             ("A3", "A2", "Residual-target effect | solo    (A3-A2)")]
    print("\n" + "-" * 104)
    print("PAIRED PER-FOLD CONTRASTS  (%s)   positive = first arm better, n=%d folds"
          % (label, len(folds)))
    print("-" * 104)
    for key, short in METRICS:
        print("\n  %s" % short)
        for hi, lo, name in pairs:
            d = []
            for f in folds:
                a, b = sel(R[hi][f])[key], sel(R[lo][f])[key]
                d.append((a - b) if key in HIGHER_BETTER else (b - a))
            m, s = ms(d)
            t, p = stats.ttest_rel(
                [sel(R[hi][f])[key] for f in folds],
                [sel(R[lo][f])[key] for f in folds])
            try:
                w, pw = stats.wilcoxon(d)
            except Exception:
                pw = float("nan")
            print("    %-42s %+.4f ± %.4f   t=%+7.2f  p=%.2e  wilcoxon p=%.4f  "
                  "wins %d/%d"
                  % (name, m, s, t, p, pw, sum(1 for x in d if x > 0), len(d)))
        # interaction: (A3-A1) - (A2-A0), in "better" orientation
        inter = []
        for f in folds:
            g = lambda arm: sel(R[arm][f])[key]
            if key in HIGHER_BETTER:
                inter.append((g("A3") - g("A1")) - (g("A2") - g("A0")))
            else:
                inter.append((g("A1") - g("A3")) - (g("A0") - g("A2")))
        m, s = ms(inter)
        t, p = stats.ttest_1samp(inter, 0.0)
        print("    %-42s %+.4f ± %.4f   t=%+7.2f  p=%.2e                      "
              "wins %d/%d"
              % ("INTERACTION (A3-A1)-(A2-A0)", m, s, t, p,
                 sum(1 for x in inter if x > 0), len(inter)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=P["checkpoints"])
    ap.add_argument("--out", default=P["cv_dir"] + "/aggregate.json")
    a = ap.parse_args()

    R = load(a.runs)
    folds = sorted(set.intersection(*[set(R[k]) for k in ARMS]))
    print("arms %s | complete folds %s (n=%d)" % (sorted(R), folds, len(folds)))
    n_runs = sum(len(v) for v in R.values())
    print("runs loaded: %d" % n_runs)

    block(R, folds, lambda r: r["test"]["pooled"],
          "MAIN POOLED FACTORIAL TABLE  (10-fold mean +/- sample std, held-out test)")
    for d in ("2", "3", "4"):
        block(R, folds, lambda r, d=d: r["test"]["per_depth"][d],
              "DEPTH-WISE TABLE  -  %s-DIE STACKS" % d)

    contrasts(R, folds, lambda r: r["test"]["pooled"], "pooled")
    for d in ("2", "3", "4"):
        contrasts(R, folds, lambda r, d=d: r["test"]["per_depth"][d],
                  "%s dies" % d)

    # ---- depth-scaling data ------------------------------------------------
    print("\n" + "=" * 104)
    print("DEPTH-SCALING DATA  (10-fold mean +/- std; for the figure)")
    print("=" * 104)
    scaling = {}
    for key, short in (("r2_pattern", "Pattern R2"),
                       ("mae_pattern_k", "Pattern MAE (K)")):
        print("\n  %s" % short)
        print("  %-4s %-18s %-18s %-18s" % ("arm", "2 dies", "3 dies", "4 dies"))
        for arm in ARMS:
            row = []
            for d in ("2", "3", "4"):
                v = [R[arm][f]["test"]["per_depth"][d][key] for f in folds]
                m, s = ms(v)
                row.append("%.4f±%.4f" % (m, s))
                scaling.setdefault(key, {}).setdefault(arm, {})[d] = {
                    "mean": m, "std": s,
                    "per_fold": [R[arm][f]["test"]["per_depth"][d][key]
                                 for f in folds]}
            print("  %-4s %-18s %-18s %-18s" % (arm, row[0], row[1], row[2]))

    # ---- per-fold raw table ------------------------------------------------
    print("\n" + "=" * 104)
    print("PER-FOLD RAW POOLED METRICS")
    print("=" * 104)
    print("%-4s %-5s %-9s %9s %9s %9s %9s %8s %8s %8s"
          % ("arm", "fold", "test grp", "PatR2", "PatMAE", "AbsMAE", "OffMAE",
             "p99", "max", "hot"))
    for arm in ARMS:
        for f in folds:
            r = R[arm][f]
            m = r["test"]["pooled"]
            print("%-4s %-5d %-9s %9.4f %9.4f %9.4f %9.4f %8.4f %8.3f %8.3f"
                  % (arm, f, r["groups"]["test"][0].replace("group_", "g"),
                     m["r2_pattern"], m["mae_pattern_k"], m["mae_k"],
                     m["mae_offset_k"], m["p99_k"], m["max_k"],
                     m["hotspot_top1%_overlap"]))

    out = {"folds": folds,
           "pooled": {arm: {k: {"mean": ms([R[arm][f]["test"]["pooled"][k]
                                            for f in folds])[0],
                                "std": ms([R[arm][f]["test"]["pooled"][k]
                                           for f in folds])[1],
                                "per_fold": [R[arm][f]["test"]["pooled"][k]
                                             for f in folds]}
                            for k, _ in METRICS} for arm in ARMS},
           "per_depth": {arm: {d: {k: {"mean": ms([R[arm][f]["test"]["per_depth"][d][k]
                                                   for f in folds])[0],
                                       "std": ms([R[arm][f]["test"]["per_depth"][d][k]
                                                  for f in folds])[1],
                                       "per_fold": [R[arm][f]["test"]["per_depth"][d][k]
                                                    for f in folds]}
                                   for k, _ in METRICS}
                               for d in ("2", "3", "4")} for arm in ARMS},
           "per_layer": {arm: {f: R[arm][f]["test"].get("per_layer", {})
                               for f in folds} for arm in ARMS},
           "depth_scaling": scaling,
           "test_groups": {f: R["A3"][f]["groups"]["test"][0] for f in folds}}
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nsaved -> %s" % a.out)


if __name__ == "__main__":
    main()
