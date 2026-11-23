#!/usr/bin/env python3
"""Task 1, max-error rows only, on the PAPER's definition of max error.

result.json["test"]["pooled"]["max_k"] is NOT the fold maximum: train_cv.score()
batch-weight-averages every metric, so that field is the mean of per-batch-slice
maxima and always sits well below the true maximum (~6.7 K where the true
maximum is ~11.7 K).  The paper's Max column is the largest |pred - true| over
every grid cell of every valid layer of every test stack in the fold, which is
what eval_global.py accumulates as global.max_global_k.

This script reads that field for the final microbump+hybrid campaign and redoes
the four paired tests.  The per-fold worst-cell errors recorded independently by
task45_model_pass.py are used as a cross-check of the same quantity.
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
import glob
import json
import os
import csv

import numpy as np
from scipy.stats import wilcoxon

EG = P["eval_global"] + "/C_new_mbhb.json"
PASS = P["results"] + "/pass"
OUT = P["results"]
ARMS = ["A0", "A1", "A2", "A3"]
LABEL = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "Solo2Stack"}
PAIRS = [("A0", "A1", "residual target"), ("A2", "A3", "residual target"),
         ("A0", "A2", "solo input"), ("A1", "A3", "solo input")]

eg = json.load(open(EG))
assert eg["base_dir"] == P["data_root"], eg["base_dir"]
assert eg["runs"] == P["checkpoints"], eg["runs"]
assert eg["ic_filter"] is None

gmax = {}
for arm in ARMS:
    v = []
    for f in range(10):
        r = eg["results"]["%s_fold%d" % (arm, f)]
        assert r["n_samples_scored"] == 6000, r["n_samples_scored"]
        assert sorted(r["technologies_scored"]) == ["hybrid", "microbump"]
        v.append(float(r["global"]["max_global_k"]))
    gmax[arm] = np.array(v)

print("=== fold maxima, paper definition (global over all cells/layers/stacks) ===")
for arm in ARMS:
    print("  %-10s mean %.4f  std %.4f   folds: %s"
          % (LABEL[arm], gmax[arm].mean(), gmax[arm].std(ddof=1),
             " ".join("%.2f" % x for x in gmax[arm])))

# independent cross-check against this session's own inference pass
chk = {}
for p in sorted(glob.glob(os.path.join(PASS, "*_fold*.json"))):
    j = json.load(open(p))
    chk[(j["arm"], j["fold"])] = j["worst_cell"]["err"]
if len(chk) == 40:
    d = np.array([chk[(a, f)] - gmax[a][f] for a in ARMS for f in range(10)])
    print("\ncross-check vs this session's pass: max |delta| = %.2e K "
          "(bf16 autocast is not bit-deterministic)" % np.abs(d).max())
else:
    print("\ncross-check: %d/40 pass records available so far" % len(chk))

rows = []
print("\n=== Task 1, max error, two-sided Wilcoxon, n=10 ===")
for a, b, eff in PAIRS:
    x, y = gmax[a], gmax[b]
    dif = y - x
    md = dif.mean()
    st, p = wilcoxon(x, y, alternative="two-sided")
    same = int((np.sign(dif) == np.sign(md)).sum())
    rows.append(dict(effect=eff, pair="%s vs %s" % (LABEL[a], LABEL[b]),
                     metric="Max error (K), global",
                     mean_a=x.mean(), mean_b=y.mean(), mean_diff=md,
                     wilcoxon_stat=st, p_value=p, folds_same_direction=same,
                     favours=LABEL[b] if md < 0 else LABEL[a],
                     significant_at_0p05=bool(p <= 0.05)))
    print("  %-26s mean_diff %+8.4f   p=%.6f   %2d/10   favours %s%s"
          % (rows[-1]["pair"], md, p, same, rows[-1]["favours"],
             "" if p <= 0.05 else "   <-- NOT SIGNIFICANT"))

with open(os.path.join(OUT, "task1_max_global.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(OUT, "task1_max_global_per_fold.csv"), "w") as fh:
    w = csv.writer(fh)
    w.writerow(["fold"] + [LABEL[a] for a in ARMS])
    for f in range(10):
        w.writerow([f] + ["%.6f" % gmax[a][f] for a in ARMS])
print("\nwrote task1_max_global.csv and task1_max_global_per_fold.csv")
