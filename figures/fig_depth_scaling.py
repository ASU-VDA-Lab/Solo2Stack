#!/usr/bin/env python3
"""Fig. 3: pattern R^2 against number of dies, four arms.

A copy of statcheck/fig_depth_exact.py with the run directory switched to the
microbump+hybrid campaign and the expected-value table replaced by the new
campaign's own numbers, so a silent drift still fails the assertion.

The only style change is the left y-limit, and it is forced by the data: A0's
four-die mean minus its fold SD is 0.2471, below the published lower limit of
0.30, so the published limit would clip A0's error bar off the figure. Palette,
markers, marker sizes, figure size, fonts, grid and legend are untouched --
Solo2Stack keeps its green star.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

RUNS = P["checkpoints"]
OUTDIR = os.environ.get("FIGDIR", P["data_root"] + "/figs_mbhb")
d = [2, 3, 4]


def folds(arm, depth):
    return np.array([json.load(open("%s/%s_fold%d/result.json" % (RUNS, arm, f)))
                     ["test"]["per_depth"][str(depth)]["r2_pattern"]
                     for f in range(10)])


# exact ten-fold mean and sample SD (ddof=1) straight from the MB+HB runs,
# not hand-read; the expected values are repeated here so a silent drift fails
series = []
EXPECT = {
    "A0": [(0.320713, 0.035048), (0.284017, 0.020008), (0.268489, 0.021410)],
    "A2": [(0.639587, 0.030373), (0.465479, 0.031673), (0.386365, 0.027782)],
    "A1": [(0.750423, 0.017955), (0.692655, 0.018606), (0.648611, 0.020371)],
    "A3": [(0.909780, 0.009712), (0.844787, 0.017066), (0.799987, 0.021691)],
}
for lab, arm, c, m in [("A0", "A0", "#888888", "^"),
                       ("A2", "A2", "#e69f00", "D"),
                       ("A1", "A1", "#0072b2", "s"),
                       ("Solo2Stack", "A3", "#009e73", "*")]:
    y = [folds(arm, k).mean() for k in d]
    e = [folds(arm, k).std(ddof=1) for k in d]
    for i in range(3):
        assert round(y[i], 6) == EXPECT[arm][i][0], (lab, d[i], y[i])
        assert round(e[i], 6) == EXPECT[arm][i][1], (lab, d[i], e[i])
    series.append((lab, y, e, c, m))

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7})
fig, ax = plt.subplots(figsize=(2.6, 2.0))
for lab, y, e, c, m in series:
    ax.errorbar(d, y, yerr=e, marker=m, color=c, ms=6 if m != "*" else 9, lw=1.6,
                label=lab, mec="k", mew=0.4,
                capsize=1.8, elinewidth=0.7, capthick=0.7)
ax.set_xticks(d); ax.set_xlabel("Number of dies", labelpad=1)
ax.set_ylabel("Pattern $R^2$", labelpad=1)
# 0.2 not 0.3: A0 at four dies is 0.2685 +- 0.0214, so 0.30 would clip it
ax.set_ylim(0.2, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.tick_params(length=2.5, pad=2)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
          handletextpad=0.3, columnspacing=0.9, borderaxespad=0.1)
fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.19)
os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(os.path.join(OUTDIR, "depth_scaling_exact_mbhb.pdf"))
fig.savefig(os.path.join(OUTDIR, "depth_scaling_exact_mbhb.png"), dpi=200)
print("all 24 mean/SD values matched the expected table")
for lab, y, e, c, m in series:
    print("  %-11s marker %-2s  %s" % (lab, m, "  ".join(
        "%d dies %.6f+-%.6f" % (d[i], y[i], e[i]) for i in range(3))))
