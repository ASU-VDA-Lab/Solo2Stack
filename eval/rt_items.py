# --- repo bootstrap -------------------------------------------------------
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d):
    if _os.path.isdir(_os.path.join(_d, "solo2stack")):
        _sys.path.insert(0, _d)
        break
    _d = _os.path.dirname(_d)
from solo2stack.paths import P, repo_path  # noqa: E402
from solo2stack.bootstrap import add_repo  # noqa: E402
# --------------------------------------------------------------------------
import json
import os

import numpy as np
import pandas as pd

rt = json.load(open(next(
    (c for c in (repo_path("results", "runtime_mbhb.json"),
                 os.path.join(P["results"], "runtime_mbhb.json"))
     if os.path.exists(c)), repo_path("results", "runtime_mbhb.json"))))
new = rt["arms"]["new"]
solo = rt["solo_s"]
inf = rt["infer_ms_batch1"] / 1000.0
fs = {d: new["full_stack_s"][str(d)] for d in (2, 3, 4)}
print("full_stack_s", fs, "solo", solo, "infer_s", inf)

print("\nspeedup, solo runs IN SERIES (N sequential solo + 1 inference):")
for d in (2, 3, 4):
    c = d * solo + inf
    print("   N=%d  cost %.5f s  speedup %.4f" % (d, c, fs[d] / c))
print("speedup, solo runs IN PARALLEL (N concurrent solo + 1 inference):")
for d in (2, 3, 4):
    c = solo + inf
    print("   N=%d  cost %.5f s  speedup %.4f" % (d, c, fs[d] / c))

n4 = 20000
t4 = n4 * fs[4]
print("\n20,000 four-die stacks one at a time: %.1f s = %.2f h" % (t4, t4 / 3600))
n_solo = 500 * 5
s2s = n_solo * solo + n4 * inf
print("Solo2Stack for that library: %d solo x %.3f + %d x %.5f = %.1f s = %.4f h"
      % (n_solo, solo, n4, inf, s2s, s2s / 3600))
print("   solo part %.1f s, inference part %.1f s" % (n_solo * solo, n4 * inf))
print("   ratio %.2f" % (t4 / s2s))

st = pd.read_csv(P["data_root"] + "/pact_stack_4L/index/stacks.csv")
cols = ["d1_chiplet", "d2_chiplet", "d3_chiplet", "d4_chiplet"]
cnt = pd.concat([st[c] for c in cols]).value_counts()
print("\nfour-die configurations per chiplet (one cooling condition): "
      "min %d max %d mean %.1f  (n chiplets %d, n stacks %d)"
      % (cnt.min(), cnt.max(), cnt.mean(), len(cnt), len(st)))

tot = sum(20000 * fs[d] for d in (2, 3, 4))
print("\nall 60,000 stacks: %.1f s = %.2f CPU-hours" % (tot, tot / 3600))

ws = [json.load(open(P["checkpoints"] + "/A3_fold%d/result.json" % f))
      ["wall_seconds"] for f in range(10)]
print("\nA3 training wall seconds per fold: mean %.1f s = %.3f h"
      % (np.mean(ws), np.mean(ws) / 3600))
print("   all ten folds (sum): %.1f s = %.2f GPU-hours" % (sum(ws), sum(ws) / 3600))

share = n_solo * solo / n4
per_eval = share + inf
print("\nper four-die evaluation incl. its share of solo: %.4f + %.5f = %.4f s"
      % (share, inf, per_eval))
print("   saved per evaluation: %.4f - %.4f = %.4f s" % (fs[4], per_eval,
                                                         fs[4] - per_eval))
setup = n_solo * solo + sum(ws)
print("\nsetup (2500 solo + 10 folds training) = %.1f s = %.2f h"
      % (setup, setup / 3600))
print("break-even four-die evaluations = %.1f -> %d"
      % (setup / (fs[4] - per_eval), int(np.ceil(setup / (fs[4] - per_eval)))))
setup1 = n_solo * solo + np.mean(ws)
print("break-even if only ONE model is trained = %d"
      % int(np.ceil(setup1 / (fs[4] - per_eval))))
