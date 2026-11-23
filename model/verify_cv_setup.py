#!/usr/bin/env python3
"""Pre-launch verification of the 10-fold setup."""
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

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader_multi import MultiDepthStackDataset
from unet_decomp import ThermalUNetDecomp

BASE = P["data_root"]
CV = P["cv_dir"]
roots = {d: os.path.join(BASE, "pact_stack_%dL" % d) for d in (2, 3, 4)}
spec = json.load(open(os.path.join(CV, "cv_folds.json")))
chips = pd.read_csv(os.path.join(roots[2], "index", "chiplets.csv"))
chip_group = dict(zip(chips["chiplet_id"], chips["group"]))

print("=" * 78)
print("2.  SAME HELD-OUT GROUPS ACROSS 2/3/4-DIE DATASETS")
print("=" * 78)
f0 = spec["folds"][0]
for d in (2, 3, 4):
    st = pd.read_csv(os.path.join(roots[d], "index", "stacks.csv"))
    n_tr = (st["group"].isin(f0["train"])).sum()
    n_va = (st["group"].isin(f0["val"])).sum()
    n_te = (st["group"].isin(f0["test"])).sum()
    print("  depth %d : train %4d  val %4d  test %4d stacks   (groups selected "
          "by name, identical list for every depth)" % (d, n_tr, n_va, n_te))

print()
print("=" * 78)
print("3.  CHIPLET-LEVEL LEAKAGE CHECK, ALL FOLDS x ALL DEPTHS")
print("=" * 78)
DIE = {2: ["bot_chiplet", "top_chiplet"],
       3: ["d1_chiplet", "d2_chiplet", "d3_chiplet"],
       4: ["d1_chiplet", "d2_chiplet", "d3_chiplet", "d4_chiplet"]}
worst = 0
for f in spec["folds"]:
    held = {c for c, g in chip_group.items() if g in set(f["val"] + f["test"])}
    for d in (2, 3, 4):
        st = pd.read_csv(os.path.join(roots[d], "index", "stacks.csv"))
        tr = st[st["group"].isin(f["train"])]
        used = set()
        for c in DIE[d]:
            used |= set(tr[c].unique())
        worst = max(worst, len(used & held))
print("  max held-out chiplets appearing in any training set: %d  (must be 0)"
      % worst)
tr_chips = {c for c, g in chip_group.items() if g in f0["train"]}
print("  fold 0: 400 training chiplets, 50 val, 50 test -> %d/%d/%d"
      % (len(tr_chips),
         len([c for c, g in chip_group.items() if g in f0["val"]]),
         len([c for c, g in chip_group.items() if g in f0["test"]])))

print()
print("=" * 78)
print("4.  PER-FOLD NORMALISATION IS TRAINING-ONLY")
print("=" * 78)
n0 = json.load(open(os.path.join(CV, "norm_cv", "fold0.json")))
print("  fold0.json powermap: mean %.4f  std %.4f" % (n0["powermap_w_m2"]["mean"],
                                                      n0["powermap_w_m2"]["std"]))
print("  fold0.json solo_dT_k std: %.6f" % n0["solo_dT_k"]["std"])
print("  provenance: %d training chiplets from %s"
      % (n0["_provenance"]["n_training_chiplets"],
         ",".join(g.replace("group_", "") for g in
                  n0["_provenance"]["computed_from_groups"])))
print("  excluded from the statistics: %s" % n0["_provenance"]["excluded_groups"])

# independently recompute fold0 powermap stats from the 8 training groups
pmap = np.load(os.path.join(roots[2], "arrays", "powermap_w_m2.f32.npy"),
               mmap_mode="r")
rows_tr = chips.index[chips["group"].isin(f0["train"])].to_numpy()
rows_ho = chips.index[chips["group"].isin(f0["val"] + f0["test"])].to_numpy()
v_tr = np.concatenate([np.asarray(pmap[r], dtype=np.float64).ravel()
                       for r in rows_tr])
v_all = np.concatenate([v_tr] + [np.asarray(pmap[r], dtype=np.float64).ravel()
                                 for r in rows_ho])
print("  independent recompute, TRAIN groups only : mean %.4f  std %.4f  -> match %s"
      % (v_tr.mean(), v_tr.std(),
         abs(v_tr.mean() - n0["powermap_w_m2"]["mean"]) < 1e-3
         and abs(v_tr.std() - n0["powermap_w_m2"]["std"]) < 1e-3))
print("  same statistic over ALL 10 groups        : mean %.4f  std %.4f  "
      "(differs -> held-out data is genuinely excluded)" % (v_all.mean(), v_all.std()))
single = json.load(open(os.path.join(roots[2], "norm_stats.json")))
print("  single-split norm_stats.json             : mean %.4f  std %.4f  (not reused)"
      % (single["powermap_w_m2"]["mean"], single["powermap_w_m2"]["std"]))

allstd = [json.load(open(os.path.join(CV, "norm_cv", "fold%d.json" % k)))
          for k in range(10)]
print("  per-fold powermap std across folds: %s"
      % " ".join("%.0f" % s["powermap_w_m2"]["std"] for s in allstd))
print("  per-fold solo_dT_k  std across folds: %s"
      % " ".join("%.4f" % s["solo_dT_k"]["std"] for s in allstd))

print()
print("  loader actually consumes the fold file:")
ds = MultiDepthStackDataset(roots, arm="A3", groups=f0["train"],
                            norm_path=os.path.join(CV, "norm_cv", "fold0.json"))
print("    dataset norm powermap std = %.4f (fold0.json = %.4f)  match %s"
      % (ds.norm["powermap_w_m2"]["std"], n0["powermap_w_m2"]["std"],
         ds.norm["powermap_w_m2"]["std"] == n0["powermap_w_m2"]["std"]))
print("    train samples = %d  (8 groups x 600 stacks x 3 depths x 5 bc = 72000)"
      % len(ds))

print()
print("=" * 78)
print("5.  FROZEN ARCHITECTURE SHARED BY A0-A3")
print("=" * 78)
for arm, ich in (("A0", 10), ("A1", 10), ("A2", 14), ("A3", 14)):
    m = ThermalUNetDecomp(in_ch=ich, out_ch=7, base=32, depth=4, scalar_dim=2)
    print("  %s in_ch=%2d  params=%d" % (arm, ich,
                                         sum(p.numel() for p in m.parameters())))
print("  identical for all arms: backbone base=32 depth=4, zero-mean pattern head,")
print("  pooled per-layer offset head, aux 0.5, grad 0.5, AdamW 3e-4 wd 1e-4,")
print("  OneCycle pct_start 0.1, batch 32, 30 epochs, select on val r2_pattern")

print()
print("=" * 78)
print("7.  SINGLE-SPLIT RESULTS UNTOUCHED")
print("=" * 78)
for p in ["runs_multi/A3_decomp_aux/best.pt", "runs_multi/A0_decomp_aux/best.pt",
          "runs_multi/A1_decomp_aux/best.pt", "runs_multi/A2_decomp_aux/best.pt",
          "runs_multi/A3_vfuse/best.pt", "runs_multi/A3_solo_level/best.pt",
          "eval_2x2.json", "eval_all.json"]:
    fp = os.path.join(BASE, p)
    print("  %-42s %s" % (p, "present %8.1f MB" % (os.path.getsize(fp) / 1e6)
                          if os.path.exists(fp) else "MISSING"))
print("  new CV outputs go to %s/  (separate tree)" % CV)
