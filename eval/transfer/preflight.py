#!/usr/bin/env python3
"""Pre-flight for the Table VI transferability runs.

Defines the four held-out settings on ONE fixed chiplet fold (cv fold 0),
counts the resulting train/val/test populations, proves the held-out condition
is absent from train and val, and writes the restricted normalisation files.

Normalisation follows make_cv_folds.py exactly, with one change: every statistic
is recomputed over the population the restricted training set actually sees.

  powermap_w_m2   per chiplet          -> depends on groups only
  solo_dT_k       per (chiplet, bc)    -> depends on groups AND the bc subset
  gmap_log10      per technology       -> depends on the technology subset
  log10_htc       per bc               -> depends on the bc subset

so the depth hold-out inherits fold 0's statistics unchanged, while the cooling
and bonding hold-outs get their own.
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

import numpy as np
import pandas as pd

BASE = P["data_root"]
CVDIR = P["cv_dir"]
OUT = P["transfer_dir"]
ROOTS = {d: "%s/mbhb/pact_stack_%dL" % (BASE, d) for d in (2, 3, 4)}
DEPTHS = (2, 3, 4)

fold = json.load(open("%s/cv_folds.json" % CVDIR))["folds"][0]
TRG, VAG, TEG = fold["train"], fold["val"], fold["test"]

ref = ROOTS[2]
chips = pd.read_csv("%s/index/chiplets.csv" % ref)
bc = pd.read_csv("%s/index/bc.csv" % ref)
ic = pd.read_csv("%s/index/interconnect.csv" % ref)
meta = json.load(open("%s/meta.json" % ref))
stacks = {d: pd.read_csv("%s/index/stacks.csv" % ROOTS[d]) for d in DEPTHS}

HTC = bc["htc_w_m2k"].to_numpy(dtype=np.float64)
TECH = dict(zip(ic["ic_row"], ic["interconnect"]))
BCROW = {int(h): i for i, h in enumerate(HTC)}

# ---- the four settings -------------------------------------------------------
# each: (name, train/val depths, train/val bc rows, train/val ic rows,
#              test depths,      test bc rows,      test ic rows)
ALL_D, ALL_B, ALL_I = list(DEPTHS), list(range(len(bc))), sorted(TECH)
MB = [k for k, v in TECH.items() if v == "microbump"]
NOT_MB = [k for k in ALL_I if k not in MB]
SETTINGS = [
    ("depth",  [2, 4], ALL_B, ALL_I,
                [3],   ALL_B, ALL_I),
    ("h_interp", ALL_D, [b for b in ALL_B if b != BCROW[25000]], ALL_I,
                 ALL_D, [BCROW[25000]], ALL_I),
    ("h_extrap", ALL_D, [b for b in ALL_B if b != BCROW[100000]], ALL_I,
                 ALL_D, [BCROW[100000]], ALL_I),
    ("tech",   ALL_D, ALL_B, NOT_MB,
               ALL_D, ALL_B, MB),
    ("tech_reverse", ALL_D, ALL_B, MB,
                     ALL_D, ALL_B, NOT_MB),
]


def count(groups, dsel, bsel, isel):
    n = 0
    for d in dsel:
        st = stacks[d]
        rows = st.index[st["group"].isin(groups) & st["ic_row"].isin(isel)]
        n += len(rows) * len(bsel)
    return n


print("=" * 78)
print("FIXED CHIPLET SPLIT -- cv fold 0, identical for all eight runs")
print("=" * 78)
print("  train  8 groups : %s" % " ".join(g.replace("group_", "") for g in TRG))
print("  val    1 group  : %s" % VAG[0])
print("  test   1 group  : %s" % TEG[0])
print("  chiplets        : %d train / %d val / %d test"
      % tuple(int((chips["group"].isin(g)).sum()) for g in (TRG, VAG, TEG)))
print()
print("  cooling grid : %s W/m2K  (bc_row %s)"
      % (", ".join("%g" % h for h in HTC), list(range(len(bc)))))
print("  technologies : %s" % ", ".join("%d=%s" % (k, v) for k, v in sorted(TECH.items())))
print()

summary = {}
for name, dT, bT, iT, dE, bE, iE in SETTINGS:
    ntr, nva, nte = (count(TRG, dT, bT, iT), count(VAG, dT, bT, iT),
                     count(TEG, dE, bE, iE))
    full_tr = count(TRG, ALL_D, ALL_B, ALL_I)
    print("-" * 78)
    print("SETTING %-9s  train/val on  depths=%s  h=%s  tech=%s"
          % (name, dT, [int(HTC[b]) for b in bT], [TECH[i] for i in iT]))
    print("%-18s  test     on  depths=%s  h=%s  tech=%s"
          % ("", dE, [int(HTC[b]) for b in bE], [TECH[i] for i in iE]))
    print("   samples   train %6d   val %5d   test %5d      (train is %.1f%% of the"
          " unrestricted %d)" % (ntr, nva, nte, 100.0 * ntr / full_tr, full_tr))
    # leakage proof: enumerate the pairs the loader will actually build for
    # train and val under this restriction, then look for the held-out value
    hd = [d for d in ALL_D if d not in dT]
    hb = [b for b in ALL_B if b not in bT]
    hi = [i for i in ALL_I if i not in iT]
    held_label = (("depth %s" % hd) if hd else
                  ("h %s" % [int(HTC[b]) for b in hb]) if hb else
                  ("tech %s" % [TECH[i] for i in hi]))

    def pairs(groups, dsel, bsel, isel):
        out = []
        for d in dsel:
            st = stacks[d]
            rows = st.index[st["group"].isin(groups) & st["ic_row"].isin(isel)]
            for r in rows:
                for b in bsel:
                    out.append((d, int(st.at[r, "ic_row"]), int(b)))
        return out

    ptr, pva = pairs(TRG, dT, bT, iT), pairs(VAG, dT, bT, iT)
    pte = pairs(TEG, dE, bE, iE)
    assert len(ptr) == ntr and len(pva) == nva and len(pte) == nte
    bad_tr = sum(1 for d, i, b in ptr if d in hd or b in hb or i in hi)
    bad_va = sum(1 for d, i, b in pva if d in hd or b in hb or i in hi)
    # and the test set must be entirely the held-out condition
    bad_te = sum(1 for d, i, b in pte if not (d in hd or b in hb or i in hi))
    withheld = count(TRG, ALL_D, ALL_B, ALL_I) - ntr
    print("   held out  %s   (%d otherwise-available training samples withheld)"
          % (held_label, withheld))
    print("   leakage   held-out value among %d TRAIN pairs: %d | among %d VAL "
          "pairs: %d | non-held-out among %d TEST pairs: %d   %s"
          % (len(ptr), bad_tr, len(pva), bad_va, len(pte), bad_te,
             "OK" if bad_tr == bad_va == bad_te == 0 else "FAIL"))
    assert bad_tr == 0 and bad_va == 0 and bad_te == 0
    # the test chiplet group never appears in train or val
    assert not (set(TEG) & set(TRG + VAG))

    summary[name] = dict(train=ntr, val=nva, test=nte,
                         train_depths=dT, train_bcs=bT, train_ics=iT,
                         test_depths=dE, test_bcs=bE, test_ics=iE,
                         groups=dict(train=TRG, val=VAG, test=TEG))

# ---- restricted normalisation ------------------------------------------------
print()
print("=" * 78)
print("RESTRICTED NORMALISATION (training groups x training conditions only)")
print("=" * 78)
arr = "%s/arrays" % ref
pmap = np.load("%s/powermap_w_m2.f32.npy" % arr, mmap_mode="r")
gmap = np.load("%s/gmap_log10.f32.npy" % arr)
cdt = np.load("%s/solo_dT_ck.i16.npy" % arr, mmap_mode="r")
log10_htc = bc["log10_htc"].to_numpy(dtype=np.float64)
rows = chips.index[chips["group"].isin(TRG)].to_numpy()
assert len(rows) == 400

# powermap: groups only -> identical in all four settings
n = s = ss = 0.0
pmax = -1e30
for r in rows:
    v = np.asarray(pmap[r], dtype=np.float64).ravel()
    n += v.size; s += v.sum(); ss += np.dot(v, v)
    pmax = max(pmax, float(v.max()))
pmean = s / n
pstd = max(ss / n - pmean * pmean, 0.0) ** 0.5

base = json.load(open("%s/norm_cv/fold0.json" % CVDIR))
assert abs(pmean - base["powermap_w_m2"]["mean"]) < 1e-6
assert abs(pstd - base["powermap_w_m2"]["std"]) < 1e-6
print("powermap reproduces cv/norm_cv/fold0.json exactly: mean %.4f std %.4f"
      % (pmean, pstd))

os.makedirs("%s/norm" % OUT, exist_ok=True)
for name, dT, bT, iT, dE, bE, iE in SETTINGS:
    # solo_dT_k over training chiplets AND training bcs
    n2 = s2 = ssq = 0.0
    smax = 0.0
    for r in rows:
        dv = (cdt[r][bT].astype(np.float64) / 100.0).ravel()
        n2 += dv.size; s2 += dv.sum(); ssq += np.dot(dv, dv)
        smax = max(smax, float(np.abs(dv).max()))
    dmean = s2 / n2
    sstd = max(ssq / n2 - dmean * dmean, 0.0) ** 0.5
    g = gmap[iT]
    lh = log10_htc[bT]
    norm = {
        "powermap_w_m2": {"mean": float(pmean), "std": float(pstd), "max": float(pmax)},
        "gmap_log10": {"mean": float(g.mean()), "std": float(g.std()),
                       "min": float(g.min()), "max": float(g.max())},
        "log10_htc": {"mean": float(lh.mean()), "std": float(lh.std())},
        "solo_dT_k": {"std": float(sstd), "absmax": float(smax)},
        "_provenance": {
            "setting": name, "fold": 0, "computed_from_groups": TRG,
            "n_training_chiplets": 400, "excluded_groups": VAG + TEG,
            "training_depths": dT,
            "training_htc_w_m2k": [float(HTC[b]) for b in bT],
            "training_technologies": [TECH[i] for i in iT],
            "note": "every statistic is computed over the restricted TRAINING "
                    "population only: powermap and solo_dT_k over the 8 training "
                    "groups (solo_dT_k additionally restricted to the training "
                    "cooling conditions), gmap_log10 over the training "
                    "technologies, log10_htc over the training cooling "
                    "conditions. Nothing from the held-out condition, the "
                    "validation group or the test group enters any statistic.",
        },
    }
    p = "%s/norm/%s.json" % (OUT, name)
    json.dump(norm, open(p, "w"), indent=2)
    d_base = ("same as fold0" if abs(sstd - base["solo_dT_k"]["std"]) < 1e-12
              and abs(g.mean() - base["gmap_log10"]["mean"]) < 1e-6
              and abs(lh.mean() - base["log10_htc"]["mean"]) < 1e-12
              else "DIFFERS from fold0")
    print("%-9s  gmap %.6f/%.6f  log10_htc %.6f/%.6f  solo_std %.9f   %s"
          % (name, g.mean(), g.std(), lh.mean(), lh.std(), sstd, d_base))

json.dump({"fold": 0, "groups": dict(train=TRG, val=VAG, test=TEG),
           "settings": summary,
           "htc_w_m2k": [float(h) for h in HTC],
           "technologies": {str(k): v for k, v in TECH.items()}},
          open("%s/plan.json" % OUT, "w"), indent=2)
print()
print("wrote %s/plan.json and %s/norm/{depth,h_interp,h_extrap,tech}.json" % (OUT, OUT))
