#!/usr/bin/env python3
"""Build the 10-fold assignment and per-fold normalisation for the A0-A3 factorial.

Fold k:  test = groups[k],  val = groups[(k+1) % 10],  train = the other eight.
Every group is the test group exactly once.

The fold is defined purely by GROUP NAME, and the group names are identical
across the 2-, 3- and 4-die datasets, so one fold definition holds out the same
chiplet population at every stack depth simultaneously. Stacks only draw
chiplets from within a single group (verified here), so holding out a group
holds out every chiplet in it, at every depth.

Normalisation is recomputed from scratch for each fold using ONLY that fold's
eight training groups. Nothing from the validation or test groups enters the
statistics. The dataset-level norm_stats.json from the single-split experiment
is not reused.
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
import json
import os

import numpy as np
import pandas as pd

DIE_COLS = {2: ["bot_chiplet", "top_chiplet"],
            3: ["d1_chiplet", "d2_chiplet", "d3_chiplet"],
            4: ["d1_chiplet", "d2_chiplet", "d3_chiplet", "d4_chiplet"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=P["data_root"])
    ap.add_argument("--out", default=P["cv_dir"])
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "norm_cv"), exist_ok=True)

    roots = {d: os.path.join(a.base_dir, "pact_stack_%dL" % d) for d in (2, 3, 4)}
    ref = roots[2]
    chips = pd.read_csv(os.path.join(ref, "index", "chiplets.csv"))
    bc = pd.read_csv(os.path.join(ref, "index", "bc.csv"))
    meta = json.load(open(os.path.join(ref, "meta.json")))
    groups = sorted(chips["group"].unique())
    assert len(groups) == 10, "expected 10 groups, found %d" % len(groups)

    # ---- fold assignment -------------------------------------------------
    folds = []
    for k in range(10):
        test = [groups[k]]
        val = [groups[(k + 1) % 10]]
        train = [g for g in groups if g not in test + val]
        assert len(train) == 8
        folds.append({"fold": k, "train": train, "val": val, "test": test})

    # ---- leakage verification across all three depths --------------------
    chip_group = dict(zip(chips["chiplet_id"], chips["group"]))
    leak = []
    stacks_by_depth = {}
    for d, root in roots.items():
        st = pd.read_csv(os.path.join(root, "index", "stacks.csv"))
        stacks_by_depth[d] = st
        for c in DIE_COLS[d]:
            bad = (st[c].map(chip_group) != st["group"]).sum()
            if bad:
                leak.append("depth %d column %s: %d rows" % (d, c, bad))
    fold_leak = []
    for f in folds:
        held = set(f["val"] + f["test"])
        held_chips = {c for c, g in chip_group.items() if g in held}
        for d, st in stacks_by_depth.items():
            tr = st[st["group"].isin(f["train"])]
            used = set()
            for c in DIE_COLS[d]:
                used |= set(tr[c].unique())
            inter = used & held_chips
            if inter:
                fold_leak.append("fold %d depth %d: %d leaked chiplets"
                                 % (f["fold"], d, len(inter)))

    # ---- per-fold normalisation, training groups only --------------------
    arr = os.path.join(ref, "arrays")
    pmap = np.load(os.path.join(arr, "powermap_w_m2.f32.npy"), mmap_mode="r")
    gmap = np.load(os.path.join(arr, "gmap_log10.f32.npy"))
    cdt = np.load(os.path.join(arr, "solo_dT_ck.i16.npy"), mmap_mode="r")
    coff = np.load(os.path.join(arr, "solo_off_ck.i32.npy"))
    t_amb, area = meta["ambient_k"], meta["die_area_m2"]
    htc = bc["htc_w_m2k"].to_numpy(dtype=np.float64)
    pw = chips["total_power_w"].to_numpy(dtype=np.float64)
    ana_solo = t_amb + pw[:, None] / (htc[None, :] * area)
    log10_htc = bc["log10_htc"].to_numpy(dtype=np.float64)

    for f in folds:
        rows = chips.index[chips["group"].isin(f["train"])].to_numpy()
        assert len(rows) == 400, "fold %d has %d training chiplets" % (f["fold"], len(rows))

        n = s = ss = 0.0
        pmax = -1e30
        for r in rows:
            v = np.asarray(pmap[r], dtype=np.float64).ravel()
            n += v.size
            s += v.sum()
            ss += np.dot(v, v)
            pmax = max(pmax, float(v.max()))
        pmean = s / n
        pstd = max(ss / n - pmean * pmean, 0.0) ** 0.5

        # solo_dT_k must keep the PACKER's definition -- the std of the packed
        # deviation from each field's own offset -- because the frozen loader
        # divides the analytic solo residual by this constant. Redefining it as
        # the residual's own std (2.755 K rather than 0.423 K) would rescale the
        # solo input channels ~6.5x and silently change the frozen setup. Only
        # the population it is computed over changes: training groups only.
        n2 = s2 = ssq = 0.0
        smax = 0.0
        for r in rows:
            dv = (cdt[r].astype(np.float64) / 100.0).ravel()
            n2 += dv.size
            s2 += dv.sum()
            ssq += np.dot(dv, dv)
            smax = max(smax, float(np.abs(dv).max()))
        dmean = s2 / n2
        sstd = max(ssq / n2 - dmean * dmean, 0.0) ** 0.5

        norm = {
            "powermap_w_m2": {"mean": float(pmean), "std": float(pstd),
                              "max": float(pmax)},
            "gmap_log10": {"mean": float(gmap.mean()), "std": float(gmap.std()),
                           "min": float(gmap.min()), "max": float(gmap.max())},
            "log10_htc": {"mean": float(log10_htc.mean()),
                          "std": float(log10_htc.std())},
            "solo_dT_k": {"std": float(sstd), "absmax": float(smax)},
            "_provenance": {
                "fold": f["fold"],
                "computed_from_groups": f["train"],
                "n_training_chiplets": int(len(rows)),
                "excluded_groups": f["val"] + f["test"],
                # The technology count is read from the gmap array rather than
                # written out as a word, so a root holding any subset of the
                # bonding technologies records its own count.
                "note": "powermap and solo_dT_k use ONLY the 8 training "
                        "groups. gmap_log10 and log10_htc are properties of "
                        "the %d bonding technologies and the five cooling "
                        "conditions, all of which appear in every group, so "
                        "they are fold-invariant by construction."
                        % gmap.shape[0],
            },
        }
        json.dump(norm, open(os.path.join(a.out, "norm_cv",
                                          "fold%d.json" % f["fold"]), "w"), indent=2)
        print("fold %d norm written (pmap mean %.1f std %.1f | solo std %.6f)"
              % (f["fold"], pmean, pstd, sstd), flush=True)

    json.dump({"folds": folds, "groups": groups}, open(
        os.path.join(a.out, "cv_folds.json"), "w"), indent=2)

    print("\n" + "=" * 72)
    print("FOLD ASSIGNMENT")
    print("=" * 72)
    print("%-5s %-10s %-10s %s" % ("fold", "test", "val", "train (8 groups)"))
    for f in folds:
        print("%-5d %-10s %-10s %s"
              % (f["fold"], f["test"][0], f["val"][0],
                 " ".join(g.replace("group_", "") for g in f["train"])))
    cnt = {}
    for f in folds:
        cnt[f["test"][0]] = cnt.get(f["test"][0], 0) + 1
    print("\nevery group is test exactly once: %s" % (set(cnt.values()) == {1}))
    print("stacks never cross group boundaries (all depths): %s"
          % (not leak) + ("" if not leak else "  %s" % leak))
    print("no held-out chiplet appears in any fold's training set: %s"
          % (not fold_leak) + ("" if not fold_leak else "  %s" % fold_leak))
    print("\nsaved -> %s" % a.out)


if __name__ == "__main__":
    main()
