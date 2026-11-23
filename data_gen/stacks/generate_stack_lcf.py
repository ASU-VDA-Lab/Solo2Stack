#!/usr/bin/env python3
"""Generate stack LCF files for any number of dies.

Replaces generate_stack_lcf_{2L,3L,4L}.py with one depth-generic generator.
A stack of N dies has 2N-1 physical layers, D1 I12 D2 I23 D3 ... DN, and every
interconnect layer in a stack uses the same bonding technology.

Which stacks get generated
--------------------------
By default the die tuples are read from the committed list in lists/, which is
the authoritative record of the published campaign:

    lists/stacks_2die.json   lists/stacks_3die.json   lists/stacks_4die.json

`--resample` redraws them instead, with the original per-depth seed and the
original sampling algorithm:

  * N = 3  enumerate every ordered triplet of the group, shuffle, take the
           first `samples_per_group`
  * N = 4  rejection-sample distinct ordered quadruplets (the full ordered set
           is 50*49*48*47 = 5,527,200 per group, too large to enumerate)
  * N = 2  NOT AVAILABLE.  The published 2-die draw was made with an unseeded
           random.shuffle(), so it cannot be reproduced.  The committed list is
           its only record.

Byte-compatibility
------------------
The emitted .lcf.csv files are byte-identical to the originals, which is what
gate G1 checks.  That requires keeping the original per-depth stack_id spelling
(`bot_/top_` at N=2, `d1_/d2_/...` at N>=3) because the stack_id is the file
name.  Layer paths inside the LCF stay relative to the chiplet-library
directory, because PACT is invoked with that directory's parent as cwd.
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
# --------------------------------------------------------------------------

import argparse
import json
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIST_DIR = os.path.join(HERE, "lists")

# Bonding technologies. Thicknesses are effective modelling parameters, not
# exact physical ones: they preserve realistic thermal behaviour while keeping
# the LU factorisation well conditioned. Copied verbatim from the per-depth
# generators so the geometry and conductance maps are unchanged.
BOND = {
    "microbump": {                      # pitch 50 um, pad 30 um -> 40x40 grid
        "flp": "interconnect_flp_grid/microbump_40x40_30um_flp.csv",
        "thickness": 0.000025,          # 25 um
    },
    "hybrid": {                         # pitch 10 um, pad 5 um -> 200x200 pads
        "flp": "interconnect_flp_grid/hybrid_100x100_5um_flp.csv",
        "thickness": 0.0000015,         # 1.5 um
    },
}

CHIPLET_THICKNESS = 0.0005              # 500 um

# Seeds used for the published draws. N=2 is absent on purpose: it was unseeded.
RESAMPLE_SEED = {3: 20260908, 4: 20260910}


def stack_id(n_die, group, idx, dies, tech):
    """The published stack_id spelling, which differs between N=2 and N>=3."""
    if n_die == 2:
        body = "bot_%s_top_%s" % (dies[0], dies[1])
    else:
        body = "_".join("d%d_%s" % (i + 1, c) for i, c in enumerate(dies))
    return "%s_s%04d_%s_%s" % (group, idx, body, tech)


def lcf_frame(dies, tech, prefix):
    """The 2N-1 layer table for one stack, in PACT's LCF column order."""
    bond = BOND[tech]
    rows = []
    for i, cid in enumerate(dies):
        if i:                                        # bond layer below this die
            rows.append({
                "Layer": len(rows),
                "FloorplanFile": "%s/%s" % (prefix, bond["flp"]),
                "Thickness (m)": bond["thickness"],
                "PtraceFile": "",
                "LateralHeatFlow": True,
            })
        rows.append({
            "Layer": len(rows),
            "FloorplanFile": "%s/raw_floorplans/%s_flp.csv" % (prefix, cid),
            "Thickness (m)": CHIPLET_THICKNESS,
            "PtraceFile": "%s/raw_ptraces/%s_ptrace.csv" % (prefix, cid),
            "LateralHeatFlow": True,
        })
    return pd.DataFrame(rows)


def load_list(n_die):
    path = os.path.join(LIST_DIR, "stacks_%ddie.json" % n_die)
    if not os.path.exists(path):
        raise SystemExit("no committed list for N=%d: %s" % (n_die, path))
    doc = json.load(open(path))
    if doc["n_die"] != n_die:
        raise SystemExit("%s says n_die=%s" % (path, doc["n_die"]))
    return doc


def resample(n_die, groups, per_group):
    """Redraw the die tuples with the original seed and algorithm."""
    if n_die not in RESAMPLE_SEED:
        raise SystemExit(
            "--resample is not available for N=%d.\n"
            "The published %d-die draw used an unseeded random.shuffle(), so it "
            "cannot be reproduced; data_gen/stacks/lists/stacks_%ddie.json is "
            "its only record." % (n_die, n_die, n_die))
    seed = RESAMPLE_SEED[n_die]
    out = {}
    for gname in sorted(groups):
        chiplets = list(groups[gname])
        rng = random.Random("%s|%d" % (gname, seed))
        if n_die == 3:
            # full ordered enumeration, then shuffle: the nested-loop order
            # below is part of the reproducible stream and must not change
            tuples = [(a, b, c)
                      for a in chiplets
                      for b in chiplets
                      for c in chiplets
                      if a != b and a != c and b != c]
            rng.shuffle(tuples)
            out[gname] = [list(t) for t in tuples[:per_group]]
        else:
            seen, picked = set(), []
            while len(picked) < per_group:
                q = tuple(rng.sample(chiplets, n_die))
                if q in seen:
                    continue
                seen.add(q)
                picked.append(list(q))
            out[gname] = picked
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Generate N-die stack LCF files (depth-generic).")
    ap.add_argument("--n-die", type=int, required=True,
                    help="number of dies in the stack (2, 3, 4, ...)")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to "
                         "<chiplet_library>/stack_lcf_<N>L")
    ap.add_argument("--resample", action="store_true",
                    help="redraw the die tuples instead of reading the "
                         "committed list (N>=3 only)")
    ap.add_argument("--technologies", nargs="+", default=None,
                    help="bonding technologies to emit (default: the list's)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output directory")
    a = ap.parse_args()

    if a.n_die < 2:
        raise SystemExit("--n-die must be at least 2")

    lib = P["chiplet_library"]
    prefix = os.path.basename(lib.rstrip("/"))   # PACT runs with lib's parent as cwd
    out_dir = a.out or os.path.join(lib, "stack_lcf_%dL" % a.n_die)

    doc = load_list(a.n_die)
    techs = a.technologies or doc["technologies"]
    unknown = [t for t in techs if t not in BOND]
    if unknown:
        raise SystemExit("unknown bonding technology: %s (have %s)"
                         % (unknown, sorted(BOND)))

    groups_json = os.path.join(lib, "groups", "chiplet_groups.json")
    if a.resample:
        if not os.path.exists(groups_json):
            raise SystemExit("--resample needs %s" % groups_json)
        groups = resample(a.n_die, json.load(open(groups_json)),
                          doc["samples_per_group"])
        source = "resampled, seed %d" % RESAMPLE_SEED[a.n_die]
    else:
        groups = doc["groups"]
        source = "committed list stacks_%ddie.json" % a.n_die

    if os.path.isdir(out_dir) and os.listdir(out_dir) and not a.force:
        raise SystemExit("%s is not empty; pass --force to overwrite" % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("N=%d dies, %d layers, technologies %s"
          % (a.n_die, 2 * a.n_die - 1, ", ".join(techs)))
    print("source : %s" % source)
    print("out    : %s" % out_dir)

    records = []
    for gname in sorted(groups):
        tuples = groups[gname]
        for idx, dies in enumerate(tuples):
            if len(dies) != a.n_die:
                raise SystemExit("%s[%d] has %d dies, expected %d"
                                 % (gname, idx, len(dies), a.n_die))
            for tech in techs:
                sid = stack_id(a.n_die, gname, idx, dies, tech)
                path = os.path.join(out_dir, gname, tech, "%s.lcf.csv" % sid)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                lcf_frame(dies, tech, prefix).to_csv(path, index=False)

                rec = {"stack_id": sid, "group": gname, "stack_idx": idx,
                       "interconnect": tech, "lcf_path": path}
                for i, cid in enumerate(dies):
                    rec["d%d" % (i + 1)] = cid
                records.append(rec)
        print("  %s: %d stacks x %d technologies" % (gname, len(tuples), len(techs)))

    man = os.path.join(out_dir, "manifest_%ddie.json" % a.n_die)
    json.dump({"n_die": a.n_die, "n_layers": 2 * a.n_die - 1,
               "technologies": list(techs), "source": source,
               "stacks": records}, open(man, "w"), indent=1)

    print("\ntotal LCF files : %d" % len(records))
    print("manifest        : %s" % man)


if __name__ == "__main__":
    main()
