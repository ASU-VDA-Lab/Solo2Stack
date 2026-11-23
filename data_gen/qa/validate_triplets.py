#!/usr/bin/env python3
"""Structural validation of the sampled stack set, at any depth.

Checks the committed die lists and the generated manifest against the rules the
campaign relies on:

  * every stack uses `n_die` distinct chiplets
  * no stack crosses a chiplet-group boundary, so holding out a group really
    does hold out unseen dies at every depth
  * no duplicate die tuple within a group
  * one manifest record per (stack, bonding technology), and the manifest's
    tuples agree with the committed list
  * the group partition is identical across depths

    python data_gen/qa/validate_triplets.py --n-die 3
    python data_gen/qa/validate_triplets.py --n-die 2 3 4
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
from solo2stack.paths import P, repo_path  # noqa: E402
# --------------------------------------------------------------------------

import argparse
import collections
import json
import os

LIST_DIR = repo_path("data_gen", "stacks", "lists")


def check(n_die, groups, lcf_root, fail):
    doc = json.load(open(os.path.join(LIST_DIR, "stacks_%ddie.json" % n_die)))
    tuples = [(g, t) for g in sorted(doc["groups"]) for t in doc["groups"][g]]
    print("\n=== %d dies (%d layers) ===" % (n_die, 2 * n_die - 1))
    print("  resample seed            : %s" % doc["resample_seed"])
    print("  stacks total             : %d" % len(tuples))

    per_group = collections.Counter(g for g, _ in tuples)
    print("  groups                   : %d" % len(per_group))
    print("  stacks per group         : %s" % sorted(set(per_group.values())))

    dup = [t for _, t in tuples if len(set(t)) != n_die]
    fail("repeated chiplet in a stack", len(dup))

    cross = [(g, t) for g, t in tuples if not set(t) <= set(groups.get(g, ()))]
    fail("stack crosses a group", len(cross))

    seen = collections.Counter((g, tuple(t)) for g, t in tuples)
    fail("duplicate tuple in a group",
         sum(1 for v in seen.values() if v > 1))

    used = {c for _, t in tuples for c in t}
    allc = {c for v in groups.values() for c in v}
    print("  distinct chiplets used   : %d of %d" % (len(used), len(allc)))

    man_path = os.path.join(lcf_root, "manifest_%ddie.json" % n_die)
    if not os.path.exists(man_path):
        print("  manifest                 : not generated yet")
        print("                             (%s)" % man_path)
        return set(per_group)
    recs = json.load(open(man_path))["stacks"]
    techs = collections.Counter(r["interconnect"] for r in recs)
    print("  manifest records         : %d" % len(recs))
    print("  per technology           : %s" % dict(techs))
    fail("records != stacks x techs",
         abs(len(recs) - len(tuples) * len(techs)))
    listed = {g: [list(t) for t in v] for g, v in doc["groups"].items()}
    fail("manifest tuple not in list",
         sum(1 for r in recs
             if [r["d%d" % (i + 1)] for i in range(n_die)]
             not in listed.get(r["group"], [])))
    return set(per_group)


def main():
    ap = argparse.ArgumentParser(
        description="Structural validation of the sampled stack set.")
    ap.add_argument("--n-die", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--lcf-root", default=None,
                    help="where the manifests live; default "
                         "<chiplet_library>/stack_lcf_<N>L per depth")
    a = ap.parse_args()

    problems = []

    def fail(label, count):
        print("  %-26s : %d  (expect 0)" % (label, count))
        if count:
            problems.append(label)

    gj = os.path.join(P["chiplet_library"], "groups", "chiplet_groups.json")
    if not os.path.exists(gj):
        raise SystemExit("chiplet groups not found at %s\n"
                         "run data_gen/floorset/build_chiplet_library.py first"
                         % gj)
    groups = json.load(open(gj))
    print("chiplet groups           : %d, %d chiplets"
          % (len(groups), sum(len(v) for v in groups.values())))

    seen_groups = []
    for n in a.n_die:
        root = a.lcf_root or os.path.join(P["chiplet_library"],
                                          "stack_lcf_%dL" % n)
        seen_groups.append(check(n, groups, root, fail))

    if len(seen_groups) > 1:
        same = all(s == seen_groups[0] for s in seen_groups)
        print("\ngroup partition identical across depths : %s" % same)
        if not same:
            problems.append("group partition differs across depths")

    print("\n%s" % ("all structural checks passed" if not problems
                    else "FAILED: %s" % ", ".join(problems)))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
