#!/usr/bin/env python3
"""Aggregate the transfer experiments into the paper's transfer table.

Four held-out settings, two arms each (A1 and Solo2Stack): a cooling condition
interior to the training range, a cooling condition at its edge, an unseen
stack depth, and an unseen bonding technology.  Reads each run's result.json
and writes comparison.json next to them.
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

NEW = P["transfer_dir"]

SETTINGS = ["h_interp", "h_extrap", "depth", "tech_reverse"]
LABEL = {"h_interp": "Cooling, interpolation (hold out htc 25k)",
         "h_extrap": "Cooling, extrapolation (hold out htc 100k)",
         "depth": "Number of dies (train 2+4, test 3)",
         "tech_reverse": "Bonding (microbump -> hybrid)"}

MET = [("mae_k", "MAE"), ("mae_pattern_k", "Pat. MAE"),
       ("r2_pattern", "R2_pat"), ("hotspot_top1%_overlap", "Hotspot"),
       ("p99_k", "p99"), ("max_k", "Max")]


def load(base, tag):
    f = os.path.join(base, "runs", tag, "result.json")
    return json.load(open(f)) if os.path.exists(f) else None


def names(base):
    return json.load(open(os.path.join(base, "plan.json")))["technologies"]


def main():
    nn = names(NEW)
    out = {"technologies": nn, "settings": {}}
    lines = []
    for s in SETTINGS:
        for arm in ("A1", "A3"):
            n = load(NEW, "%s_%s" % (s, arm))
            assert n is not None, "missing run %s_%s" % (s, arm)
            out["settings"]["%s_%s" % (s, arm)] = {
                "arm": arm,
                "setting": s,
                "label": LABEL[s],
                "restriction": n["restriction"],
                "train_tech": [nn[str(i)] for i in n["restriction"]["train_ics"]],
                "test_tech": [nn[str(i)] for i in n["restriction"]["test_ics"]],
                "n": n["n_samples"],
                "test": n["test"]["pooled"],
                "best_val_r2_pattern": n["best_val_r2_pattern"],
                "per_depth": n["test"].get("per_depth"),
                "wall_s": n["wall_seconds"],
            }
            for k, lab in MET:
                lines.append((LABEL[s], arm, lab, n["test"]["pooled"][k]))

    dst = os.path.join(NEW, "comparison.json")
    json.dump(out, open(dst, "w"), indent=2)

    w = max(len(l[0]) for l in lines)
    hdr = "%-*s %-4s %-9s %12s" % (w, "setting", "arm", "metric", "value")
    print(hdr)
    print("-" * len(hdr))
    for lab, arm, met, v in lines:
        print("%-*s %-4s %-9s %12.4f" % (w, lab, arm, met, v))
    print("\nwrote %s" % dst)


if __name__ == "__main__":
    main()
