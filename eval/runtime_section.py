#!/usr/bin/env python3
"""Runtime speedups and the break-even point, model against PACT at 400x400.

The timing campaign measured each (depth, bonding technology) separately
(runtime_bench/timing.csv), so this is an aggregation of existing measurements
-- no PACT solve and no benchmark is repeated.

Three speedup conventions appear in the paper and are all reported here,
because they answer different questions:

  * single-stack  full_stack_s / (N * solo_s + infer_s): one solo solve per
    die, nothing shared. The pessimistic reading.
  * sweep         full_stack_s / (solo_s * 500 / 4000 + infer_s): the 500 solo
    solves are shared across the 4,000 stacks of each depth.
  * dataset       the same over all 60,000 cases at once.

Writes runtime_mbhb.json into the results directory (`results` in
configs/paths.yaml), where rt_items.py and fig_accuracy_vs_runtime.py read
it. Nothing existing is modified.
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

# timing.csv and runtime_summary.json are measurement records that live beside
# the PACT checkout, not in this repository.
BENCH = os.path.join(P["pact_runs"], "runtime_bench")

TECHS = ["microbump", "hybrid"]
DEPTHS = [2, 3, 4]

# dataset shape, from the packed roots' meta.json (verified, not assumed)
CHIPLETS = 500
STACKS_PER_DEPTH = 4000
N_BC = 5

# measured training wall time per 30-epoch run, from the history.json "secs"
# field of the fold-0 run (mean over epochs x 30).
TRAIN_H = 548.1 * 30 / 3600.0


def read_timing():
    """mean wall time per config over the reps in timing.csv"""
    rows = {}
    with open(os.path.join(BENCH, "timing.csv")) as fh:
        next(fh)
        for line in fh:
            cfg, rep, wall, rss = line.strip().split(",")
            rows.setdefault(cfg, []).append(float(wall))
    return {k: sum(v) / len(v) for k, v in rows.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Runtime speedups and break-even at 400x400.")
    ap.add_argument("--out", default=None,
                    help="default <results>/runtime_mbhb.json")
    ap.add_argument("--force", action="store_true",
                    help="replace the output even if it was written by "
                         "something else")
    opts = ap.parse_args(argv)

    t = read_timing()
    pub = json.load(open(os.path.join(BENCH, "runtime_summary.json")))
    solo_s = t["solo"]
    infer_s = pub["infer_ms"] / 1000.0

    def full_stack(depth, techs):
        v = [t["d%d_%s" % (depth, x)] for x in techs]
        return sum(v) / len(v)

    def amortised(stacks):
        """Solo2Stack cost per stack: solo sims shared across the sweep."""
        return solo_s * CHIPLETS / stacks + infer_s

    out = {"note": "Runtime aggregation over microbump + hybrid at 400x400.",
           "generated_by": "eval/runtime_section.py",
           "source": "runtime_bench/timing.csv (2 reps, one job at a time, "
                     "single-threaded, SciPy SuperLU); no solve repeated",
           "solo_s": round(solo_s, 4),
           "infer_ms_batch1": pub["infer_ms"],
           "per_technology_full_stack_s": {
               "d%d_%s" % (d, x): round(t["d%d_%s" % (d, x)], 4)
               for d in DEPTHS for x in TECHS},
           }

    arms = {}
    for tag, techs in (("new", TECHS),):
        stacks = STACKS_PER_DEPTH
        cost = amortised(stacks)
        a = {
            "technologies": techs,
            "stacks_per_depth": stacks,
            "stacks_per_solo_sim": stacks / CHIPLETS,
            "solo2stack_cost_per_stack_s": round(cost, 6),
            "full_stack_s": {},
            "sweep_speedup": {},
            "single_stack_speedup": {},
            "break_even_stacks": {},
        }
        for d in DEPTHS:
            fs = full_stack(d, techs)
            a["full_stack_s"][str(d)] = round(fs, 4)
            a["sweep_speedup"][str(d)] = round(fs / cost, 4)
            # one stack in isolation: d solo solves are not amortised at all
            single = d * solo_s + infer_s
            a["single_stack_speedup"][str(d)] = round(fs / single, 4)
            # break-even: n stacks over the same d chiplets.
            #   PACT    = n * fs
            #   Solo2Stack = d * solo_s + n * infer_s
            # equal at n = d*solo_s / (fs - infer_s)
            a["break_even_stacks"][str(d)] = round(
                d * solo_s / (fs - infer_s), 4)
        arms[tag] = a

    # total full-stack simulation cost for the whole dataset
    for tag, techs in (("new", TECHS),):
        stacks = STACKS_PER_DEPTH
        per_tech_stacks = stacks // len(techs)        # tuples per technology
        tot = 0.0
        for d in DEPTHS:
            for x in techs:
                tot += per_tech_stacks * N_BC * t["d%d_%s" % (d, x)]
        arms[tag]["dataset_cases"] = len(DEPTHS) * stacks * N_BC
        arms[tag]["full_stack_total_s"] = round(tot, 1)
        arms[tag]["full_stack_total_h"] = round(tot / 3600.0, 1)
        arms[tag]["full_stack_total_days_1cpu"] = round(tot / 86400.0, 2)
        solo_tot = CHIPLETS * N_BC * solo_s
        arms[tag]["solo_total_s"] = round(solo_tot, 1)
        arms[tag]["solo_total_h"] = round(solo_tot / 3600.0, 2)
        arms[tag]["dataset_speedup_vs_full_stack"] = round(
            tot / (solo_tot + arms[tag]["dataset_cases"] * infer_s), 2)

        # Break-even including the one-time cost of producing the surrogate:
        # the labelled dataset (full-stack + solo PACT) plus one training run.
        #   PACT        = n * fs
        #   Solo2Stack  = setup + n * amortised_cost
        # n = setup / (fs - amortised_cost)
        setup_s = tot + solo_tot + TRAIN_H * 3600.0
        arms[tag]["train_h_per_run"] = round(TRAIN_H, 3)
        arms[tag]["setup_total_h"] = round(setup_s / 3600.0, 1)
        cost = arms[tag]["solo2stack_cost_per_stack_s"]
        arms[tag]["break_even_stacks_incl_setup"] = {
            str(d): int(round(setup_s / (arms[tag]["full_stack_s"][str(d)] - cost)))
            for d in DEPTHS}

    out["arms"] = arms

    dst = opts.out or os.path.join(P["results"], "runtime_mbhb.json")
    if os.path.exists(dst) and not opts.force:
        # only ever replace this script's own output
        prev = json.load(open(dst))
        assert prev.get("generated_by") == out["generated_by"], \
            "refusing to overwrite %s" % dst
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)

    n = arms["new"]
    print("solo %.3f s   inference %.3f ms" % (solo_s, pub["infer_ms"]))
    print("amortised Solo2Stack cost/stack: %.4f s (1 solo : %g stacks)"
          % (n["solo2stack_cost_per_stack_s"], n["stacks_per_solo_sim"]))
    print("\n%-6s | %19s | %13s | %16s | %s"
          % ("dies", "full-stack PACT (s)", "sweep sp.",
             "single-stack sp.", "break-even stacks"))
    for d in DEPTHS:
        k = str(d)
        print("%-6d | %19.2f | %13.1f | %16.2f | %.2f"
              % (d, n["full_stack_s"][k], n["sweep_speedup"][k],
                 n["single_stack_speedup"][k], n["break_even_stacks"][k]))
    print("\nbreak-even including the one-time cost of producing the "
          "surrogate\n  (labelled dataset + one %s h training run):"
          % n["train_h_per_run"])
    for d in DEPTHS:
        print("  %d dies %14d" % (d, n["break_even_stacks_incl_setup"][str(d)]))

    print("\n%-34s %14s" % ("dataset-level", "value"))
    for key, label in (("dataset_cases", "cases"),
                       ("full_stack_total_h", "full-stack PACT total (h)"),
                       ("full_stack_total_days_1cpu", "   = CPU-days"),
                       ("solo_total_h", "solo PACT total (h)"),
                       ("train_h_per_run", "training (h / run)"),
                       ("setup_total_h", "one-time setup total (h)"),
                       ("dataset_speedup_vs_full_stack", "dataset speedup")):
        print("%-34s %14s" % (label, n[key]))
    print("\nwrote %s" % dst)


if __name__ == "__main__":
    main()
