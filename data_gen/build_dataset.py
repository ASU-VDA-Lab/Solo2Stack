#!/usr/bin/env python3
"""Build the stack dataset end to end, for one or more stack depths.

Three stages per depth, each skipped when its output is already there, so an
interrupted build can simply be rerun:

    1  stacks   data_gen/stacks/generate_stack_lcf.py --n-die N
    2  solve    data_gen/run/run_stack.py             --n-die N
    3  pack     data_gen/pack/pack_dataset.py         --n-die N

Stage 2 is the expensive one: tens of thousands of CPU-hours for the full
campaign.  Nothing here writes into an existing data root unless you point it
at one; --out-root and --pack-root default to the locations configured in
configs/paths.yaml, and --dry-run prints the three command lines per depth
without running anything.

    python data_gen/build_dataset.py --depth 2 3 4 --dry-run
    python data_gen/build_dataset.py --depth 3 --jobs 24
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
import glob
import os
import subprocess
import sys


def done_stacks(lcf_root, n_die):
    man = os.path.join(lcf_root, "manifest_%ddie.json" % n_die)
    return os.path.exists(man)


def done_solve(pact_out, n_die):
    """A solve stage counts as done when every layer of the last stack is there."""
    if not os.path.isdir(pact_out):
        return False
    return bool(glob.glob(os.path.join(
        pact_out, "*.grid.steady.layer%d" % (2 * n_die - 2))))


def done_pack(pack_dir):
    return os.path.exists(os.path.join(pack_dir, "meta.json"))


def run(cmd, dry):
    print("    $ %s" % " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(
        description="Build the N-die stack datasets end to end.")
    ap.add_argument("--depth", type=int, nargs="+", required=True,
                    help="stack depths to build, e.g. --depth 2 3 4")
    ap.add_argument("--lcf-root", default=None,
                    help="where stack LCFs live; default <chiplet_library>/stack_lcf_<N>L")
    ap.add_argument("--out-root", default=None,
                    help="where PACT output goes; default pact_runs")
    ap.add_argument("--pack-root", default=None,
                    help="where packed datasets go; default <data_root>")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--resample", action="store_true",
                    help="redraw the die tuples (N>=3 only)")
    ap.add_argument("--force", action="store_true",
                    help="rerun stages even when their output exists")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bad = [d for d in a.depth if d < 2]
    if bad:
        raise SystemExit("depths must be at least 2: %s" % bad)

    py = sys.executable
    out_root = a.out_root or P["pact_runs"]
    pack_root = a.pack_root or P["data_root"]
    rc_total = 0

    for n in a.depth:
        lcf_root = a.lcf_root or os.path.join(P["chiplet_library"],
                                              "stack_lcf_%dL" % n)
        pact_out = os.path.join(out_root, "output_stack_%ddie" % n)
        pack_dir = os.path.join(pack_root, "pact_stack_%ddie" % n)

        print("=== %d dies (%d layers) ===" % (n, 2 * n - 1))

        # ---- 1. stack LCFs ----
        if done_stacks(lcf_root, n) and not a.force:
            print("    [skip] stacks: %s/manifest_%ddie.json exists"
                  % (lcf_root, n))
        else:
            cmd = [py, repo_path("data_gen", "stacks", "generate_stack_lcf.py"),
                   "--n-die", str(n), "--out", lcf_root]
            if a.resample:
                cmd.append("--resample")
            if a.force:
                cmd.append("--force")
            rc_total |= run(cmd, a.dry_run)

        # ---- 2. PACT solves ----
        if done_solve(pact_out, n) and not a.force:
            print("    [skip] solve: %s already has output" % pact_out)
        else:
            rc_total |= run([py, repo_path("data_gen", "run", "run_stack.py"),
                             "--n-die", str(n), "--lcf-root", lcf_root,
                             "--out-root", out_root, "--jobs", str(a.jobs)],
                            a.dry_run)

        # ---- 3. pack ----
        if done_pack(pack_dir) and not a.force:
            print("    [skip] pack: %s/meta.json exists" % pack_dir)
        else:
            rc_total |= run([py, repo_path("data_gen", "pack", "pack_dataset.py"),
                             "--n-die", str(n), "--lcf-root", lcf_root,
                             "--pact-out", pact_out, "--out", pack_dir,
                             "--workers", str(a.workers)], a.dry_run)

    if a.dry_run:
        print("\ndry run: nothing executed")
    elif rc_total:
        raise SystemExit("one or more stages failed")
    else:
        print("\nall depths built")


if __name__ == "__main__":
    main()
