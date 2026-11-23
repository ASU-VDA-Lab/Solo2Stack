#!/usr/bin/env python3
"""Run the PACT sweep over an N-die stack dataset.

Replaces run_stack_2L.sh, run_stack_3L.py and run_stack_4L.py with one
depth-generic driver, and adopts the 3-/4-die behaviour at every depth:

  * SciPy sparse LU (`splu`), the only backend this repository ships.  The
    published 2-die and solo labels were originally produced with an external
    SuperLU_MT driver, which has been removed; `splu` reproduces a stored 2-die
    label to within the 0.01 K resolution of PACT's own output files (max
    0.01 K, mean 2.0e-7 K).  See pact/README.md.
  * job-level parallelism with internal threading pinned to 1
  * per-sample validation (layer count, finiteness, shape, energy balance,
    die monotonicity) through data_gen/run/grid_io.py
  * resume that re-validates an existing result instead of trusting that
    layer0 exists
  * an append-only JSONL manifest carrying status, runtime and solver stats

Everything outside the repository is resolved through configs/paths.yaml.

    python data_gen/run/run_stack.py --n-die 3 --jobs 24
    python data_gen/run/run_stack.py --n-die 4 --out-root /scratch/try --limit 10
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
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_io import t_analytic, validate_stack  # noqa: E402

# (name, htc) for the five cooling conditions, matching configs/pact/bc_*.config
BCS = [("bc_0", 5000), ("bc_1", 10000), ("bc_2", 25000),
       ("bc_3", 50000), ("bc_4", 100000)]

_lock = threading.Lock()
_counts = {"completed": 0, "failed": 0, "skipped": 0}
_cfg = {}


def emit(rec):
    with _lock:
        with open(_cfg["manifest"], "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())


def run_one(job):
    rec = dict(job)
    n_die = job["n_die"]
    tag = "%s_%s" % (job["stack_id"], job["bc_name"])
    prefix = os.path.join(_cfg["outdir"], "%s.grid.steady" % tag)
    rec["output_prefix"] = prefix
    rec["T_ana"] = t_analytic(job["htc"], n_die)

    # resume: accept only a fully valid existing result
    ok, info = validate_stack(prefix, job["htc"], n_die)
    if ok:
        rec.update(status="completed", reused=True, runtime_s=None,
                   validation=info)
        with _lock:
            _counts["skipped"] += 1
        emit(rec)
        return rec

    log = os.path.join(_cfg["logdir"], "%s.log" % tag)
    stats_path = os.path.join(_cfg["statdir"], "%s.stats.json" % tag)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PACT3L_STATS=stats_path)
    cmd = [_cfg["python"], _cfg["pact"], job["lcf_path"],
           repo_path("configs", "pact", "%s.config" % job["bc_name"]),
           _cfg["model"], "--gridSteadyFile", prefix]
    rec["command"] = " ".join(cmd)

    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.run(cmd, cwd=_cfg["cwd"], env=env, stdout=lf,
                           stderr=subprocess.STDOUT)
    rec["runtime_s"] = round(time.time() - t0, 2)
    rec["returncode"] = p.returncode

    if os.path.exists(stats_path):
        try:
            rec["solver"] = json.load(open(stats_path))
        except Exception:                                   # noqa: BLE001
            pass
        os.unlink(stats_path)

    if p.returncode != 0:
        rec.update(status="failed",
                   error="PACT exited %d; see %s" % (p.returncode, log))
        with _lock:
            _counts["failed"] += 1
        emit(rec)
        return rec

    ok, info = validate_stack(prefix, job["htc"], n_die)
    rec["validation"] = info
    if not ok:
        rec.update(status="failed",
                   error="validation: %s" % info.get("error", "unknown"))
        with _lock:
            _counts["failed"] += 1
    else:
        rec.update(status="completed", reused=False)
        with _lock:
            _counts["completed"] += 1
    emit(rec)
    return rec


def build_jobs(manifest, n_die, group=None, technologies=None, bcs=None):
    jobs = []
    for r in manifest["stacks"]:
        if group and r["group"] != group:
            continue
        if technologies and r["interconnect"] not in technologies:
            continue
        dies = [r["d%d" % (i + 1)] for i in range(n_die)]
        for bc_name, htc in BCS:
            if bcs and bc_name not in bcs:
                continue
            jobs.append(dict(
                stack_id=r["stack_id"], group=r["group"],
                stack_idx=r["stack_idx"], n_die=n_die,
                dies=dies, technology=r["interconnect"],
                bc_name=bc_name, htc=htc, lcf_path=r["lcf_path"],
                solo_refs=[os.path.join(P["solo_runs"],
                                        "%s_%s.grid.steady.layer0" % (c, bc_name))
                           for c in dies],
            ))
    return jobs


def main():
    ap = argparse.ArgumentParser(
        description="PACT sweep over an N-die stack dataset.")
    ap.add_argument("--n-die", type=int, required=True)
    ap.add_argument("--lcf-root", default=None,
                    help="directory holding manifest_<N>die.json; defaults to "
                         "<chiplet_library>/stack_lcf_<N>L")
    ap.add_argument("--out-root", default=None,
                    help="where outputs/logs/manifest go; defaults to pact_runs")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--group", default=None)
    ap.add_argument("--technologies", nargs="+", default=None)
    ap.add_argument("--bc", nargs="+", default=None,
                    help="cooling conditions to run (default: all five)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.n_die < 2:
        raise SystemExit("--n-die must be at least 2")

    lcf_root = a.lcf_root or os.path.join(P["chiplet_library"],
                                          "stack_lcf_%dL" % a.n_die)
    out_root = a.out_root or P["pact_runs"]
    tag = "%ddie" % a.n_die

    _cfg.update(
        python=P["pact_python"],
        pact=os.path.join(P["pact_src"], "PACT.py"),
        model=repo_path("configs", "pact", "modelParam.config"),
        # PACT resolves the relative layer paths inside each LCF against its
        # cwd, which must therefore be the chiplet library's parent
        cwd=os.path.dirname(P["chiplet_library"].rstrip("/")),
        outdir=os.path.join(out_root, "output_stack_%s" % tag),
        logdir=os.path.join(out_root, "logs_stack_%s" % tag),
        statdir=os.path.join(out_root, "workdirs_%s" % tag),
        manifest=os.path.join(out_root, "manifest", "manifest_%s.jsonl" % tag),
    )

    man_path = os.path.join(lcf_root, "manifest_%s.json" % tag)
    if not os.path.exists(man_path):
        raise SystemExit("no stack manifest at %s\n"
                         "run data_gen/stacks/generate_stack_lcf.py --n-die %d"
                         % (man_path, a.n_die))
    man = json.load(open(man_path))
    jobs = build_jobs(man, a.n_die, a.group, a.technologies, a.bc)
    if a.limit:
        jobs = jobs[:a.limit]

    print("N=%d dies (%d layers)   jobs: %d   concurrency: %d"
          % (a.n_die, 2 * a.n_die - 1, len(jobs), a.jobs))
    print("PACT      : %s" % _cfg["pact"])
    print("python    : %s" % _cfg["python"])
    print("cwd       : %s" % _cfg["cwd"])
    print("outputs   : %s" % _cfg["outdir"])
    if a.dry_run:
        for j in jobs[:3]:
            print("  would run %s_%s" % (j["stack_id"], j["bc_name"]))
        print("dry run: nothing executed")
        return

    for d in (_cfg["outdir"], _cfg["logdir"], _cfg["statdir"],
              os.path.dirname(_cfg["manifest"])):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for _ in ex.map(run_one, jobs):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print("[%6d/%6d] ok=%d reused=%d fail=%d  %.1f job/min  ETA %.1f h"
                      % (done, len(jobs), _counts["completed"],
                         _counts["skipped"], _counts["failed"],
                         rate * 60, eta / 3600), flush=True)
    print("finished in %.1f h: %s" % ((time.time() - t0) / 3600, _counts))
    if _counts["failed"]:
        raise SystemExit("%d jobs failed" % _counts["failed"])


if __name__ == "__main__":
    main()
