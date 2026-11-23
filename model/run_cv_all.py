#!/usr/bin/env python3
"""Orchestrate the 40 (arm, fold) runs of the frozen decomp+aux factorial.

One worker per GPU, each pulling from a shared queue. Jobs are ordered
fold-major -- (A0,f0) (A1,f0) (A2,f0) (A3,f0) (A0,f1) ... -- so that each
completed wave of four yields a whole fold across all arms, and a partial
experiment is still balanced.

Resumable: a run whose result.json already exists is skipped, so an
interruption costs at most the runs that were in flight.
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
import queue
import subprocess
import threading
import time

ARMS = ["A0", "A1", "A2", "A3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", default=P["cv_dir"])
    ap.add_argument("--out", default=P["checkpoints"])
    ap.add_argument("--gpus", nargs="+", default=["cuda:0", "cuda:1",
                                                  "cuda:2", "cuda:3"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-dir", default=P["data_root"],
                    help="holds pact_stack_{2,3,4}L")
    ap.add_argument("--python", default=P["ml_python"])
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(a.out, exist_ok=True)
    logdir = os.path.join(a.cv_dir, "logs")
    os.makedirs(logdir, exist_ok=True)

    jobs = queue.Queue()
    todo = skipped = 0
    for fold in a.folds:
        for arm in a.arms:
            if os.path.exists(os.path.join(a.out, "%s_fold%d" % (arm, fold),
                                           "result.json")):
                skipped += 1
                continue
            jobs.put((arm, fold))
            todo += 1
    print("queued %d runs, %d already complete" % (todo, skipped), flush=True)

    lock = threading.Lock()
    state = {"done": 0, "failed": []}
    t_start = time.time()

    def worker(gpu):
        while True:
            try:
                arm, fold = jobs.get_nowait()
            except queue.Empty:
                return
            tag = "%s_fold%d" % (arm, fold)
            log = os.path.join(logdir, "%s.log" % tag)
            cmd = [a.python, os.path.join(here, "train_cv.py"),
                   "--arm", arm, "--fold", str(fold), "--device", gpu,
                   "--cv-dir", a.cv_dir, "--out", a.out,
                   "--base-dir", a.base_dir,
                   "--workers", str(a.workers)]
            t0 = time.time()
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=here, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            with lock:
                state["done"] += 1
                if rc != 0:
                    state["failed"].append(tag)
                el = time.time() - t_start
                rate = state["done"] / el if el else 0
                left = (todo - state["done"]) / rate / 3600 if rate else 0
                print("[%2d/%2d] %-10s %-8s rc=%d  %.1f min   ETA %.1f h"
                      % (state["done"], todo, tag, gpu, rc,
                         (time.time() - t0) / 60, left), flush=True)
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in a.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\nall runs finished in %.1f h" % ((time.time() - t_start) / 3600))
    if state["failed"]:
        print("FAILED: %s" % state["failed"])
    else:
        print("no failures")
    done = sum(1 for fold in range(10) for arm in ARMS
               if os.path.exists(os.path.join(a.out, "%s_fold%d" % (arm, fold),
                                              "result.json")))
    print("complete runs on disk: %d / 40" % done)


if __name__ == "__main__":
    main()
