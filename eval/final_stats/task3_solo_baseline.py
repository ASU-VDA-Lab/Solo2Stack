#!/usr/bin/env python3
"""Task 3 -- solo-only baseline, no ML model anywhere.

For every DIE layer of every test stack, the prediction of the mean-removed
full-stack temperature is the mean-removed solo residual of the chiplet that
occupies that die slot, at the SAME boundary condition.  Nothing is fitted,
scaled or tuned.

Die-slot -> solo-map matching (read from the dataset index, never assumed):
    depth 2 : stacks.csv columns  bot_row, top_row
    depth 3 : stacks.csv columns  d1_row, d2_row, d3_row
    depth 4 : stacks.csv columns  d1_row, d2_row, d3_row, d4_row
  The k-th die (0-based, bottom first) sits at stack layer 2*k
  (LAYER_NAMES = D1 I12 D2 I23 D3 I34 D4) and its solo field is
  solo_dT_ck[die_row[k], bc].  Interface layers have no solo map and are
  excluded, so only layers 0, 2, 4, 6 are ever read.

Mean removal makes the per-field offset irrelevant, so solo_off_ck and
stack_off_ck cancel and the raw centi-Kelvin arrays are used directly:
    pattern = (dT - mean(dT)) / 100       [K]

Metrics are defined exactly as in loader_multi.thermal_metrics /
eval_global.Acc so they are comparable with the model arms:
    r2_pattern  = 1 - sum (pred_pat - true_pat)^2 / sum true_pat^2, pooled
                  jointly over all scored die-layer fields
    mae_pattern = mean |pred_pat - true_pat| over all scored cells
    hotspot     = per field, fraction of the predicted top-1% cells that fall
                  in the true top-1%; averaged over fields

The dataset lives on a latency-bound filesystem (~2.3 s for a single 6 MB row
read, ~73 MB/s with 48 concurrent readers), so rows are prefetched by a thread
pool; numpy memmap reads release the GIL.  Read order does not affect any
result because every accumulator is a sum.
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
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

BASE = P["data_root"]
CV = P["cv_dir"]
OUT = P["results"]
DIE_COLS = {2: ["bot_row", "top_row"],
            3: ["d1_row", "d2_row", "d3_row"],
            4: ["d1_row", "d2_row", "d3_row", "d4_row"]}
N_HOT = max(1, int(0.01 * 400 * 400))      # 1600 cells, as in thermal_metrics


class Acc:
    """Streaming accumulator of exact sums; chunking cannot change the result."""

    def __init__(self):
        self.n_cells = 0
        self.n_fields = 0
        self.abs_sum = 0.0
        self.sse = 0.0
        self.sst = 0.0
        self.hot_sum = 0.0

    def add_sums(self, n_cells, n_fields, abs_sum, sse, sst, hot_sum):
        self.n_cells += n_cells
        self.n_fields += n_fields
        self.abs_sum += abs_sum
        self.sse += sse
        self.sst += sst
        self.hot_sum += hot_sum

    def result(self):
        if self.n_cells == 0:
            return None
        return {"n_cells": int(self.n_cells), "n_fields": int(self.n_fields),
                "r2_pattern": 1.0 - self.sse / (self.sst + 1e-12),
                "mae_pattern_k": self.abs_sum / self.n_cells,
                "hotspot_top1%_overlap": self.hot_sum / self.n_fields}


def field_sums(pred_pat, true_pat):
    """Per-chunk sums for a batch of mean-removed fields, computed once."""
    d = pred_pat - true_pat
    tf, pf = true_pat.flatten(1), pred_pat.flatten(1)
    th = tf.topk(N_HOT, dim=1).indices
    ph = pf.topk(N_HOT, dim=1).indices
    hit = torch.zeros_like(tf, dtype=torch.bool)
    hit.scatter_(1, th, True)
    return (d.numel(), int(pred_pat.shape[0]), d.abs().sum().item(),
            d.pow(2).sum().item(), true_pat.pow(2).sum().item(),
            hit.gather(1, ph).float().mean(dim=1).sum().item())


def prefetch(sdt, rows, dies, workers, depth_q):
    """Yield (row, die-layer block) with reads running ahead of the consumer."""
    def read(r):
        return np.ascontiguousarray(sdt[r][:, dies])

    with ThreadPoolExecutor(workers) as ex:
        q = deque()
        it = iter(rows)
        for _ in range(depth_q):
            try:
                r = next(it)
            except StopIteration:
                break
            q.append((r, ex.submit(read, r)))
        while q:
            r, fut = q.popleft()
            try:
                r2 = next(it)
                q.append((r2, ex.submit(read, r2)))
            except StopIteration:
                pass
            yield r, fut.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(OUT, "task3_solo_only.json"))
    ap.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--queue", type=int, default=128)
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="smoke test: only this many stack rows per depth")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(OUT, exist_ok=True)

    spec = json.load(open(os.path.join(CV, "cv_folds.json")))
    # test group -> fold; each stack is in exactly one group, so a single pass
    # over the dataset produces every fold's test set.
    g2fold = {}
    for f in spec["folds"]:
        assert len(f["test"]) == 1, f
        g2fold[f["test"][0]] = int(f["fold"])
    assert len(g2fold) == 10, g2fold

    # The solo arrays are one shared file across the three roots (symlinked in
    # the microbump+hybrid root), so they are loaded and centred exactly once.
    t0 = time.time()
    cdt = np.load(os.path.join(BASE, "pact_stack_4L", "arrays",
                               "solo_dT_ck.i16.npy"), mmap_mode="r")
    solo = torch.from_numpy(np.array(cdt)).to(dev).float() / 100.0
    solo = solo - solo.mean(dim=(-2, -1), keepdim=True)
    print("solo loaded %s in %.1fs" % (tuple(solo.shape), time.time() - t0),
          flush=True)

    acc = defaultdict(Acc)          # (fold, scope) -> Acc
    for depth in a.depths:
        root = os.path.join(BASE, "pact_stack_%dL" % depth)
        st = pd.read_csv(os.path.join(root, "index", "stacks.csv"))
        sdt = np.load(os.path.join(root, "arrays", "stack_dT_ck.i16.npy"),
                      mmap_mode="r")
        assert sdt.shape[2] == 2 * depth - 1
        dies = [2 * k for k in range(depth)]
        die_rows = st[DIE_COLS[depth]].to_numpy(dtype=np.int64)
        folds = np.array([g2fold[g] for g in st["group"]])
        techs = st["interconnect"].to_numpy()
        rows = st.index.to_numpy()
        if a.limit_rows:
            rows = rows[:a.limit_rows]
        t0 = time.time()
        done = 0
        for row, arr in prefetch(sdt, rows, dies, a.workers, a.queue):
            scopes = [(int(folds[row]), s) for s in
                      ("overall", "N=%d" % depth, "tech=%s" % techs[row])]
            blk = torch.from_numpy(arr).to(dev).float() / 100.0   # (nbc,nd,H,W)
            for k in range(depth):
                true_pat = blk[:, k]
                true_pat = true_pat - true_pat.mean(dim=(-2, -1), keepdim=True)
                sums = field_sums(solo[die_rows[row, k]], true_pat)
                for key in scopes:
                    acc[key].add_sums(*sums)
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                print("depth %d  %d/%d rows  %.0fs  (%.0f rows/min)"
                      % (depth, done, len(rows), el, 60.0 * done / el),
                      flush=True)
        print("depth %d done: %d rows in %.0fs" % (depth, len(rows),
                                                   time.time() - t0), flush=True)

    out = {"base_dir": BASE, "cv_dir": CV,
           "definition": "mean-removed solo residual of each die's chiplet at "
                         "the same h, used as the prediction of the mean-removed "
                         "full-stack die-layer temperature; no model, no "
                         "fitting, no scaling",
           "die_slot_matching": {str(k): v for k, v in DIE_COLS.items()},
           "n_hot_cells": N_HOT, "per_fold": {}}
    for (fold, scope), ac in sorted(acc.items()):
        out["per_fold"].setdefault(str(fold), {})[scope] = ac.result()
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
