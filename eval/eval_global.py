#!/usr/bin/env python3
"""Inference-only re-evaluation with an unambiguous global-max definition.

Two problems this answers.

1. `train_cv.py`'s score() batch-weight-averages every metric, including
   max_k, so the "Max" column in result.json is the MEAN OF PER-SLICE MAXIMA,
   not the largest error anywhere. It is therefore always below the true max
   and it moves with batch size. This script reports, alongside those legacy
   numbers, the true global figures:

       max_global_k  = largest |pred - true| over every grid cell of every
                       valid layer of every test stack in the fold
       p99_global_k  = 99th percentile of that same error population
                       (histogram, 0.1 mK bins -- exact to the bin width)
       and exact pooled MAE / pattern MAE / pattern R2 accumulated as sums
       rather than as means of per-batch means.

2. Scoring restricted to chosen bonding technologies. Passing --ic changes
   only the set of scored stacks, leaving the checkpoint's input pipeline
   bit-identical, so a model can be re-scored on a subset without retraining.

Nothing is trained and nothing is written except the --out JSON.
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
from solo2stack.bootstrap import add_repo  # noqa: E402
add_repo(__file__, "model")
# --------------------------------------------------------------------------

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from loader_multi import MultiDepthStackDataset, ic_names, thermal_metrics
from unet_decomp import ThermalUNetDecomp

LAYER_NAMES = {2: ["D1", "I12", "D2"],
               3: ["D1", "I12", "D2", "I23", "D3"],
               4: ["D1", "I12", "D2", "I23", "D3", "I34", "D4"]}

# |error| histogram: 0 .. 100 K in 0.1 mK bins
HIST_MAX = 100.0
HIST_BINS = 1000000


class Acc:
    """Exact streaming accumulator for one slice of the test set."""

    def __init__(self):
        self.n = 0                 # number of cells
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.max = 0.0
        self.pat_abs_sum = 0.0
        self.pat_sse = 0.0
        self.pat_sst = 0.0
        self.off_abs_sum = 0.0
        self.n_fields = 0          # number of (sample, layer) fields
        self.hist = np.zeros(HIST_BINS, dtype=np.int64)
        self.n_over = 0            # cells beyond the histogram range

    def add(self, pred, true):
        """pred/true: (B, L, H, W) float32 on GPU, already sliced to valid layers."""
        err = (pred - true).abs()
        self.n += err.numel()
        self.abs_sum += err.sum().item()
        self.sq_sum += err.pow(2).sum().item()
        self.max = max(self.max, err.max().item())

        tm = true.mean(dim=(-2, -1), keepdim=True)
        pm = pred.mean(dim=(-2, -1), keepdim=True)
        pat_t, pat_p = true - tm, pred - pm
        self.pat_abs_sum += (pat_p - pat_t).abs().sum().item()
        self.pat_sse += (pat_p - pat_t).pow(2).sum().item()
        self.pat_sst += pat_t.pow(2).sum().item()
        self.off_abs_sum += (pm - tm).abs().sum().item()
        self.n_fields += int(tm.numel())

        idx = (err.flatten() * (HIST_BINS / HIST_MAX)).long().clamp_(
            0, HIST_BINS - 1)
        self.hist += torch.bincount(idx, minlength=HIST_BINS).cpu().numpy()
        if self.n_over is not None and err.max().item() >= HIST_MAX:
            self.n_over += int((err >= HIST_MAX).sum().item())

    def result(self):
        if self.n == 0:
            return None
        c = self.hist.cumsum()
        tgt = 0.99 * c[-1]
        p99 = int(np.searchsorted(c, tgt)) * (HIST_MAX / HIST_BINS)
        return {
            "n_cells": int(self.n),
            "n_fields": int(self.n_fields),
            "max_global_k": self.max,
            "p99_global_k": p99,
            "mae_global_k": self.abs_sum / self.n,
            "rmse_global_k": (self.sq_sum / self.n) ** 0.5,
            "mae_pattern_global_k": self.pat_abs_sum / self.n,
            "r2_pattern_global": 1.0 - self.pat_sse / (self.pat_sst + 1e-12),
            "mae_offset_global_k": self.off_abs_sum / max(1, self.n_fields),
            "cells_above_hist_range": int(self.n_over),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="dir holding <arm>_fold<k>/best.pt")
    ap.add_argument("--cv-dir", required=True,
                    help="holds cv_folds.json and norm_cv/fold<k>.json")
    ap.add_argument("--base-dir", required=True,
                    help="holds pact_stack_{2,3,4}L")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ic", nargs="+", default=None,
                    help="restrict scoring to these bonding technologies "
                         "(default: all present in the root)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop after N batches per run (0 = all); smoke tests only")
    a = ap.parse_args()

    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp = torch.bfloat16
    spec = json.load(open(os.path.join(a.cv_dir, "cv_folds.json")))
    roots = {d: os.path.join(a.base_dir, "pact_stack_%dL" % d) for d in (2, 3, 4)}

    # ic_row -> technology name, read from THIS root's index. The mapping
    # depends on which technologies a root holds, so it must never be
    # hard-coded.
    IC = ic_names(roots[2])
    for d in (3, 4):
        assert ic_names(roots[d]) == IC, "ic_row order differs across depths"
    print("ic_row order in %s: %s" % (a.base_dir, IC), flush=True)

    out = {"runs": a.runs, "cv_dir": a.cv_dir, "base_dir": a.base_dir,
           "ic_filter": a.ic, "batch_size": a.batch_size,
           "note": "max_global_k/p99_global_k are over every grid cell of every "
                   "valid layer of every scored stack. The legacy_* block "
                   "reproduces train_cv.score()'s batch-weighted means.",
           "results": {}}

    for arm in a.arms:
        for fold in a.folds:
            tag = "%s_fold%d" % (arm, fold)
            ck = os.path.join(a.runs, tag, "best.pt")
            if not os.path.exists(ck):
                print("%s: no checkpoint, skipped" % tag, flush=True)
                continue
            f = spec["folds"][fold]
            assert f["fold"] == fold
            norm_path = os.path.join(a.cv_dir, "norm_cv", "fold%d.json" % fold)

            ds = MultiDepthStackDataset(roots, arm=arm, groups=f["test"],
                                        norm_path=norm_path)
            n_all = len(ds.pairs)
            if a.ic:
                keep = []
                for (d, row, b) in ds.pairs:
                    if ds.ds[d].stacks.loc[row, "interconnect"] in a.ic:
                        keep.append((d, row, b))
                ds.pairs = keep
            # record which technologies actually survived
            techs = sorted({ds.ds[d].stacks.loc[r, "interconnect"]
                            for (d, r, _) in ds.pairs})

            model = ThermalUNetDecomp(in_ch=ds.in_channels, base=a.base,
                                      depth=a.depth, scalar_dim=2).to(dev)
            sd = torch.load(ck, map_location=dev)
            model.load_state_dict(sd["model"] if "model" in sd else sd)
            model.eval()

            dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.workers, pin_memory=True)

            pooled = Acc()
            per_depth = defaultdict(Acc)
            per_layer = defaultdict(Acc)
            per_ic = defaultdict(Acc)
            per_ic_layer = defaultdict(Acc)
            legacy = defaultdict(lambda: defaultdict(float))
            lcnt = defaultdict(int)

            with torch.no_grad():
                for bi, (x, y, mask, aux) in enumerate(dl):
                    if a.max_batches and bi >= a.max_batches:
                        break
                    x = x.to(dev, non_blocking=True)
                    y = y.to(dev, non_blocking=True)
                    with torch.autocast(dev.type, dtype=amp, enabled=cuda):
                        p, _, _ = model(
                            x, torch.stack([aux["log10_htc_norm"].to(dev),
                                            aux["depth_norm"].to(dev)], dim=1))
                    p, y = p.float(), y.float()
                    dep = aux["depth"]
                    icr = aux["ic_row"]
                    for d in torch.unique(dep).tolist():
                        dsel = (dep == d).nonzero(as_tuple=True)[0]
                        sel = dsel.to(dev)
                        nv = 2 * d - 1
                        pd_, yd_ = p[sel][:, :nv], y[sel][:, :nv]
                        pooled.add(pd_, yd_)
                        per_depth[d].add(pd_, yd_)
                        for li in range(nv):
                            key = "%d|%s" % (d, LAYER_NAMES[d][li])
                            per_layer[key].add(pd_[:, li:li + 1],
                                               yd_[:, li:li + 1])
                        # per-technology, resolved through this root's ic_row order
                        rows_ic = icr[dsel]
                        for j in torch.unique(rows_ic).tolist():
                            t = IC[j]
                            m2 = (rows_ic == j).nonzero(as_tuple=True)[0].to(dev)
                            per_ic[t].add(pd_[m2], yd_[m2])
                            for li in range(nv):
                                per_ic_layer["%s|%d|%s"
                                             % (t, d, LAYER_NAMES[d][li])].add(
                                    pd_[m2][:, li:li + 1],
                                    yd_[m2][:, li:li + 1])
                        # legacy batch-weighted means, same as train_cv.score()
                        m = thermal_metrics(pd_, yd_)
                        b = int(sel.numel())
                        for k, v in m.items():
                            legacy[d][k] += v * b
                        lcnt[d] += b

            tot = sum(lcnt.values())
            leg_pooled = {k: sum(legacy[d][k] for d in legacy) / tot
                          for k in next(iter(legacy.values()))}
            rec = {
                "arm": arm, "fold": fold,
                "test_groups": f["test"],
                "n_samples_scored": len(ds.pairs),
                "n_samples_available": n_all,
                "technologies_scored": techs,
                "global": pooled.result(),
                "global_per_depth": {str(d): per_depth[d].result()
                                     for d in sorted(per_depth)},
                "global_per_layer": {k: per_layer[k].result()
                                     for k in sorted(per_layer)},
                "global_per_ic": {k: per_ic[k].result() for k in sorted(per_ic)},
                "global_per_ic_layer": {k: per_ic_layer[k].result()
                                        for k in sorted(per_ic_layer)},
                "legacy_pooled_batch_averaged": leg_pooled,
            }
            out["results"][tag] = rec
            g = rec["global"]
            print("%-10s n=%-6d %-22s max_global %8.4f  p99 %7.4f  "
                  "patMAE %7.4f  R2pat %7.4f  | legacy max %8.4f"
                  % (tag, len(ds.pairs), ",".join(techs),
                     g["max_global_k"], g["p99_global_k"],
                     g["mae_pattern_global_k"], g["r2_pattern_global"],
                     leg_pooled["max_k"]), flush=True)
            del model, ds, dl
            if cuda:
                torch.cuda.empty_cache()

    json.dump(out, open(a.out, "w"), indent=2)
    print("\nwrote %s  (%d runs)" % (a.out, len(out["results"])))


if __name__ == "__main__":
    main()
