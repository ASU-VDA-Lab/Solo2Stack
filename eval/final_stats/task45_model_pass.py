#!/usr/bin/env python3
"""Inference-only pass over the final microbump+hybrid campaign.

One pass per (arm, fold) collects everything Tasks 3, 4 and 5 need:

  Task 3 companion : pooled pattern metrics restricted to DIE layers only
                     (overall / by N / by bonding technology), so the model
                     arms are measured on exactly the layers the solo-only
                     baseline can predict.
  Task 4           : per-sample peak absolute temperature, predicted and true,
                     the layer each peak sits in, and the group key
                     (N, bonding, h) needed for the per-group Spearman.
  Task 5           : the single largest grid-cell error of the fold, with its
                     full location -- stack, bonding, h, layer, (x, y) cell,
                     the error and the true temperature there.

Nothing is trained; no PACT is run.  Metric definitions are copied from
loader_multi.thermal_metrics / eval_global.Acc so these numbers sit on the
same footing as the published ones.

Residual arms (A1, Solo2Stack) are trained on T - analytic(stack, h); the
analytic term is a per-(stack, h) scalar, so |pred - true| is identical in
residual and absolute space, but PEAK TEMPERATURE is not -- analytic_k is
added back before any peak is taken.
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
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

add_repo(__file__, "model")
from loader_multi import MultiDepthStackDataset, ic_names, ARMS   # noqa: E402
from unet_decomp import ThermalUNetDecomp                          # noqa: E402

LAYER_NAMES = {2: ["D1", "I12", "D2"],
               3: ["D1", "I12", "D2", "I23", "D3"],
               4: ["D1", "I12", "D2", "I23", "D3", "I34", "D4"]}
N_HOT = max(1, int(0.01 * 400 * 400))


class Acc:
    """Exact streaming sums over a set of (sample, layer) fields."""

    def __init__(self):
        self.n_cells = self.n_fields = 0
        self.abs_sum = self.sse = self.sst = self.hot_sum = 0.0

    def add(self, pred, true):
        """pred/true: (B, L, H, W) float32, already restricted to the layers
        being scored.  Pattern = per-field mean removed."""
        pm = pred.mean(dim=(-2, -1), keepdim=True)
        tm = true.mean(dim=(-2, -1), keepdim=True)
        pp, tp = pred - pm, true - tm
        d = pp - tp
        self.n_cells += d.numel()
        self.n_fields += int(tm.numel())
        self.abs_sum += d.abs().sum().item()
        self.sse += d.pow(2).sum().item()
        self.sst += tp.pow(2).sum().item()
        tf, pf = tp.flatten(0, 1).flatten(1), pp.flatten(0, 1).flatten(1)
        th = tf.topk(N_HOT, dim=1).indices
        ph = pf.topk(N_HOT, dim=1).indices
        hit = torch.zeros_like(tf, dtype=torch.bool)
        hit.scatter_(1, th, True)
        self.hot_sum += hit.gather(1, ph).float().mean(dim=1).sum().item()

    def result(self):
        if self.n_cells == 0:
            return None
        return {"n_cells": int(self.n_cells), "n_fields": int(self.n_fields),
                "r2_pattern": 1.0 - self.sse / (self.sst + 1e-12),
                "mae_pattern_k": self.abs_sum / self.n_cells,
                "hotspot_top1%_overlap": self.hot_sum / self.n_fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=P["data_root"])
    ap.add_argument("--cv-dir", default=P["cv_dir"])
    ap.add_argument("--runs", default=P["checkpoints"])
    ap.add_argument("--out-dir", default=P["results"] + "/pass")
    ap.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp = torch.bfloat16

    spec = json.load(open(os.path.join(a.cv_dir, "cv_folds.json")))
    roots = {d: os.path.join(a.base_dir, "pact_stack_%dL" % d) for d in (2, 3, 4)}
    IC = ic_names(roots[2])
    for d in (3, 4):
        assert ic_names(roots[d]) == IC, "ic_row order differs across depths"
    print("ic_row order in %s: %s" % (a.base_dir, IC), flush=True)

    for arm in a.arms:
        residual = ARMS[arm]["target"] == "residual"
        for fold in a.folds:
            tag = "%s_fold%d" % (arm, fold)
            ck = os.path.join(a.runs, tag, "best.pt")
            assert os.path.exists(ck), ck
            f = spec["folds"][fold]
            assert f["fold"] == fold
            norm_path = os.path.join(a.cv_dir, "norm_cv", "fold%d.json" % fold)

            ds = MultiDepthStackDataset(roots, arm=arm, groups=f["test"],
                                        norm_path=norm_path)
            stacks = {d: ds.ds[d].stacks for d in (2, 3, 4)}
            model = ThermalUNetDecomp(in_ch=ds.in_channels, base=a.base,
                                      depth=a.depth, scalar_dim=2).to(dev)
            sd = torch.load(ck, map_location=dev)
            model.load_state_dict(sd["model"] if "model" in sd else sd)
            model.eval()
            dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.workers, pin_memory=True)

            die_acc = defaultdict(Acc)        # scope -> Acc (die layers only)
            peaks = []                        # one record per sample
            worst = {"err": -1.0}
            t0 = time.time()
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
                    for d in torch.unique(dep).tolist():
                        idx = (dep == d).nonzero(as_tuple=True)[0]
                        sel = idx.to(dev)
                        nv = 2 * d - 1
                        pd_, yd_ = p[sel][:, :nv], y[sel][:, :nv]

                        # ---- Task 3 companion: die layers only ----
                        dsl = [2 * k for k in range(d)]
                        pdie, ydie = pd_[:, dsl], yd_[:, dsl]
                        icr = aux["ic_row"][idx]
                        die_acc["overall"].add(pdie, ydie)
                        die_acc["N=%d" % d].add(pdie, ydie)
                        for j in torch.unique(icr).tolist():
                            m = (icr == j).nonzero(as_tuple=True)[0].to(dev)
                            die_acc["tech=%s" % IC[j]].add(pdie[m], ydie[m])

                        # ---- Task 4: peak absolute temperature ----
                        off = aux["analytic_k"][idx].to(dev) if residual else \
                            torch.zeros(len(idx), device=dev)
                        pabs = pd_ + off[:, None, None, None]
                        yabs = yd_ + off[:, None, None, None]
                        pf_ = pabs.flatten(1)
                        yf_ = yabs.flatten(1)
                        pmax, pami = pf_.max(dim=1)
                        ymax, yami = yf_.max(dim=1)
                        cells = 400 * 400
                        play = (pami // cells).cpu().numpy()
                        ylay = (yami // cells).cpu().numpy()
                        pmax = pmax.cpu().numpy()
                        ymax = ymax.cpu().numpy()
                        srow = aux["stack_row"][idx].numpy()
                        brow = aux["bc_row"][idx].numpy()
                        htc = aux["htc"][idx].numpy()
                        icn = [IC[j] for j in icr.tolist()]
                        for i in range(len(idx)):
                            peaks.append({
                                "arm": arm, "fold": fold, "depth": d,
                                "stack_row": int(srow[i]),
                                "stack_id": stacks[d].at[int(srow[i]), "stack_id"],
                                "tech": icn[i], "bc_row": int(brow[i]),
                                "htc": float(htc[i]),
                                "true_peak_k": float(ymax[i]),
                                "pred_peak_k": float(pmax[i]),
                                "abs_peak_err_k": float(abs(pmax[i] - ymax[i])),
                                "true_peak_layer": LAYER_NAMES[d][int(ylay[i])],
                                "pred_peak_layer": LAYER_NAMES[d][int(play[i])],
                                "same_peak_layer": int(ylay[i] == play[i])})

                        # ---- Task 5: largest grid-cell error in the fold ----
                        err = (pd_ - yd_).abs()
                        e_flat = err.flatten(1)
                        emax, eami = e_flat.max(dim=1)
                        j = int(emax.argmax().item())
                        if float(emax[j]) > worst["err"]:
                            lin = int(eami[j].item())
                            lay, rem = divmod(lin, cells)
                            yy, xx = divmod(rem, 400)
                            r = int(srow[j])
                            worst = {
                                "err": float(emax[j]), "arm": arm, "fold": fold,
                                "depth": d, "stack_row": r,
                                "stack_id": stacks[d].at[r, "stack_id"],
                                "tech": icn[j], "bc_row": int(brow[j]),
                                "htc": float(htc[j]),
                                "layer_index": lay,
                                "layer": LAYER_NAMES[d][lay],
                                "x": xx, "y": yy,
                                "true_k": float(yabs[j, lay, yy, xx].item()),
                                "pred_k": float(pabs[j, lay, yy, xx].item()),
                                "chiplets": [stacks[d].at[r, c] for c in
                                             stacks[d].columns if
                                             c.endswith("_chiplet")]}
            rec = {"arm": arm, "fold": fold, "test_groups": f["test"],
                   "n_samples": len(ds.pairs), "ic_order": IC,
                   "die_layers_only": {k: v.result()
                                       for k, v in sorted(die_acc.items())},
                   "worst_cell": worst,
                   "seconds": time.time() - t0}
            json.dump(rec, open(os.path.join(a.out_dir, "%s.json" % tag), "w"),
                      indent=2)
            with open(os.path.join(a.out_dir, "%s_peaks.csv" % tag), "w") as fh:
                w = csv.DictWriter(fh, fieldnames=list(peaks[0].keys()))
                w.writeheader()
                w.writerows(peaks)
            print("%s  n=%d  %.0fs  die-only R2pat %.4f  worst %.4f K at %s|%s "
                  "(%d,%d) h=%.0f" %
                  (tag, len(ds.pairs), rec["seconds"],
                   rec["die_layers_only"]["overall"]["r2_pattern"],
                   worst["err"], worst["tech"], worst["layer"],
                   worst["x"], worst["y"], worst["htc"]), flush=True)


if __name__ == "__main__":
    main()
