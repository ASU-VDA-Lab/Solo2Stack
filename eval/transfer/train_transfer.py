#!/usr/bin/env python3
"""One transferability run for Table VI: one arm, one held-out setting, fold 0.

Architecture, loss, optimiser, schedule, batch size, epoch count, precision and
checkpoint-selection rule are imported or copied verbatim from train_cv.py, so
the only thing that differs from a main-experiment run is which samples the
training population contains.

The chiplet split is cv fold 0 for every setting: train on groups 03-10,
validate on group_02, test on group_01. On top of that the setting withholds one
physical condition -- a stack depth, a cooling coefficient or a bonding
technology -- from train AND val, and the test set is exactly that condition on
the held-out chiplet group.

Normalisation comes from transfer/norm/<setting>.json, recomputed over the
restricted training population by preflight.py.

Writes only into <transfer_dir>/runs/. Nothing in the cross-validation
directory is touched.
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
import sys
import time

import torch
from torch.utils.data import DataLoader

add_repo(__file__, "model")
from loader_multi import (MAX_LAYERS, MultiDepthStackDataset,  # noqa: E402
                          masked_gradient_loss, masked_l1)
from unet_decomp import ThermalUNetDecomp, masked_scalar_l1  # noqa: E402
from train_cv import scalars, score  # noqa: E402  (identical scoring code)

BASE = P["data_root"]
ROOTS = {d: "%s/mbhb/pact_stack_%dL" % (BASE, d) for d in (2, 3, 4)}


class Restricted(MultiDepthStackDataset):
    """MultiDepthStackDataset limited to a subset of bcs and technologies.

    Depth is already a constructor argument. Bonding technology and cooling
    condition are not, so the sample index is filtered after construction and
    the filter is asserted to have removed every held-out sample.
    """

    def __init__(self, *a, bcs=None, ics=None, **kw):
        super().__init__(*a, **kw)
        if bcs is None and ics is None:
            return
        keep = []
        for (d, row, b) in self.pairs:
            if bcs is not None and b not in bcs:
                continue
            if ics is not None and int(self.ds[d].ic_row[row]) not in ics:
                continue
            keep.append((d, row, b))
        self.pairs = keep
        if bcs is not None:
            assert all(b in bcs for _, _, b in self.pairs)
        if ics is not None:
            assert all(int(self.ds[d].ic_row[r]) in ics for d, r, _ in self.pairs)


def build(arm, depths, bcs, ics, groups, norm_path):
    return Restricted(ROOTS, arm=arm, groups=groups, depths=depths,
                      norm_path=norm_path, bcs=bcs, ics=ics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A1", "A3"])
    ap.add_argument("--setting", required=True,
                    choices=["depth", "h_interp", "h_extrap", "tech",
                             "tech_reverse"])
    ap.add_argument("--out", default=P["transfer_dir"] + "/runs")
    ap.add_argument("--device", default="cuda:0")
    # frozen main-experiment hyperparameters -- do not change
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad-weight", type=float, default=0.5)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    # resume is purely operational: with it off the code path is unchanged
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pt if present")
    args = ap.parse_args()

    tag = "%s_%s" % (args.setting, args.arm)
    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)
    done = os.path.join(outdir, "result.json")
    if os.path.exists(done):
        print("%s already complete -- skipping" % tag, flush=True)
        return

    plan = json.load(open(P["transfer_dir"] + "/plan.json"))
    S = plan["settings"][args.setting]
    G = S["groups"]
    norm_path = P["transfer_dir"] + "/norm/%s.json" % args.setting

    dev = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tr = build(args.arm, S["train_depths"], S["train_bcs"], S["train_ics"],
               G["train"], norm_path)
    va = build(args.arm, S["train_depths"], S["train_bcs"], S["train_ics"],
               G["val"], norm_path)
    te = build(args.arm, S["test_depths"], S["test_bcs"], S["test_ics"],
               G["test"], norm_path)
    assert (len(tr), len(va), len(te)) == (S["train"], S["val"], S["test"]), \
        "sample counts disagree with the pre-flight plan"

    # independent re-assertion that the held-out condition is absent from the
    # populations the model will actually see
    hd = [d for d in (2, 3, 4) if d not in S["train_depths"]]
    hb = [b for b in range(5) if b not in S["train_bcs"]]
    all_ics = sorted(int(k) for k in plan["technologies"])
    hi = [i for i in all_ics if i not in S["train_ics"]]
    for nm, ds in (("train", tr), ("val", va)):
        for d, row, b in ds.pairs:
            assert d not in hd and b not in hb
            assert int(ds.ds[d].ic_row[row]) not in hi
    for d, row, b in te.pairs:
        assert d in hd or b in hb or int(te.ds[d].ic_row[row]) in hi
    print("%s | train %d val %d test %d | held-out absent from train and val"
          % (tag, len(tr), len(va), len(te)), flush=True)

    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0, drop_last=True,
                       prefetch_factor=4 if args.workers else None)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                       num_workers=max(4, args.workers // 2), pin_memory=True)
    dl_te = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                       num_workers=max(4, args.workers // 2), pin_memory=True)

    model = ThermalUNetDecomp(in_ch=tr.in_channels, out_ch=MAX_LAYERS,
                              base=args.base, depth=args.depth,
                              scalar_dim=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr), pct_start=0.1)

    hist, best, t_start = [], -1e9, time.time()
    ckpt = os.path.join(outdir, "best.pt")
    last = os.path.join(outdir, "last.pt")
    start_ep = 0
    if args.resume and os.path.exists(last):
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        hist, best, start_ep = st["hist"], st["best"], st["epoch"] + 1
        print("%s resuming at epoch %d (best val r2_pat %.4f so far)"
              % (tag, start_ep, best), flush=True)
    for ep in range(start_ep, args.epochs):
        t0, run, nb = time.time(), 0.0, 0
        for x, y, mask, aux in dl_tr:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            mask = mask.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp):
                field, pat, off = model(x, scalars(aux, dev))
                loss = (masked_l1(field, y, mask)
                        + args.grad_weight * masked_gradient_loss(field, y, mask))
                t_off = y.float().mean(dim=(-2, -1))
                loss = loss + args.aux_weight * (
                    masked_scalar_l1(off.float(), t_off, mask)
                    + masked_l1(pat.float(),
                                y.float() - t_off[:, :, None, None], mask))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            nb += 1

        m = score(model, dl_va, dev, amp)
        hist.append({"epoch": ep, "train_loss": run / max(1, nb),
                     "secs": time.time() - t0, "val": m})
        print("%s ep %2d loss %.4f val r2_pat %.4f pat_mae %.4f (%.0fs)"
              % (tag, ep, hist[-1]["train_loss"], m["pooled"]["r2_pattern"],
                 m["pooled"]["mae_pattern_k"], hist[-1]["secs"]), flush=True)
        if m["pooled"]["r2_pattern"] > best:
            best = m["pooled"]["r2_pattern"]
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "arch": "decomp", "epoch": ep, "val": m}, ckpt)
        json.dump(hist, open(os.path.join(outdir, "history.json"), "w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "hist": hist}, last + ".tmp")
        os.replace(last + ".tmp", last)

    model.load_state_dict(torch.load(ckpt, map_location=dev,
                                     weights_only=False)["model"])
    test = score(model, dl_te, dev, amp, layerwise=True)

    json.dump({
        "arm": args.arm, "setting": args.setting, "tag": tag, "fold": 0,
        "groups": G, "restriction": {k: S[k] for k in
                                     ("train_depths", "train_bcs", "train_ics",
                                      "test_depths", "test_bcs", "test_ics")},
        "norm_path": norm_path, "norm": json.load(open(norm_path)),
        "config": vars(args),
        "n_samples": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_val_r2_pattern": best,
        "best_epoch": int(max(range(len(hist)),
                              key=lambda i: hist[i]["val"]["pooled"]["r2_pattern"])),
        "val_history": [h["val"]["pooled"] for h in hist],
        "test": test,
        "wall_seconds": time.time() - t_start,
    }, open(done, "w"), indent=2)

    print("%s DONE  best val r2_pat %.4f | TEST r2_pat %.4f pat_mae %.4f"
          % (tag, best, test["pooled"]["r2_pattern"],
             test["pooled"]["mae_pattern_k"]), flush=True)


if __name__ == "__main__":
    main()
