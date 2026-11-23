#!/usr/bin/env python3
"""One (arm, fold) run of the 10-fold A0-A3 factorial on the frozen decomp+aux model.

Frozen configuration, identical for all four arms and all ten folds:
  architecture  ThermalUNetDecomp -- zero-mean pattern head + pooled per-layer
                offset head, base 32, U-Net depth 4, FiLM scalar_dim 2
  loss          masked L1 + 0.5 * masked gradient, plus 0.5 * (offset L1 +
                zero-mean pattern L1)
  optimiser     AdamW lr 3e-4 wd 1e-4, OneCycle pct_start 0.1
  batch 32, 30 epochs, bf16 autocast
  checkpoint    best validation pooled r2_pattern

The only things that vary are the two factorial variables (solo inputs, and
absolute vs residual target) and the fold. No per-arm hyperparameter tuning.

Normalisation comes from cv/norm_cv/fold{k}.json, built from that fold's eight
training groups only.

The test group is touched exactly once, by the selected checkpoint, at the end.
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
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader_multi import (MAX_LAYERS, MultiDepthStackDataset,  # noqa: E402
                          masked_gradient_loss, masked_l1, thermal_metrics)
from unet_decomp import ThermalUNetDecomp, masked_scalar_l1  # noqa: E402

LAYER_NAMES = {2: ["D1", "I12", "D2"],
               3: ["D1", "I12", "D2", "I23", "D3"],
               4: ["D1", "I12", "D2", "I23", "D3", "I34", "D4"]}


def scalars(aux, dev):
    return torch.stack([aux["log10_htc_norm"].to(dev),
                        aux["depth_norm"].to(dev)], dim=1)


def score(model, dl, dev, amp, layerwise=False):
    model.eval()
    acc = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    lacc = defaultdict(lambda: defaultdict(float))
    lcnt = defaultdict(int)
    with torch.no_grad():
        for x, y, mask, aux in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp):
                p, _, _ = model(x, scalars(aux, dev))
            p, y = p.float(), y.float()
            dep = aux["depth"]
            for d in torch.unique(dep).tolist():
                sel = (dep == d).nonzero(as_tuple=True)[0].to(dev)
                nv = 2 * d - 1
                m = thermal_metrics(p[sel][:, :nv], y[sel][:, :nv])
                b = int(sel.numel())
                for k, v in m.items():
                    acc[d][k] += v * b
                cnt[d] += b
                if layerwise:
                    for li in range(nv):
                        ml = thermal_metrics(p[sel][:, li:li + 1],
                                             y[sel][:, li:li + 1])
                        for k, v in ml.items():
                            lacc[(d, li)][k] += v * b
                        lcnt[(d, li)] += b
    model.train()
    per = {int(d): {k: acc[d][k] / cnt[d] for k in acc[d]} for d in acc}
    tot = sum(cnt.values())
    pooled = {k: sum(acc[d][k] for d in acc) / tot for k in next(iter(acc.values()))}
    out = {"pooled": pooled, "per_depth": per,
           "counts": {int(d): cnt[d] for d in cnt}}
    if layerwise:
        out["per_layer"] = {
            "%d|%s" % (d, LAYER_NAMES[d][li]):
                {k: v / lcnt[(d, li)] for k, v in lacc[(d, li)].items()}
            for (d, li) in lacc}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A0", "A1", "A2", "A3"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--base-dir", default=P["data_root"])
    ap.add_argument("--cv-dir", default=P["cv_dir"])
    ap.add_argument("--out", default=P["checkpoints"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grad-weight", type=float, default=0.5)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    args = ap.parse_args()

    tag = "%s_fold%d" % (args.arm, args.fold)
    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)
    done_flag = os.path.join(outdir, "result.json")
    if os.path.exists(done_flag):
        print("%s already complete -- skipping" % tag, flush=True)
        return

    spec = json.load(open(os.path.join(args.cv_dir, "cv_folds.json")))
    f = spec["folds"][args.fold]
    assert f["fold"] == args.fold
    norm_path = os.path.join(args.cv_dir, "norm_cv", "fold%d.json" % args.fold)

    dev = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    roots = {d: os.path.join(args.base_dir, "pact_stack_%dL" % d) for d in (2, 3, 4)}
    mk = lambda gs: MultiDepthStackDataset(roots, arm=args.arm, groups=gs,
                                           norm_path=norm_path)
    tr, va, te = mk(f["train"]), mk(f["val"]), mk(f["test"])
    print("%s | train %d val %d test %d | norm %s"
          % (tag, len(tr), len(va), len(te), os.path.basename(norm_path)), flush=True)

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
    for ep in range(args.epochs):
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
        rec = {"epoch": ep, "train_loss": run / max(1, nb),
               "secs": time.time() - t0, "val": m}
        hist.append(rec)
        print("%s ep %2d loss %.4f val r2_pat %.4f pat_mae %.4f (%.0fs)"
              % (tag, ep, rec["train_loss"], m["pooled"]["r2_pattern"],
                 m["pooled"]["mae_pattern_k"], rec["secs"]), flush=True)
        if m["pooled"]["r2_pattern"] > best:
            best = m["pooled"]["r2_pattern"]
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "arch": "decomp", "epoch": ep, "val": m}, ckpt)
        json.dump(hist, open(os.path.join(outdir, "history.json"), "w"), indent=2)

    # ---- single, final evaluation of the selected checkpoint on the test group
    model.load_state_dict(torch.load(ckpt, map_location=dev,
                                     weights_only=False)["model"])
    test = score(model, dl_te, dev, amp, layerwise=True)

    json.dump({
        "arm": args.arm, "fold": args.fold, "tag": tag,
        "groups": {"train": f["train"], "val": f["val"], "test": f["test"]},
        "norm_path": norm_path,
        "norm": json.load(open(norm_path)),
        "config": vars(args),
        "n_samples": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_val_r2_pattern": best,
        "best_epoch": int(max(range(len(hist)),
                              key=lambda i: hist[i]["val"]["pooled"]["r2_pattern"])),
        "val_history": [h["val"]["pooled"] for h in hist],
        "test": test,
        "wall_seconds": time.time() - t_start,
    }, open(done_flag, "w"), indent=2)

    print("%s DONE  best val r2_pat %.4f | TEST r2_pat %.4f pat_mae %.4f"
          % (tag, best, test["pooled"]["r2_pattern"],
             test["pooled"]["mae_pattern_k"]), flush=True)


if __name__ == "__main__":
    main()
