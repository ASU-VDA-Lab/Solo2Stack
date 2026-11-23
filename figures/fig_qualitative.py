#!/usr/bin/env python3
"""Fig. 4: qualitative maps, three rows (D4, I34, D1).

A copy of make_qual_jet_3row.py. The canvas, row order, five columns, per-row
shared vmin/vmax from the PACT field, colour bars, fonts, headers, figure size
and the absolute cell-wise error annotation are all carried over unchanged.
Four things differ:

  * the dataset and checkpoints are the microbump+hybrid ones;
  * the two assertions that pinned the selection and the per-panel metrics to
    the published figure are removed. They existed to prove the published
    figure had not drifted; here the case is deliberately re-selected, so
    asserting against the old one would always fail. Nothing else was relaxed;
  * the technology label is resolved by name from interconnect.csv rather than
    indexed into a hard-coded three-element list;
  * it emits TWO figures, because the published selection rule and the
    published caption no longer have to agree:

      _rulematch : the published rule applied verbatim -- median Solo2Stack
                   pattern error among ALL held-out four-die stacks at the
                   chosen bc. Whatever bonding technology that lands on is
                   reported, not forced.
      _hybrid    : the same rule restricted to hybrid stacks, which is what the
                   published caption describes.

    If the two coincide, the second is skipped and that is stated.

Writes *_mbhb*.{pdf,png} and a selection record; refuses to overwrite anything.
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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

add_repo(__file__, "model")
from loader_multi import MAX_LAYERS, MultiDepthStackDataset  # noqa: E402
from unet_decomp import ThermalUNetDecomp  # noqa: E402

ROWS = ["$D_1$", "$I_{12}$", "$D_2$", "$I_{23}$", "$D_3$", "$I_{34}$", "$D_4$"]
ORDER = [6, 5, 0]                       # canvas row i shows physical layer ORDER[i]
ARMS = ("A0", "A1", "A2", "A3")
HEADERS = ("PACT", "A0", "A1", "A2", "Solo2Stack")

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load_arm(arm, fold, runs, in_ch, dev):
    sd = torch.load(os.path.join(runs, "%s_fold%d" % (arm, fold), "best.pt"),
                    map_location="cpu", weights_only=False)
    net = ThermalUNetDecomp(in_ch=in_ch, out_ch=MAX_LAYERS,
                            base=sd["args"]["base"], depth=sd["args"]["depth"],
                            scalar_dim=2).to(dev).eval()
    net.load_state_dict(sd["model"])
    return net


def draw_cbar(cax, lo, hi, nticks=5):
    grad = np.linspace(lo, hi, 512).reshape(-1, 1)
    cax.imshow(grad, aspect="auto", cmap="jet", vmin=lo, vmax=hi,
               origin="lower", extent=[0.0, 1.0, lo, hi],
               interpolation="nearest")
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(lo, hi)
    cax.set_xticks([])
    t = np.linspace(lo, hi, nticks)
    d = 1
    while d < 4 and len({"%.*f" % (d, v) for v in t}) < nticks:
        d += 1
    cax.yaxis.set_ticks_position("right")
    cax.set_yticks(t)
    cax.set_yticklabels(["%.*f" % (d, v) for v in t])
    cax.tick_params(labelsize=6.0, width=0.4, length=1.8, pad=1.0)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)


def render(pick, ds, nets, dev, ic_name, stacks, outdir, suffix, note):
    fields = {}
    for arm in ARMS:
        x, y, m, aux = ds[arm][pick]
        ana = float(aux["analytic_k"])
        s = torch.tensor([[float(aux["log10_htc_norm"]),
                           float(aux["depth_norm"])]], device=dev)
        with torch.no_grad():
            p, _, _ = nets[arm](x[None].to(dev), s)
        pred = p.float()[0, :7].cpu().numpy()
        true = y[:7].numpy()
        if ds[arm].target == "residual":
            pred, true = pred + ana, true + ana
        fields[arm], fields["gt"] = pred, true
    gt = fields["gt"]
    _, _, _, aux = ds["A3"][pick]
    htc = float(aux["htc"])
    depth, srow, brow = ds["A3"].pairs[pick]
    tech = ic_name[int(aux["ic_row"])]
    stack_id = str(stacks.loc[srow, "stack_id"])
    assert stacks.loc[srow, "interconnect"] == tech, (stack_id, tech)

    stats = {}
    for r in range(7):
        for name in ARMS:
            e = np.abs(fields[name][r] - gt[r])
            stats[(r, name)] = (float(e.mean()), float(e.max()))

    nr = len(ORDER)
    fig = plt.figure(figsize=(7.16, 7.62 * nr / 7.0))
    gs = GridSpec(nr, 7, figure=fig,
                  width_ratios=[1, 1, 1, 1, 1, 0.085, 0.075],
                  wspace=0.04, hspace=0.25)
    for i, r in enumerate(ORDER):
        lo, hi = float(gt[r].min()), float(gt[r].max())
        for c, (name, img) in enumerate(
                [("PACT", gt[r])] + [(k, fields[k][r]) for k in ARMS]):
            ax = fig.add_subplot(gs[i, c])
            ax.imshow(img, cmap="jet", vmin=lo, vmax=hi, origin="lower",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if c == 0:
                ax.set_ylabel(ROWS[r], fontsize=9.5, labelpad=2, rotation=0,
                              va="center", ha="right")
            else:
                mae, mx = stats[(r, name)]
                ax.set_xlabel("MAE %.2f K\nMax %.2f K" % (mae, mx),
                              fontsize=6.2, labelpad=1, linespacing=1.10)
            if i == 0:
                ax.set_title(HEADERS[c], fontsize=8.2, pad=2.5,
                             fontweight="bold" if c == 4 else "normal")
        draw_cbar(fig.add_subplot(gs[i, 6]), lo, hi)

    out = os.path.join(outdir, "qual_jet_metrics_3row_mbhb%s.pdf" % suffix)
    assert not os.path.exists(out), "refusing to overwrite %s" % out
    fig.savefig(out, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out.replace(".pdf", ".png"), dpi=400,
                bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print("wrote %s" % out)
    print("   stack %s | %s | %d dies | htc %.0f | sample index %d"
          % (stack_id, tech, depth, htc, pick))
    for r in ORDER:
        print("     %-7s " % ROWS[r].replace("$", "")
              + "  ".join("%s MAE %.3f Max %.3f"
                          % (k, stats[(r, k)][0], stats[(r, k)][1])
                          for k in ARMS))
    return {"sample_index": int(pick), "stack_id": stack_id,
            "technology": tech, "n_dies": int(depth), "htc": htc,
            "bc_row": int(brow), "stack_row": int(srow),
            "selection": note, "figure": os.path.basename(out),
            "row_order_top_to_bottom": [ROWS[r] for r in ORDER],
            "layers_omitted_from_canvas":
                [ROWS[r] for r in range(7) if r not in ORDER],
            "shared_scale": "per-row vmin/vmax from PACT reference field",
            "annotation": "absolute cell-wise |T_hat - T_PACT|",
            "per_panel": {"%d|%s" % (r, k): {"abs_mae": v[0], "abs_max": v[1]}
                          for (r, k), v in stats.items() if r in ORDER}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=P["data_root"])
    ap.add_argument("--cv", default=P["cv_dir"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bc", type=int, default=4)
    ap.add_argument("--outdir", default=P["data_root"] + "/figs_mbhb")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = torch.device(os.environ.get("INFER_DEV", "cuda:0"))

    spec = json.load(open(os.path.join(a.cv, "cv_folds.json")))["folds"][a.fold]
    norm = os.path.join(a.cv, "norm_cv", "fold%d.json" % a.fold)
    roots = {d: os.path.join(a.base_dir, "pact_stack_%dL" % d) for d in (2, 3, 4)}

    ic = pd.read_csv(os.path.join(roots[4], "index", "interconnect.csv"))
    ic_name = {int(r): n for r, n in zip(ic["ic_row"], ic["interconnect"])}
    stacks = pd.read_csv(os.path.join(roots[4], "index", "stacks.csv"))

    ds = {arm: MultiDepthStackDataset(roots, arm=arm, groups=spec["test"],
                                      norm_path=norm) for arm in ARMS}
    nets = {arm: load_arm(arm, a.fold, os.path.join(a.cv, "runs"),
                          ds[arm].in_channels, dev) for arm in ARMS}

    cand = [i for i, (d, r, b) in enumerate(ds["A3"].pairs)
            if d == 4 and b == a.bc]
    errs = []
    for x, y, m, aux in DataLoader(Subset(ds["A3"], cand), batch_size=16,
                                   num_workers=8):
        x, y = x.to(dev), y.to(dev)
        s = torch.stack([aux["log10_htc_norm"].to(dev),
                         aux["depth_norm"].to(dev)], 1)
        with torch.no_grad():
            p, _, _ = nets["A3"](x, s)
        p, y = p.float()[:, :7], y.float()[:, :7]
        pat = ((p - p.mean((-2, -1), keepdim=True))
               - (y - y.mean((-2, -1), keepdim=True)))
        errs.extend(pat.abs().mean((1, 2, 3)).cpu().tolist())
    errs = np.asarray(errs)
    order = np.argsort(errs)

    pick_all = cand[int(order[len(order) // 2])]
    tech_all = ic_name[int(ds["A3"][pick_all][3]["ic_row"])]
    print("candidate pool: %d held-out four-die stacks at bc_%d"
          % (len(cand), a.bc))
    print("rule-faithful median pick is a %s stack" % tech_all)

    recs = {}
    recs["rulematch"] = render(
        pick_all, ds, nets, dev, ic_name, stacks, a.outdir, "_rulematch",
        "median Solo2Stack pattern error among ALL held-out four-die stacks "
        "at bc_%d (the published rule, verbatim)" % a.bc)

    if tech_all == "hybrid":
        print("\nrule-faithful pick is already hybrid; no separate "
              "hybrid-restricted figure is needed")
        recs["hybrid"] = "same as rulematch"
    else:
        hyb = [i for i in cand
               if stacks.loc[ds["A3"].pairs[i][1], "interconnect"] == "hybrid"]
        hset = {i: e for i, e in zip(cand, errs)}
        hyb_sorted = sorted(hyb, key=lambda i: hset[i])
        pick_h = hyb_sorted[len(hyb_sorted) // 2]
        print("\nhybrid-restricted pool: %d stacks" % len(hyb))
        recs["hybrid"] = render(
            pick_h, ds, nets, dev, ic_name, stacks, a.outdir, "_hybrid",
            "median Solo2Stack pattern error among held-out four-die HYBRID "
            "stacks at bc_%d (matches the published caption)" % a.bc)

    sel = os.path.join(a.outdir, "qual_jet_selection_3row_mbhb.json")
    assert not os.path.exists(sel), "refusing to overwrite %s" % sel
    json.dump({"fold": a.fold, "test_group": spec["test"],
               "cv_dir": a.cv, "base_dir": a.base_dir,
               "bc_row": a.bc,
               "candidate_pool_size": len(cand),
               "picks": recs}, open(sel, "w"), indent=2)
    print("wrote %s" % sel)


if __name__ == "__main__":
    main()
