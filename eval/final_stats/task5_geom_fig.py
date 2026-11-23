#!/usr/bin/env python3
"""Task 5 -- where the largest grid-cell error of each run lands.

Reads the 40 worst-cell records written by task45_model_pass.py, adds the
geometry of each cell, and renders one PNG for the most frequent worst stack.

Geometry, all in the array's own (row=y, col=x) convention:
  die edge          : min(x, y, 399-x, 399-y) cells -> um, i.e. the distance to
                      the nearest boundary of the 2 mm x 2 mm die
  microbump edge    : distance to the nearest cell that sits on a Cu-pad
                      boundary in this stack's conductance map.  The map is
                      read from gmap_log10[ic_row]; microbump has exactly two
                      levels (pad / underfill) and hybrid is uniform, so for a
                      hybrid stack this distance is undefined and reported as
                      None rather than faked.
  block boundary    : distance to the nearest floorplan block edge of the two
                      dies that bound the interface, rasterised at 5 um/cell.

The cell -> floorplan convention is not assumed: it is verified by rasterising
the microbump interconnect floorplan and requiring an exact match with the
packed conductance map.
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
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

add_repo(__file__, "model")

FS = P["results"]
PASS = os.path.join(FS, "pass")
BASE = P["data_root"]
CV = P["cv_dir"]
FLP = P["data_root"] + "/pactgen/floorset_pipeline_out"
GRID = 400
DIE_M = 0.002
UM_PER_CELL = DIE_M / GRID * 1e6          # 5.0 um
LAYER_NAMES = {2: ["D1", "I12", "D2"],
               3: ["D1", "I12", "D2", "I23", "D3"],
               4: ["D1", "I12", "D2", "I23", "D3", "I34", "D4"]}
# which two dies bound each interface layer
IFACE_DIES = {"I12": ("d1", "d2"), "I23": ("d2", "d3"), "I34": ("d3", "d4")}


def rasterise(df, label=None):
    """Mark cells whose centre falls inside a block (optionally one label)."""
    m = np.zeros((GRID, GRID), dtype=bool)
    xc = (np.arange(GRID) + 0.5) * (DIE_M / GRID)
    yc = (np.arange(GRID) + 0.5) * (DIE_M / GRID)
    for _, r in df.iterrows():
        if label is not None and r["Label"] != label:
            continue
        x0, y0 = float(r["X"]), float(r["Y"])
        x1, y1 = x0 + float(r["Length (m)"]), y0 + float(r["Width (m)"])
        ix = np.where((xc >= x0) & (xc < x1))[0]
        iy = np.where((yc >= y0) & (yc < y1))[0]
        if len(ix) and len(iy):
            m[np.ix_(iy, ix)] = True
    return m


def boundary_cells(mask):
    """Cells on the inside edge of a True region (4-neighbour)."""
    b = np.zeros_like(mask)
    b[:-1, :] |= mask[:-1, :] != mask[1:, :]
    b[1:, :] |= mask[:-1, :] != mask[1:, :]
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    b[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return b & mask


def block_boundary_mask(df):
    """Cells adjacent to a change of block id -> the floorplan's internal edges."""
    ids = np.full((GRID, GRID), -1, dtype=np.int32)
    xc = (np.arange(GRID) + 0.5) * (DIE_M / GRID)
    yc = (np.arange(GRID) + 0.5) * (DIE_M / GRID)
    for k, (_, r) in enumerate(df.iterrows()):
        x0, y0 = float(r["X"]), float(r["Y"])
        x1, y1 = x0 + float(r["Length (m)"]), y0 + float(r["Width (m)"])
        ix = np.where((xc >= x0) & (xc < x1))[0]
        iy = np.where((yc >= y0) & (yc < y1))[0]
        if len(ix) and len(iy):
            ids[np.ix_(iy, ix)] = k
    b = np.zeros((GRID, GRID), dtype=bool)
    b[:-1, :] |= ids[:-1, :] != ids[1:, :]
    b[1:, :] |= ids[:-1, :] != ids[1:, :]
    b[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    b[:, 1:] |= ids[:, :-1] != ids[:, 1:]
    return b


def dist_to(mask, x, y):
    """Euclidean distance in cells from (x, y) to the nearest True cell."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    d = np.hypot(xs.astype(np.float64) - x, ys.astype(np.float64) - y)
    return float(d.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-out", default=os.path.join(FS, "task5_worst_stack.png"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    gmap = np.load(os.path.join(BASE, "pact_stack_4L", "arrays",
                                "gmap_log10.f32.npy"))
    ic = pd.read_csv(os.path.join(BASE, "pact_stack_4L", "index",
                                  "interconnect.csv")).sort_values("ic_row")
    ic_name = ic["interconnect"].tolist()
    mb_row = ic_name.index("microbump")

    # ---- verify the cell <-> floorplan convention before using it ----
    mb_flp = pd.read_csv(os.path.join(FLP, "interconnect_flp_grid",
                                      ic.loc[ic["ic_row"] == mb_row,
                                             "flp_csv"].iloc[0]))
    cu_label = [l for l in mb_flp["Label"].unique() if l.lower().startswith("cu")]
    assert len(cu_label) == 1, mb_flp["Label"].unique()
    pad_raster = rasterise(mb_flp, cu_label[0])
    pad_from_gmap = gmap[mb_row] > gmap[mb_row].min()
    assert np.array_equal(pad_raster, pad_from_gmap), (
        "cell<->floorplan convention mismatch: rasterised Cu pads %d cells, "
        "conductance-map pads %d cells, overlap %d"
        % (pad_raster.sum(), pad_from_gmap.sum(),
           (pad_raster & pad_from_gmap).sum()))
    print("convention check OK: %d Cu-pad cells match the conductance map"
          % pad_raster.sum())
    pad_edge = boundary_cells(pad_from_gmap)

    stacks = {d: pd.read_csv(os.path.join(BASE, "pact_stack_%dL" % d,
                                          "index", "stacks.csv"))
              for d in (2, 3, 4)}

    # ---- collect the 40 worst cells ----
    recs = []
    for p in sorted(glob.glob(os.path.join(PASS, "*_fold*.json"))):
        j = json.load(open(p))
        w = j["worst_cell"]
        d = w["depth"]
        st = stacks[d]
        row = w["stack_row"]
        chips = [st.at[row, c] for c in st.columns if c.endswith("_chiplet")]
        r = dict(arm=j["arm"], fold=j["fold"], stack_id=w["stack_id"],
                 depth=d, tech=w["tech"], htc=w["htc"], layer=w["layer"],
                 x=w["x"], y=w["y"], err_k=w["err"], true_k=w["true_k"],
                 pred_k=w["pred_k"], chiplets="|".join(map(str, chips)),
                 stack_row=row, bc_row=w["bc_row"])

        r["dist_die_edge_um"] = UM_PER_CELL * min(w["x"], w["y"],
                                                  GRID - 1 - w["x"],
                                                  GRID - 1 - w["y"])
        if w["tech"] == "microbump":
            dd = dist_to(pad_edge, w["x"], w["y"])
            r["dist_microbump_edge_um"] = None if dd is None else UM_PER_CELL * dd
            r["inside_cu_pad"] = bool(pad_from_gmap[w["y"], w["x"]])
        else:
            # hybrid bonding has a single conductance level: no bump geometry
            r["dist_microbump_edge_um"] = None
            r["inside_cu_pad"] = None

        if w["layer"] in IFACE_DIES:
            ds = []
            for slot in IFACE_DIES[w["layer"]]:
                cid = st.at[row, "%s_chiplet" % slot]
                f = pd.read_csv(os.path.join(FLP, "raw_floorplans",
                                             "%s_flp.csv" % cid))
                ds.append((slot, cid,
                           UM_PER_CELL * dist_to(block_boundary_mask(f),
                                                 w["x"], w["y"])))
            for slot, cid, dv in ds:
                r["dist_block_edge_%s_um" % slot] = dv
                r["%s_chiplet" % slot] = cid
            r["dist_block_edge_min_um"] = min(d for _, _, d in ds)
        recs.append(r)

    keys = sorted({k for r in recs for k in r})
    with open(os.path.join(FS, "task5_worst_cells.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)

    # ---- agreement across arms ----
    print("\n## Task 5 - the 40 fold-level maxima\n")
    print("| Arm | Fold | Stack | N | Bonding | h | Layer | cell (x,y) | err (K) | "
          "true T (K) | die edge (um) | bump edge (um) | block edge (um) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda r: (r["arm"], r["fold"])):
        print("| %s | %d | %s | %d | %s | %.0f | %s | (%d,%d) | %.4f | %.2f | %.1f | %s | %s |"
              % (r["arm"], r["fold"], r["stack_id"][:34], r["depth"], r["tech"],
                 r["htc"], r["layer"], r["x"], r["y"], r["err_k"], r["true_k"],
                 r["dist_die_edge_um"],
                 "n/a" if r["dist_microbump_edge_um"] is None
                 else "%.1f" % r["dist_microbump_edge_um"],
                 "n/a" if r.get("dist_block_edge_min_um") is None
                 else "%.1f" % r["dist_block_edge_min_um"]))

    print("\nlayer of the maximum: %s" % dict(Counter(r["layer"] for r in recs)))
    print("bonding: %s" % dict(Counter(r["tech"] for r in recs)))
    print("depth: %s" % dict(Counter(r["depth"] for r in recs)))
    print("h: %s" % dict(Counter(r["htc"] for r in recs)))
    sc = Counter(r["stack_id"] for r in recs)
    print("distinct stacks: %d; most common: %s" % (len(sc), sc.most_common(3)))
    print("\nper fold, do all four arms hit the same stack?")
    for f in range(10):
        sub = [r for r in recs if r["fold"] == f]
        same = len({r["stack_id"] for r in sub}) == 1
        cells = {r["arm"]: (r["x"], r["y"]) for r in sub}
        if same:
            pts = np.array(list(cells.values()), dtype=float)
            dmax = max(np.hypot(*(p - q)) for p in pts for q in pts)
            print("  fold %d: same stack (%s), max cell separation %.1f cells "
                  "(%.0f um)  %s" % (f, sub[0]["stack_id"][:40], dmax,
                                     dmax * UM_PER_CELL, cells))
        else:
            print("  fold %d: DIFFERENT stacks %s" % (f, cells))

    if a.no_fig:
        return
    make_fig(sc.most_common(1)[0][0], recs, gmap, ic_name, stacks, a)


def make_fig(stack_id, recs, gmap, ic_name, stacks, a):
    """PACT temperature / Solo2Stack error / conductance map for one stack."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from loader_multi import MultiDepthStackDataset
    from unet_decomp import ThermalUNetDecomp

    sub = [r for r in recs if r["stack_id"] == stack_id]
    a3 = [r for r in sub if r["arm"] == "A3"] or sub
    r = a3[0]
    d, row, bc = r["depth"], r["stack_row"], r["bc_row"]
    fold = r["fold"]
    st = stacks[d]
    group = st.at[row, "group"]
    li = LAYER_NAMES[d].index(r["layer"])

    roots = {k: os.path.join(BASE, "pact_stack_%dL" % k) for k in (2, 3, 4)}
    norm = os.path.join(CV, "norm_cv", "fold%d.json" % fold)
    ds = MultiDepthStackDataset(roots, arm="A3", groups=[group], norm_path=norm,
                               depths=(d,))
    idx = ds.pairs.index((d, row, bc))
    x, y, mask, aux = ds[idx]

    dev = torch.device(a.device)
    model = ThermalUNetDecomp(in_ch=ds.in_channels, base=32, depth=4,
                              scalar_dim=2).to(dev)
    sd = torch.load(os.path.join(CV, "runs", "A3_fold%d" % fold, "best.pt"),
                    map_location=dev)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    with torch.no_grad():
        p, _, _ = model(x[None].to(dev),
                        torch.tensor([[float(aux["log10_htc_norm"]),
                                       float(aux["depth_norm"])]],
                                     device=dev))
    ana = float(aux["analytic_k"])
    pred = p[0, li].float().cpu().numpy() + ana      # residual arm -> absolute
    true = y[li].numpy() + ana
    err = pred - true
    g = gmap[int(st.at[row, "ic_row"])]

    # the recorded cell must still be the worst cell of this field
    yy, xx = np.unravel_index(np.abs(err).argmax(), err.shape)
    print("\nfigure stack %s  layer %s  recorded cell (%d,%d) err %.4f; "
          "recomputed worst (%d,%d) err %.4f"
          % (stack_id, r["layer"], r["x"], r["y"], r["err_k"], xx, yy,
             abs(err[yy, xx])))

    # Two rows: the whole interface, and a zoom around the worst cell.  A
    # single 5 um cell is far below one screen pixel at full-die scale, so the
    # marked maximum is only legible in the zoom.
    Z = 25                                     # +- 25 cells = +- 125 um
    x0, x1 = max(0, r["x"] - Z), min(GRID, r["x"] + Z + 1)
    y0, y1 = max(0, r["y"] - Z), min(GRID, r["y"] + Z + 1)
    sub_ext = [x0 * 2.0 / GRID, x1 * 2.0 / GRID, y0 * 2.0 / GRID, y1 * 2.0 / GRID]
    v = float(np.abs(err).max())
    panels = [(true, "jet", None, None, "T (K)",
               "PACT %s, h=%.0e W/m$^2$K (%s)" % (r["layer"], r["htc"], r["tech"])),
              (err, "bwr", -v, v, "error (K)",
               "Solo2Stack error (pred - PACT), max |err| = %.2f K" % v),
              (g, "viridis", None, None, "$\\log_{10}$ G",
               "conductance map $\\log_{10}G$ (%s)" % r["tech"])]

    fig, ax = plt.subplots(2, 3, figsize=(13.6, 8.6))
    ext = [0, 2, 0, 2]      # mm
    for k, (arr, cm, lo, hi, cl, ttl) in enumerate(panels):
        im = ax[0, k].imshow(arr, origin="lower", extent=ext, cmap=cm,
                             vmin=lo, vmax=hi)
        ax[0, k].set_title(ttl, fontsize=9)
        plt.colorbar(im, ax=ax[0, k], label=cl, fraction=0.046)
        ax[0, k].add_patch(plt.Rectangle(
            (sub_ext[0], sub_ext[2]), sub_ext[1] - sub_ext[0],
            sub_ext[3] - sub_ext[2], fill=False, ec="k", lw=1.0))

        im = ax[1, k].imshow(arr[y0:y1, x0:x1], origin="lower", extent=sub_ext,
                             cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        ax[1, k].set_title("zoom $\\pm$%d $\\mu$m" % int(Z * UM_PER_CELL),
                           fontsize=9)
        plt.colorbar(im, ax=ax[1, k], label=cl, fraction=0.046)

    for row, ms_ in ((0, 13), (1, 22)):
        for k in range(3):
            ax[row, k].plot(r["x"] * 2.0 / GRID, r["y"] * 2.0 / GRID, marker="o",
                            mfc="none", mec="k", ms=ms_, mew=1.8)
            ax[row, k].set_xlabel("x (mm)")
            ax[row, k].set_ylabel("y (mm)")
    fig.suptitle("%s\nfold %d, worst cell x=%d y=%d, |error| = %.2f K"
                 % (stack_id, fold, r["x"], r["y"], abs(err[r["y"], r["x"]])),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(a.fig_out, dpi=200)
    print("wrote", a.fig_out)


if __name__ == "__main__":
    main()
