#!/usr/bin/env python3
"""Pack the PACT output of an N-die stack dataset into a transferable ML dataset.

Replaces pack_dataset_2L.py and pack_dataset_nl.py with one depth-generic
packer.  Storage is lossless: PACT emits temperatures quantised to 0.01 K, so
each field is stored as an int16 deviation in centi-kelvin from a per-field
int32 offset -- 2 bytes a cell, the same as float16 but exact.

    T_kelvin = (off_ck[..., None, None] + dT_ck) / 100.0

Unified against the two per-depth packers it replaces:

  * index/stacks.csv always names dies d1_row/d1_chiplet ... dN_row/dN_chiplet.
    The old 2-die packer wrote bot_row/top_row/bot_chiplet/top_chiplet;
    model/loader_multi.py accepts both, so existing packed data still loads.
  * total_power_w is round(sum, 9) at every depth.  The old 2-die packer left
    the sum unrounded.
  * arrays/analytic_k.f64.npy is no longer written.  Nothing read it: the
    loader recomputes T_ana from total_power_w and htc.
  * bonding technologies come from the data, never from a table in this file.
    --ic-order fixes their row order when it has to match an existing dataset.

    python data_gen/pack/pack_dataset.py --n-die 3 --out /scratch/pack_3die
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
from solo2stack.paths import P as PATHS  # noqa: E402
# --------------------------------------------------------------------------

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pact_io as IO  # noqa: E402

BC_HTC = [5000, 10000, 25000, 50000, 100000]
T_AMBIENT_K = 318.15
SOLO_LAYER = 0

# bonding geometry: floorplan file name -> effective bond-layer thickness.
# Keyed by technology name as it appears in the dataset, so adding or removing
# a technology does not touch this file's logic.
BOND_FLP = {
    "hybrid":    ("hybrid_100x100_5um_flp.csv",   1.5e-6),
    "microbump": ("microbump_40x40_30um_flp.csv", 2.5e-5),
}

G = IO.GRID
NBC = len(BC_HTC)
_shared = {}
_cfg = {}


def layer_names(n_die):
    """D1 I12 D2 I23 D3 ... DN."""
    out = []
    for i in range(n_die):
        out.append("D%d" % (i + 1))
        if i != n_die - 1:
            out.append("I%d%d" % (i + 1, i + 2))
    return out


def read_manifest(path, n_die):
    """Normalise either manifest schema into (stack_id, group, dies, tech).

    Accepts the new manifest_<N>die.json (d1..dN keys) and the legacy
    manifest_{2,3,4}L.json (bottom_chiplet/top_chiplet, or D1..DN).
    """
    doc = json.load(open(path))
    recs = doc["stacks"] if isinstance(doc, dict) else doc
    out = []
    for r in recs:
        if "d1" in r:
            dies = [r["d%d" % (i + 1)] for i in range(n_die)]
        elif "D1" in r:
            dies = [r["D%d" % (i + 1)] for i in range(n_die)]
        elif "bottom_chiplet" in r:
            dies = [r["bottom_chiplet"], r["top_chiplet"]]
        else:
            raise SystemExit("%s: unrecognised stack record keys %s"
                             % (path, sorted(r)))
        if len(dies) != n_die or any(d is None for d in dies):
            raise SystemExit("%s: %s does not have %d dies"
                             % (path, r.get("stack_id"), n_die))
        out.append({"stack_id": r["stack_id"], "group": r["group"],
                    "dies": dies, "interconnect": r["interconnect"]})
    return out


def _init(out, outdir, nsl):
    _shared["sdt"] = np.lib.format.open_memmap(
        os.path.join(out, "arrays", "stack_dT_ck.i16.npy"), mode="r+")
    _shared["soff"] = np.lib.format.open_memmap(
        os.path.join(out, "arrays", "stack_off_ck.i32.npy"), mode="r+")
    _shared["cdt"] = np.lib.format.open_memmap(
        os.path.join(out, "arrays", "solo_dT_ck.i16.npy"), mode="r+")
    _shared["coff"] = np.lib.format.open_memmap(
        os.path.join(out, "arrays", "solo_off_ck.i32.npy"), mode="r+")
    _cfg["outdir"] = outdir
    _cfg["nsl"] = nsl


def _pack_field(ck):
    """int32 centi-kelvin field -> (int32 offset, int16 deviation).

    The offset is the rounded field mean, so the stored deviation is signed and
    centred; this is the encoding the published datasets use.
    """
    off = int(np.rint(ck.mean()))
    d = ck - off
    lo, hi = int(d.min()), int(d.max())
    if lo < -32768 or hi > 32767:
        raise OverflowError("deviation %d..%d exceeds int16" % (lo, hi))
    return off, d.astype(np.int16)


def _do_stack(job):
    row, stack_id = job
    for b in range(NBC):
        for j in range(_cfg["nsl"]):
            p = os.path.join(_cfg["outdir"],
                             "%s_bc_%d.grid.steady.layer%d" % (stack_id, b, j))
            off, d = _pack_field(IO.read_grid_ck(p))
            _shared["soff"][row, b, j] = off
            _shared["sdt"][row, b, j] = d.reshape(G, G)
    return row


def _do_chiplet(job):
    row, chip_id = job
    for b in range(NBC):
        p = os.path.join(PATHS["solo_runs"],
                         "%s_bc_%d.grid.steady.layer%d" % (chip_id, b, SOLO_LAYER))
        off, d = _pack_field(IO.read_grid_ck(p))
        _shared["coff"][row, b] = off
        _shared["cdt"][row, b] = d.reshape(G, G)
    return row


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="Pack an N-die PACT stack dataset (depth-generic).")
    ap.add_argument("--n-die", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lcf-root", default=None,
                    help="directory holding the stack manifest; defaults to "
                         "<chiplet_library>/stack_lcf_<N>L")
    ap.add_argument("--manifest", default=None,
                    help="explicit path to the stack manifest JSON")
    ap.add_argument("--pact-out", default=None,
                    help="directory of .grid.steady files; defaults to "
                         "<pact_runs>/output_stack_<N>die")
    ap.add_argument("--ic-order", nargs="+", default=None,
                    help="bonding technology row order (default: sorted)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-checksums", action="store_true")
    a = ap.parse_args()

    if a.n_die < 2:
        raise SystemExit("--n-die must be at least 2")

    n_die = a.n_die
    names = layer_names(n_die)
    nsl = len(names)
    die_keys = ["d%d" % (i + 1) for i in range(n_die)]
    lib = PATHS["chiplet_library"]

    lcf_root = a.lcf_root or os.path.join(lib, "stack_lcf_%dL" % n_die)
    man_path = a.manifest or os.path.join(lcf_root,
                                          "manifest_%ddie.json" % n_die)
    if not os.path.exists(man_path):
        raise SystemExit("no stack manifest at %s" % man_path)
    outdir = a.pact_out or os.path.join(PATHS["pact_runs"],
                                        "output_stack_%ddie" % n_die)

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "arrays"), exist_ok=True)
    os.makedirs(os.path.join(out, "index"), exist_ok=True)
    t_start = time.time()

    stacks = sorted(read_manifest(man_path, n_die), key=lambda s: s["stack_id"])
    chiplets = json.load(open(os.path.join(lib, "chiplet_manifest.json")))["chiplets"]
    chip_ids = sorted(c["chiplet_id"] for c in chiplets)
    chip_group = {c["chiplet_id"]: c["group"] for c in chiplets}

    if a.limit:
        stacks = stacks[:a.limit]
        keep = set()
        for s in stacks:
            keep |= set(s["dies"])
        chip_ids = [c for c in chip_ids if c in keep]
    chip_row = {c: i for i, c in enumerate(chip_ids)}

    ic_types = a.ic_order or sorted({s["interconnect"] for s in stacks})
    missing = [t for t in ic_types if t not in BOND_FLP]
    if missing:
        raise SystemExit("no bond geometry for %s (have %s)"
                         % (missing, sorted(BOND_FLP)))

    n_stacks, n_chip = len(stacks), len(chip_ids)
    print("packing %d stacks x %d BC x %d layers (%s), %d chiplets, ic order %s"
          % (n_stacks, NBC, nsl, " ".join(names), n_chip, ic_types), flush=True)

    chip_meta = []
    for c in chip_ids:
        m = json.load(open(os.path.join(lib, "meta", "%s.json" % c)))
        pt = pd.read_csv(os.path.join(lib, "raw_ptraces", "%s_ptrace.csv" % c))
        pt.columns = [x.strip() for x in pt.columns]
        chip_meta.append({
            "row": chip_row[c], "chiplet_id": c, "group": chip_group[c],
            "n_blocks": m["n_blocks"],
            "total_power_w": round(float(pt["Power"].sum()), 9),
            "die_size_m": m["target_size_m"],
        })
    chip_df = pd.DataFrame(chip_meta).sort_values("row")
    chip_df.to_csv(os.path.join(out, "index", "chiplets.csv"), index=False)
    pw = dict(zip(chip_df["chiplet_id"], chip_df["total_power_w"]))

    stack_rows = []
    for i, s in enumerate(stacks):
        r = {"row": i, "stack_id": s["stack_id"], "group": s["group"],
             "interconnect": s["interconnect"],
             "ic_row": ic_types.index(s["interconnect"])}
        for k, cid in zip(die_keys, s["dies"]):
            r["%s_chiplet" % k] = cid
            r["%s_row" % k] = chip_row[cid]
        r["total_power_w"] = round(sum(pw[c] for c in s["dies"]), 9)
        stack_rows.append(r)
    stack_df = pd.DataFrame(stack_rows)
    stack_df.to_csv(os.path.join(out, "index", "stacks.csv"), index=False)

    area = IO.DIE_SIZE_M ** 2
    bc_df = pd.DataFrame([{
        "bc_row": i, "name": "bc_%d" % i, "htc_w_m2k": h,
        "r_conv_m2k_w": 1.0 / h,
        "log10_htc": float(np.log10(h)),
        "dT_lumped_per_watt_k": 1.0 / (h * area),
    } for i, h in enumerate(BC_HTC)])
    bc_df.to_csv(os.path.join(out, "index", "bc.csv"), index=False)

    gmap = np.zeros((len(ic_types), G, G), dtype=np.float32)
    ic_meta = []
    for i, ic in enumerate(ic_types):
        fn, th = BOND_FLP[ic]
        path = os.path.join(lib, "interconnect_flp_grid", fn)
        g, sub = IO.rasterize_flp_conductance(path, th)
        gmap[i] = np.log10(g).astype(np.float32)
        ic_meta.append({
            "ic_row": i, "interconnect": ic, "flp_csv": fn, "thickness_m": th,
            "cu_area_fraction": round(IO.cu_area_fraction(path), 6),
            "g_min_w_m2k": float(g.min()), "g_mean_w_m2k": float(g.mean()),
            "g_max_w_m2k": float(g.max()), "supersample": sub,
        })
    np.save(os.path.join(out, "arrays", "gmap_log10.f32.npy"), gmap)
    pd.DataFrame(ic_meta).to_csv(os.path.join(out, "index", "interconnect.csv"),
                                 index=False)
    print("  gmap done", flush=True)

    pmap = np.zeros((n_chip, G, G), dtype=np.float32)
    for c in chip_ids:
        pmap[chip_row[c]] = np.load(
            os.path.join(lib, "powermap_npy", "%s_powermap.npy" % c))
    np.save(os.path.join(out, "arrays", "powermap_w_m2.f32.npy"), pmap)
    print("  powermaps done", flush=True)

    for name, shape, dt in [
        ("stack_dT_ck.i16.npy",  (n_stacks, NBC, nsl, G, G), np.int16),
        ("stack_off_ck.i32.npy", (n_stacks, NBC, nsl),       np.int32),
        ("solo_dT_ck.i16.npy",   (n_chip, NBC, G, G),        np.int16),
        ("solo_off_ck.i32.npy",  (n_chip, NBC),              np.int32),
    ]:
        np.lib.format.open_memmap(os.path.join(out, "arrays", name),
                                  mode="w+", dtype=dt, shape=shape).flush()

    jobs_c = [(chip_row[c], c) for c in chip_ids]
    jobs_s = [(r["row"], r["stack_id"]) for r in stack_rows]

    with Pool(args_workers(a), initializer=_init,
              initargs=(out, outdir, nsl)) as pool:
        for n, _ in enumerate(pool.imap_unordered(_do_chiplet, jobs_c, chunksize=4), 1):
            if n % 100 == 0 or n == len(jobs_c):
                print("  chiplets %d/%d" % (n, len(jobs_c)), flush=True)
        t0 = time.time()
        for n, _ in enumerate(pool.imap_unordered(_do_stack, jobs_s, chunksize=4), 1):
            if n % 200 == 0 or n == len(jobs_s):
                el = time.time() - t0
                print("  stacks %d/%d  %.0fs elapsed, ~%.0fs left"
                      % (n, len(jobs_s), el, el / n * (len(jobs_s) - n)), flush=True)

    groups = sorted(stack_df["group"].unique())
    splits = {
        "by_group": {
            "note": "Chiplets are partitioned into disjoint groups and stacks only draw "
                    "within a group, so holding out whole groups gives an unseen-chiplet "
                    "test set. A random split over stacks leaks dies across the split.",
            "train": groups[:-2], "val": groups[-2:-1], "test": groups[-1:],
        },
        "by_interconnect": {
            "note": "Generalisation to an unseen bonding technology: train on every "
                    "technology but the last in ic_row order, test on that one.",
            "train": list(ic_types[:-1]), "test": list(ic_types[-1:]),
        },
        "by_bc": {
            "note": "Generalisation to an unseen htc. Interpolation, since 25000 is "
                    "interior to the training range.",
            "train": [0, 1, 3, 4], "test": [2],
        },
    }
    json.dump(splits, open(os.path.join(out, "index", "splits.json"), "w"), indent=2)

    sdt = np.load(os.path.join(out, "arrays", "stack_dT_ck.i16.npy"), mmap_mode="r")
    cdt = np.load(os.path.join(out, "arrays", "solo_dT_ck.i16.npy"), mmap_mode="r")
    tr = stack_df.index[stack_df["group"].isin(splits["by_group"]["train"])].to_numpy()
    if len(tr) == 0:
        tr = stack_df.index.to_numpy()
    samp = tr[:: max(1, len(tr) // 400)]

    sd = sdt[samp].astype(np.float32) / 100.0
    cd = cdt[:: max(1, n_chip // 200)].astype(np.float32) / 100.0
    norm = {
        "computed_from": {"stack_rows": len(samp),
                          "chiplet_rows": int(np.ceil(n_chip / max(1, n_chip // 200)))},
        "stack_dT_k": {
            "per_layer_std": [float(sd[:, :, j].std()) for j in range(nsl)],
            "per_layer_absmax": [float(np.abs(sd[:, :, j]).max()) for j in range(nsl)],
            "global_std": float(sd.std()),
        },
        "solo_dT_k": {"std": float(cd.std()), "absmax": float(np.abs(cd).max())},
        "powermap_w_m2": {"mean": float(pmap.mean()), "std": float(pmap.std()),
                          "max": float(pmap.max())},
        "gmap_log10": {"mean": float(gmap.mean()), "std": float(gmap.std()),
                       "min": float(gmap.min()), "max": float(gmap.max())},
        "log10_htc": {"mean": float(np.log10(BC_HTC).mean()),
                      "std": float(np.log10(BC_HTC).std())},
    }
    json.dump(norm, open(os.path.join(out, "norm_stats.json"), "w"), indent=2)

    meta = {
        "name": "pact_stack_%ddie" % n_die,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stack_manifest": man_path,
        "pact_output": outdir,
        "solver": "PACT steady-state, SciPy SuperLU "
                  "(MMD_AT_PLUS_A, SymmetricMode, diag_pivot_thresh=0.001), "
                  "400x400 uniform grid",
        "grid": G,
        "n_dies": n_die,
        "die_size_m": IO.DIE_SIZE_M,
        "die_area_m2": area,
        "ambient_k": T_AMBIENT_K,
        "bc_htc_w_m2k": BC_HTC,
        "interconnect_types": list(ic_types),
        "interconnect_rule": "all interconnect layers in a stack share one technology",
        "stack_layer_index": {str(i): n for i, n in enumerate(names)},
        "omitted_layer": {str(nsl): "NoPackage convective boundary; analytic"},
        "solo_layer_index": {"0": "die"},
        "reconstruct": "T_kelvin = (off_ck[..., None, None] + dT_ck) / 100.0",
        "residual": "T_r = T_kelvin - (ambient_k + total_power_w / (htc * die_area_m2))",
        "lossless": True,
        "counts": {"stacks": n_stacks, "chiplets": n_chip, "bcs": NBC,
                   "stack_layers": nsl},
        "pack_seconds": round(time.time() - t_start, 1),
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)

    if not a.skip_checksums:
        lines = []
        for d in ("arrays", "index"):
            for fn in sorted(os.listdir(os.path.join(out, d))):
                p = os.path.join(out, d, fn)
                lines.append("%s  %s/%s" % (sha256(p), d, fn))
        open(os.path.join(out, "checksums.sha256"), "w").write(
            "\n".join(lines) + "\n")
        print("  checksums done", flush=True)

    print("packed %s in %.1f s" % (out, time.time() - t_start))


def args_workers(a):
    return max(1, min(a.workers, os.cpu_count() or 1))


if __name__ == "__main__":
    main()
