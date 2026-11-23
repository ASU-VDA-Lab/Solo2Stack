"""Unified loader over the 2-, 3- and 4-chiplet stack datasets.

One model handles all three depths by using fixed slots with a presence mask
rather than a depth-specific architecture:

    inputs   4 power-map slots          (bottom-aligned; absent dies are zero)
             4 presence planes          (1.0 where a die exists, else 0.0)
             1 interconnect conductance map (log10; one technology per stack)
             1 boundary-condition plane
             [A3 only] 4 solo-residual slots
             -> 10 channels for A1, 14 for A3

    target   7 layer slots  D1 I12 D2 I23 D3 I34 D4, bottom-aligned
             valid = 2*n_dies - 1  (3, 5 or 7); the rest are zero and masked out

Slots are bottom-aligned so slot k is always stack layer k, which keeps the
input and target indexing consistent. The die count reaches the network two
ways: explicitly through the presence planes (available to the very first
convolution) and through the FiLM scalars at the bottleneck.

Performance note: every normalised input that depends only on (chiplet, bc) or
(technology) is precomputed once in __init__ rather than per sample. Building a
14-channel 400x400 input with per-sample arithmetic left the GPUs at 0%
utilisation, entirely data-starved; precomputing turns __getitem__ into memory
copies. The arrays are built before the DataLoader forks, so workers share them
copy-on-write instead of each holding a copy.

The per-input normalisation constants, the power maps, the conductance maps and
the solo fields are byte-identical across the three dataset roots -- they derive
from the same 500 chiplets, the same three bonding technologies and the same
five cooling conditions -- so one shared copy is used. This was verified by
checksum, not assumed.
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

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_DIES = 4
MAX_LAYERS = 2 * MAX_DIES - 1  # 7

ARMS = {
    "A0": dict(solo=False, target="absolute"),
    "A1": dict(solo=False, target="residual"),
    "A2": dict(solo=True,  target="absolute"),
    "A3": dict(solo=True,  target="residual"),
}

# stacks.csv names its die-row columns d1_row ... dN_row at every depth.  The
# superseded 2-chiplet packer wrote bot_row/top_row instead, so that spelling is
# still accepted and existing packed datasets keep loading unchanged.
LEGACY_DIE_COLS = {2: ["bot_row", "top_row"]}


def die_columns(df, depth):
    """Die-row column names for `depth`, in die order, from either schema."""
    unified = ["d%d_row" % (i + 1) for i in range(depth)]
    if all(c in df.columns for c in unified):
        return unified
    legacy = LEGACY_DIE_COLS.get(depth)
    if legacy and all(c in df.columns for c in legacy):
        return legacy
    raise KeyError(
        "stacks.csv has no usable die-row columns for depth %d: expected %s "
        "or %s; present: %s" % (depth, unified, legacy, sorted(df.columns)))


def ic_names(root):
    """Bonding technology names in ic_row order, read from the dataset index.

    Read rather than hard-coded so a root holding any subset of the bonding
    technologies reports its own labels and column count.
    """
    df = pd.read_csv(os.path.join(root, "index", "interconnect.csv"))
    return df.sort_values("ic_row")["interconnect"].tolist()


class _OneDepth:
    """Per-root indices and the stack temperature arrays (which differ by depth)."""

    def __init__(self, root, depth):
        self.root = os.path.abspath(root)
        self.depth = depth
        self.n_spatial = 2 * depth - 1
        self.meta = json.load(open(os.path.join(self.root, "meta.json")))
        self.splits = json.load(open(os.path.join(self.root, "index", "splits.json")))
        self.stacks = pd.read_csv(os.path.join(self.root, "index", "stacks.csv"))
        self.bc = pd.read_csv(os.path.join(self.root, "index", "bc.csv"))

        a = os.path.join(self.root, "arrays")
        self.sdt = np.load(os.path.join(a, "stack_dT_ck.i16.npy"), mmap_mode="r")
        self.soff = np.load(os.path.join(a, "stack_off_ck.i32.npy"))

        self.area = self.meta["die_area_m2"]
        self.t_amb = self.meta["ambient_k"]
        self.die_cols = die_columns(self.stacks, depth)
        assert self.sdt.shape[2] == self.n_spatial, (
            "%s has %d layers, expected %d" % (root, self.sdt.shape[2], self.n_spatial))

        # per-stack die rows and analytic level, materialised once
        self.die_rows = self.stacks[self.die_cols].to_numpy(dtype=np.int64)
        self.ic_row = self.stacks["ic_row"].to_numpy(dtype=np.int64)
        p = self.stacks["total_power_w"].to_numpy(dtype=np.float64)
        htc = self.bc["htc_w_m2k"].to_numpy(dtype=np.float64)
        self.analytic = (self.t_amb
                         + p[:, None] / (htc[None, :] * self.area)).astype(np.float64)


class MultiDepthStackDataset(Dataset):
    """One sample = one (dataset, stack, boundary condition) triple."""

    def __init__(self, roots, split="train", arm="A3", normalise=True,
                 norm_path=None, depths=None, groups=None):
        """roots: {depth: path}.

        groups: explicit list of chiplet groups for this split. Overrides the
        dataset's stored by_group split, which is what cross-validation needs --
        the same group list is applied to every depth, so a group held out for
        test is held out at 2, 3 and 4 dies simultaneously.
        """
        if arm not in ARMS:
            raise ValueError("arm must be one of %s" % sorted(ARMS))
        self.arm = arm
        self.use_solo = ARMS[arm]["solo"]
        self.target = ARMS[arm]["target"]
        self.normalise = normalise

        depths = sorted(depths or roots)
        self.depths = depths
        self.ds = {d: _OneDepth(roots[d], d) for d in depths}

        ref = self.ds[depths[0]]
        self.norm = json.load(open(norm_path or
                                   os.path.join(ref.root, "norm_stats.json")))
        self.grid = G = ref.meta["grid"]

        # ---- shared, precomputed, normalised inputs -------------------------
        a = os.path.join(ref.root, "arrays")
        chips = pd.read_csv(os.path.join(ref.root, "index", "chiplets.csv"))

        pmap = np.load(os.path.join(a, "powermap_w_m2.f32.npy"))
        if normalise:
            s = self.norm["powermap_w_m2"]
            pmap = (pmap - s["mean"]) / (s["std"] + 1e-12)
        self.pmap_n = np.ascontiguousarray(pmap, dtype=np.float32)

        gmap = np.load(os.path.join(a, "gmap_log10.f32.npy"))
        if normalise:
            s = self.norm["gmap_log10"]
            gmap = (gmap - s["mean"]) / (s["std"] + 1e-12)
        self.gmap_n = np.ascontiguousarray(gmap, dtype=np.float32)

        nbc = len(ref.bc)
        self.bc_planes = np.empty((nbc, G, G), dtype=np.float32)
        for b in range(nbc):
            v = float(ref.bc.loc[b, "log10_htc"])
            if normalise:
                st = self.norm["log10_htc"]
                v = (v - st["mean"]) / (st["std"] + 1e-12)
            self.bc_planes[b] = v

        self.presence = {}
        for d in depths:
            pr = np.zeros((MAX_DIES, G, G), dtype=np.float32)
            pr[:d] = 1.0
            self.presence[d] = pr

        if self.use_solo:
            cdt = np.load(os.path.join(a, "solo_dT_ck.i16.npy"), mmap_mode="r")
            coff = np.load(os.path.join(a, "solo_off_ck.i32.npy"))
            pw = chips["total_power_w"].to_numpy(dtype=np.float64)
            htc = ref.bc["htc_w_m2k"].to_numpy(dtype=np.float64)
            ana = ref.t_amb + pw[:, None] / (htc[None, :] * ref.area)
            sd = self.norm["solo_dT_k"]["std"] if normalise else 1.0
            n_chip = cdt.shape[0]
            self.solo_n = np.empty((n_chip, nbc, G, G), dtype=np.float32)
            for c in range(n_chip):
                t = (coff[c][:, None, None]
                     + cdt[c].astype(np.float32)) / 100.0
                self.solo_n[c] = ((t - ana[c][:, None, None]) / sd).astype(np.float32)
        else:
            self.solo_n = None

        # ---- sample index ---------------------------------------------------
        self.pairs = []
        for d in depths:
            o = self.ds[d]
            gsel = groups if groups is not None else o.splits["by_group"][split]
            rows = o.stacks.index[o.stacks["group"].isin(gsel)].to_numpy()
            for r in rows:
                for b in range(len(o.bc)):
                    self.pairs.append((d, int(r), int(b)))

        self.log10_htc_norm = np.array(
            [(float(ref.bc.loc[b, "log10_htc"]) - self.norm["log10_htc"]["mean"])
             / (self.norm["log10_htc"]["std"] + 1e-12) for b in range(nbc)],
            dtype=np.float32)
        self.htc = ref.bc["htc_w_m2k"].to_numpy(dtype=np.float32)

    # -- Dataset ---------------------------------------------------------------

    def __len__(self):
        return len(self.pairs)

    @property
    def in_channels(self):
        return 2 * MAX_DIES + 2 + (MAX_DIES if self.use_solo else 0)

    def __getitem__(self, i):
        d, row, b = self.pairs[i]
        o = self.ds[d]
        G = self.grid
        die_rows = o.die_rows[row]

        x = np.zeros((self.in_channels, G, G), dtype=np.float32)
        for k in range(d):
            x[k] = self.pmap_n[die_rows[k]]
        x[MAX_DIES:2 * MAX_DIES] = self.presence[d]
        x[2 * MAX_DIES] = self.gmap_n[o.ic_row[row]]
        x[2 * MAX_DIES + 1] = self.bc_planes[b]
        if self.use_solo:
            base = 2 * MAX_DIES + 2
            for k in range(d):
                x[base + k] = self.solo_n[die_rows[k], b]

        n = o.n_spatial
        y = np.zeros((MAX_LAYERS, G, G), dtype=np.float32)
        mask = np.zeros((MAX_LAYERS,), dtype=np.float32)
        t = (o.soff[row, b, :n][:, None, None]
             + o.sdt[row, b, :n].astype(np.float32)) / 100.0
        ana = float(o.analytic[row, b])
        y[:n] = (t - ana) if self.target == "residual" else t
        mask[:n] = 1.0

        aux = {
            "depth": np.int64(d),
            "n_valid": np.int64(n),
            "analytic_k": np.float32(ana),
            "htc": np.float32(self.htc[b]),
            "log10_htc_norm": np.float32(self.log10_htc_norm[b]),
            "depth_norm": np.float32(d - 3.0),
            "total_power_w": np.float32(o.stacks.at[row, "total_power_w"]),
            "ic_row": np.int64(o.ic_row[row]),
            "stack_row": np.int64(row),
            "bc_row": np.int64(b),
        }
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(mask), aux)


# --- masked losses and metrics -------------------------------------------------

def masked_l1(pred, true, mask):
    m = mask[:, :, None, None]
    return ((pred - true).abs() * m).sum() / (m.sum() * pred.shape[-1] * pred.shape[-2])


def masked_gradient_loss(pred, true, mask):
    m = mask[:, :, None, None]
    dx = ((pred.diff(dim=-1) - true.diff(dim=-1)).abs() * m).sum()
    dy = ((pred.diff(dim=-2) - true.diff(dim=-2)).abs() * m).sum()
    n = m.sum() * pred.shape[-1] * pred.shape[-2]
    return (dx + dy) / n


def thermal_metrics(pred_k, true_k, hot_frac=0.01):
    """Identical definition to the 2-chiplet loader; call it on valid layers only.

    Report r2_pattern and mae_pattern_k. r2_field sits near 0.98 regardless,
    because the per-sample offset carries ~99% of the variance.
    """
    err = (pred_k - true_k).abs()
    flat = err.flatten(1)

    tm = true_k.mean(dim=(-2, -1), keepdim=True)
    pm = pred_k.mean(dim=(-2, -1), keepdim=True)
    pat_t, pat_p = true_k - tm, pred_k - pm

    n_hot = max(1, int(hot_frac * true_k.shape[-1] * true_k.shape[-2]))
    tf = pat_t.reshape(-1, true_k.shape[-1] * true_k.shape[-2])
    pf = pat_p.reshape(-1, pred_k.shape[-1] * pred_k.shape[-2])
    th = tf.topk(n_hot, dim=1).indices
    ph = pf.topk(n_hot, dim=1).indices
    hit = torch.zeros_like(tf, dtype=torch.bool)
    hit.scatter_(1, th, True)
    overlap = hit.gather(1, ph).float().mean(dim=1).mean()

    pat_var = pat_t.pow(2).mean()
    fld_var = (true_k - true_k.mean()).pow(2).mean()
    return {
        "mae_k": flat.mean().item(),
        "p99_k": flat.quantile(0.99, dim=1).mean().item(),
        "max_k": flat.max().item(),
        "mae_offset_k": (pm - tm).abs().mean().item(),
        "mae_pattern_k": (pat_p - pat_t).abs().mean().item(),
        "r2_pattern": (1 - (pat_p - pat_t).pow(2).mean() / (pat_var + 1e-12)).item(),
        "r2_field": (1 - (pred_k - true_k).pow(2).mean() / (fld_var + 1e-12)).item(),
        "hotspot_top%d%%_overlap" % max(1, int(hot_frac * 100)): overlap.item(),
    }


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else P["data_root"]
    roots = {2: base + "/pact_stack_2L", 3: base + "/pact_stack_3L",
             4: base + "/pact_stack_4L"}
    for arm in ("A1", "A3"):
        ds = MultiDepthStackDataset(roots, split="train", arm=arm)
        x, y, m, aux = ds[0]
        print("%s  n=%-7d x=%-18s y=%-18s mask=%s depth=%d target=%s"
              % (arm, len(ds), tuple(x.shape), tuple(y.shape),
                 m.tolist(), aux["depth"], ds.target))
