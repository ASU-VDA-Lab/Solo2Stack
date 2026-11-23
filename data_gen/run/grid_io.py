#!/usr/bin/env python3
"""Reader and validation helpers for PACT grid.steady output, any stack depth.

Replaces grid_io_3L.py and grid_io_4L.py.

PACT writes temperatures with np.savetxt(fmt="%.2f"), which is NOT fixed width:
a cell is "<int>.<2dp>\\n", so 929.94 takes 7 bytes but 1146.46 takes 8.  The
original 2- and 3-die readers hard-coded 7 bytes, which was safe there only
because those stacks stay below 1000 K.  A 4-die stack at h = 5,000 reaches
T_ana = 318.15 + 16/(5000 * 4e-6) = 1118.15 K, so its files are 8 bytes a cell.
This reader detects the width instead of assuming it, which makes it correct at
every depth; gate G1b checks it against both original readers on stored files.
"""
import numpy as np

NR = NC = 400
T_AMB = 318.15
AREA = 4e-6              # 2 mm x 2 mm die
DIE_POWER_W = 4.0        # every chiplet in the library is normalised to 4 W


def n_layers(n_die):
    """Physical layer count: D1 I12 D2 ... DN."""
    return 2 * n_die - 1


def layer_names(n_die):
    out = []
    for i in range(n_die):
        out.append("D%d" % (i + 1))
        if i != n_die - 1:
            out.append("I%d%d" % (i + 1, i + 2))
    return out


def read_grid_ck(path, nr=NR, nc=NC):
    """Read one PACT grid.steady layer into float64 (nr, nc), any cell width."""
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    ncell = nr * nc
    if raw.size % ncell:
        raise ValueError("%s: %d bytes is not a multiple of %d cells"
                         % (path, raw.size, ncell))
    w = raw.size // ncell
    if w < 7:
        raise ValueError("%s: implausible cell width %d" % (path, w))

    cells = raw.reshape(ncell, w)
    # layout: <int digits> '.' d d '\n'  -> the integer part is w-4 bytes
    ndig = w - 4
    if not (np.all(cells[:, ndig] == ord(".")) and
            np.all(cells[:, -1] == ord("\n"))):
        raise ValueError("%s: unexpected layout at width %d" % (path, w))

    digits = cells[:, :ndig].astype(np.int64) - 48
    whole = np.zeros(ncell, dtype=np.int64)
    for k in range(ndig):
        whole = whole * 10 + digits[:, k]
    frac = ((cells[:, -3].astype(np.int64) - 48) * 10 +
            (cells[:, -2].astype(np.int64) - 48))
    return (whole + frac / 100.0).reshape(nr, nc)


def t_analytic(htc, n_die, t_amb=T_AMB, area=AREA, die_power=DIE_POWER_W):
    """T_ana = T_amb + P_total / (h * A), with P_total = n_die * 4 W."""
    return t_amb + (die_power * n_die) / (htc * area)


def validate_stack(prefix, htc, n_die):
    """Validate one completed solve of an n_die stack. Returns (ok, info)."""
    n_spatial = n_layers(n_die)
    info = {"prefix": prefix, "htc": htc, "n_die": n_die}
    layers = []
    for i in range(n_spatial):
        try:
            layers.append(read_grid_ck("%s.layer%d" % (prefix, i)))
        except Exception as e:                            # noqa: BLE001
            info["error"] = "layer%d: %s" % (i, e)
            return False, info

    stack = np.stack(layers)
    info["shape"] = list(stack.shape)
    info["all_finite"] = bool(np.all(np.isfinite(stack)))
    info["n_nan"] = int(np.isnan(stack).sum())
    info["n_inf"] = int(np.isinf(stack).sum())
    if not info["all_finite"]:
        info["error"] = "non-finite values present"
        return False, info
    if stack.shape != (n_spatial, NR, NC):
        info["error"] = "bad shape %s" % (stack.shape,)
        return False, info

    tana = t_analytic(htc, n_die)
    info["T_ana"] = tana
    info["stack_min"] = float(stack.min())
    info["stack_max"] = float(stack.max())
    info["min_minus_Tana"] = float(stack.min() - tana)
    dies = [layers[2 * k].mean() for k in range(n_die)]
    info["monotonic_dies"] = bool(
        all(dies[k] > dies[k + 1] for k in range(n_die - 1)))
    return True, info
