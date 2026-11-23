"""Shared readers for PACT grid outputs and interconnect floorplans.

PACT writes its steady-state grids as ASCII, one cell per line: ``<int>.<2dp>``.
That is 7 bytes below 1000 K and 8 bytes above.  Either way it lets us parse with integer arithmetic instead of
``np.loadtxt`` (~12x faster), and -- more usefully -- it means the data is
quantised to 0.01 K, so an int16 centi-kelvin deviation from a per-field
integer offset stores the grid *losslessly* in 2 bytes per cell.
"""

import numpy as np
import pandas as pd

GRID = 400
NCELL = GRID * GRID
RECLEN = 7  # "DDD.DD\n"

# Thermal conductivity (W/m-K) = 1 / thermalresistivity, from Config_files/bc_*.config
K_MAT = {
    "Si": 1.0 / 0.0077,
    "Diel": 1.0 / 0.724,
    "Cu": 1.0 / 2.56e-3,
    "Air": 1.0 / 50.0,
}

DIE_SIZE_M = 0.002  # 2 mm x 2 mm, from meta/chiplet*.json target_size_m


def read_grid_ck(path):
    """Read a PACT grid file as int32 centi-kelvin, shape (NCELL,).

    Cells are ``<int>.<2dp>\\n``.  Width is 7 bytes below 1000 K (2- and
    3-chiplet stacks) and 8 bytes above it (a 4-die stack at h=5,000 sits near
    1118 K), so the width is detected rather than assumed.  Falls back to
    ``np.loadtxt`` for anything that does not match that layout.
    """
    with open(path, "rb") as fh:
        buf = fh.read()
    n = len(buf)
    if n and n % NCELL == 0:
        w = n // NCELL
        if RECLEN <= w <= 12:
            a = np.frombuffer(buf, dtype=np.uint8).reshape(NCELL, w)
            ndig = w - 4  # "<ndig digits>" "." "d" "d" "\n"
            if (a[:, ndig] == 46).all() and (a[:, w - 1] == 10).all():
                cols = list(range(ndig)) + [w - 3, w - 2]
                d = a[:, cols].astype(np.int32) - 48
                if d.min() >= 0 and d.max() <= 9:
                    whole = np.zeros(NCELL, dtype=np.int32)
                    for k in range(ndig):
                        whole = whole * 10 + d[:, k]
                    return whole * 100 + d[:, ndig] * 10 + d[:, ndig + 1]
    return np.rint(np.loadtxt(path, dtype=np.float64) * 100).astype(np.int32)


def read_grid_k(path):
    """Read a PACT grid file as float64 kelvin, shape (GRID, GRID)."""
    return (read_grid_ck(path).astype(np.float64) / 100.0).reshape(GRID, GRID)


def rasterize_flp_conductance(csv_path, thickness_m, grid=GRID, extent_m=DIE_SIZE_M):
    """Rasterize a PACT floorplan CSV to a per-area vertical conductance map.

    Returns ``(grid, grid)`` float64 in W/(m^2-K): the area-weighted mean of
    ``k_material / thickness`` over each output cell.  Area weighting is exact --
    rectangle edges are snapped to a supersampling lattice whose pitch divides
    every coordinate and extent in the file.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    vals = np.concatenate([
        df["X"].to_numpy(), df["Y"].to_numpy(),
        (df["X"] + df["Length (m)"]).to_numpy(),
        (df["Y"] + df["Width (m)"]).to_numpy(),
    ])
    # Find a supersample factor per output cell such that every edge lands on a
    # lattice point.  Search small factors; these floorplans are regular arrays.
    cell = extent_m / grid
    for sub in (1, 2, 4, 5, 10, 20, 25, 40, 50, 100):
        idx = vals / (cell / sub)
        if np.abs(idx - np.rint(idx)).max() < 1e-6:
            break
    else:
        raise ValueError("no supersample factor divides %s" % csv_path)

    n = grid * sub
    fine = np.zeros((n, n), dtype=np.float64)
    x0 = np.rint(df["X"].to_numpy() / (cell / sub)).astype(np.int64)
    y0 = np.rint(df["Y"].to_numpy() / (cell / sub)).astype(np.int64)
    x1 = np.rint((df["X"] + df["Length (m)"]).to_numpy() / (cell / sub)).astype(np.int64)
    y1 = np.rint((df["Y"] + df["Width (m)"]).to_numpy() / (cell / sub)).astype(np.int64)
    kv = np.array([K_MAT[str(s).strip()] for s in df["Label"]], dtype=np.float64)

    for i in range(len(df)):
        fine[y0[i]:y1[i], x0[i]:x1[i]] = kv[i]

    if (fine == 0).any():
        raise ValueError("floorplan %s leaves %d fine cells uncovered"
                         % (csv_path, int((fine == 0).sum())))

    coarse = fine.reshape(grid, sub, grid, sub).mean(axis=(1, 3))
    return coarse / thickness_m, sub


def cu_area_fraction(csv_path, grid=GRID, extent_m=DIE_SIZE_M):
    """Fraction of the layer area occupied by Cu, by exact area."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    area = df["Length (m)"] * df["Width (m)"]
    is_cu = df["Label"].astype(str).str.strip() == "Cu"
    return float(area[is_cu].sum() / area.sum())
