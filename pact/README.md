# PACT, and what Solo2Stack changes in it

Solo2Stack's thermal labels are produced by [PACT](https://github.com/peaclab/PACT),
which is third-party code under **GPL-3.0**. PACT is therefore *not* vendored
here. This directory ships only the patch:

```bash
./pact/fetch_pact.sh            # clones upstream at bb7f05a and applies the patch
```

| | |
|---|---|
| Upstream | `https://github.com/peaclab/PACT.git` |
| Commit | `bb7f05a0a893eac4826b1bc55b3e1f2415ba057d` ("Full Workflow", 2026-01-16) |
| Patch | `solo2stack.patch` — 3 files, +107 / −25 lines |
| Upstream licence | GPL-3.0; the clone carries its own `LICENSE` |
| Patch licence | GPL-3.0 — see below |

## Licence

`solo2stack.patch` is a modification of PACT, which is licensed under
**GPL-3.0**. The patch is therefore a derivative of GPL-3.0 code and stays
under PACT's licence — the MIT licence in this repository's root `LICENSE`
covers the rest of the repository but does **not** cover this patch or the PACT
checkout that `fetch_pact.sh` creates. Anyone redistributing the patched PACT
tree is bound by GPL-3.0.

## The solver

**The solver is SciPy's sparse LU (`scipy.sparse.linalg.splu`, i.e. SuperLU.)**
It is the only backend in this repository, and it is the backend that produced
every 3-die and 4-die label in the paper.

Two points of provenance, stated plainly because they matter for anyone trying
to reproduce the dataset exactly:

* The **3-die and 4-die** labels were produced with this SciPy `splu` backend.
* The **solo (single-chiplet) and 2-die** labels were originally produced with
  a different backend, an external **SuperLU_MT** driver, which is not part of
  this repository.
* Re-solving a stored 2-die stack with `splu` reproduces the SuperLU_MT label
  to within the resolution of PACT's own output files: **max 0.01 K, mean
  2.0e-7 K** over 640,000 cells. PACT writes `.grid.steady` with two decimals,
  so 0.01 K is the smallest non-zero difference those files can express; the
  mean corresponds to roughly a dozen cells of 640,000 falling on the other
  side of a rounding boundary.

SuperLU_MT was dropped because it does not scale to the deeper stacks: at three
dies (960,000 unknowns) `pdgssv` aborts with

```
Storage for U columns exceeded; Current column 815436; Need at least 319523463;
Memory allocation failed at line 188 in file pmemory.c
```

for both bonding technologies, where `splu` solves the same system in ~125 s.
With `splu` as the single backend there is no external solver binary, no
`PACT_SUPERLU_WORKDIR` / `PACT_SUPERLU_BIN` environment contract, and no
SuperLU_MT build dependency.

## What the patch changes

### 1. `src/Layer.py` — die dimensions from the floorplan bounding box

Upstream derives the chip length and width from the **last row** of the
floorplan after sorting by `['Y', 'X']`, which assumes that block sits at the
top-right corner:

```python
self.length = self.flp_df.iloc[-1].X + self.flp_df.iloc[-1]['Length (m)']
self.width  = self.flp_df.iloc[-1].Y + self.flp_df.iloc[-1]['Width (m)']
```

That assumption does not hold for arbitrary floorplans. On the 500 chiplets
used here it is wrong for **271 of the first 300**: `chiplet000`, for example,
comes out 0.681 mm wide instead of its true 2 mm, which would rescale every
grid cell and therefore every thermal resistance in the stack.

The patch uses the bounding box instead:

```python
self.length = float((self.flp_df["X"] + self.flp_df["Length (m)"]).max())
self.width  = float((self.flp_df["Y"] + self.flp_df["Width (m)"]).max())
```

It also normalises the origin from the column minima rather than from the first
sorted row, and resets the index after sorting.

### 2. `src/PACT.py` — the singular-matrix guard

Grid cells not covered by any floorplan block keep the zero initialised by
`GridManager`, and PACT substitutes a small resistance so the matrix is not
singular. Upstream substitutes `1e-6`; the patch substitutes `0.0077`
(the silicon thermal resistivity used throughout PACT's own config files):

```python
value[value == 0.0] = 0.0077   # was 1e-6
```

Roughly 0.09 % of cells are affected (11 of 12,800 on a 40x40 probe of a
four-die stack). The effect on temperature is negligible — re-solving a
400x400 four-die stack with `1e-6` instead changes the field by **MAE 0.002 K,
max 0.02 K**, which is at the output files' quantisation — but the conditioning
is not: the relative residual improves from **1.18e-04 to 1.64e-08**.

### 3. `src/SuperLUSolver.py` — SciPy `splu` with a statistics record

Upstream already factorises with `splu`; the patch keeps the same options
(`permc_spec="MMD_AT_PLUS_A"`, `SymmetricMode=True`,
`diag_pivot_thresh=0.001`) and adds a per-solve record — matrix size, nnz,
L/U fill, solve time, relative residual, peak RSS, SciPy version — written as
JSON to the file named by the `PACT3L_STATS` environment variable when it is
set, and summarised on stdout:

```
PACT3L backend=scipy_splu solve=276.8s rel_res=2.70e-07 peak_rss=16.29GB
```

The shipped file drops a block of dead commented-out code and three unused
imports (`sys`, `subprocess`, `pathlib.Path`) that the original carried. The
`SuperLUSolver` class itself is unchanged: its AST is identical to the file
that produced the labels.

## Not in the patch

Upstream's `M3D/config_files/M3D.config` differs in the working tree this was
developed in (an `htc` default flipped between `1e5` and `1e4`). Nothing in the
Solo2Stack pipeline reads that file, so it is deliberately left out.
