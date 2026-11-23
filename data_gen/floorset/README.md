# Chiplet library from FloorSet-Lite

`build_chiplet_library.py` turns FloorSet-Lite floorplans into the 500-chiplet
library every Solo2Stack experiment is built from. **FloorSet itself is not
redistributed here** — neither the dataset nor its `iccad2026contest` package.
Download it, then point `floorset_root` in `configs/paths.yaml` at the clone.

## Get FloorSet

FloorSet is Intel's open floorplanning benchmark suite, released for the
ICCAD 2026 contest:

* Repository: <https://github.com/IntelLabs/FloorSet>
* Paper: Uttam Dhakal, Rachel Selina Rajarathnam, et al., *"FloorSet — a VLSI
  Floorplanning Dataset with Design Constraints of Real-World SoCs"*.

Follow the upstream instructions to clone the repository and fetch the
`LiteTensorDataTest` tensors. Respect the licence that ships with it
(`LICENSE` in the FloorSet repository) and cite the paper if you use the data.

```bash
git clone https://github.com/IntelLabs/FloorSet.git ~/FloorSet
# then download the Lite dataset tensors as the FloorSet README describes
```

Set in `configs/paths.yaml`:

```yaml
floorset_root: /path/to/FloorSet
chiplet_library: /where/the/library/should/be/written
```

## Build the library

The FloorSet dataloader is a torch dataset, so this step needs torch (CPU is
enough) and therefore Python >= 3.8 — it cannot run in the Python 3.6
environment PACT uses. See `requirements-floorset.txt`.

```bash
~/venvs/s2s_floorset/bin/python data_gen/floorset/build_chiplet_library.py
```

Defaults reproduce the published library exactly:

| Option | Default | Meaning |
|---|---|---|
| `--num-samples` | 500 | first 500 floorplans of the FloorSet-Lite training set, in dataset order, unfiltered |
| `--target-mm` | 2.0 | each floorplan is scaled independently in x and y to a 2 mm x 2 mm die |
| `--power-mode` | `area_weighted_gaussian` | block power proportional to block area, times an independent factor drawn from N(1, 0.3^2) |
| `--cv` | 0.3 | the 0.3 above |
| `--total-power-w` | 4.0 | factors are clipped at 0.05 and renormalised so every chiplet dissipates exactly 4 W |
| `--seed` | 0 | chiplet *i* uses seed `0 + i`, so each chiplet is reproducible on its own |
| `--grid-size` | 400 | rasterised onto a 400 x 400 grid (5 um cells) |
| `--num-groups`, `--group-size` | 10, 50 | the ten disjoint chiplet groups used for cross-validation |

Outputs, under `chiplet_library`:

```
raw_floorplans/chipletNNN_flp.csv     PACT floorplan (X, Y, Length, Width, Label)
raw_ptraces/chipletNNN_ptrace.csv     per-block power
raw_lcf/chipletNNN_lcf.csv            PACT layer config
powermap_npy/chipletNNN_powermap.npy  400 x 400 power density
meta/chipletNNN.json                  scaling and power-assignment record
groups/…                              the ten chiplet groups
chiplet_manifest.json                 index over all of the above
```

## Reproducibility check

Chiplets 0, 249 and 499 regenerated from this script reproduce the stored
library **exactly**, on a clean Python 3.11 environment with torch 2.15.0+cpu,
numpy 2.4.6 and pandas 3.1.1 (the library was originally built with much older
versions):

| Chiplet | Blocks | Geometry after scaling | Per-block power | Total power | Power map |
|---|---|---|---|---|---|
| `chiplet000` | 37 | max delta 0.0 | max delta 0.0 | 4.000000 W | bitwise equal |
| `chiplet249` | 102 | max delta 0.0 | max delta 0.0 | 4.000000 W | bitwise equal |
| `chiplet499` | 67 | max delta 0.0 | max delta 0.0 | 4.000000 W | bitwise equal |

Block names and labels match row for row as well. The per-chiplet seed
(`seed + index`) makes each chiplet reproducible independently of the others.
