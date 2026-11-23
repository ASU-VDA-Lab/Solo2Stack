# Solo2Stack

Solo2Stack is a thermal surrogate for heterogeneous chiplet stacks. Thermal
ground truth comes from PACT, a compact thermal simulator that solves a
thermal RC network of the stack. Solving a full 3D stack that way is
expensive, but a *single-chiplet* ("solo") solve of each die is cheap and can
be reused across every stack that die appears in. Solo2Stack feeds those solo temperature fields into a
U-Net, together with the per-die power maps, a bonding-conductance map and the
cooling condition, and predicts the steady-state temperature field of every
layer of the assembled stack. Two design choices carry the result: the solo
fields as input, and an *analytic residual* target — the network predicts the
departure from a closed-form uniform-spreading estimate rather than absolute
temperature.

The campaign is 12,000 stack configurations — 2, 3 and 4 dies, microbump and
hybrid bonding — under 5 cooling conditions, which the paper counts as 60,000
stacks, with 10-fold cross-validation over disjoint chiplet groups. On unseen
chiplets Solo2Stack reaches a pattern R² of **0.852 ± 0.016**, against
**0.291** without the solo input and the residual target (A0) and **0.697**
without the solo input (A1).

Against a full-stack PACT solve of one unseen four-die stack, Solo2Stack is
**9.8x** faster when the solo simulations run sequentially and **39.0x** faster
when they run in parallel, and **311x** faster over a sweep of 20,000 four-die
stacks, where 2,500 solo simulations (500 chiplets under 5 cooling conditions)
are shared across all stacks. These are measured runtimes, not a cost estimate.

---

## 1. Environments

Three, because the pieces have incompatible requirements. None of them needs
PyYAML — `configs/paths.yaml` is read by `solo2stack/miniyaml.py`.

| Environment | Python | For | Install |
|---|---|---|---|
| PACT | 3.6 | running the thermal simulator | `requirements-pact.txt` |
| Model | 3.10 | training, evaluation, figures | `requirements.txt` |
| FloorSet | >= 3.8 | building the chiplet library only | `requirements-floorset.txt` |

```bash
python3.6  -m venv ~/venvs/s2s_pact     && ~/venvs/s2s_pact/bin/pip install -r requirements-pact.txt
python3.10 -m venv ~/venvs/s2s_model    && ~/venvs/s2s_model/bin/pip install -r requirements.txt
python3.11 -m venv ~/venvs/s2s_floorset && ~/venvs/s2s_floorset/bin/pip install -r requirements-floorset.txt
```

## 2. Configure paths

Everything inside the repository is addressed relative to the repo root.
Everything outside it — datasets, raw simulator output, checkpoints, result
trees, interpreters — comes from `configs/paths.yaml`, which is per-machine
and not committed.

```bash
cp configs/paths.example.yaml configs/paths.yaml
$EDITOR configs/paths.yaml
```

Python reads it through `solo2stack.paths`:

```python
from solo2stack.paths import P, repo_path
root = P["data_root"]
cfg  = repo_path("configs", "pact", "bc_4.config")
```

and shell scripts through `solo2stack/env.sh`, which exports each key in upper
case (`data_root` -> `$DATA_ROOT`).

## 3. Get PACT

PACT is third-party (GPL-3.0) and is not vendored here; only the Solo2Stack
patch is. See `pact/README.md` for what the patch changes and why.

```bash
./pact/fetch_pact.sh          # clones upstream at bb7f05a, applies the patch
```

## 4. Pipeline, in order

Each step writes where `configs/paths.yaml` says. Sizes are for the published
campaign. The simulation steps are the expensive ones — generating labels for
all 60,000 stacks takes about 3,300 CPU-hours — so in practice you run them
once and keep the packed arrays. Training one model takes about 5 GPU-hours,
47 for all ten folds.

### 4.1 Download FloorSet and build the chiplet library

FloorSet is not redistributed here. `data_gen/floorset/README.md` has the
download link, the licence and the citation.

```bash
~/venvs/s2s_floorset/bin/python data_gen/floorset/build_chiplet_library.py
```

500 chiplets: the first 500 FloorSet-Lite training floorplans in dataset order,
each scaled independently in x and y to a 2 mm x 2 mm die, each block given
power proportional to its area times an N(1, 0.3²) factor clipped at 0.05 and
renormalised so every chiplet dissipates exactly 4 W (seed 0 + chiplet index),
rasterised onto a 400 x 400 grid.

### 4.2 Interconnect floorplans

```bash
~/venvs/s2s_model/bin/python data_gen/interconnect/generate_interconnect_flp.py
```

`microbump_40x40_30um_flp.csv` is committed (512 KB). The hybrid-bond
floorplan is 12 MB and is **regenerated** rather than committed — it is a
200 x 200 array of 5 um Cu pads on a 10 um pitch.

### 4.3 Solo PACT runs

Every chiplet on its own, at all five cooling conditions. One solo solve per
chiplet is reused by every stack that chiplet appears in, and it is the extra
input that distinguishes the A2 and A3 arms.

```bash
bash data_gen/run/run_solo_pact.sh
```

### 4.4 Build the stack datasets

One command per depth, or all three at once. It samples the stacks, writes
their LCF files, runs the PACT sweep and packs the result into arrays; each
stage is skipped when its output already exists, so an interrupted build is
resumed by rerunning the same command.

```bash
~/venvs/s2s_pact/bin/python data_gen/build_dataset.py --depth 2 3 4 --dry-run
~/venvs/s2s_pact/bin/python data_gen/build_dataset.py --depth 2 3 4
```

The three stages are also runnable on their own, all taking `--n-die N`:

```bash
~/venvs/s2s_model/bin/python data_gen/stacks/generate_stack_lcf.py --n-die 3
~/venvs/s2s_pact/bin/python  data_gen/run/run_stack.py --n-die 3 --jobs 24
~/venvs/s2s_model/bin/python data_gen/pack/pack_dataset.py --n-die 3 --out <dir>
```

2,000 stacks per depth per bonding technology, grouped so a chiplet group held
out for test is held out at every depth simultaneously. A stack of N dies has
2N-1 physical layers, D1 I12 D2 ... DN, and every bond layer in a stack uses
the same technology. 400 x 400 uniform grid, ambient 318.15 K, h in
{5, 10, 25, 50, 100} x 10³ W/m²K. The solver is SciPy's sparse LU at every
depth, including N=2 — see `pact/README.md`.

**Which stacks.** The die tuples come from the committed lists in
`data_gen/stacks/lists/stacks_{2,3,4}die.json`, which are the record of the
published campaign. `--resample` redraws them with the original per-depth seed
(20260908 for 3 dies, 20260910 for 4, with rejection sampling at 4 dies), and
reproduces those lists exactly. There is no `--resample` for 2 dies: that draw
was made with an unseeded `random.shuffle()`, so the committed list is its only
record.

### 4.5 Train, 10-fold cross-validation

```bash
~/venvs/s2s_model/bin/python model/make_cv_folds.py
~/venvs/s2s_model/bin/python model/verify_cv_setup.py
~/venvs/s2s_model/bin/python model/run_cv_all.py --arms A0 A1 A2 A3
~/venvs/s2s_model/bin/python model/aggregate_cv.py
```

Four arms: `A0` no solo + absolute target, `A1` no solo + residual target,
`A2` solo + absolute target, `A3` = **Solo2Stack**, solo + residual target.
Ten folds each; test group *k*, validation group *k+1*, the other eight train.

### 4.6 Evaluate

```bash
~/venvs/s2s_model/bin/python eval/extract_tables.py                   # Tables III-V
```

**Max-error evaluation.** `train_cv.score()` batch-weight-averages its metrics,
so its per-fold maximum is a mean of per-slice maxima, not a global maximum.
Table IV's Max column comes from here instead:

```bash
~/venvs/s2s_model/bin/python eval/eval_global.py --arms A0 A1 A2 A3   # global max / p99
```

**Statistical tests** — paired significance, depth scaling, the solo-only
baseline, peak ranking and the worst-cell analysis:

```bash
~/venvs/s2s_model/bin/python eval/final_stats/task12_sig_depth.py     # Wilcoxon + depth scaling
~/venvs/s2s_model/bin/python eval/final_stats/task1_max_fix.py        # max error, global definition
~/venvs/s2s_model/bin/python eval/final_stats/task3_solo_baseline.py  # zero-parameter baseline
bash eval/final_stats/run_pass.sh                                     # per-arm inference pass
~/venvs/s2s_model/bin/python eval/final_stats/task34_aggregate.py
~/venvs/s2s_model/bin/python eval/final_stats/task5_geom_fig.py
```

**Transfer experiment** — held-out depth, cooling condition and bonding
technology (Table VI):

```bash
~/venvs/s2s_model/bin/python eval/transfer/preflight.py
bash eval/transfer/orchestrate.sh                                     # the transfer runs
~/venvs/s2s_model/bin/python eval/eval_indist.py                      # in-distribution column
~/venvs/s2s_model/bin/python eval/agg_transfer.py
```

**Runtime comparison, model against PACT at 400 x 400:**

```bash
bash eval/runtime_bench/build_lcf.sh                                  # benchmark LCFs
bash eval/runtime_bench/run_timing.sh                                 # PACT wall time, 2/3/4 dies
~/venvs/s2s_model/bin/python eval/bench_infer_cv.py                   # surrogate latency
~/venvs/s2s_model/bin/python eval/runtime_section.py                  # speedups, break-even
~/venvs/s2s_model/bin/python eval/rt_items.py
```

### 4.7 Figures

```bash
~/venvs/s2s_model/bin/python figures/fig_accuracy_vs_runtime.py
~/venvs/s2s_model/bin/python figures/fig_depth_scaling.py
~/venvs/s2s_model/bin/python figures/fig_qualitative.py
```

## 5. Paper item -> script

| Paper item | Produced by |
|---|---|
| Table I — comparison against published ML models | not generated here |
| Table II — bond geometry | not generated here |
| Table III — hyperparameters and model configuration | `eval/extract_tables.py` |
| Table IV — cross-validation accuracy, four arms | `eval/extract_tables.py`; the Max column from `eval/eval_global.py` |
| Table V — Solo2Stack per-layer accuracy, four-die | `eval/extract_tables.py` |
| Table VI — transfer to unseen depth / cooling / bonding | `eval/transfer/train_transfer.py`, `eval/eval_indist.py`, `eval/agg_transfer.py` |
| Fig 1 — accuracy vs runtime | `figures/fig_accuracy_vs_runtime.py` |
| Fig 3 — pattern R² vs number of dies | `figures/fig_depth_scaling.py` |
| Fig 4 — qualitative median-error case | `figures/fig_qualitative.py` |
| Runtime section | `eval/runtime_section.py`, `eval/rt_items.py`, `eval/bench_infer_cv.py`, `eval/runtime_bench/*` |
| Paired significance, depth scaling, solo-only baseline, peak ranking, worst-cell analysis | `eval/final_stats/*` |

`results/` holds the committed JSON/CSV these scripts produce, so every figure
can be regenerated without re-running the campaign. `results/PAPER_NUMBERS.md`
and `results/final_stats/SUMMARY.md` are the written-out numbers.

## 6. Repository layout

```
pact/            patch against upstream PACT + fetch script (PACT is not vendored)
data_gen/
  floorset/      FloorSet -> the 500-chiplet library
  interconnect/  microbump and hybrid bond floorplans
  stacks/        depth-generic stack sampling + the committed die lists
  run/           PACT drivers (solo and N-die) + the grid reader
  pack/          PACT output -> .npy arrays
  qa/            structural validation of the sampled stacks
  build_dataset.py   the three stages above, per depth, resumable
model/           loader, U-Net + FiLM, training, 10-fold cross-validation
eval/
  eval_global.py     max error on the global definition
  extract_tables.py  Tables III-V
  final_stats/       paired significance tests, solo-only baseline, peaks
  transfer/          held-out depth / cooling / bonding (Table VI)
  runtime_bench/     PACT wall-time benchmark at 400 x 400
figures/         one script per paper figure (Figs 1, 3, 4)
configs/         paths.example.yaml + the PACT .config files actually used
results/         committed numbers and figure inputs behind the paper
solo2stack/      path resolution shared by every script
```

## 7. License

The code in this repository is licensed under the BSD 3-Clause License;
see `LICENSE`.

Two dependencies are not covered by it and are not included here, as `NOTICE`
records. **PACT** is GPL-3.0; `pact/fetch_pact.sh` clones it from upstream and
the patch in `pact/solo2stack.patch` stays under PACT's licence (see
`pact/README.md`). **FloorSet** has its own licence and citation requirement,
documented in `data_gen/floorset/README.md`.
