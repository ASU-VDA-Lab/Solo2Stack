# Solo2Stack paper numbers — microbump + hybrid campaign

Every number below comes from the MB+HB campaign only. Nothing is carried over
from the three-technology results, and nothing is interpolated or hand-read.

Paths below are written as placeholders — `<data_root>`, `<ckpt_dir>`,
`<transfer_dir>` — each the correspondingly named key in
`configs/paths.yaml`.

**Scope.** Table numbering follows the submitted paper: I ML-model comparison,
II bond geometry, III hyperparameters, IV cross-validation accuracy, V
per-layer accuracy, VI transfer. Table I is a comparison against published
work and has no generator here; Table II's generator is not in this repository
either. Everything from Table III onward is reproduced by the scripts under
`eval/`.

* Dataset: `<data_root>/pact_stack_{2,3,4}L` (400x400), plus `g300`
  and `g500` with a load-time technology filter for the resolution study.
* Models: `<ckpt_dir>/{A0,A1,A2,A3}_fold{0..9}`, plus
  `cv300_mbhb` and `cv500_mbhb` A3 fold 0.
* Transfer: `<transfer_dir>/runs/`.
* Extractors: `eval/extract_tables.py`, `eval/agg_transfer.py`,
  `eval/eval_indist.py`.
* Machine-readable copy of every table: `data/paper_numbers_mbhb.json`.

Metric directions: **Hotspot is `hotspot_top1%_overlap`, a fraction of the
hottest 1% of cells that the prediction gets right — higher is better.** R2_pat
is higher-better. Everything else is an error in K, lower-better.

---

## Table II — Bond geometry

Geometry read from the interconnect floorplans themselves
(`interconnect_flp_grid/*.csv`), conductances from
`mbhb/pact_stack_4L/index/interconnect.csv`.

| Technology | Cu pad array (per 2 mm die) | Pad size | Pitch | Cu area fraction | Bond thickness | Mean G (W/m²K) |
|---|---|---|---|---|---|---|
| Microbump | 40 × 40 (1,600 pads) | 30 µm | 50 µm | 0.36 | 25 µm | 5.660 × 10⁶ |
| Hybrid | 200 × 200 (40,000 pads) | 5 µm | 10 µm | 0.25 | 1.5 µm | 6.579 × 10⁷ |

Spatial spread of the conductance map:

| Technology | G min | G mean | G max | Spatially uniform? |
|---|---|---|---|---|
| Microbump | 5.525 × 10⁴ | 5.660 × 10⁶ | 1.563 × 10⁷ | no |
| Hybrid | 6.579 × 10⁷ | 6.579 × 10⁷ | 6.579 × 10⁷ | **yes** |

Hybrid is a continuous blanket bond, so its conductance map is spatially
constant — `g_min = g_mean = g_max` and the `gmap_log10` channel has std
exactly 0 within hybrid. This is why the hybrid→microbump transfer direction
is not well posed and Table V reports microbump→hybrid instead.

The paper's Table II lists Cu pad size and pitch only, not the array size, so
the two columns it prints are **30 µm / 50 µm** for microbump and
**5 µm / 10 µm** for hybrid. Those are measured from the floorplans. The
array-size column above is supplementary.

## Table III — Hyperparameters and model configuration

| Arm | Solo conditioning | Target | Input channels | Parameters |
|---|---|---|---|---|
| A0 | no | absolute | 10 | 14,593,980 |
| A1 | no | residual | 10 | 14,593,980 |
| A2 | yes | absolute | 14 | 14,595,260 |
| **Solo2Stack (A3)** | yes | residual | **14** | **14,595,260** |

**Confirmed: neither the channel count nor the parameter count changed.**
Bonding technology enters the network as one continuous `gmap_log10`
conductance-map channel, not as a one-hot technology code, so dropping a
technology removes rows from the dataset without touching the input tensor or
the architecture. Output head is 7 (`MAX_LAYERS`), base 32, depth 4,
`scalar_dim` 2 in every arm.

Rounding note: 14,595,260 is **14.60 M** at two decimals, 14.59 M only if
truncated. If the manuscript prints "14.59 M" it is quoting the A0/A1 count
(14,593,980 → 14.59 M) or truncating A3's. Both counts are listed above so the
right one can be chosen.

## Table IV — Accuracy, 10-fold group-wise CV (mean ± sample std, n = 10)

| Metric | A0 | A2 | A1 | **Solo2Stack** |
|---|---|---|---|---|
| Pat. MAE (K) | 0.4590 ± 0.0162 | 0.3808 ± 0.0143 | 0.2719 ± 0.0052 | **0.2048 ± 0.0109** |
| R²_pat | 0.2911 ± 0.0216 | 0.4971 ± 0.0263 | 0.6972 ± 0.0177 | **0.8515 ± 0.0155** |
| Abs. MAE (K) | 0.7427 ± 0.0322 | 0.4652 ± 0.0197 | 0.2920 ± 0.0075 | **0.2629 ± 0.0118** |
| Offset MAE (K) | 0.5641 ± 0.0407 | 0.2426 ± 0.0160 | **0.0907 ± 0.0101** | 0.1340 ± 0.0091 |
| p99 (K) | 2.2170 ± 0.0609 | 1.7529 ± 0.0399 | 1.4406 ± 0.0246 | **0.9368 ± 0.0494** |
| Max (K) | 11.7 ± 0.5 | 10.9 ± 0.5 | 11.1 ± 0.4 | **10.5 ± 0.9** |
| Hotspot | 0.0926 ± 0.0116 | 0.2074 ± 0.0178 | 0.2142 ± 0.0154 | **0.4785 ± 0.0216** |

Solo2Stack is best on six of seven metrics. A1 keeps the lowest Offset MAE, as
in the published table.

**Estimator provenance — these columns are not all the same kind of average.**

* Pat. MAE, R²_pat, Abs. MAE, Offset MAE and Hotspot are `aggregate_cv.py`
  per-stack means (every test stack weighted equally), then averaged over the
  ten folds. This is the estimator Table IV uses.
* **Max is the true global maximum** over every cell of every valid layer of
  every test stack (`eval_global.py`). `aggregate_cv.py`'s `max_k` is a mean of
  per-slice maxima and understates the real maximum by 1.6–1.9×; it is not used
  here. The same evaluator reproduces the published 17.8 ± 0.2 K / 19.1 K
  exactly on the old data, so the definition is verified.
* **p99 uses the published estimator** (a mean of per-slice p99s), and stays
  that way. Decided 2026-11-08: the paper defines pooled linear metrics as
  averages over valid layers and test stacks, and defines only Max as a global
  maximum, so the published estimator is the internally consistent choice and
  Table IV is not switched to a global p99. The global values below are
  recorded for reference only and are **not** used anywhere in the paper:

| Arm | p99, published estimator | p99, true global |
|---|---|---|
| A0 | 2.2170 ± 0.0609 | 3.1877 ± 0.1647 |
| A2 | 1.7529 ± 0.0399 | 2.3178 ± 0.0741 |
| A1 | 1.4406 ± 0.0246 | 1.6674 ± 0.0417 |
| **Solo2Stack** | 0.9368 ± 0.0494 | 1.4440 ± 0.0813 |

  Table IV therefore carries two kinds of statistic by design, not by
  oversight: the linear metrics and p99 are pooled averages over valid layers
  and test stacks, while Max is a global maximum, which is the only way a
  maximum can be defined. The reference column differs from the printed one by
  up to 54%, so the two must not be mixed when quoting.

Full precision Max: A0 11.6840 ± 0.5447, A2 10.9329 ± 0.5062,
A1 11.0816 ± 0.4368, Solo2Stack 10.5023 ± 0.8989.

## Table V — Solo2Stack accuracy by layer, four-die stacks (10-fold mean ± std)

Layers bottom to top; D1 is the bottom die, I34 the top bonding interface,
D4 the top die.

| Layer | Pat. MAE (K) | R²_pat | Abs. MAE (K) | Offset MAE (K) | p99 (K) | Max (K) | Hotspot |
|---|---|---|---|---|---|---|---|
| D1 | 0.2609 ± 0.0194 | 0.6929 ± 0.0454 | 0.4629 ± 0.0275 | 0.3727 ± 0.0299 | 1.1117 ± 0.0586 | 3.7 ± 0.4 | 0.4130 ± 0.0301 |
| I12 | 0.2779 ± 0.0155 | 0.7396 ± 0.0326 | 0.4576 ± 0.0235 | 0.3512 ± 0.0261 | 1.4654 ± 0.0861 | 5.2 ± 0.5 | 0.2205 ± 0.0474 |
| D2 | 0.2704 ± 0.0166 | 0.7450 ± 0.0362 | 0.4465 ± 0.0236 | 0.3498 ± 0.0262 | 1.1722 ± 0.0582 | 4.1 ± 0.5 | 0.3054 ± 0.0450 |
| I23 | 0.3044 ± 0.0215 | 0.8374 ± 0.0249 | 0.4171 ± 0.0221 | 0.2677 ± 0.0168 | 1.5781 ± 0.1508 | 7.8 ± 0.8 | 0.4734 ± 0.0504 |
| D3 | 0.3072 ± 0.0197 | 0.7777 ± 0.0273 | 0.4086 ± 0.0195 | 0.2625 ± 0.0164 | 1.2728 ± 0.0641 | 4.8 ± 0.5 | 0.4793 ± 0.0362 |
| **I34** | 0.3858 ± 0.0204 | 0.8465 ± 0.0185 | 0.3929 ± 0.0199 | 0.0568 ± 0.0034 | 2.0739 ± 0.1848 | **10.5 ± 0.9** | 0.4663 ± 0.0399 |
| D4 | 0.3913 ± 0.0171 | 0.7508 ± 0.0235 | 0.3929 ± 0.0172 | 0.0354 ± 0.0048 | 1.3812 ± 0.0704 | 5.6 ± 0.8 | 0.4566 ± 0.0258 |

Pat. MAE and p99 both grow monotonically from bottom to top; I34 carries the
stack's global maximum (see below). Offset MAE falls sharply with height — the
top layers sit closest to the convective boundary, where the analytic offset is
most accurate. Max is the true global per-layer maximum, same estimator as
Table IV, so the I34 row equals Table IV's Solo2Stack Max exactly.

## Table VI — Transfer to held-out conditions (pattern R², fold 0)

"Solo2Stack (all)" is the in-distribution fold-0 A3 model from the main
campaign scored on exactly the same held-out test subset. It is an upper
reference: the gap to the Solo2Stack column is the cost of holding the
condition out, not a training failure. No leakage — every test subset is drawn
from group_01, which is fold 0's test group for the in-distribution model too.

| Held-out condition | n test | A1 | **Solo2Stack** | Solo2Stack (all) | Fraction retained |
|---|---|---|---|---|---|
| N = 3 dies | 2,000 | 0.2560 | **0.3823** | 0.8424 | 45% |
| h = 25,000 (interpolation) | 1,200 | 0.6903 | **0.8432** | 0.8489 | **99%** |
| h = 100,000 (extrapolation) | 1,200 | 0.6823 | **0.8520** | 0.8628 | **99%** |
| Bonding, microbump → hybrid | 3,000 | −0.3014 | **0.5212** | 0.8527 | 61% |

* Cooling transfer is essentially free in both directions — Solo2Stack retains
  99% of in-distribution pattern R² whether the held-out condition is interior
  or at the edge of the range.
* Bonding transfer is where the arms separate most: A1 is **worse than
  predicting the mean** (−0.3014) while Solo2Stack keeps 61% of its
  in-distribution quality. Per depth, Solo2Stack holds R²_pat 0.6472 / 0.4726 /
  0.4436 at 2 / 3 / 4 dies while A1 is negative at every depth.
* Depth transfer is the weakest: 45% retained, and the absolute error is
  3.31 K against about 0.26 K in distribution.

**Only one direction of bonding transfer is well posed.** A hybrid-only
training set has a spatially constant `gmap_log10` channel (std = 0,
Table II), so the model cannot learn any dependence on bonding conductance
structure. microbump → hybrid is therefore the only usable direction in a
two-technology dataset.

---

## Argmax of the 40 fold-level global maxima

Over all four arms × ten folds, located in the per-(technology, depth, layer)
breakdown from `eval_global.py`:

| Technology | N | Layer | Count |
|---|---|---|---|
| microbump | 4 | I34 | **40 / 40** |

By field: technology — microbump 40/40. Depth — 4 dies 40/40. Layer — I34 40/40.

**Sentence for the paper:** *All 40 fold-level maxima occur in microbump
stacks, every one of them at I34 of a four-die stack.* Unanimous on both
the technology and the layer.

## Interaction: (A3 − A1) − (A2 − A0)

Does solo conditioning help the residual target more than the absolute target?
Paired over the ten folds.

| Quantity | R²_pat | Hotspot |
|---|---|---|
| Solo effect on residual (A3 − A1) | +0.1543 ± 0.0212 | +0.2643 ± 0.0290 |
| Solo effect on absolute (A2 − A0) | +0.2061 ± 0.0207 | +0.1148 ± 0.0127 |
| **Interaction** | **−0.0518 ± 0.0279** | **+0.1494 ± 0.0271** |
| Folds with positive interaction | **0 / 10** | **10 / 10** |
| t-test p (two-sided) | 2.36 × 10⁻⁴ | 3.04 × 10⁻⁸ |
| Wilcoxon signed-rank p | 0.00195 | 0.00195 |

0.00195 is the minimum attainable two-sided p for n = 10, so both Wilcoxon
results are as strong as a ten-fold paired test can be.

**This is a sign change on R²_pat and it is the most consequential finding of
the regeneration.** See the flagged-changes section below.

## Dataset

| Depth | Stacks | Chiplets | Cooling conditions | Cases | Physical layers |
|---|---|---|---|---|---|
| 2 dies | 4,000 | 500 | 5 | 20,000 | 3 |
| 3 dies | 4,000 | 500 | 5 | 20,000 | 5 |
| 4 dies | 4,000 | 500 | 5 | 20,000 | 7 |
| **Total** | **12,000** | 500 | 5 | **60,000** | — |

60,000 = 3 depths × 4,000 stacks × 5 cooling conditions
= 3 N × 2 bonding technologies × 5 cooling × 2,000 chiplet draws.
Per fold: 48,000 train / 6,000 validation / 6,000 test, by group-wise split
(8 training groups, 1 validation, 1 test; every group is the test group once).

## Runtime

Measured single-threaded, one job at a time, SciPy SuperLU, 400×400, mean of
2 reps (`runtime_bench/timing.csv`); no solve was repeated for this campaign.
Solo characterisation 8.685 s per chiplet, batch-1 inference 4.39 ms.

**Full-stack PACT cost (mean over microbump and hybrid)**

| Depth | Full-stack PACT (s) |
|---|---|
| 2 dies | 75.12 |
| 3 dies | 183.41 |
| 4 dies | 338.92 |

**Speedups.** Three conventions, each answering a different question. The
paper's single-stack numbers are the sequential and parallel columns; the sweep
column is the 20,000-stack figure.

| Depth | Sequential solo | Parallel solo | Sweep (amortised) |
|---|---|---|---|
| 2 dies | 4.32× | 8.64× | 68.9× |
| 3 dies | 7.04× | 21.11× | 168.3× |
| **4 dies** | **9.75×** | **39.00×** | **310.9×** |

* Sequential solo = `full_stack_s / (N × solo_s + infer_s)`: one solo solve per
  die, run one after another, nothing shared. The pessimistic reading.
* Parallel solo = `full_stack_s / (solo_s + infer_s)`: the N solo solves run
  concurrently, so the wall time is one solo solve plus one inference. This is
  the convention Fig. 1 plots.
* Sweep = `full_stack_s / (solo_s × chiplets/stacks_per_depth + infer_s)`
  = `full_stack_s / 1.090 s`: 2,500 solo simulations (500 chiplets under 5
  cooling conditions) are shared across all stacks — 4,000 stacks per depth
  at five cooling conditions, i.e. 20,000 cases. At four dies that is
  1,882.9 h of full-stack PACT against 6.03 h of solo solves.
* Dataset-level, building all 60,000 cases: **543.7×**
  (11,948,800 s full-stack against 21,712.5 s of solo solves).

**Break-even**

| Definition | 2 dies | 3 dies | 4 dies |
|---|---|---|---|
| Deployment only (solo solves already done) | 0.23 | 0.14 | **0.10** stacks |
| Including setup (dataset + training) | 161,931 | 65,747 | **35,483** stacks |

The two definitions answer different questions and differ by six orders of
magnitude. Deployment-only is below one stack at every depth — the surrogate
pays for itself before the first stack finishes. Including setup counts the
full 3,319.1 h of dataset generation plus 4.567 h of training per run.
**Pick one definition and say which.**

**Total full-stack simulation time for 60,000 stacks**

| | |
|---|---|
| Full-stack PACT, all 60,000 cases | 11,948,800 s = **3,319.1 h = 138.3 CPU-days** |
| Solo characterisation, 500 chiplets × 5 conditions | 21,712.5 s = **6.03 h** |
| Ratio | **543.7×** |

---

## Figures

PDF + PNG, suffix `_mbhb`, written where `configs/paths.yaml` points. No
existing figure is overwritten; each generator refuses to write over one.

| Paper figure | File | Generator |
|---|---|---|
| Fig. 1 — accuracy vs runtime, four-die stacks | `accuracy_vs_runtime_4die_mbhb.pdf` | `fig1_mbhb.py` |
| Fig. 3 — pattern R² vs number of dies | `depth_scaling_exact_mbhb.pdf` | `fig3_mbhb.py` |
| Fig. 4 — qualitative maps, D4 / I34 / D1 | `qual_jet_metrics_3row_mbhb_rulematch.pdf` | `fig4_mbhb.py` |

Selection record for Fig. 4: `figs/qual_jet_selection_3row_mbhb.json`.

**Fig. 4 — which stack it is.** Applying the published rule verbatim (median
Solo2Stack pattern error among all held-out four-die stacks at h = 100,000),
the pick is

> `group_01_s0193_d1_chiplet251_d2_chiplet440_d3_chiplet408_d4_chiplet055_hybrid`

— **hybrid** bonding, four dies, h = 100,000 W/m²K, sample index 5934 of 400
candidates, test group group_01, fold 0. **The median case is hybrid again, so
the published caption's "hybrid stack" stays correct** and no technology
restriction had to be imposed. Per-panel errors (MAE / Max, K):

| Layer | A0 | A1 | A2 | **Solo2Stack** |
|---|---|---|---|---|
| D4 | 1.17 / 3.34 | 0.61 / 3.20 | 1.06 / 3.77 | **0.41 / 2.68** |
| I34 | 1.23 / 3.81 | 0.62 / 4.74 | 1.20 / 4.39 | **0.43 / 4.00** |
| D1 | 0.65 / 2.32 | 0.31 / 1.67 | 1.40 / 2.71 | **0.18 / 1.04** |

**One style limit had to open, forced by data, not taste:**

* Fig. 3 left y-limit 0.30 → 0.20. A0 at four dies is 0.2685 ± 0.0214, so 0.30
  would clip its error bar off the figure.

Everything else — palette, markers, marker sizes, figure sizes, fonts, grid,
legend placement, output format — is unchanged. Solo2Stack remains the green
star throughout.

### Fig. 1 — plotted values

| Series | Four-die pattern R² (10-fold mean ± std) | Wall time per stack (s) |
|---|---|---|
| A0 | 0.268489 ± 0.021410 | 0.00439 (inference only) |
| A2 | 0.386365 ± 0.027782 | 8.68939 |
| A1 | 0.648611 ± 0.020371 | 0.00439 (inference only) |
| **Solo2Stack (A3)** | **0.799987 ± 0.021691** | **8.68939** |
| Full-stack PACT | 1.0 (reference) | **338.915** |

* 8.68939 s = one solo simulation (8.685 s) + one batch-1 inference (4.39 ms):
  the parallel-solo convention, where the four solo solves run concurrently.
  Solo2Stack is 39.00× faster than full-stack PACT at four dies on it.
* 338.915 s is the mean PACT wall time over four-die stacks only, averaged over
  microbump and hybrid (241.985 / 435.845 s).
* **No arrow, connector or shared-cost marker appears on this figure, and no
  sweep-level cost is claimed on it.** The sweep-level amortised figure
  (1.09 s per stack, 310.9×) is a different quantity and appears only in the
  runtime section above.
* y-limit 0.20: A0 at four dies is 0.2685 ± 0.0214, reaching down to 0.2471,
  so a limit of 0.28 or higher would clip it — the same change Fig. 3 needed
  and for the same reason.

The published Fig. 1 generator is not in this repository (the copy sent on
2026-11-08 arrived with an empty `[paste tradeoff script]` placeholder), so
`fig1_mbhb.py` builds it in the Fig. 3 visual language, with PACT taking the
vermillion circle — the one marker Fig. 3 does not use. The data values,
cost convention and axis treatment above are as specified; only the styling is
unverified against the original.

Solo2Stack runtimes are the measured 2026-10-23 values, carried over unchanged:
solo characterisation solves one chiplet with no bonding layer, so its cost does
not depend on which bonding technologies the dataset contains, and inference is
the same architecture at the same parameter count. A re-measurement taken
alongside the new inference (4.57 / 7.26 / 6.23 ms batch-1) was discarded as
contended — it is non-monotonic in resolution and GPU 3 was carrying another
user's 45 GB job throughout.

---
