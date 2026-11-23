# Solo2Stack final campaign (microbump + hybrid) — Tasks 1–5

All numbers below come from the final MB+HB campaign only. Nothing was
retrained and PACT was not run. Source artifacts, with each path written as
the correspondingly named key in `configs/paths.yaml`:

| What | Path |
|---|---|
| Per-fold metrics | `<ckpt_dir>/{A0,A1,A2,A3}_fold{0..9}/result.json` |
| Checkpoints | same dirs, `best.pt` |
| Fold spec / norms | `<cv_dir>/cv_folds.json`, `norm_cv/fold{k}.json` |
| Global-max metrics | `<eval_global>/C_new_mbhb.json` |
| PACT labels | `<data_root>/pact_stack_{2,3,4}L/arrays/stack_dT_ck.i16.npy` |
| Solo maps | `…/arrays/solo_dT_ck.i16.npy` (500, 5, 400, 400) |

Provenance checks that passed: all 40 runs have `base_dir=<data_root>`,
test group = `group_{k+1}`, 48000/6000/6000 samples; `index/interconnect.csv`
has exactly two rows (microbump, hybrid); 4000 stacks per depth = 2000 + 2000.
The campaign is 60,000 samples = 12,000 stacks x 5 h.

## Task 1 — paired Wilcoxon (n = 10 folds, two-sided)

Max error uses the paper definition (`global.max_global_k`, the largest
|pred − true| over every cell of every valid layer of every test stack).
The other six metrics use the pooled per-stack estimator in `result.json`,
which is the paper's definition for linear metrics.

| Effect | Pair | Metric | mean diff | p | Dir. | Favours |
|---|---|---|---|---|---|---|
| residual | A0→A1 | Pattern MAE (K) | −0.1871 | 0.001953 | 10/10 | A1 |
| | | Pattern R² | +0.4062 | 0.001953 | 10/10 | A1 |
| | | Absolute MAE (K) | −0.4507 | 0.001953 | 10/10 | A1 |
| | | Offset MAE (K) | −0.4734 | 0.001953 | 10/10 | A1 |
| | | p99 error (K) | −0.7763 | 0.001953 | 10/10 | A1 |
| | | Max error (K) | −0.6024 | 0.013672 | 8/10 | A1 |
| | | Hotspot overlap | +0.1216 | 0.001953 | 10/10 | A1 |
| residual | A2→Solo2Stack | Pattern MAE (K) | −0.1759 | 0.001953 | 10/10 | Solo2Stack |
| | | Pattern R² | +0.3544 | 0.001953 | 10/10 | Solo2Stack |
| | | Absolute MAE (K) | −0.2023 | 0.001953 | 10/10 | Solo2Stack |
| | | Offset MAE (K) | −0.1086 | 0.001953 | 10/10 | Solo2Stack |
| | | p99 error (K) | −0.8161 | 0.001953 | 10/10 | Solo2Stack |
| | | Max error (K) | −0.4307 | 0.048828 | 7/10 | Solo2Stack |
| | | Hotspot overlap | +0.2711 | 0.001953 | 10/10 | Solo2Stack |
| solo | A0→A2 | Pattern MAE (K) | −0.0782 | 0.001953 | 10/10 | A2 |
| | | Pattern R² | +0.2061 | 0.001953 | 10/10 | A2 |
| | | Absolute MAE (K) | −0.2775 | 0.001953 | 10/10 | A2 |
| | | Offset MAE (K) | −0.3216 | 0.001953 | 10/10 | A2 |
| | | p99 error (K) | −0.4640 | 0.001953 | 10/10 | A2 |
| | | Max error (K) | −0.7510 | 0.009766 | 9/10 | A2 |
| | | Hotspot overlap | +0.1148 | 0.001953 | 10/10 | A2 |
| solo | A1→Solo2Stack | Pattern MAE (K) | −0.0670 | 0.001953 | 10/10 | Solo2Stack |
| | | Pattern R² | +0.1543 | 0.001953 | 10/10 | Solo2Stack |
| | | Absolute MAE (K) | −0.0291 | 0.003906 | 9/10 | Solo2Stack |
| | | Offset MAE (K) | +0.0433 | 0.001953 | 10/10 | **A1** |
| | | p99 error (K) | −0.5038 | 0.001953 | 10/10 | Solo2Stack |
| | | Max error (K) | −0.5793 | **0.064453** | 7/10 | not significant |
| | | Hotspot overlap | +0.2643 | 0.001953 | 10/10 | Solo2Stack |

Notes.
* A0 vs A2 absolute/offset MAE: p = 0.001953, 10/10. The earlier campaign's
  p = 0.084 is gone.
* p = 0.001953125 is the smallest attainable two-sided p at n = 10; 24 of 28
  rows sit on that floor. Quote as "p ≤ 0.002 (minimum attainable at n = 10)".
* A1 → Solo2Stack on max error is NOT significant (p = 0.0645).
* Offset MAE is the one metric where the solo input hurts (A1 0.091 K,
  Solo2Stack 0.134 K, worse in 10/10 folds).

## Task 2 — pattern R² vs number of dies

| Arm | N = 2 | N = 3 | N = 4 | Drop N2→N4 | Folds | p |
|---|---|---|---|---|---|---|
| A0 | 0.3207 ± 0.0350 | 0.2840 ± 0.0200 | 0.2685 ± 0.0214 | 0.0522 ± 0.0302 | 10/10 | 0.001953 |
| A1 | 0.7504 ± 0.0180 | 0.6927 ± 0.0186 | 0.6486 ± 0.0204 | 0.1018 ± 0.0143 | 10/10 | 0.001953 |
| A2 | 0.6396 ± 0.0304 | 0.4655 ± 0.0317 | 0.3864 ± 0.0278 | 0.2532 ± 0.0271 | 10/10 | 0.001953 |
| Solo2Stack | 0.9098 ± 0.0097 | 0.8448 ± 0.0171 | 0.8000 ± 0.0217 | 0.1098 ± 0.0149 | 10/10 | 0.001953 |

## Task 3 — solo-only baseline, die layers only

Die-slot → solo-map matching: depth 2 `bot_row`/`top_row`, depth 3
`d1_row`…`d3_row`, depth 4 `d1_row`…`d4_row`; die k sits at stack layer 2k;
its solo field is `solo_dT_ck[die_row[k], bc]`, same h. Prediction =
mean-removed solo field, unscaled and unfitted.

| Method | Scope | Pattern R² | Pattern MAE (K) | Hotspot top-1% |
|---|---|---|---|---|
| Solo-only | overall | 0.1914 ± 0.0384 | 0.4594 ± 0.0167 | 0.1878 ± 0.0200 |
| Solo-only | N=2 | 0.4605 ± 0.0480 | 0.2596 ± 0.0106 | 0.2592 ± 0.0255 |
| Solo-only | N=3 | 0.2303 ± 0.0427 | 0.4251 ± 0.0174 | 0.1850 ± 0.0171 |
| Solo-only | N=4 | 0.1342 ± 0.0371 | 0.5850 ± 0.0218 | 0.1541 ± 0.0230 |
| Solo-only | microbump | 0.2252 ± 0.0441 | 0.4333 ± 0.0179 | 0.1985 ± 0.0205 |
| Solo-only | hybrid | 0.1620 ± 0.0338 | 0.4855 ± 0.0162 | 0.1771 ± 0.0201 |
| A1 | overall | 0.6828 ± 0.0236 | 0.2797 ± 0.0062 | 0.2737 ± 0.0189 |
| A1 | N=2 | 0.7900 ± 0.0199 | 0.1563 ± 0.0038 | 0.3262 ± 0.0225 |
| A1 | N=3 | 0.7255 ± 0.0242 | 0.2438 ± 0.0062 | 0.2833 ± 0.0247 |
| A1 | N=4 | 0.6484 ± 0.0253 | 0.3683 ± 0.0093 | 0.2402 ± 0.0173 |
| A1 | microbump | 0.7208 ± 0.0226 | 0.2553 ± 0.0064 | 0.2900 ± 0.0204 |
| A1 | hybrid | 0.6498 ± 0.0246 | 0.3041 ± 0.0063 | 0.2574 ± 0.0185 |
| Solo2Stack | overall | 0.7919 ± 0.0206 | 0.2245 ± 0.0116 | 0.4666 ± 0.0163 |
| Solo2Stack | N=2 | 0.9136 ± 0.0105 | 0.1000 ± 0.0052 | 0.5730 ± 0.0120 |
| Solo2Stack | N=3 | 0.8259 ± 0.0189 | 0.1968 ± 0.0119 | 0.4663 ± 0.0201 |
| Solo2Stack | N=4 | 0.7590 ± 0.0245 | 0.3074 ± 0.0158 | 0.4135 ± 0.0223 |
| Solo2Stack | microbump | 0.7461 ± 0.0252 | 0.2398 ± 0.0117 | 0.4395 ± 0.0140 |
| Solo2Stack | hybrid | 0.8317 ± 0.0186 | 0.2091 ± 0.0128 | 0.4937 ± 0.0199 |
| A0 | overall | 0.1587 ± 0.0278 | 0.4785 ± 0.0189 | 0.1224 ± 0.0157 |
| A2 | overall | 0.3668 ± 0.0384 | 0.3973 ± 0.0163 | 0.2598 ± 0.0196 |

## Task 4 — peak temperature and ranking

| Arm | Peak abs. error (K) | Same peak layer | Mean Spearman | Groups/fold | Stacks/group |
|---|---|---|---|---|---|
| A0 | 0.8254 ± 0.0825 | 1.0000 | 0.7431 ± 0.0352 | 30 | 200 |
| A1 | 0.2354 ± 0.0132 | 0.9999 | 0.9796 ± 0.0033 | 30 | 200 |
| A2 | 0.3695 ± 0.0281 | 1.0000 | 0.9453 ± 0.0092 | 30 | 200 |
| Solo2Stack | 0.2895 ± 0.0223 | 0.9999 | 0.9671 ± 0.0043 | 30 | 200 |

Spread of the TRUE peak within a (N, bonding, h) group, averaged over folds:

| N | std (K) | range (K) |
|---|---|---|
| 2 | 0.89–0.91 | 4.39–4.50 |
| 3 | 1.34–1.37 | 6.92–7.01 |
| 4 | 1.93–1.94 | 9.91–10.06 |

The true peak lies in D1 in all 60,000 samples, so "same peak layer" is
near-degenerate: always answering D1 scores 1.000.

## Task 5 — where the maximum error occurs

All 40 maxima (4 arms x 10 folds): layer I34, microbump, N = 4 — 40/40.
h distribution: 5e3 x24, 1e5 x7, 1e4 x4, 5e4 x3, 2.5e4 x2.
32 distinct stacks; per fold the four arms hit 2–4 different stacks.

Geometry of the 40 worst cells:
* nearest Cu-pad boundary: 0 um x21, 5 um x17, 10 um x2 → 38/40 within one cell
* inside a Cu pad: 36/40
* die edge: median 242 um, mean 255 um (only 9/40 within 50 um)
* nearest floorplan block boundary of the two bounding dies: median 15 um,
  26/40 within 20 um

Full per-record table in `csv/task5_worst_cells.csv`.
Figure: `figs/task5_worst_stack.png`.
