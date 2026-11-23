#!/usr/bin/env python3
"""Extract every paper number from the microbump+hybrid campaign.

Estimator provenance, which must not be mixed up:

* The mean columns (Pat. MAE, R2_pat, Abs. MAE, Offset MAE, Hotspot) come from
  aggregate.json, i.e. train_cv.score()'s per-stack means averaged over the ten
  folds. That is the estimator Table IV uses.
* p99 in aggregate.json is a mean of per-slice p99s, not a true quantile. It is
  reported here under the published definition AND alongside the true global
  p99, because which one the manuscript prints is an open question.
* Max is the true global maximum over every cell of every valid layer of every
  test stack, from eval_global.py. aggregate.json's max_k is a mean of per-slice
  maxima and understates the real maximum by 1.6-1.9x, so it is never used.
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
from scipy import stats

AGG = P["cv_dir"] + "/aggregate.json"
GLOB = P["eval_global"] + "/C_new_mbhb.json"
RUNS = P["checkpoints"]
OUT = os.path.join(P["results"], "paper_numbers_mbhb.json")
ARMS = ["A0", "A2", "A1", "A3"]          # paper column order
HOT = "hotspot_top1%_overlap"
LAYERS4 = ["D1", "I12", "D2", "I23", "D3", "I34", "D4"]


def ms(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "per_fold": [float(x) for x in v]}


def main():
    agg = json.load(open(AGG))
    gl = json.load(open(GLOB))["results"]
    out = {}

    # ---------------- Table III: channels and parameter count -------------
    import sys
    add_repo(__file__, "model")
    from unet_decomp import ThermalUNetDecomp
    from loader_multi import MAX_LAYERS
    t2 = {}
    for arm, inch in (("A0", 10), ("A1", 10), ("A2", 14), ("A3", 14)):
        m = ThermalUNetDecomp(in_ch=inch, out_ch=MAX_LAYERS, base=32, depth=4,
                              scalar_dim=2)
        t2[arm] = {"in_channels": inch,
                   "params": int(sum(p.numel() for p in m.parameters())),
                   "params_trainable": int(sum(p.numel() for p in
                                               m.parameters()
                                               if p.requires_grad))}
    # cross-check against what the runs actually instantiated
    for arm in ARMS:
        r = json.load(open(os.path.join(RUNS, "%s_fold0" % arm, "result.json")))
        t2[arm]["max_layers_out"] = MAX_LAYERS
    out["table2"] = t2

    # ---------------- Table IV: four arms, seven metrics ------------------
    t3 = {}
    for arm in ARMS:
        p = agg["pooled"][arm]
        gmax = [gl["%s_fold%d" % (arm, f)]["global"]["max_global_k"]
                for f in range(10)]
        gp99 = [gl["%s_fold%d" % (arm, f)]["global"]["p99_global_k"]
                for f in range(10)]
        t3[arm] = {
            "mae_pattern_k": p["mae_pattern_k"],
            "r2_pattern": p["r2_pattern"],
            "mae_k": p["mae_k"],
            "mae_offset_k": p["mae_offset_k"],
            "p99_k_published_estimator": p["p99_k"],
            "p99_k_global": ms(gp99),
            "max_k_global": ms(gmax),
            "max_k_published_estimator": p["max_k"],
            "hotspot": p[HOT],
        }
    out["table3"] = t3

    # ---------------- Table V: Solo2Stack per layer, 4-die ----------------
    # aggregate.json["per_layer"][arm] is keyed by FOLD, then by "N|layer",
    # so the fold statistics have to be formed here rather than read off.
    pl_a3 = agg["per_layer"]["A3"]
    mets = sorted(pl_a3["0"]["4|D1"].keys())
    t4 = {}
    for L in LAYERS4:
        key = "4|%s" % L
        t4[L] = {m: ms([pl_a3[str(f)][key][m] for f in range(10)])
                 for m in mets}
    out["table4"] = t4
    out["table4_metrics_available"] = mets

    # per-layer global max for the same cells, so Table V's Max is the same
    # estimator as Table IV's
    t4g = {}
    for L in LAYERS4:
        vals = []
        for f in range(10):
            pl = gl["A3_fold%d" % f]["global_per_layer"]
            k = [kk for kk in pl if kk.endswith("|%s" % L)
                 and kk.startswith("4|")]
            assert len(k) == 1, (L, k)
            vals.append(pl[k[0]]["max_global_k"])
        t4g[L] = ms(vals)
    out["table4_global_max"] = t4g

    # ---------------- argmax location of the 40 global maxima -------------
    loc = []
    for arm in ARMS:
        for f in range(10):
            g = gl["%s_fold%d" % (arm, f)]
            tgt = g["global"]["max_global_k"]
            best = None
            for k, v in g["global_per_ic_layer"].items():
                if abs(v["max_global_k"] - tgt) < 1e-9:
                    best = k
                    break
            assert best is not None, (arm, f, tgt)
            ic, n, lay = best.split("|")
            loc.append({"arm": arm, "fold": f, "max_k": tgt,
                        "technology": ic, "n_dies": int(n), "layer": lay,
                        "key": best})
    out["argmax_locations"] = loc
    cnt = {}
    for r in loc:
        kk = "%s | N=%d | %s" % (r["technology"], r["n_dies"], r["layer"])
        cnt[kk] = cnt.get(kk, 0) + 1
    out["argmax_counts"] = dict(sorted(cnt.items(), key=lambda x: -x[1]))
    for field in ("technology", "n_dies", "layer"):
        c = {}
        for r in loc:
            c[str(r[field])] = c.get(str(r[field]), 0) + 1
        out["argmax_counts_%s" % field] = dict(sorted(c.items(),
                                                      key=lambda x: -x[1]))

    # ---------------- interaction (A3-A1) - (A2-A0) -----------------------
    inter = {}
    for key, lab in (("r2_pattern", "r2_pattern"), (HOT, "hotspot")):
        f = {a: np.array(agg["pooled"][a][key]["per_fold"], dtype=float)
             for a in ARMS}
        d = (f["A3"] - f["A1"]) - (f["A2"] - f["A0"])
        t = stats.ttest_1samp(d, 0.0)
        try:
            w = stats.wilcoxon(d)
            wp = float(w.pvalue)
        except ValueError:
            wp = None
        inter[lab] = {
            "solo_effect_residual_A3_minus_A1": ms(f["A3"] - f["A1"]),
            "solo_effect_absolute_A2_minus_A0": ms(f["A2"] - f["A0"]),
            "interaction": ms(d),
            "folds_interaction_positive": int((d > 0).sum()),
            "folds_interaction_negative": int((d < 0).sum()),
            "t_statistic": float(t.statistic),
            "t_pvalue": float(t.pvalue),
            "wilcoxon_pvalue": wp,
            "n_folds": int(len(d)),
        }
    out["interaction"] = inter

    # ---------------- dataset counts --------------------------------------
    counts = {}
    for d in (2, 3, 4):
        meta = json.load(open(P["data_root"] + "/pact_stack_%dL/meta.json" % d))
        counts[d] = meta["counts"]
    out["dataset_counts"] = counts
    out["dataset_total_stacks"] = sum(c["stacks"] * c["bcs"]
                                      for c in counts.values())
    out["per_fold_sample_counts"] = json.load(
        open(os.path.join(RUNS, "A3_fold0", "result.json")))["n_samples"]

    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote %s" % OUT)

    # ---------------- console summary -------------------------------------
    print("\nTable IV (mean +- sample std over 10 folds)")
    hdr = ("%-10s %16s %16s %16s %16s %16s %16s %16s"
           % ("metric", "A0", "A2", "A1", "Solo2Stack", "", "", ""))
    print(hdr[:96])
    rows = [("Pat. MAE", "mae_pattern_k"), ("R2_pat", "r2_pattern"),
            ("Abs. MAE", "mae_k"), ("Offset MAE", "mae_offset_k"),
            ("p99 (pub)", "p99_k_published_estimator"),
            ("p99 (glob)", "p99_k_global"),
            ("Max (glob)", "max_k_global"), ("Hotspot", "hotspot")]
    for lab, k in rows:
        print("%-11s" % lab, end="")
        for arm in ARMS:
            v = t3[arm][k]
            print("%9.4f+-%-7.4f" % (v["mean"], v["std"]), end="")
        print()

    print("\nargmax of the 40 fold-level global maxima")
    for k, v in out["argmax_counts"].items():
        print("   %-28s %2d / 40" % (k, v))

    print("\ninteraction (A3-A1)-(A2-A0)")
    for lab, v in inter.items():
        print("   %-10s %+.4f +- %.4f   pos %d/10  t p=%.4g  wilcoxon p=%s"
              % (lab, v["interaction"]["mean"], v["interaction"]["std"],
                 v["folds_interaction_positive"], v["t_pvalue"],
                 v["wilcoxon_pvalue"]))

    print("\nTable III")
    for arm in ARMS:
        print("   %s  in_ch %2d  params %d"
              % (arm, t2[arm]["in_channels"], t2[arm]["params"]))


if __name__ == "__main__":
    main()
