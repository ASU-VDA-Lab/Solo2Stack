#!/usr/bin/env python3
"""Fig. 1: accuracy against runtime for four-die stacks.

No standalone Fig. 1 script existed in the repository. The published figure's
content lives in make_paper_figs.py:runtime() panel (b), which plots pattern R^2
against wall time per stack for the four arms with a full-stack PACT reference,
but pools accuracy over all three depths and places PACT at the geometric mean
of its 2-die and 4-die cost. This script rebuilds that panel as a single figure
restricted to FOUR-DIE stacks, as specified, and in the Fig. 3 visual language
(statcheck/fig_depth_exact.py): same palette, same per-arm markers with black
edges, same dotted grid, same fonts, Solo2Stack as the green star. PACT takes
the vermillion circle, the one marker Fig. 3 does not use.

Cost model, following the published figure's convention:

  * A0 and A1 need no solo characterisation, so a stack costs one inference.
  * A2 and Solo2Stack are placed at ONE solo simulation plus one inference,
    8.685 + 0.00439 = 8.68939 s. That is the convention the published figure
    used, and it is all this figure shows -- a single point per arm.
  * Full-stack PACT at four dies is the measured mean over the two bonding
    technologies (microbump 241.985 s, hybrid 435.845 s).

Nothing on this figure is drawn or labelled at a shared-cost position, and no
sweep-level cost is claimed here. That separate quantity lives in the runtime
section of PAPER_NUMBERS.md.

Accuracy is the ten-fold mean pattern R^2 at depth 4 with its sample SD
(ddof=1); PACT sits at 1.0 by construction, being the reference the surrogate
is scored against, not a measurement.
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
from solo2stack.paths import P, repo_path  # noqa: E402
from solo2stack.bootstrap import add_repo  # noqa: E402
# --------------------------------------------------------------------------
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
import numpy as np                    # noqa: E402

RUNS = P["checkpoints"]
# the committed copy under results/ is the published one, so it is the
# default; a fresh runtime_section.py run writes into the results directory
RT = next((c for c in (repo_path("results", "runtime_mbhb.json"),
                       os.path.join(P["results"], "runtime_mbhb.json"))
           if os.path.exists(c)), repo_path("results", "runtime_mbhb.json"))
OUTDIR = os.environ.get("FIGDIR", P["data_root"] + "/figs_mbhb")
DEPTH = 4

# Fig. 3 palette and markers, reused verbatim; PACT gets the unused circle.
STYLE = {
    "A0":         ("#888888", "^", 6.0, "A0"),
    "A2":         ("#e69f00", "D", 6.0, "A2"),
    "A1":         ("#0072b2", "s", 6.0, "A1"),
    "A3":         ("#009e73", "*", 9.0, "Solo2Stack"),
    "PACT":       ("#d55e00", "o", 6.0, "Full-stack PACT"),
}
NEEDS_SOLO = {"A0": False, "A1": False, "A2": True, "A3": True}


def r2(arm):
    v = np.array([json.load(open("%s/%s_fold%d/result.json" % (RUNS, arm, f)))
                  ["test"]["per_depth"][str(DEPTH)]["r2_pattern"]
                  for f in range(10)])
    return float(v.mean()), float(v.std(ddof=1))


def main():
    rt = json.load(open(RT))
    new = rt["arms"]["new"]
    solo_s = rt["solo_s"]
    infer_s = rt["infer_ms_batch1"] / 1000.0
    pact_s = new["full_stack_s"][str(DEPTH)]
    # one solo simulation + one inference for the solo-conditioned arms
    single = {a: (solo_s + infer_s) if NEEDS_SOLO[a] else infer_s
              for a in ("A0", "A1", "A2", "A3")}
    # components are read from the runtime record, never hard-coded
    assert abs(solo_s - 8.685) < 1e-6 and abs(infer_s - 0.00439) < 1e-9, \
        (solo_s, infer_s)
    assert abs(pact_s - 338.915) < 1e-6, pact_s

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7})
    fig, ax = plt.subplots(figsize=(3.40, 2.25))

    vals = {}
    for arm in ("A0", "A2", "A1", "A3"):
        c, m, ms, lab = STYLE[arm]
        mu, sd = r2(arm)
        vals[arm] = (single[arm], mu, sd)
        ax.errorbar([single[arm]], [mu], yerr=[sd], marker=m, color=c,
                    ms=ms, mec="k", mew=0.4, ls="none", capsize=1.8,
                    elinewidth=0.7, capthick=0.7, zorder=4)
    c, m, ms, lab = STYLE["PACT"]
    ax.plot([pact_s], [1.0], marker=m, color=c, ms=ms, mec="k", mew=0.4,
            ls="none", zorder=4)
    vals["PACT"] = (pact_s, 1.0, 0.0)

    ax.set_xscale("log")
    ax.set_xlabel("Wall time per four-die stack (s)", labelpad=2)
    ax.set_ylabel("Pattern $R^2$", labelpad=1)
    ax.set_xlim(2e-3, 2.2e3)
    ax.set_ylim(0.2, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1],
                      ls="none", ms=STYLE[k][2], mec="k", mew=0.4,
                      label=STYLE[k][3])
               for k in ("A0", "A2", "A1", "A3", "PACT")]
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
                    handletextpad=0.3, columnspacing=1.0, labelspacing=0.3,
                    borderaxespad=0.0)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "accuracy_vs_runtime_4die_mbhb.pdf")
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=[leg])
    fig.savefig(out.replace(".pdf", ".png"), dpi=400, bbox_inches="tight",
                bbox_extra_artists=[leg])
    print("wrote %s (+ .png)" % out)
    print("%-16s %14s %10s %8s" % ("series", "wall_s/stack", "r2_pat", "sd"))
    for k in ("A0", "A2", "A1", "A3", "PACT"):
        x, mu, sd = vals[k]
        print("%-16s %14.5f %10.4f %8.4f" % (STYLE[k][3], x, mu, sd))
    print("Solo2Stack speedup at %d dies, one solo + inference: %.2fx"
          % (DEPTH, pact_s / single["A3"]))


if __name__ == "__main__":
    main()
