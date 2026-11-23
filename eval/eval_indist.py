#!/usr/bin/env python3
"""Table VI's third column: the in-distribution Solo2Stack on the transfer tests.

"Solo2Stack (all)" is the fold-0 A3 model from the main campaign -- trained on
every depth, cooling condition and bonding technology of fold 0's eight
training groups -- scored on exactly the same held-out test subset each transfer
model was scored on. It is an upper reference: it answers "how well would the
model do here if the condition had NOT been held out", so the gap to the
transfer column is the cost of holding the condition out.

No leakage: the transfer test subsets are drawn from group_01, which is fold 0's
test group, so the in-distribution model never saw any group_01 chiplet either.

Each model is evaluated under its OWN normalisation, as trained: the transfer
runs use transfer_mbhb/norm/<setting>.json and this one uses
cv_mbhb/norm_cv/fold0.json.
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
import sys

import torch
from torch.utils.data import DataLoader

add_repo(__file__, "model")
add_repo(__file__, "eval/transfer")
from loader_multi import MAX_LAYERS                       # noqa: E402
from unet_decomp import ThermalUNetDecomp                 # noqa: E402
from train_cv import score                                # noqa: E402
from train_transfer import Restricted, ROOTS              # noqa: E402

PLAN = P["transfer_dir"] + "/plan.json"
CKPT = P["checkpoints"] + "/A3_fold0/best.pt"
NORM = P["cv_dir"] + "/norm_cv/fold0.json"
OUT = P["transfer_dir"] + "/indist_reference.json"
SETTINGS = ["depth", "h_interp", "h_extrap", "tech_reverse"]
DEV = torch.device(os.environ.get("INFER_DEV", "cuda:0"))
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main():
    plan = json.load(open(PLAN))
    G = plan["groups"]
    assert G["test"] == ["group_01"], G
    by_name = {s["name"] if isinstance(s, dict) else s: s
               for s in plan["settings"]}
    # plan["settings"] is a list of names; the restrictions live in each run's
    # result.json, which is the authoritative record of what was actually used.
    out = {"ckpt": CKPT, "norm": NORM, "groups": G, "results": {}}
    for st in SETTINGS:
        res = json.load(open(P["transfer_dir"] + "/runs/%s_A3/"
                             "result.json" % st))
        R = res["restriction"]
        te = Restricted(ROOTS, arm="A3", groups=G["test"],
                        depths=tuple(R["test_depths"]), norm_path=NORM,
                        bcs=R["test_bcs"], ics=R["test_ics"])
        n_tr = res["n_samples"]["test"]
        assert len(te) == n_tr, (st, len(te), n_tr)
        dl = DataLoader(te, batch_size=32, shuffle=False, num_workers=8,
                        pin_memory=True)
        model = ThermalUNetDecomp(in_ch=te.in_channels, out_ch=MAX_LAYERS,
                                  base=32, depth=4, scalar_dim=2).to(DEV)
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        m = score(model, dl, DEV, AMP, layerwise=True)
        out["results"][st] = {
            "restriction": R, "n_test": len(te),
            "indist_A3_all": m["pooled"],
            "indist_A3_all_per_depth": m.get("per_depth"),
            "transfer_A3": res["test"]["pooled"],
            "transfer_A1": json.load(open(
                P["transfer_dir"] + "/runs/%s_A1/result.json"
                % st))["test"]["pooled"],
        }
        print("%-13s n=%4d   A1 %.4f   A3 %.4f   A3(all) %.4f"
              % (st, len(te),
                 out["results"][st]["transfer_A1"]["r2_pattern"],
                 out["results"][st]["transfer_A3"]["r2_pattern"],
                 m["pooled"]["r2_pattern"]), flush=True)
        del model, te, dl
        torch.cuda.empty_cache()
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
