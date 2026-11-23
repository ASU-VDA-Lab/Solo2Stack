#!/usr/bin/env python3
"""Inference latency of the frozen decomp+aux surrogate used in the 10-fold study.

The unified model always consumes the same padded tensor and emits the same
seven layer slots regardless of stack depth, so a forward pass costs the same
at 2, 3 and 4 dies. That is measured here rather than assumed.
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
add_repo(__file__, "model")
# --------------------------------------------------------------------------

import argparse
import json
import os
import statistics as st
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader_multi import MAX_LAYERS
from unet_decomp import ThermalUNetDecomp


def bench(net, x, s, mask_depth, reps, dev):
    with torch.no_grad():
        for _ in range(10):                       # warm-up
            net(x, s)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            net(x, s)
            torch.cuda.synchronize(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts), st.stdev(ts) if len(ts) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=P["checkpoints"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", default=P["cv_dir"] + "/runtime_infer.json")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    res = {"gpu": torch.cuda.get_device_name(dev), "reps": a.reps}

    for arm, ich in (("A1", 10), ("A3", 14)):
        ck = os.path.join(a.ckpt_dir, "%s_fold0" % arm, "best.pt")
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        net = ThermalUNetDecomp(in_ch=ich, out_ch=MAX_LAYERS,
                                base=sd["args"]["base"], depth=sd["args"]["depth"],
                                scalar_dim=2).to(dev).eval()
        net.load_state_dict(sd["model"])
        nparam = sum(p.numel() for p in net.parameters())

        entry = {"in_ch": ich, "params": nparam, "ckpt": ck}
        for d in (2, 3, 4):                       # depth-independence check
            x = torch.randn(1, ich, 400, 400, device=dev)
            s = torch.randn(1, 2, device=dev)
            m, sdev = bench(net, x, s, d, a.reps, dev)
            entry["ms_b1_depth%d" % d] = m
            entry["ms_b1_depth%d_std" % d] = sdev
        for bs in (1, 8, 32):
            x = torch.randn(bs, ich, 400, 400, device=dev)
            s = torch.randn(bs, 2, device=dev)
            m, _ = bench(net, x, s, 4, max(20, a.reps // 4), dev)
            entry["ms_batch%d" % bs] = m
            entry["ms_per_sample_batch%d" % bs] = m / bs
        res[arm] = entry
        print("%s in_ch=%d params=%d  b1=%.2f ms  per-sample@bs32=%.2f ms"
              % (arm, ich, nparam, entry["ms_b1_depth4"],
                 entry["ms_per_sample_batch32"]), flush=True)

    json.dump(res, open(a.out, "w"), indent=2)
    print("\ndepth-independence (A3 batch-1 median ms): d2 %.2f  d3 %.2f  d4 %.2f"
          % (res["A3"]["ms_b1_depth2"], res["A3"]["ms_b1_depth3"],
             res["A3"]["ms_b1_depth4"]))
    print("saved -> %s" % a.out)


if __name__ == "__main__":
    main()
