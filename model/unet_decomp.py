"""U-Net with an explicit offset / pattern decomposition of the residual field.

Identical backbone to unet.ThermalUNet -- same stem, downs, mid, FarField FiLM,
ups and 1x1 pattern head. The only change is how the output is formed:

    pattern   the existing 1x1 conv head, with its per-layer spatial mean
              subtracted so it is exactly zero-mean by construction
    offset    one scalar per physical layer, read from the globally pooled
              bottleneck (plus the FiLM scalars) through a single Linear
    residual  pattern + offset

The motivation is that the per-sample offset carries ~99% of the residual
variance but is a *global* quantity, and a 1x1 convolution over local decoder
features is a poor way to represent one. Making it an explicit pooled scalar
stops offset error from leaking into the pattern, which is the component the
surrogate exists to capture.

Parameter cost over the baseline is one Linear(C_bottleneck + scalar_dim,
out_ch): 3,591 parameters on the default configuration, or +0.025%.

The head and the mean subtraction run in float32 even under autocast. The
pattern is ~0.5 K against a residual of ~4-19 K, and a bfloat16 spatial mean
over 160,000 cells carries enough error to be comparable to the pattern MAE
being measured.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from unet import Block, FarField  # noqa: E402


class ThermalUNetDecomp(nn.Module):
    def __init__(self, in_ch, out_ch=7, base=32, depth=4, scalar_dim=0):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.stem = Block(in_ch, chs[0])
        self.down = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(depth)])
        self.mid = Block(chs[-1], chs[-1])
        self.far = FarField(chs[-1], scalar_dim)
        self.up = nn.ModuleList(
            [Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)          # zero-mean pattern
        self.offset = nn.Linear(chs[-1] + scalar_dim, out_ch)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x, s=None):
        h = self.stem(x)
        skips = []
        for blk in self.down:
            skips.append(h)
            h = blk(F.avg_pool2d(h, 2))
        h = self.far(self.mid(h), s)

        with torch.autocast("cuda", enabled=False):
            g = h.float().mean(dim=(2, 3))
            if s is not None:
                g = torch.cat([g, s.float()], dim=1)
            off = self.offset(g)                          # (B, out_ch)

        for blk, sk in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=sk.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, sk], dim=1))

        with torch.autocast("cuda", enabled=False):
            pat = self.head(h.float())
            pat = pat - pat.mean(dim=(-2, -1), keepdim=True)

        return pat + off[:, :, None, None], pat, off


def true_offset(y, mask):
    """Per-layer spatial mean of the target, for optional direct supervision."""
    return y.mean(dim=(-2, -1))


def masked_scalar_l1(pred, true, mask):
    return ((pred - true).abs() * mask).sum() / mask.sum().clamp(min=1)
