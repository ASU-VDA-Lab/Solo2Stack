"""Conditioned U-Net for 2-layer chiplet stack thermal residuals.

400 = 16 x 25, so four downsamples (400 -> 200 -> 100 -> 50 -> 25) divide cleanly
and a fifth would need padding.

The one non-standard piece is the far-field branch. Steady-state heat spreading is
long range -- the Green's function decays like 1/r, so every cell couples to every
other cell. A stack of 3x3 convolutions weights the near field far too heavily, and
the failure mode is a model that gets hotspot shapes right and inter-hotspot
coupling wrong. The bottleneck therefore carries a global-average-pooling path
whose output is broadcast back as a per-channel FiLM modulation, giving the network
a cheap route to represent "how hot is this die overall" independently of local
structure. That also happens to be the component that dominates the target: the
per-sample offset is ~99% of the residual variance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = gn(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class FarField(nn.Module):
    """Global pooling -> FiLM. Gives the bottleneck an unbounded receptive field."""

    def __init__(self, c, scalar_dim=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c + scalar_dim, c * 2), nn.SiLU(), nn.Linear(c * 2, c * 2))

    def forward(self, x, s=None):
        g = x.mean(dim=(2, 3))
        if s is not None:
            g = torch.cat([g, s], dim=1)
        gamma, beta = self.mlp(g).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ThermalUNet(nn.Module):
    def __init__(self, in_ch, out_ch=2, base=32, depth=4, scalar_dim=0):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]  # 32 64 128 256 512
        self.stem = Block(in_ch, chs[0])
        self.down = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(depth)])
        self.mid = Block(chs[-1], chs[-1])
        self.far = FarField(chs[-1], scalar_dim)
        self.up = nn.ModuleList(
            [Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, s=None):
        h = self.stem(x)
        skips = []
        for blk in self.down:
            skips.append(h)
            h = blk(F.avg_pool2d(h, 2))
        h = self.far(self.mid(h), s)
        for blk, sk in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=sk.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, sk], dim=1))
        return self.head(h)


def gradient_loss(pred, true):
    """Keeps hotspot edges sharp; plain L1/MSE smooths them."""
    dx_p, dx_t = pred.diff(dim=-1), true.diff(dim=-1)
    dy_p, dy_t = pred.diff(dim=-2), true.diff(dim=-2)
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()
