"""Rotation-friendly surgery for the log-polar front end.

In the log-polar images this project feeds the backbone, theta is the ROW axis
(dim -2) and log-r is the COLUMN axis (dim -1) -- see LogPolar.compute_map in
trans.py, where `theta` indexes output_shape[0]. A rotation about the fixation
point is therefore a CYCLIC shift along rows, verified empirically: the best
matching roll between LP(rotate180(x)) and roll(LP(x), k) is exactly k=90 of
180 rows, residual 0.046 against 0.273 unrolled.

Two things stop a stock ResNet exploiting that (Azulay & Weiss 2019):

  1. Zero padding treats the theta=0/2pi seam as a hard edge, so wrapped content
     meets padding that upright content never sees. CylConv2d pads theta
     CIRCULARLY and log-r with zeros. The asymmetry matters: r runs from the
     fovea outward and genuinely has ends, so circular padding there would be
     wrong. torch's padding_mode applies to both axes at once, hence this
     wrapper rather than padding_mode='circular'.

  2. Strided convs and pooling alias, breaking shift-equivariance. BlurPool2d
     (Zhang 2019) replaces "stride s" with "stride 1, low-pass, subsample s".

Measured motivation: on r19_rfwh_s42 a 1-row shift (2 degrees) already moves the
256-d code by 0.056, 17% of the full between-identity distance, and 20% inverted
training exposure (r18_inv20) does not improve that at all -- it is structural,
not learnable.
"""

# Support direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F


class CylConv2d(nn.Module):
    """Conv2d padding circularly on theta (rows) and with zeros on log-r (cols)."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.pad_t, self.pad_r = conv.padding if isinstance(conv.padding, tuple) else (conv.padding,) * 2
        conv.padding = (0, 0)
        self.conv = conv

    def forward(self, x):
        if self.pad_t:
            # circular pad needs pad < dim; fall back to replicate if a tiny
            # feature map ever violates that.
            if self.pad_t < x.shape[-2]:
                x = F.pad(x, (0, 0, self.pad_t, self.pad_t), mode="circular")
            else:
                x = F.pad(x, (0, 0, self.pad_t, self.pad_t), mode="replicate")
        if self.pad_r:
            x = F.pad(x, (self.pad_r, self.pad_r, 0, 0), mode="constant", value=0.0)
        return self.conv(x)


class BlurPool2d(nn.Module):
    """Low-pass then subsample (Zhang 2019).

    circular=True wraps theta in the blur's own padding, matching CylConv2d.
    circular=False reflect-pads both axes, as in Zhang's reference code, so the
    anti-aliasing adds NO theta-wrap prior -- resnet18_blur depends on this,
    since a circular blur would quietly reintroduce the rotation structure that
    the blur-only arm exists to leave out.
    """

    def __init__(self, channels, stride=2, filt_size=3, circular=True):
        super().__init__()
        self.stride, self.channels, self.circular = stride, channels, circular
        a = {1: [1.], 2: [1., 1.], 3: [1., 2., 1.],
             4: [1., 3., 3., 1.], 5: [1., 4., 6., 4., 1.]}[filt_size]
        a = torch.tensor(a)
        filt = (a[:, None] * a[None, :])
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None].repeat(channels, 1, 1, 1))
        self.pad = filt_size // 2

    def forward(self, x):
        if self.pad and not self.circular:
            x = F.pad(x, (self.pad,) * 4, mode="reflect")
        elif self.pad:
            if self.pad < x.shape[-2]:
                x = F.pad(x, (0, 0, self.pad, self.pad), mode="circular")
            else:
                x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="constant", value=0.0)
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)


def cylindrify(module):
    """Replace every padded Conv2d with a CylConv2d, in place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            pad = child.padding if isinstance(child.padding, tuple) else (child.padding,) * 2
            if pad != (0, 0):
                setattr(module, name, CylConv2d(child))
        else:
            cylindrify(child)
    return module


def antialias(module, filt_size=3, _ch=[3], circular=True):
    """Replace strided convs / maxpool with stride-1 + BlurPool, in place.
    circular is passed to every BlurPool2d (see its docstring)."""
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv2d, CylConv2d)):
            conv = child.conv if isinstance(child, CylConv2d) else child
            _ch[0] = conv.out_channels
            if tuple(conv.stride) != (1, 1):
                s = conv.stride[0]
                conv.stride = (1, 1)
                setattr(module, name, nn.Sequential(child, BlurPool2d(conv.out_channels, s, filt_size, circular)))
        elif isinstance(child, nn.MaxPool2d) and child.stride not in (1, (1, 1)):
            s = child.stride if isinstance(child.stride, int) else child.stride[0]
            setattr(module, name, nn.Sequential(
                nn.MaxPool2d(child.kernel_size, stride=1, padding=child.padding),
                BlurPool2d(_ch[0], s, filt_size, circular)))
        else:
            antialias(child, filt_size, _ch, circular)
    return module
