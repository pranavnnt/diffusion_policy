"""Photometric augmentation, which this repository does not otherwise have.

``MultiImageObsEncoder`` ships a ``CropRandomizer`` and ImageNet normalisation
and nothing else. That is a reasonable omission upstream — its benchmarks are
simulated under fixed lighting — and a real problem here: the bed-front camera
faces a window.

Measured on ``zigzag_bed_0910`` (2026-09-10), front camera:

    within-episode brightness std    2.5
    between-episode brightness std   6.7   (episodes cluster at ~152 and ~166)
    per-channel means at the two clusters   R 144.6 G 156.5 B 159.3
                                            R 159.5 G 169.8 B 171.5

So illumination is stable *during* a recording and shifts *between* them, and
it shifts in colour as well as level — daylight is bluer, and R sits ~10 below
G and B by a margin that tracks the brightness. Five episodes recorded in one
session already show that; different times of day would show much more.

Two consequences, and the second is the dangerous one:

1. A policy trained on a handful of episodes can key on illumination as a cue
   for *which episode it is in*, which is a shortcut to the demonstrated
   trajectory that does not survive a different hour.
2. Every episode here is a success, so nothing in the data pushes back on that.

``imagenet_norm`` does not help: it subtracts a fixed dataset-wide mean, not a
per-image one, so a whole-image brightness shift passes straight through.

The jitter below is **per sample**, not per batch. torchvision's ``ColorJitter``
draws one parameter set per call, so a batch spanning several episodes would be
shifted together and the model could still separate them; drawing per image is
what makes the cue unusable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

#: Luminance weights, so a saturation change keeps perceived brightness.
_LUMA = (0.299, 0.587, 0.114)


class PhotometricJitter(nn.Module):
    """Per-image brightness, contrast, saturation and colour-temperature jitter.

    Operates on ``[N, 3, H, W]`` in ``[0, 1]``, before ImageNet normalisation.
    A no-op in ``eval()``: augmentation belongs to training, and a diagnostic
    that perturbs the input deliberately (see
    :func:`~diffusion_policy.drim.diagnose.illumination_sensitivity`) must be
    the only thing changing it at evaluation time.

    ``channel_gain`` is the one aimed at daylight. Brightness and contrast move
    all three channels together; a window moves them apart, and a per-channel
    gain is the cheapest thing that reproduces that.

    ``brightness`` takes either a fraction — ``0.3`` meaning ``x[0.7, 1.3]`` —
    or an explicit ``(lo, hi)`` pair of multipliers. The pair exists because a
    train/inference gap has a *direction*: training on the sunlit 0910
    recordings and running under the curtained 0911 conditions asks for x0.66
    on the bed-back camera and nothing above x1.11, so a symmetric range wide
    enough to reach 0.66 also opens 1.34 — pushing the brightest training
    episodes towards saturation for nothing.
    """

    def __init__(self, brightness=0.3, contrast: float = 0.3,
                 saturation: float = 0.3, channel_gain: float = 0.12):
        super().__init__()
        self.brightness = (float(brightness) if np.isscalar(brightness)
                           else tuple(float(v) for v in brightness))
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.channel_gain = float(channel_gain)

    def extra_repr(self) -> str:
        return (f"brightness={self.brightness}, contrast={self.contrast}, "
                f"saturation={self.saturation}, channel_gain={self.channel_gain}")

    def _u(self, n: int, amount, x: torch.Tensor,
           gen: Optional[torch.Generator]) -> torch.Tensor:
        """``[N, 1, 1, 1]`` multipliers, uniform over the declared range.

        A scalar ``a`` means ``[1 - a, 1 + a]``; a pair is taken as it is.
        """
        if np.isscalar(amount):
            if amount <= 0:
                return torch.ones(n, 1, 1, 1, device=x.device, dtype=x.dtype)
            lo, hi = 1.0 - float(amount), 1.0 + float(amount)
        else:
            lo, hi = (float(v) for v in amount)
        r = torch.rand(n, 1, 1, 1, device=x.device, dtype=x.dtype, generator=gen)
        return lo + (hi - lo) * r

    def forward(self, x: torch.Tensor,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if not self.training:
            return x
        n = x.shape[0]
        luma = torch.tensor(_LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

        if self.brightness != 0:
            x = x * self._u(n, self.brightness, x, generator)
        if self.contrast > 0:
            mean = (x * luma).sum(1, keepdim=True).mean((2, 3), keepdim=True)
            x = mean + (x - mean) * self._u(n, self.contrast, x, generator)
        if self.saturation > 0:
            grey = (x * luma).sum(1, keepdim=True)
            x = grey + (x - grey) * self._u(n, self.saturation, x, generator)
        if self.channel_gain > 0:
            g = torch.rand(n, 3, 1, 1, device=x.device, dtype=x.dtype,
                           generator=generator)
            x = x * (1.0 + (2.0 * g - 1.0) * self.channel_gain)
        return x.clamp(0.0, 1.0)


def shift_illumination(x: torch.Tensor, brightness: float = 0.0,
                       channel_gain: Optional[Sequence[float]] = None
                       ) -> torch.Tensor:
    """A **deterministic** illumination change, for probing a trained policy.

    Deliberately separate from :class:`PhotometricJitter`: one is random and
    trains, the other is fixed and measures, and mixing them would make the
    measurement depend on a seed.
    """
    y = x * (1.0 + brightness)
    if channel_gain is not None:
        g = torch.tensor(channel_gain, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        y = y * g
    return y.clamp(0.0, 1.0)
