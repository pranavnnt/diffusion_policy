"""The IRUM networks, ported from dap with every dimension made an argument.

Sources, all under ``~/dap``:

``FlowChunkPolicy``       ``dynamics_aligned_policy/irum_adapter/slow_policy.py``
``FastCorrector``         ``drawer_.../v1/policy/flow/models_flow.py``
``ResidualVectorField``   same file
``MessageReconciler``     same file
``MessageEncoder``        ``drawer_.../v1/policy/full_online/models_full.py``

Two substantive changes, both forced by the target task and both marked at the
site:

* the per-channel fast ceiling from ``liftoff_v5.ChannelBoundedFastCorrector`` is
  folded into the one corrector rather than kept as a subclass — dap needed both
  because v4's scalar was already in published checkpoints, which is not a
  constraint here;
* ``wrench`` may be zero-width, so every ``cat`` that consumed it is written to
  tolerate an empty tensor.

``ConditionalUnet1D`` is this repository's own
(``diffusion_policy.model.diffusion.conditional_unet1d``) — it is what dap
imported too, so the backbone is not a re-implementation on either side.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def mlp(sizes: Sequence[int], act=nn.SiLU) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def _cat(*parts: Optional[torch.Tensor]) -> torch.Tensor:
    """Concatenate, skipping ``None`` and zero-width tensors.

    ``wrench_dim == 0`` is a real configuration here (the dressing rig has no
    force channel), and a ``[B, 0]`` tensor concatenates fine — but callers that
    have no wrench at all pass ``None``, and silently dropping it is better than
    making every call site build an empty tensor.
    """
    keep = [p for p in parts if p is not None and p.shape[-1] > 0]
    return torch.cat(keep, dim=-1)


# --------------------------------------------------------------------------- #
# Slow level
# --------------------------------------------------------------------------- #


class FlowChunkPolicy(nn.Module):
    """Rectified-flow action-chunk policy over a stack of slow frames.

    Ordinary and well understood on purpose: the question IRUM asks is whether a
    plug-in adapter improves a *frozen competent* nominal policy, so the nominal
    policy is a standard chunk policy and nothing more.

    ``obs_mean``/``obs_std`` are buffers, not parameters: they must travel with
    the checkpoint and never receive gradient.  When the loader already
    standardises its arrays these stay at identity — normalising twice is
    harmless but puts a second, undocumented set of statistics in the file.
    """

    def __init__(self, frame_dim: int, act_dim: int, horizon: int,
                 n_obs_steps: int = 2, down_dims: Sequence[int] = (128, 256, 512),
                 step_embed: int = 128, n_flow_steps: int = 10):
        super().__init__()
        self.frame_dim = int(frame_dim)
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_flow_steps = int(n_flow_steps)
        self.context_dim = self.frame_dim * self.n_obs_steps

        self.net = ConditionalUnet1D(
            input_dim=act_dim, local_cond_dim=None,
            global_cond_dim=self.context_dim,
            diffusion_step_embed_dim=step_embed, down_dims=list(down_dims))

        self.register_buffer("obs_mean", torch.zeros(self.frame_dim))
        self.register_buffer("obs_std", torch.ones(self.frame_dim))

    def fit_normalizer(self, frames: torch.Tensor) -> None:
        flat = frames.reshape(-1, self.frame_dim)
        self.obs_mean.copy_(flat.mean(dim=0))
        self.obs_std.copy_(flat.std(dim=0).clamp_min(1e-3))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, n_obs_steps, frame_dim] -> [B, context_dim]``."""
        return ((frames - self.obs_mean) / self.obs_std).flatten(start_dim=1)


# --------------------------------------------------------------------------- #
# Fast level
# --------------------------------------------------------------------------- #


class FastCorrector(nn.Module):
    """Stateless per-step residual, bounded per channel.

    Inputs are the current observation and wrench, the nominal action for *this*
    step, a summary of the nominal chunk still to be executed, and the phase
    within the chunk.  Everything is present-tense or part of the plan being
    tracked: no previous observation, no previous action, no recurrent state.
    The level stays a reactive controller rather than a second memory — which is
    what makes the upward message the only route by which history can reach the
    slow level.

    The remaining-chunk summary is there because the flow policy's nominal is a
    *sample*: two draws at the same state can commit to different manoeuvres, and
    a corrector seeing only the current step cannot tell which it is tracking.

    ``limits`` is a **buffer**, so a checkpoint carries the authority it was
    trained under and a loader can refuse a mismatch.  ``limits[c] == 0`` makes
    channel ``c`` identically zero for any parameter values, because the output
    is ``limits * tanh(...)`` — a forbidden channel is absent, not discouraged.
    """

    def __init__(self, prop_dim: int, wrench_dim: int, act_dim: int,
                 limits: np.ndarray, hidden: int = 256):
        super().__init__()
        lim = torch.as_tensor(np.asarray(limits, dtype=np.float32))
        assert lim.shape == (act_dim,), f"expected {act_dim} ceilings, got {tuple(lim.shape)}"
        assert bool((lim >= 0).all()), "a ceiling may not be negative"
        self.register_buffer("max_residual", lim)
        self.net = mlp([prop_dim + wrench_dim + 3 * act_dim + 1,
                        hidden, hidden, act_dim])

    @property
    def writable(self) -> torch.Tensor:
        return self.max_residual > 0

    def forward(self, prop, wrench, nominal_t, chunk_mean, chunk_remaining, phase):
        x = _cat(prop, wrench, nominal_t, chunk_mean, chunk_remaining, phase)
        return self.max_residual * torch.tanh(self.net(x))

    @staticmethod
    def chunk_features(nominal: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step ``(mean, remaining, phase)`` for a ``[B, H, A]`` chunk.

        ``remaining`` at step ``t`` averages steps ``t..H-1``: a function of the
        *plan*, which the policy itself emitted, not of anything the environment
        has yet revealed.
        """
        b, h, a = nominal.shape
        mean = nominal.mean(1, keepdim=True).expand(b, h, a)
        rev = torch.flip(nominal, dims=[1])
        cum = torch.cumsum(rev, dim=1) / torch.arange(
            1, h + 1, device=nominal.device, dtype=nominal.dtype).view(1, h, 1)
        remaining = torch.flip(cum, dims=[1])
        phase = (torch.arange(h, device=nominal.device, dtype=nominal.dtype)
                 / max(h - 1, 1)).view(1, h, 1).expand(b, h, 1)
        return mean, remaining, phase


# --------------------------------------------------------------------------- #
# The message and its two consumers
# --------------------------------------------------------------------------- #


class MessageEncoder(nn.Module):
    """The compact upward message: a window of per-step tokens to ``message_dim``.

    The bottleneck is the point.  A wide conditioning vector would let the slow
    level read the fast level's whole trajectory, and "the message helped" would
    become indistinguishable from "more of the past was visible".
    """

    def __init__(self, a_dim: int, b_dim: int, act_dim: int,
                 message_dim: int = 16, hidden: int = 128,
                 use_actions: bool = False):
        super().__init__()
        self.use_actions = bool(use_actions)
        step_in = a_dim + b_dim + (2 * act_dim if use_actions else 0)
        self.step = mlp([step_in, hidden, hidden])
        self.head = mlp([2 * hidden, hidden, message_dim])

    def forward(self, hist_a, hist_b, hist_act=None, hist_res=None):
        parts = [hist_a, hist_b]
        if self.use_actions:
            parts += [hist_act, hist_res]
        h = self.step(_cat(*parts))
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=-1))


class ResidualVectorField(nn.Module):
    """``dv(x, t, ctx, m)`` — a small conditional UNet over the chunk shape.

    Deliberately smaller than the base field: it is a correction to an already
    competent generative policy, and giving it the base's capacity would make
    "the message helped" indistinguishable from "the second network is better".
    """

    def __init__(self, act_dim: int, cond_dim: int, message_dim: int,
                 down_dims: Sequence[int] = (64, 128), step_embed: int = 64):
        super().__init__()
        self.net = ConditionalUnet1D(
            input_dim=act_dim, local_cond_dim=None,
            global_cond_dim=cond_dim + message_dim,
            diffusion_step_embed_dim=step_embed, down_dims=list(down_dims))

    def forward(self, x, t, ctx, message):
        return self.net(x, t, global_cond=torch.cat([ctx, message], dim=-1))


class MessageReconciler(nn.Module):
    """``reconcile(prop, wrench, nominal_t, m)`` — the fast level's message branch."""

    def __init__(self, prop_dim: int, wrench_dim: int, act_dim: int,
                 message_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp([prop_dim + wrench_dim + act_dim + message_dim,
                        hidden, hidden, act_dim])

    def forward(self, prop, wrench, nominal_t, message):
        return self.net(_cat(prop, wrench, nominal_t, message))
