"""Action-conditioned delta dynamics and the ``(S, U)`` surprise channels.

Ported from ``cap_constraint_benchmark/liftoff_v6_image/dyn/dynamics_lo6.py``.
The network, the modality-specific heads, the Gaussian NLL, the log-sigma clamp
and the clip constant are that file's; what changes is that the state layout is
an argument instead of a module-level table, because the dressing rig's 28-dim
state is not cap's 59-dim one.

Why only S and U
----------------

dap's full triangle has five channels, three of which need the *nominal* action
or the correction effort::

    E = a_exec - a_nom
    C = mu(y, a_exec) - mu(y, a_nom)

Teleoperated demonstrations have no fast level, so ``a_exec == a_nom`` for every
recorded step and ``E`` is structurally zero.  ``D2_SURPRISE_ONLY`` is ``(S, U)``
and needs neither::

    nu = delta_real - mu(y, a_exec)
    S  = nu / (sigma(y, a_exec) + eps)
    U  = log sigma(y, a_exec)

Both are functions of the executed action and the observed next state, which any
demonstration set has.  So D2 is computable on the dressing data and D4/D6 are
not — the same reason cap's image round built D2 and stopped.

The train/inference shift, taken seriously up front
---------------------------------------------------

dap measured on drawer that ``S`` is the one channel that genuinely shifts
between training and rollout: its p99 went 2.66 -> 7.07 and the clip-reaching
rate 0.0027 % -> 0.575 %, a factor of 213.  The cause is structural and applies
here unchanged — the dynamics model is fitted on demonstration transitions and
then queried at states the *policy* visits, so ``mu`` is out of distribution and
the residual grows.  :data:`CLIP` is 3.0 for that reason, adopted from the start.

``U``'s problem is different and clipping does not fix it: part of the channel is
a constant wherever ``sigma`` sits on its floor.  :func:`saturation` measures
that fraction so it is reported rather than discovered later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

LOGSTD_MIN, LOGSTD_MAX = -7.0, 3.0
EPS = 1e-3
#: dap used 10.0 on drawer and measured a 213x blow-up in the clip-reaching rate
#: between training and rollout, against a training p99 of 2.66.  3.0 leaves the
#: training distribution essentially intact and folds the inference tail inside
#: the range the policy was trained on.
CLIP = 3.0

#: ``(name, (lo, hi), delta_width, kind)``.  ``kind`` is ``"linear"`` or
#: ``"quat"``; a quaternion is four numbers but three degrees of freedom, so its
#: delta is a relative rotation vector.  Subtracting quaternions componentwise
#: would make the delta discontinuous at the double cover and put a spurious
#: jump into every surprise that crosses it.
Modality = Tuple[str, Tuple[int, int], int, str]


def dressing_modalities(res, names) -> List[Modality]:
    """The modality table for a resolved dataset.

    Delegates to :func:`diffusion_policy.irum.fields.modalities` so the layout is
    derived from the fields that actually resolved, never written out twice.
    """
    from diffusion_policy.irum import fields as F

    return F.modalities(res, names)


def delta_dim(mods: Sequence[Modality]) -> int:
    return sum(w for _, _, w, _ in mods)


def state_dim(mods: Sequence[Modality]) -> int:
    return max(hi for _, (_, hi), _, _ in mods)


def _quat_rel_rotvec(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Rotation vector taking ``q0`` to ``q1``, sign-resolved to the short arc."""
    w0, v0 = q0[..., :1], q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    w = w1 * w0 + (v1 * v0).sum(-1, keepdims=True)
    v = -w1 * v0 + w0 * v1 - np.cross(v1, v0)
    s = np.where(w < 0, -1.0, 1.0)
    w, v = w * s, v * s
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(n, np.clip(w, -1.0, 1.0))
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9) * ang, 0.0)


def _torch_quat_rel_rotvec(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, v0 = q0[..., :1], q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    w = w1 * w0 + (v1 * v0).sum(-1, keepdim=True)
    v = -w1 * v0 + w0 * v1 - torch.cross(v1, v0, dim=-1)
    s = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    w, v = w * s, v * s
    n = v.norm(dim=-1, keepdim=True)
    ang = 2.0 * torch.atan2(n, w.clamp(-1.0, 1.0))
    return torch.where(n > 1e-9, v / n.clamp_min(1e-9) * ang, torch.zeros_like(v))


def _torch_quat_apply_rotvec(q0: torch.Tensor, rv: torch.Tensor) -> torch.Tensor:
    """Rotate ``q0`` by the rotation vector ``rv`` — the inverse of the delta.

    Needed to *advance* a state by a predicted delta, which the forward direction
    (:func:`torch_state_delta`) never has to do.  Without it a rolled-out
    trajectory would have to add rotation vectors to quaternions componentwise,
    which is not a rotation at all.
    """
    ang = rv.norm(dim=-1, keepdim=True)
    axis = rv / ang.clamp_min(1e-9)
    half = 0.5 * ang
    w1 = torch.cos(half)
    v1 = torch.where(ang > 1e-9, axis * torch.sin(half), torch.zeros_like(rv))
    w0, v0 = q0[..., :1], q0[..., 1:]
    # q = q_rel * q0
    w = w1 * w0 - (v1 * v0).sum(-1, keepdim=True)
    v = w1 * v0 + w0 * v1 + torch.cross(v1, v0, dim=-1)
    q = torch.cat([w, v], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def torch_apply_delta(y: torch.Tensor, d: torch.Tensor,
                      mods: Sequence[Modality]) -> torch.Tensor:
    """``y_next`` from a state and a **raw** (unnormalised) delta."""
    out = y.clone()
    off = 0
    for _, (lo, hi), w, kind in mods:
        part = d[..., off:off + w]
        if kind == "quat":
            out[..., lo:hi] = _torch_quat_apply_rotvec(y[..., lo:hi], part)
        else:
            out[..., lo:hi] = y[..., lo:hi] + part
        off += w
    return out


def state_delta(y0: np.ndarray, y1: np.ndarray,
                mods: Sequence[Modality]) -> np.ndarray:
    parts = []
    for _, (lo, hi), _, kind in mods:
        parts.append(_quat_rel_rotvec(y0[..., lo:hi], y1[..., lo:hi])
                     if kind == "quat" else y1[..., lo:hi] - y0[..., lo:hi])
    return np.concatenate(parts, axis=-1)


def torch_state_delta(y0: torch.Tensor, y1: torch.Tensor,
                      mods: Sequence[Modality]) -> torch.Tensor:
    parts = []
    for _, (lo, hi), _, kind in mods:
        parts.append(_torch_quat_rel_rotvec(y0[..., lo:hi], y1[..., lo:hi])
                     if kind == "quat" else y1[..., lo:hi] - y0[..., lo:hi])
    return torch.cat(parts, dim=-1)


class DeltaDynamics(nn.Module):
    """Probabilistic MLP over the normalised state *change*, conditioned on action.

    Delta rather than absolute prediction is the fix that mattered in dap: on
    Square, absolute prediction left the dynamics **worse than "next = current"**
    (-155 % against the trivial baseline) and the message it fed encoded model
    error rather than physical evidence.  Switching to deltas took it to +37.5 %
    over that baseline and moved rollout success 0.313 -> 0.393 in lockstep.
    Consecutive states are autocorrelated, so absolute prediction spends the
    network's capacity learning the identity map first.

    Modality-specific mean and log-sigma heads so a badly-scaled group cannot
    dominate a shared one.  No recurrence: a hidden state would make the runtime
    message stateful, which the fast level is required not to be.
    """

    def __init__(self, mods: Sequence[Modality], act_dim: int,
                 hidden: int = 256, use_dt: bool = True):
        super().__init__()
        self.mods = list(mods)
        self.state_dim = state_dim(self.mods)
        self.delta_dim = delta_dim(self.mods)
        self.act_dim = int(act_dim)
        #: This rig does not run at a fixed rate, and a state *change* means
        #: something different over 0.05 s than over 0.75 s.  Without dt the
        #: model has to average over the timing distribution, and the residual
        #: it leaves behind — which is exactly the surprise channel — would be
        #: dominated by how long the step happened to take rather than by what
        #: the arm ran into.
        self.use_dt = bool(use_dt)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim + self.act_dim + int(self.use_dt), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.ModuleDict({n: nn.Linear(hidden, w)
                                 for n, _, w, _ in self.mods})
        self.logstd = nn.ModuleDict({n: nn.Linear(hidden, w)
                                     for n, _, w, _ in self.mods})
        for head in self.logstd.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, y: torch.Tensor, a: torch.Tensor,
                dt: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = [y, a]
        if self.use_dt:
            assert dt is not None, "this dynamics was built with use_dt=True"
            parts.append(dt)
        h = self.trunk(torch.cat(parts, -1))
        mu = torch.cat([self.mu[n](h) for n, _, _, _ in self.mods], -1)
        ls = torch.cat([self.logstd[n](h) for n, _, _, _ in self.mods], -1)
        return mu, ls.clamp(LOGSTD_MIN, LOGSTD_MAX)


def gaussian_nll(mu, logstd, target) -> torch.Tensor:
    inv = torch.exp(-2.0 * logstd)
    return (0.5 * ((target - mu) ** 2) * inv + logstd).mean()


def no_change_baseline(delta: np.ndarray) -> float:
    """MSE of predicting zero change, in normalised delta units.

    dap's hardest-won diagnostic: log the dynamics MSE **against this**, always.
    A negative NLL can be driven down by shrinking sigma while the point
    predictions stay worse than doing nothing, and that is exactly what hid a
    broken dynamics model for a whole round.
    """
    return float(np.mean(np.asarray(delta, np.float64) ** 2))


class Normaliser:
    """Mean/std over states and over deltas, fitted on the train split only."""

    def __init__(self, y: Optional[np.ndarray] = None,
                 d: Optional[np.ndarray] = None,
                 state: Optional[Dict[str, Any]] = None):
        if state is not None:
            self.ym, self.ys = (np.asarray(state["ym"], np.float32),
                                np.asarray(state["ys"], np.float32))
            self.dm, self.ds = (np.asarray(state["dm"], np.float32),
                                np.asarray(state["ds"], np.float32))
            return
        f = lambda a: a.reshape(-1, a.shape[-1]).astype(np.float64)
        self.ym = f(y).mean(0).astype(np.float32)
        self.ys = np.maximum(f(y).std(0), 1e-3).astype(np.float32)
        self.dm = f(d).mean(0).astype(np.float32)
        self.ds = np.maximum(f(d).std(0), 1e-6).astype(np.float32)

    def y(self, a):
        return ((np.asarray(a, np.float32) - self.ym) / self.ys).astype(np.float32)

    def d(self, a):
        return ((np.asarray(a, np.float32) - self.dm) / self.ds).astype(np.float32)

    def torch_y(self, t):
        return (t - torch.as_tensor(self.ym, device=t.device)) / \
            torch.as_tensor(self.ys, device=t.device)

    def torch_d(self, t):
        return (t - torch.as_tensor(self.dm, device=t.device)) / \
            torch.as_tensor(self.ds, device=t.device)

    def state_dict(self):
        return {k: getattr(self, k).tolist() for k in ("ym", "ys", "dm", "ds")}

    @classmethod
    def from_state(cls, s):
        return cls(state=s)


class FrozenDynamics(nn.Module):
    """A trained :class:`DeltaDynamics` plus its normalisers, in one object."""

    def __init__(self, model: DeltaDynamics, norm: Normaliser):
        super().__init__()
        self.model = model
        self.norm = norm

    @torch.no_grad()
    def forward(self, y_raw: torch.Tensor, a: torch.Tensor,
                dt: Optional[torch.Tensor] = None):
        return self.model(self.norm.torch_y(y_raw), a, dt)

    @torch.no_grad()
    def normalise_delta(self, d_raw: torch.Tensor) -> torch.Tensor:
        return self.norm.torch_d(d_raw)


@torch.no_grad()
def surprise(dyn: FrozenDynamics, y: torch.Tensor, y_next: torch.Tensor,
             a_exec: torch.Tensor, dt: Optional[torch.Tensor] = None
             ) -> Dict[str, torch.Tensor]:
    """The ``(S, U)`` pair — ``D2_SURPRISE_ONLY``'s two channels.

    Shapes are ``[..., D]``; leading dimensions are flattened and restored, so
    one implementation serves both a ``[B, W, ...]`` training batch and a
    ``[1, W, ...]`` rollout window and they cannot drift apart.
    """
    lead = y.shape[:-1]
    f = lambda x: x.reshape(-1, x.shape[-1])
    yf, ynf, ae = f(y), f(y_next), f(a_exec)
    mu, ls = dyn(yf, ae, f(dt) if dt is not None else None)
    d_real = dyn.normalise_delta(torch_state_delta(yf, ynf, dyn.model.mods))
    nu = d_real - mu
    out = {
        "S": (nu / (torch.exp(ls) + EPS)).clamp(-CLIP, CLIP),
        "U": ls,                        # already clamped by the model
        "NU_RAW": nu,
        "S_RAW": nu / (torch.exp(ls) + EPS),
    }
    return {k: v.reshape(*lead, v.shape[-1]) for k, v in out.items()}


def saturation(u: np.ndarray) -> Dict[str, float]:
    """How much of ``U`` is a constant rather than information.

    ``sigma`` on ``LOGSTD_MIN`` means the model is as confident as it is allowed
    to be, and every such entry carries the same number whatever the state.  dap
    measured 6.68 % in training and 4.86 % at inference on drawer and reported it
    rather than fixing it, because the fix is either a lower floor or a better
    dynamics model and choosing between them needs this number first.
    """
    u = np.asarray(u, np.float64)
    return {"frac_at_floor": float(np.mean(u <= LOGSTD_MIN + 1e-6)),
            "frac_at_ceiling": float(np.mean(u >= LOGSTD_MAX - 1e-6)),
            "mean_abs": float(np.abs(u).mean()),
            "logstd_min": LOGSTD_MIN, "logstd_max": LOGSTD_MAX}


def clip_report(s_raw: np.ndarray) -> Dict[str, float]:
    """Where the surprise distribution sits relative to :data:`CLIP`."""
    a = np.abs(np.asarray(s_raw, np.float64))
    return {"clip": CLIP, "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
            "frac_reaching_clip": float(np.mean(a >= CLIP))}
