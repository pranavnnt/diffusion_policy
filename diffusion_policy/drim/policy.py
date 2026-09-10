"""``B0`` / ``B1`` / ``D2`` in one container, with dap's exact-null algebra.

Staged, each stage freezing the last::

    B0   slow flow policy alone:  vision + proprioception (+ wrench) -> chunk
    B1   + the bounded per-channel fast corrector, on a frozen B0
    D2   + the upward message, on a frozen B1.  ``D2_SURPRISE_ONLY``: the message
         reads the two dynamics channels ``S`` (standardised surprise) and ``U``
         (predicted log-sigma) over the causal window, and nothing else.

The exact-null guarantee is structural, not an initialisation trick::

    v(x, t, ctx, m) = v_base(x, t, ctx) + gate * (dv(x,t,ctx,m) - dv(x,t,ctx,0))

At ``m = 0`` the bracket is identically zero for any parameter values, so ``D2``
with a null message **is** ``B1`` — bit for bit, forever.  The same construction
guards the fast level's message branch through ``reconcile_gate``.  This is what
makes "the message helped" a measurable claim rather than a comparison between
two separately trained networks.

Ported from ``cap_constraint_benchmark/liftoff_v4/models_lo4.py`` (the container
and the algebra) and ``liftoff_v6_image/models_lo6.py`` (the vision front end).
Every dimension comes from :class:`~diffusion_policy.drim.spec.DrimSpec`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

from diffusion_policy.drim import nets as N
from diffusion_policy.drim.augment import PhotometricJitter
from diffusion_policy.drim.spec import DrimSpec

VARIANTS = ("B0", "B1", "D2")
PARENT = {"B0": None, "B1": "B0", "D2": "B1"}


def build_vision(spec: DrimSpec, random_crop: bool = True,
                 weights: Optional[str] = None) -> MultiImageObsEncoder:
    """This repository's own encoder stack: resnet18, group norm, random crop.

    Nothing new is installed and nothing is re-implemented — dap reached for the
    same ``MultiImageObsEncoder`` from this repo, so the vision front end is
    literally shared code rather than two implementations that can drift.

    Group norm rather than batch norm: the batch here is chunks drawn from a
    handful of episodes, which is exactly the correlated-batch structure batch
    norm handles badly, and EMA over batch-norm statistics is worse still.

    ``weights`` is ``None`` by default, matching dap — the encoder is trained
    from scratch. Worth knowing rather than assuming: with a few hundred chunks
    an 11.2 M-parameter backbone learned from nothing is likely the binding
    constraint, and ``imagenet_norm=True`` below normalises the input as if for
    a pretrained one regardless. ``"IMAGENET1K_V1"`` uses the pretrained
    weights; it changes what the frozen trunk means, so it is a decision to
    make once and record, not a knob to turn between runs.
    """
    c, h, w = spec.image_shape
    shape_meta = {"obs": {k: {"shape": (c, h, w), "type": "rgb"}
                          for k in spec.cameras}}
    return MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet("resnet18", weights=weights),
        crop_shape=tuple(spec.crop_shape),
        random_crop=random_crop,
        use_group_norm=True,
        share_rgb_model=True,
        imagenet_norm=True,
    )


class DrimPolicy(nn.Module):
    """One object for all three stages.

    ``core`` nests a stage on its trained parent: ``B1`` reuses ``B0``'s slow
    policy and vision, ``D2`` reuses ``B1``'s everything.  Nesting rather than
    re-instantiating is what makes the stage contrast a contrast — the frozen
    parts are the *same tensors*, not a reload that could differ.
    """

    def __init__(self, variant: str, spec: DrimSpec,
                 core: Optional["DrimPolicy"] = None,
                 message_in_dims: Optional[Tuple[int, int]] = None,
                 freeze_base: bool = True,
                 photometric: Optional[PhotometricJitter] = None,
                 vision_weights: Optional[str] = None):
        super().__init__()
        if variant not in VARIANTS:
            raise KeyError(f"unknown variant {variant!r}; have {VARIANTS}")
        self.variant = variant
        self.spec = spec
        self.horizon = spec.pred_horizon
        self.exec_horizon = spec.exec_horizon
        self.act_dim = spec.act_dim
        self.message_dim = spec.message_dim

        # -- vision ---------------------------------------------------------
        if spec.is_image:
            self.vision = (core.vision if core is not None
                           else build_vision(spec, weights=vision_weights))
            #: Inherited with the vision encoder: a later stage that augmented
            #: differently from the frozen trunk it nests on would be measuring
            #: the augmentation, not the stage.
            self.photometric = (core.photometric if core is not None
                                else (photometric or PhotometricJitter()))
        else:
            self.vision = None
            self.photometric = None

        # -- slow -----------------------------------------------------------
        #: The flow field must exist at the final frame width *before* anything
        #: else is built.  dap built the core first and swapped ``slow``
        #: afterwards, which left the residual field sized for the state context
        #: while the checkpoint carried the visual one — a mismatch only a fresh
        #: load exposed.
        self.slow = (core.slow if core is not None
                     else N.FlowChunkPolicy(
                         frame_dim=spec.frame_dim, act_dim=spec.act_dim,
                         horizon=spec.pred_horizon,
                         n_obs_steps=spec.n_obs_steps))

        # -- fast -----------------------------------------------------------
        if variant == "B0":
            self.fast = None
        elif core is not None and getattr(core, "fast", None) is not None:
            self.fast = core.fast
        else:
            self.fast = N.FastCorrector(spec.prop_dim, spec.wrench_dim,
                                        spec.act_dim, spec.limits())

        # -- message --------------------------------------------------------
        self.encoder = self.dv = self.reconcile = None
        self.cond_dim = 0
        if variant == "D2":
            if message_in_dims is None:
                raise ValueError(
                    "D2 needs message_in_dims=(S_width, U_width); they are the "
                    "delta width of the dynamics model the channels come from")
            s_dim, u_dim = message_in_dims
            self.message_in_dims = (int(s_dim), int(u_dim))
            #: Same class, same message dimension, same gated route as dap's
            #: ``B2_OBS`` — only the encoder's *inputs* change, ``S`` where the
            #: state stack put the observation history and ``U`` where it put the
            #: wrench history.  That is what makes this an input-class contrast
            #: rather than a different method.
            self.encoder = N.MessageEncoder(s_dim, u_dim, spec.act_dim,
                                            message_dim=spec.message_dim)
            self.cond_dim = spec.message_dim
            self.dv = N.ResidualVectorField(spec.act_dim, self.slow.context_dim,
                                            self.cond_dim)
            self.reconcile = N.MessageReconciler(spec.prop_dim, spec.wrench_dim,
                                                 spec.act_dim, self.cond_dim)
            self.gate = nn.Parameter(torch.zeros(1))
            self.reconcile_gate = nn.Parameter(torch.zeros(1))

        if core is not None and freeze_base:
            for p in self.slow.parameters():
                p.requires_grad_(False)
            if self.vision is not None:
                for p in self.vision.parameters():
                    p.requires_grad_(False)
            if self.fast is not None and variant == "D2":
                for p in self.fast.parameters():
                    p.requires_grad_(False)

    # ------------------------------------------------------------------ context

    def encode_frames(self, rgb: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``{camera: [B, n_obs, H, W, 3] uint8} -> [B, n_obs, vision_dim*n_cam]``."""
        any_cam = next(iter(rgb.values()))
        b, n = any_cam.shape[:2]
        flat = {}
        for k in self.spec.cameras:
            x = rgb[k]
            x = x.reshape(b * n, *x.shape[2:])
            if x.shape[-1] == 3:                       # [B*n, H, W, 3] -> NCHW
                x = x.permute(0, 3, 1, 2)
            x = x.float() / 255.0
            if self.photometric is not None:
                x = self.photometric(x)
            flat[k] = x
        f = self.vision(flat)
        return f.reshape(b, n, -1)

    def context(self, rgb: Optional[Dict[str, torch.Tensor]],
                prop: torch.Tensor,
                wrench: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, n_obs, *] -> [B, context_dim]``, the flow policy's conditioning."""
        parts = []
        if self.vision is not None:
            parts.append(self.encode_frames(rgb))
        parts.append(prop)
        if wrench is not None and wrench.shape[-1] > 0:
            parts.append(wrench)
        return self.slow.encode(torch.cat(parts, dim=-1))

    # ------------------------------------------------------------------ message

    def raw_conditioning(self, batch) -> torch.Tensor:
        return self.encoder(batch["hist_S"], batch["hist_U"])

    def conditioning(self, batch, donor_cond: Optional[torch.Tensor] = None,
                     force_null: bool = False) -> torch.Tensor:
        """The message actually used, after the structural-null mask.

        ``msg_valid`` is a function of the decision step alone — 1 only once a
        full causal window exists behind it — so every conditioned variant shares
        ``B1``'s pre-message prefix *exactly* and the comparison is confined to
        the continuation.
        """
        b = batch["prop2"]
        if force_null or self.cond_dim == 0:
            return torch.zeros(len(b), max(self.cond_dim, 1), device=b.device)
        c = donor_cond if donor_cond is not None else self.raw_conditioning(batch)
        v = batch.get("msg_valid")
        if v is not None:
            c = c * v.view(-1, 1)
        return c

    # ------------------------------------------------------------------ flow

    def velocity(self, x, t_scaled, ctx, message: Optional[torch.Tensor]):
        v = self.slow.net(x, t_scaled, global_cond=ctx)
        if self.dv is not None and message is not None:
            null = torch.zeros_like(message)
            v = v + self.gate * (self.dv(x, t_scaled, ctx, message)
                                 - self.dv(x, t_scaled, ctx, null))
        return v

    def flow_loss(self, ctx, chunk, message: Optional[torch.Tensor] = None,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Rectified flow: predict ``x1 - x0`` along the straight path."""
        x0 = torch.randn(chunk.shape, device=chunk.device, dtype=chunk.dtype,
                         generator=generator)
        t = torch.rand(len(chunk), device=chunk.device, dtype=chunk.dtype,
                       generator=generator)
        xt = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * chunk
        pred = self.velocity(xt, t * self.slow.n_flow_steps, ctx, message)
        return nn.functional.mse_loss(pred, chunk - x0)

    @torch.no_grad()
    def sample_chunk(self, ctx, message: Optional[torch.Tensor] = None,
                     noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Euler-integrate the flow from ``noise``.

        ``noise`` is used, not ignored: paired diagnostics need two chunks to
        share one draw, and dap lost time twice to comparisons where the sampler
        moved more than the effect being measured.
        """
        b = len(ctx)
        if noise is None:
            noise = torch.randn(b, self.horizon, self.act_dim,
                                device=ctx.device, dtype=ctx.dtype)
        x = noise
        n = self.slow.n_flow_steps
        dt = 1.0 / n
        for k in range(n):
            t = torch.full((b,), k * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity(x, t * n, ctx, message)
        return x.clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ fast

    def residual_prefix(self, nominal, step_prop, step_wrench=None,
                        message: Optional[torch.Tensor] = None,
                        n: Optional[int] = None) -> torch.Tensor:
        """Differentiable residual for the **executed** prefix only.

        ``pred/exec`` means the tail of every predicted chunk is discarded, so
        training the fast level on it would fit the corrector to actions the
        environment never receives.  The chunk summary features are still
        computed over the whole prediction, because that is what the corrector
        sees at rollout time.
        """
        n = int(n or self.exec_horizon)
        b, h, a = nominal.shape
        if self.fast is None:
            return torch.zeros_like(nominal[:, :n])
        mean, remaining, phase = N.FastCorrector.chunk_features(nominal)
        o = step_prop[:, :n].reshape(b * n, -1)
        w = (step_wrench[:, :n].reshape(b * n, -1)
             if step_wrench is not None and step_wrench.shape[-1] > 0 else None)
        nt = nominal[:, :n].reshape(b * n, a)
        r = self.fast(o, w, nt, mean[:, :n].reshape(b * n, a),
                      remaining[:, :n].reshape(b * n, a),
                      phase[:, :n].reshape(b * n, 1)).view(b, n, a)
        if self.reconcile is not None and message is not None:
            m = message.unsqueeze(1).expand(b, n, message.shape[-1]).reshape(b * n, -1)
            z = torch.zeros_like(m)
            d = (self.reconcile(o, w, nt, m)
                 - self.reconcile(o, w, nt, z)).view(b, n, a)
            r = torch.clamp(r + self.reconcile_gate * d,
                            -self.fast.max_residual, self.fast.max_residual)
        return r

    @torch.no_grad()
    def residual_step(self, feats, t: int, prop, wrench=None,
                      message: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One executed step, for a rollout loop."""
        if self.fast is None:
            return torch.zeros(len(prop), self.act_dim, device=prop.device)
        nominal, mean, remaining, phase = feats
        r = self.fast(prop, wrench, nominal[:, t], mean[:, t], remaining[:, t],
                      phase[:, t])
        if self.reconcile is not None and message is not None:
            z = torch.zeros_like(message)
            d = (self.reconcile(prop, wrench, nominal[:, t], message)
                 - self.reconcile(prop, wrench, nominal[:, t], z))
            r = torch.clamp(r + self.reconcile_gate * d,
                            -self.fast.max_residual, self.fast.max_residual)
        return r

    @staticmethod
    def step_features(nominal: torch.Tensor):
        return (nominal,) + N.FastCorrector.chunk_features(nominal)

    # ------------------------------------------------------------------ report

    def n_params(self) -> Dict[str, int]:
        def c(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0
        return {"vision": c(self.vision), "slow": c(self.slow),
                "fast": c(self.fast), "encoder": c(self.encoder),
                "dv": c(self.dv), "reconcile": c(self.reconcile),
                "trainable": sum(p.numel() for p in self.parameters()
                                 if p.requires_grad),
                "total": c(self)}

    def gate_norms(self) -> Dict[str, float]:
        if self.dv is None:
            return {}
        return {"gate": float(self.gate.detach().abs().item()),
                "reconcile_gate": float(self.reconcile_gate.detach().abs().item())}

    def authority(self) -> Optional[np.ndarray]:
        if self.fast is None:
            return None
        return self.fast.max_residual.detach().cpu().numpy().astype(float)

    def assert_authority(self, spec: DrimSpec) -> None:
        """Refuse a corrector whose ceilings are not the ones declared.

        dap hit this the expensive way: a nested load dropped ``max_residual``
        because it is a buffer on one side and a plain float on the other, and a
        whole round ran at the wrong authority while reporting the right one.
        ``load_state_dict(strict=False)`` — which a nested load must use — says
        nothing when it happens, so the check has to be explicit.
        """
        if self.fast is None:
            return
        want = spec.limits()
        got = self.authority()
        assert np.allclose(got, want, atol=1e-6), (
            f"corrector carries ceilings {got}, spec declares {want}")


def build(variant: str, spec: DrimSpec, core: Optional[DrimPolicy] = None,
          message_in_dims: Optional[Tuple[int, int]] = None,
          freeze_base: bool = True,
          photometric: Optional[PhotometricJitter] = None,
          vision_weights: Optional[str] = None) -> DrimPolicy:
    return DrimPolicy(variant, spec, core=core,
                      message_in_dims=message_in_dims, freeze_base=freeze_base,
                      photometric=photometric, vision_weights=vision_weights)
