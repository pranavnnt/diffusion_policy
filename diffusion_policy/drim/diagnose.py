"""Offline checks to run before a policy ever reaches the robot.

Two different jobs, deliberately not mixed:

**Trajectory divergence** (:func:`divergence`) asks *does error compound*.
Single-step action MSE cannot answer that — it is measured at demonstration
states, and the thing that breaks a behaviour-cloned policy is visiting its own.
So the policy is rolled forward through the **learned dynamics**: it replans
every ``exec_horizon`` steps exactly as it would on the robot, the fast level
corrects against the *rolled* proprioception rather than the demonstrated one,
and the rolled state is compared with what the demonstration did.

Two honesty requirements, both enforced here rather than left to the reader:

* **Vision is teacher-forced.** The dynamics models low-dimensional state, not
  pixels, so each replan sees the demonstration's camera frames.  Real drift
  would move the images too, so every number here is **optimistic**.
* **The dynamics has its own error**, and a divergence that is really the
  model's would look exactly like a divergence that is the policy's.  So the
  same rollout is run a second time driven by the **demonstrated actions**
  (:func:`divergence`'s ``replay`` row).  That is the floor: whatever it
  diverges by is the model, not the policy.  Only the gap above it is evidence
  about the policy.

**Failure detectors** (:func:`detectors`) ask *is anything broken* — a different
question with a much better answer rate.  None of them predicts success; each
catches a specific way dap's rounds went wrong:

``dynamics_skill``     negative means the model is worse than predicting no
                       change, and the message is then carrying model error.
                       dap ran a whole round in this state (-155 %).
``surprise_shift``     ``S`` is fitted on demonstration transitions and read at
                       states the policy visits.  On drawer its p99 went
                       2.66 -> 7.07 and its clip rate 0.0027 % -> 0.575 %, a
                       factor of 213.  This measures the same thing here by
                       recomputing ``S`` along the rolled trajectory.
``saturation``         the corrector pinned to its ceiling has been given less
                       authority than the task needs.
``message_reliance``   how much the conditioned policy's output moves when the
                       message is zeroed.  **Reliance, not value**: zeroing an
                       input the model trained with is out of distribution, and
                       dap measured 99 % reliance on a message that was worth
                       *negative* success.  Read it only as "is the route live".
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_policy.drim import dynamics as DY
from diffusion_policy.drim import fields as F
from diffusion_policy.drim.spec import DrimSpec


def _ctx(model, rgb, prop2, wrench2):
    return model.context(rgb, prop2, wrench2)


@torch.no_grad()
def rollout(model, dyn: DY.FrozenDynamics, mods, eps, norm, spec: DrimSpec,
            episodes: Sequence[int], n_steps: int = 32, n_starts: int = 64,
            device: str = "cpu", seed: int = 0, use_demo_actions: bool = False,
            message_of: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """Roll the policy forward through the dynamics; return per-step divergence.

    ``use_demo_actions`` drives the same loop with the demonstrated action, which
    is the model-error floor every policy number must be read against.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    starts: List[Tuple[int, int]] = []
    for j in episodes:
        n = len(eps.episodes[j]["dyn"])
        hi = n - n_steps - 1
        if hi <= spec.n_obs_steps:
            continue
        starts += [(j, int(s)) for s in
                   rng.choice(np.arange(1, hi), size=min(hi - 1, 16),
                              replace=False)]
    if not starts:
        return {}
    rng.shuffle(starts)
    starts = starts[:n_starts]

    P, W = spec.prop_dim, spec.wrench_dim
    chans = list(spec.act_channels) or list(range(spec.act_dim))
    div = np.zeros((len(starts), n_steps), np.float64)
    adiv = np.zeros((len(starts), n_steps), np.float64)
    s_rolled: List[np.ndarray] = []

    for i, (j, s0) in enumerate(starts):
        ep = eps.episodes[j]
        y = torch.as_tensor(ep["dyn"][s0:s0 + 1], dtype=torch.float32, device=device)
        chunk = None
        for k in range(n_steps):
            t = s0 + k
            if k % spec.exec_horizon == 0:
                #: replan.  Vision is the demonstration's — see the module note.
                fr = [max(t + spec.slow_offsets[0], 0), t]
                rgb = ({c: torch.as_tensor(ep[c][fr][None], device=device)
                        for c in spec.cameras} if spec.is_image else None)
                prop2 = torch.as_tensor(
                    norm.apply_vec("prop2", ep["prop"][fr][None]), device=device)
                #: the *current* frame's proprioception is the rolled state, not
                #: the demonstration's — that is the whole point of the rollout
                prop2[:, -1] = torch.as_tensor(
                    norm.apply_vec("prop2", y[:, :P].cpu().numpy()), device=device)
                w2 = torch.as_tensor(
                    norm.apply_vec("wrench2", ep["wrench"][fr][None]), device=device)
                if W:
                    w2[:, -1] = torch.as_tensor(
                        norm.apply_vec("wrench2", y[:, P:].cpu().numpy()),
                        device=device)
                ctx = _ctx(model, rgb, prop2, w2)
                msg = message_of(model, ep, t) if message_of is not None else None
                chunk = model.sample_chunk(ctx, msg, None)
            idx = k % spec.exec_horizon
            nominal = chunk[:, idx]
            sp = torch.as_tensor(norm.apply_vec("step_prop", y[:, :P].cpu().numpy()),
                                 device=device)
            sw = (torch.as_tensor(norm.apply_vec("step_wrench", y[:, P:].cpu().numpy()),
                                  device=device) if W else None)
            feats = model.step_features(chunk)
            r = model.residual_step(feats, idx, sp, sw, None)
            a_norm = torch.clamp(nominal + r, -1.0, 1.0)
            a_raw = torch.as_tensor(
                norm.invert_vec("target", a_norm.cpu().numpy()), device=device)
            if use_demo_actions:
                a_raw = torch.as_tensor(ep["action"][t:t + 1][:, chans],
                                        dtype=torch.float32, device=device)
            dt = torch.as_tensor(ep["dt"][t:t + 1], dtype=torch.float32, device=device)
            mu, _ = dyn(y, a_raw, dt)
            d_raw = mu * torch.as_tensor(dyn.norm.ds, device=device) + \
                torch.as_tensor(dyn.norm.dm, device=device)
            y_next = DY.torch_apply_delta(y, d_raw, mods)
            truth = torch.as_tensor(ep["dyn"][t + 1:t + 2], dtype=torch.float32,
                                    device=device)
            #: divergence in **normalised** state units, so channels with
            #: different physical scales contribute comparably
            sd = torch.as_tensor(dyn.norm.ys, device=device)
            div[i, k] = float(((y_next - truth) / sd).norm(dim=-1).item())
            adiv[i, k] = float((a_norm - torch.as_tensor(
                norm.apply_vec("target", ep["action"][t:t + 1][:, chans]),
                device=device)).norm(dim=-1).item())
            s_rolled.append(y_next.cpu().numpy())
            y = y_next
    model.train()
    return {"state_divergence": div, "action_divergence": adiv,
            "rolled_states": np.concatenate(s_rolled) if s_rolled else np.zeros((0, 1))}


def divergence(model, dyn, mods, eps, norm, spec, episodes, device="cpu",
               n_steps: int = 32, n_starts: int = 64, seed: int = 0
               ) -> Dict[str, Any]:
    """Policy divergence against the demonstrated-action floor.

    The reported quantity is the **gap**: how much worse the policy's own actions
    keep the state on the demonstrated trajectory than replaying the
    demonstration through the same model. A policy at the floor is not
    necessarily good, but a policy far above it is compounding.
    """
    pol = rollout(model, dyn, mods, eps, norm, spec, episodes, n_steps,
                  n_starts, device, seed, use_demo_actions=False)
    rep = rollout(model, dyn, mods, eps, norm, spec, episodes, n_steps,
                  n_starts, device, seed, use_demo_actions=True)
    if not pol:
        return {}
    marks = sorted({1, spec.exec_horizon, 2 * spec.exec_horizon, n_steps})
    out: Dict[str, Any] = {"n_starts": int(len(pol["state_divergence"])),
                           "n_steps": n_steps, "at": {}}
    for m in marks:
        if m > n_steps:
            continue
        p = pol["state_divergence"][:, m - 1]
        r = rep["state_divergence"][:, m - 1]
        out["at"][f"step{m}"] = {
            "policy_median": float(np.median(p)),
            "replay_median": float(np.median(r)),
            "gap_median": float(np.median(p) - np.median(r)),
            "policy_p90": float(np.percentile(p, 90)),
        }
    d1 = np.median(pol["state_divergence"][:, 0])
    dn = np.median(pol["state_divergence"][:, -1])
    out["compounding_factor"] = float(dn / max(d1, 1e-9))
    out["action_divergence_median"] = float(
        np.median(pol["action_divergence"]))
    out["note"] = ("vision is teacher-forced from the demonstration, so these "
                   "are optimistic; read the policy row against the replay row, "
                   "which is the dynamics model's own error")
    return out


@torch.no_grad()
def surprise_shift(dyn, mods, eps, norm, spec, rolled: np.ndarray,
                   episodes: Sequence[int], device: str = "cpu") -> Dict[str, Any]:
    """``S`` on demonstration transitions vs on the states the policy reaches.

    dap's single most expensive surprise: the channel the whole message is made
    of moves between the two, and by a lot.  A large shift here means D2 is
    conditioned at inference on values it never saw in training.
    """
    ys, yns, acts, dts = [], [], [], []
    chans = list(spec.act_channels) or list(range(spec.act_dim))
    for j in episodes:
        ep = eps.episodes[j]
        ys.append(ep["dyn"][:-1]); yns.append(ep["dyn"][1:])
        acts.append(ep["action"][:-1][:, chans]); dts.append(ep["dt"][:-1])
    f = lambda xs: torch.as_tensor(np.concatenate(xs), dtype=torch.float32,
                                   device=device)
    train = DY.surprise(dyn, f(ys), f(yns), f(acts), f(dts))
    out = {"demonstration": DY.clip_report(train["S_RAW"].cpu().numpy()),
           "saturation_U": DY.saturation(train["U"].cpu().numpy())}
    if rolled is not None and len(rolled) > 2:
        r = torch.as_tensor(rolled, dtype=torch.float32, device=device)
        a = f(acts)[:len(r) - 1]
        d = f(dts)[:len(r) - 1]
        roll = DY.surprise(dyn, r[:-1][:len(a)], r[1:][:len(a)], a, d)
        out["rolled"] = DY.clip_report(roll["S_RAW"].cpu().numpy())
        dm, rm = out["demonstration"], out["rolled"]
        out["p99_ratio"] = float(rm["p99"] / max(dm["p99"], 1e-9))
        out["clip_rate_ratio"] = float(
            rm["frac_reaching_clip"] / max(dm["frac_reaching_clip"], 1e-9)
            if dm["frac_reaching_clip"] > 0 else float("inf")
            if rm["frac_reaching_clip"] > 0 else 1.0)
    return out


@torch.no_grad()
def message_reliance(model, batch, message_of) -> Dict[str, Any]:
    """How far the conditioned output moves when the message is zeroed.

    **Reliance, not value.**  Zeroing an input the model was trained with puts it
    out of distribution, and dap measured 99.15 % reliance on a message worth
    negative success.  Useful for one thing only: confirming the route is live.
    """
    if model.cond_dim == 0:
        return {}
    model.eval()
    ctx = model.context(
        {c: batch[f"rgb2_{c}"] for c in model.spec.cameras}
        if model.spec.is_image else None, batch["prop2"], batch.get("wrench2"))
    noise = torch.zeros(len(batch["target"]), model.horizon, model.act_dim,
                        device=ctx.device)
    msg = message_of(model, batch)
    zero = torch.zeros_like(msg)
    with_msg = model.sample_chunk(ctx, msg, noise)
    without = model.sample_chunk(ctx, zero, noise)
    model.train()
    scale = with_msg.abs().mean().clamp_min(1e-9)
    return {"mean_abs_shift": float((with_msg - without).abs().mean().item()),
            "relative_shift": float(((with_msg - without).abs().mean() / scale).item()),
            "note": "reliance, not value; the honest value measure is D2 vs B1 "
                    "at equal budget on a real rollout"}


def detectors(summary: Dict[str, Any]) -> List[str]:
    """The one-line verdicts worth reading before anything reaches the robot."""
    out: List[str] = []
    dyn = summary.get("dyn") or {}
    if "skill" in dyn:
        s = dyn["skill"]
        out.append(f"{'OK  ' if s > 0 else 'FAIL'} dynamics skill {s:+.1%} vs "
                   f"no-change baseline"
                   + ("" if s > 0 else "  <-- the message is carrying model error"))
    b1 = (summary.get("B1") or {}).get("saturation")
    if b1 is not None:
        v = "OK  " if 0.001 <= b1 <= 0.5 else "WARN"
        out.append(f"{v} corrector saturation {b1:.1%}"
                   + ("  <-- never reaches its ceiling; authority unused"
                      if b1 < 0.001 else
                      "  <-- pinned; authority too small" if b1 > 0.5 else ""))
    d2 = summary.get("D2") or {}
    if "exact_null" in d2:
        ok = d2["exact_null"]["passed"]
        out.append(f"{'OK  ' if ok else 'FAIL'} exact-null identity "
                   f"(max |D2(m=0) - B1| = {d2['exact_null']['max_abs_diff']:.1e})")
    sh = (summary.get("diagnostics") or {}).get("surprise_shift") or {}
    if "p99_ratio" in sh:
        r = sh["p99_ratio"]
        out.append(f"{'OK  ' if r < 2 else 'WARN'} surprise p99 shift x{r:.1f} "
                   f"demonstration -> rolled"
                   + ("" if r < 2 else "  <-- D2 conditions on values it never "
                                       "saw in training"))
    tb = summary.get("trivial_action_baselines") or {}
    if tb:
        best = None
        for st in ("D2", "B1", "B0"):
            c = (summary.get(st) or {}).get("curve") or []
            am = [r["action_mse"] for r in c if "action_mse" in r]
            if am:
                best = min(am) if best is None else min(best, min(am))
        if best is not None:
            copy = tb["repeat_first_action"]
            ok = best < copy
            out.append(f"{'OK  ' if ok else 'FAIL'} best action_mse {best:.5f} vs "
                       f"copycat {copy:.5f}"
                       + ("" if ok else "  <-- holding the previous action beats "
                                        "every trained stage; this action target "
                                        "is degenerate"))
    dv = (summary.get("diagnostics") or {}).get("divergence") or {}
    if "compounding_factor" in dv:
        c = dv["compounding_factor"]
        out.append(f"{'OK  ' if c < 10 else 'WARN'} state divergence grows x{c:.1f} "
                   f"over {dv['n_steps']} steps")
    return out
