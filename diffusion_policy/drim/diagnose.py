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
    #: The same drift in metres.  A normalised norm says whether one policy is
    #: worse than another; millimetres say whether anyone should care.
    ee = _ee_slice(mods)
    eediv = np.zeros((len(starts), n_steps), np.float64)
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
            if ee is not None:
                eediv[i, k] = float(
                    (y_next[:, ee[0]:ee[1]] - truth[:, ee[0]:ee[1]]
                     ).norm(dim=-1).item())
            s_rolled.append(y_next.cpu().numpy())
            y = y_next
    model.train()
    return {"state_divergence": div, "action_divergence": adiv,
            "ee_divergence_m": eediv,
            "rolled_states": np.concatenate(s_rolled) if s_rolled else np.zeros((0, 1))}


def _ee_slice(mods) -> Optional[Tuple[int, int]]:
    """Where the first arm's end-effector position sits in the dynamics state."""
    for name, (lo, hi), _, _ in mods:
        if name.endswith("_ee_pos"):
            return (lo, hi)
    return None


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
            "policy_ee_mm": float(np.median(pol["ee_divergence_m"][:, m - 1]) * 1e3),
            "replay_ee_mm": float(np.median(rep["ee_divergence_m"][:, m - 1]) * 1e3),
            "seconds": round(m * float(getattr(eps, "dt_stats", {}).get("mean", 0.0)), 2),
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


# --------------------------------------------------------------------------- #
# standalone entry point: diagnose a run that has already been trained
# --------------------------------------------------------------------------- #


def diagnose_run(run_dir: str, data: Any, device: str = "cpu",
                 n_steps: int = 32, n_starts: int = 48, seed: int = 0,
                 log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run the checks against checkpoints already on disk.

    The same diagnostics the training loop ends with, decoupled from it. Two
    reasons that matters: a run trained before these existed can still be
    checked, and a checkpoint about to be driven on hardware can be re-checked
    without retraining the thing you are about to trust.
    """
    import json
    import os

    import torch

    from diffusion_policy.drim import dressing as DRESS
    from diffusion_policy.drim import policy as PL
    from diffusion_policy.drim import selection as SEL
    from diffusion_policy.drim.dataset import load_split
    from diffusion_policy.drim.spec import DrimSpec

    with open(os.path.join(run_dir, f"summary_seed{seed}.json")) as fh:
        summary = json.load(fh)
    saved = DrimSpec.from_dict(summary["spec"])
    prof = DRESS.profile(cameras=saved.cameras)
    tr, va, norm, eps, spec = load_split(
        data, cameras=saved.cameras, n_arms=max(saved.n_arms, 1),
        layout=prof["layout"], seed=seed, image_size=prof["image_size"],
        roi=prof.get("roi"), exo_key=prof["exo_key"],
        act_scale_per_arm=prof["act_scale_per_arm"],
        act_per_arm=prof["act_per_arm"], warn_scale=False,
        spec_kw=dict(action_mode=saved.action_mode,
                     pred_horizon=saved.pred_horizon,
                     exec_horizon=saved.exec_horizon,
                     message_window=saved.message_window))
    spec.assert_schema(saved)

    dyn_ck = torch.load(os.path.join(run_dir, f"dynamics_seed{seed}.pt"),
                        map_location=device, weights_only=False)
    mods = [(n, tuple(sl), w, k) for n, sl, w, k in dyn_ck["mods"]]
    m = DY.DeltaDynamics(mods, saved.act_dim, exo_dim=saved.exo_dim)
    m.load_state_dict(dyn_ck["state_dict"])
    m.eval()
    frozen = DY.FrozenDynamics(m, DY.Normaliser.from_state(dyn_ck["norm"])).to(device)

    model, parent = None, None
    for stage in ("B0", "B1", "D2"):
        path = os.path.join(run_dir, f"{stage}_seed{seed}.pt")
        if not os.path.exists(path):
            break
        kw = ({} if stage != "D2"
              else {"message_in_dims": (DY.delta_dim(mods), DY.delta_dim(mods))})
        model = PL.build(stage, saved, core=parent, **kw).to(device)
        SEL.load_selected(path, model, map_location=device)
        parent, loaded = model, stage
    log(f"[diagnose] {run_dir}: loaded up to {loaded}")

    va_eps = sorted({int(i) for i in va["episode_index"]})
    diag = {"divergence": divergence(model, frozen, mods, eps, norm, spec,
                                     va_eps, device=device, n_steps=n_steps,
                                     n_starts=n_starts, seed=seed)}
    roll = rollout(model, frozen, mods, eps, norm, spec, va_eps, n_steps,
                   n_starts, device, seed)
    diag["surprise_shift"] = surprise_shift(frozen, mods, eps, norm, spec,
                                            roll.get("rolled_states"), va_eps,
                                            device=device)
    summary["diagnostics"] = diag
    for k, v in (diag["divergence"].get("at") or {}).items():
        log(f"      {k:8s} ({v['seconds']:4.1f}s)  policy {v['policy_ee_mm']:7.2f} mm"
            f"   replay {v['replay_ee_mm']:7.2f} mm"
            f"   gap {v['policy_ee_mm'] - v['replay_ee_mm']:+7.2f} mm")
    sh = diag["surprise_shift"]
    if "p99_ratio" in sh:
        log(f"      surprise p99 x{sh['p99_ratio']:.2f}, clip rate "
            f"x{sh['clip_rate_ratio']:.1f}  (demonstration -> rolled)")
    for v in detectors(summary):
        log(f"      {v}")
    out = os.path.join(run_dir, f"diagnostics_seed{seed}.json")
    with open(out, "w") as fh:
        json.dump(diag, fh, indent=2, default=str)
    log(f"[diagnose] wrote {out}")
    return diag


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=diagnose_run.__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="a training output directory")
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--starts", type=int, default=48)
    a = ap.parse_args(argv)
    diagnose_run(a.run, a.data, device=a.device, seed=a.seed,
                 n_steps=a.steps, n_starts=a.starts)
    return 0


@torch.no_grad()
def illumination_sensitivity(model, batch, message_of=None,
                             brightness: float = 0.15,
                             warm: Sequence[float] = (1.10, 1.0, 0.92),
                             cool: Sequence[float] = (0.92, 1.0, 1.10)
                             ) -> Dict[str, Any]:
    """How far the action moves when only the lighting changes.

    The check that decides whether the augmentation is doing anything. A policy
    trained on a handful of episodes recorded in one session can key on
    illumination as a cue for *which episode it is in* — a shortcut straight to
    the demonstrated trajectory that does not survive a different hour — and
    with every episode a success, nothing in the data discourages it.

    The perturbations are sized from what was measured on this rig: a 15 %
    brightness change is about twice the between-episode spread, and the warm
    and cool channel gains reproduce the direction of the daylight shift (R
    below G and B, or the reverse).

    Reported **relative to the policy's own sampling spread**, because that is
    the scale that decides whether it matters: a shift smaller than the noise
    between two draws at the same state is not a shift the robot will feel.
    """
    from diffusion_policy.drim.augment import shift_illumination

    if not model.spec.is_image:
        return {}
    model.eval()
    n = len(batch["target"])
    noise = torch.zeros(n, model.horizon, model.act_dim,
                        device=batch["prop2"].device)

    def chunk(fn) -> torch.Tensor:
        rgb = {}
        for c in model.spec.cameras:
            x = batch[f"rgb2_{c}"]
            b, t = x.shape[:2]
            f = x.reshape(b * t, *x.shape[2:]).permute(0, 3, 1, 2).float() / 255.0
            f = fn(f).permute(0, 2, 3, 1).mul(255.0).round().clamp(0, 255)
            rgb[c] = f.reshape(b, t, *x.shape[2:]).to(x.dtype)
        ctx = model.context(rgb, batch["prop2"], batch.get("wrench2"))
        msg = message_of(model, batch) if message_of is not None else None
        return model.sample_chunk(ctx, msg, noise)

    base = chunk(lambda f: f)
    out: Dict[str, Any] = {}
    for name, fn in (("brighter", lambda f: shift_illumination(f, brightness)),
                     ("darker", lambda f: shift_illumination(f, -brightness)),
                     ("warm", lambda f: shift_illumination(f, 0.0, warm)),
                     ("cool", lambda f: shift_illumination(f, 0.0, cool))):
        out[name] = float((chunk(fn) - base).abs().mean().item())

    #: the policy's own spread between two draws at the same state
    g = torch.Generator(device="cpu").manual_seed(0)
    d1 = model.sample_chunk(
        model.context({c: batch[f"rgb2_{c}"] for c in model.spec.cameras},
                      batch["prop2"], batch.get("wrench2")), None,
        torch.randn(n, model.horizon, model.act_dim, generator=g).to(base.device))
    spread = float((d1 - base).abs().mean().item())
    model.train()
    worst = max(out.values())
    out["sampling_spread"] = spread
    out["worst_over_spread"] = float(worst / max(spread, 1e-9))
    out["note"] = ("action shift under a lighting-only change, in normalised "
                   "action units, against the spread between two draws at the "
                   "same state")
    return out


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
    il = (summary.get("diagnostics") or {}).get("illumination") or {}
    if "worst_over_spread" in il:
        r = il["worst_over_spread"]
        out.append(f"{'OK  ' if r < 0.5 else 'WARN'} lighting-only action shift "
                   f"is {r:.2f}x the policy's own sampling spread"
                   + ("" if r < 0.5 else "  <-- the policy is reading the light; "
                                         "widen the photometric jitter or record "
                                         "across more of the day"))
    dv = (summary.get("diagnostics") or {}).get("divergence") or {}
    if "compounding_factor" in dv:
        c = dv["compounding_factor"]
        out.append(f"{'OK  ' if c < 10 else 'WARN'} state divergence grows x{c:.1f} "
                   f"over {dv['n_steps']} steps")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
