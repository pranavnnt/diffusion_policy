"""Staged DRIM training: ``B0 -> B1 -> D2``, offline, no environment.

Budgets, losses, batch order, cached nominal draws and the snapshot grid follow
``cap_constraint_benchmark/liftoff_v6_image/train_lo6.py``.  Two structural
differences, both forced and both marked at the site:

* selection is :mod:`diffusion_policy.drim.selection` — an epoch grid reported by
  ``last-k`` — because there is no ``EnvRunner`` to rank snapshots with;
* the ``D2`` dynamics is trained here as stage ``dyn`` rather than shipped as a
  separate entry point, since it is small and its whole purpose is to feed the
  message.

No environment is imported anywhere in this module.  That is dap's "strictly
offline" guarantee and it is worth keeping for the same reason: it makes "no
on-policy data was used" structurally checkable instead of merely asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_policy.drim import diagnose as DG
from diffusion_policy.drim import dynamics as DY
from diffusion_policy.drim import policy as PL
from diffusion_policy.drim import selection as SEL
from diffusion_policy.drim import fields as F
from diffusion_policy.drim.dataset import (ChunkNormaliser, load_split,
                                           action_residual_demand,
                                           trivial_action_baselines)
from diffusion_policy.drim import dressing as DRESS
from diffusion_policy.drim.augment import PhotometricJitter
from diffusion_policy.drim.spec import (DrimSpec, fast_limits_from_fraction,
                                        fast_frac_from_demand)

BATCH = 64
LR_FLOW = 1e-4
LR_FAST = 3e-4
LR_COND = 3e-4
LR_DYN = 1e-3
WEIGHT_DECAY = 1e-6
#: The slow policy's nominal is a *sample*.  Training the corrector against one
#: draw fits it to one manoeuvre; dap caches eight and picks one per batch so the
#: corrector learns to track whichever plan it is given.
N_NOMINAL_DRAWS = 8
N_NOMINAL_DRAWS_VALID = 2
W_ACTION = 1.0
NOISE_SEED = 909

STAGES = ("dyn", "B0", "B1", "D2")

#: Epoch budgets. ``B0``/``B1`` are deliberately light and ``D2`` heavy: the
#: first two are the baseline machinery, the third is the thing under study, and
#: every stage is selected from a grid so a generous budget costs wall-clock
#: rather than accuracy.
#:
#: Epochs are **not** comparable across dataset sizes — at batch 64 the 0910
#: recording gives 8 optimizer steps per epoch and 0909 gives 14, so the same
#: number means very different amounts of training. The run logs steps/epoch
#: for that reason. A B0 that is "still improving at epoch 60" on 521 chunks has
#: had 480 steps, which is not much.
#:
#: The cost of a light ``B0``: an underfit backbone leaves more for the fast
#: level, which inflates the measured residual demand and makes ``B1`` look
#: better than it is. It does **not** invalidate ``D2 - B1`` — both nest on the
#: same frozen ``B0`` — but the absolute numbers sit on a weak stack.
#: The fast level's authority as a fraction of full command, and the ceiling
#: the measurement may not exceed.  See the note where it is applied.
FAST_FRAC_DEFAULT = 0.15
FAST_FRAC_CAP = 0.30


def run_name(prof: Dict[str, Any], epochs: Dict[str, int], a) -> str:
    """A directory name that says which run this was without opening it.

    Every part of it is something that has actually been varied in this project
    and that changes what the numbers mean, so two runs cannot be told apart by
    their timestamps alone: how many cameras, at what input size, which crop,
    which target, how long each stage ran, and whether the encoder started from
    scratch. The timestamp goes last so a sort groups the comparable runs.
    """
    cams = prof.get("cameras") or ()
    hw = prof.get("image_size") or (0, 0)
    roi = a.roi if a.profile == "dressing" else "none"
    vis = "scratch" if not a.vision_weights else str(a.vision_weights).lower()
    ep = "-".join(str(epochs[k]) for k in ("dyn", "B0", "B1", "D2"))
    bits = [f"{len(cams)}cam", f"{hw[0]}x{hw[1]}", f"roi-{roi}",
            str(prof.get("action_mode", "absolute")).replace("_", "-"),
            vis, f"e{ep}", f"s{a.seed}", time.strftime("%Y%m%d-%H%M%S")]
    if a.keep:
        bits.insert(4, "keep-" + a.keep.replace(",", "+"))
    return "drim_" + "_".join(bits)


def _ablate(excluded: Sequence[str], keep: str) -> Tuple[str, ...]:
    """The profile's exclusions minus anything ``--keep`` puts back."""
    back = {k.strip() for k in keep.split(",") if k.strip()}
    unknown = sorted(back - set(excluded))
    assert not unknown, (
        f"--keep names field(s) the profile does not exclude: {unknown}; "
        f"it excludes {sorted(excluded)}")
    return tuple(f for f in excluded if f not in back)


def _parse_frac(given: str, profile_default=None):
    """``--fast-frac`` as 'auto', one number, or one per channel.

    The profile's own per-channel ceiling wins only when the flag was left at
    its default: an explicit ``--fast-frac`` on the command line is a statement
    and must not be silently overridden by the rig profile.
    """
    if given == "auto":
        return "auto"
    if given == str(FAST_FRAC_DEFAULT) and profile_default is not None:
        return tuple(float(v) for v in profile_default)
    if "," in given:
        return tuple(float(v) for v in given.split(",") if v.strip())
    return float(given)


BUDGETS: Dict[str, Dict[str, int]] = {
    "fast":     {"dyn": 40, "B0": 30, "B1": 20, "D2": 60},
    "standard": {"dyn": 60, "B0": 60, "B1": 30, "D2": 200},
    "long":     {"dyn": 100, "B0": 100, "B1": 50, "D2": 400},
}


def _tee(out_dir: str, also: Callable[[str], None] = print
         ) -> Callable[[str], None]:
    """Log to the console and to ``<out_dir>/train.log``.

    Line-buffered and opened in append mode, so a run that is killed still
    leaves everything it had printed — which is when the log is most wanted.
    Warnings are routed here too: the field report and the control-rate warning
    are the two things worth reading afterwards, and they arrive through
    ``warnings``, not through this function.
    """
    path = os.path.join(out_dir, "train.log")
    fh = open(path, "a", buffering=1)
    fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
             f"{' '.join(sys.argv)}\n")

    def log(msg: str = "") -> None:
        also(msg)
        fh.write(str(msg) + "\n")

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        fh.write(f"WARNING {category.__name__}: {message}\n")
        _prev(message, category, filename, lineno, file, line)

    _prev = warnings.showwarning
    warnings.showwarning = _showwarning
    log.path = path
    return log


def _batches(n, size, rng):
    idx = rng.permutation(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def _to_torch(d: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if k.startswith("rgb2_"):
            out[k] = torch.from_numpy(np.ascontiguousarray(v))   # uint8, on CPU
        elif v.dtype.kind == "f":
            out[k] = torch.as_tensor(v, dtype=torch.float32)
    return out


def _batch(d: Dict[str, torch.Tensor], sel, device: str) -> Dict[str, torch.Tensor]:
    return {k: v[sel].to(device) for k, v in d.items()}


def _ctx(model: PL.DrimPolicy, b: Dict[str, torch.Tensor]) -> torch.Tensor:
    rgb = ({c: b[f"rgb2_{c}"] for c in model.spec.cameras}
           if model.spec.is_image else None)
    return model.context(rgb, b["prop2"], b.get("wrench2"))


# --------------------------------------------------------------------------- #
# stage: dynamics (the source of D2's S and U)
# --------------------------------------------------------------------------- #


def train_dynamics(spec: DrimSpec, tr, va, seed: int, epochs: int, device: str,
                   log: Callable[[str], None], mods,
                   estimator: str = SEL.DEFAULT_ESTIMATOR) -> Dict[str, Any]:
    """Fit the delta dynamics whose residual becomes the message.

    Reported against the **no-change baseline**, always.  dap ran a whole round
    on a dynamics model that was 155 % *worse* than predicting no change while
    its NLL looked fine — NLL can be driven down by shrinking sigma while the
    point predictions stay useless, and the message then encodes model error
    rather than physical evidence.  If ``skill`` below is negative the message is
    noise and D2 is not worth training yet.
    """
    torch.manual_seed(seed + 3000)
    assert DY.state_dim(mods) == spec.dyn_dim, (
        f"modality table covers {DY.state_dim(mods)} channels, spec declares "
        f"{spec.dyn_dim} for the dynamics (prop {spec.prop_dim} + wrench "
        f"{spec.wrench_dim})")

    def flat(d, key):
        a = d[key]
        #: A zero-width stream (no exogenous command declared) cannot go
        #: through ``reshape(-1, 0)`` — numpy cannot infer a free dimension
        #: against a zero one — so build it explicitly.
        if a.shape[-1] == 0:
            return np.zeros((int(np.prod(a.shape[:-1])), 0), a.dtype)
        return a.reshape(-1, a.shape[-1])

    def prep(d):
        y = flat(d, "hist_y")
        yn = flat(d, "hist_y_next")
        a = flat(d, "hist_a")
        dt = flat(d, "hist_dt")
        xo = flat(d, "hist_exo")
        keep = np.repeat(d["msg_valid"] > 0, d["hist_y"].shape[1])
        return y[keep], yn[keep], a[keep], dt[keep], xo[keep]

    ytr, yntr, atr, dttr, xotr = prep(tr)
    yva, ynva, ava, dtva, xova = prep(va)
    assert len(ytr), (
        "no decision step has a full message window behind it; lower "
        "spec.message_window or record longer episodes")
    dtr = DY.state_delta(ytr, yntr, mods)
    norm = DY.Normaliser(y=ytr, d=dtr)
    baseline = DY.no_change_baseline(norm.d(dtr))

    model = DY.DeltaDynamics(mods, spec.act_dim,
                             exo_dim=spec.exo_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_DYN,
                            weight_decay=WEIGHT_DECAY)
    #: The stage that used to be exempt.  Its last weights are what feed D2's
    #: message, so a budget that was slightly wrong made the message worse and
    #: nothing caught it — held-out skill ran +68.5 % at 30 epochs and +40.1 %
    #: at 200.  It now snapshots on the same grid and is selected the same way,
    #: which removes the sensitivity rather than asking the budget to be right.
    bank = SEL.SnapshotBank(epochs)
    ty = torch.as_tensor(norm.y(ytr), device=device)
    ta = torch.as_tensor(atr, dtype=torch.float32, device=device)
    tdt = torch.as_tensor(dttr, dtype=torch.float32, device=device)
    txo = torch.as_tensor(xotr, dtype=torch.float32, device=device)
    td = torch.as_tensor(norm.d(dtr), device=device)
    rng = np.random.default_rng(seed + 3000)
    vy = torch.as_tensor(norm.y(yva), device=device)
    va_ = torch.as_tensor(ava, dtype=torch.float32, device=device)
    vdt = torch.as_tensor(dtva, dtype=torch.float32, device=device)
    vxo = torch.as_tensor(xova, dtype=torch.float32, device=device)
    dva = DY.state_delta(yva, ynva, mods)
    vd = torch.as_tensor(norm.d(dva), device=device)
    base_va = DY.no_change_baseline(norm.d(dva))

    @torch.no_grad()
    def _val() -> float:
        model.eval()
        mu, _ = model(vy, va_, vdt, vxo if spec.exo_dim else None)
        v = float(((vd - mu) ** 2).mean().item())
        model.train()
        return v

    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for sel in _batches(len(ty), BATCH, rng):
            i = torch.as_tensor(sel, device=device)
            mu, ls = model(ty[i], ta[i], tdt[i],
                           txo[i] if spec.exo_dim else None)
            loss = DY.gaussian_nll(mu, ls, td[i])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(sel)
        #: selected on held-out delta MSE, which is what ``skill`` is computed
        #: from — the quantity the message's usefulness actually depends on,
        #: not the NLL, which can be driven down by shrinking sigma while the
        #: point predictions get worse.
        vm = _val() if (ep + 1) in bank.grid else None
        if bank.observe(model, ep + 1, action_mse=vm, val_loss=vm,
                        train_loss=tot / len(ty)):
            log(f"      dyn epoch {ep + 1}/{epochs} nll={tot / len(ty):.5f} "
                f"val_mse={vm:.5f} (skill {(base_va - vm) / base_va:+.1%}) [snapshot]")

    sel_eps = SEL.select_epochs(sorted(bank.states), estimator, curve=bank.curve)
    model.load_state_dict(SEL.average_states(bank, sel_eps))
    log(f"      dyn selected epoch(s) {sel_eps} by {estimator}")
    model.eval()
    frozen = DY.FrozenDynamics(model, norm).to(device)
    with torch.no_grad():
        t_ava = torch.as_tensor(ava, dtype=torch.float32, device=device)
        t_dtva = torch.as_tensor(dtva, dtype=torch.float32, device=device)
        t_xova = (torch.as_tensor(xova, dtype=torch.float32, device=device)
                  if spec.exo_dim else None)
        mu, _ = frozen(torch.as_tensor(yva, device=device), t_ava, t_dtva, t_xova)
        mse = float(((torch.as_tensor(norm.d(dva), device=device) - mu) ** 2)
                    .mean().item())
        ch = DY.surprise(frozen, torch.as_tensor(yva, device=device),
                         torch.as_tensor(ynva, device=device), t_ava, t_dtva,
                         t_xova)
    base_va = DY.no_change_baseline(norm.d(dva))
    skill = (base_va - mse) / max(base_va, 1e-12)
    log(f"      dyn  mse={mse:.6f}  no-change baseline={base_va:.6f}  "
        f"skill={skill:+.1%}"
        + ("" if skill > 0 else "   <-- WORSE THAN DOING NOTHING"))
    return {"dynamics": frozen, "mods": mods, "norm": norm,
            "use_dt": bool(model.use_dt), "curve": bank.curve,
            "selected_epochs": sel_eps,
            "delta_dim": DY.delta_dim(mods),
            "mse": mse, "no_change_baseline": base_va, "skill": skill,
            "train_no_change_baseline": baseline,
            "saturation": DY.saturation(ch["U"].cpu().numpy()),
            "clip": DY.clip_report(ch["S_RAW"].cpu().numpy()),
            "seconds": round(time.time() - t0, 1)}


def attach_su(d: Dict[str, np.ndarray], frozen: DY.FrozenDynamics,
              device: str = "cpu") -> Dict[str, np.ndarray]:
    """Compute ``S`` and ``U`` over each row's window and mask before valid."""
    out = dict(d)
    y = torch.as_tensor(d["hist_y"], dtype=torch.float32, device=device)
    yn = torch.as_tensor(d["hist_y_next"], dtype=torch.float32, device=device)
    a = torch.as_tensor(d["hist_a"], dtype=torch.float32, device=device)
    dt = torch.as_tensor(d["hist_dt"], dtype=torch.float32, device=device)
    xo = (torch.as_tensor(d["hist_exo"], dtype=torch.float32, device=device)
          if d["hist_exo"].shape[-1] else None)
    with torch.no_grad():
        ch = DY.surprise(frozen, y, yn, a, dt, xo)
    mv = d["msg_valid"][:, None, None]
    out["hist_S"] = ch["S"].cpu().numpy().astype(np.float32) * mv
    out["hist_U"] = ch["U"].cpu().numpy().astype(np.float32) * mv
    return out


# --------------------------------------------------------------------------- #
# stage B0: the slow policy alone
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _valid_flow(model, va, device, batch=64, n_repeat=2, seed=0) -> float:
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    tot = n = 0.0
    for _ in range(n_repeat):
        for i in range(0, len(va["target"]), batch):
            sel = slice(i, i + batch)
            b = _batch(va, sel, device)
            tot += model.flow_loss(_ctx(model, b), b["target"],
                                   generator=g).item() * len(b["target"])
            n += len(b["target"])
    model.train()
    return tot / max(n, 1.0)


#: Draws averaged into the reported action error.  The nominal is a *sample*,
#: so one draw is a noisy read of the policy; four is enough to make the number
#: stable without making validation the expensive part of a run.
N_EVAL_DRAWS = 4


@torch.no_grad()
def action_mse(model, va, device, batch=64, seed=NOISE_SEED,
               message_of=None, n_draws: int = N_EVAL_DRAWS) -> float:
    """Error of the **executed prefix** against the demonstrated action.

    The one metric every stage is scored by, computed by this one function with
    the same draws and the same seed. ``B0`` has no corrector, so its executed
    action is the nominal; ``D2`` gets its message. That is the whole difference
    between the stages, which is what makes ``D2 - B1`` a comparison rather than
    two different rulers — an earlier version scored ``B1`` with a separate
    routine over cached nominals and the two numbers were not on the same scale.

    It is the closest offline stand-in for what the robot receives, and on this
    track it is also the default **selector**: there is no rollout to rank with
    and one checkpoint has to be chosen anyway.
    """
    model.eval()
    e = model.exec_horizon
    tot = n = 0.0
    for d in range(n_draws):
        g = torch.Generator(device="cpu").manual_seed(seed + 1000 * d)
        for i in range(0, len(va["target"]), batch):
            sel = slice(i, i + batch)
            b = _batch(va, sel, device)
            noise = torch.randn(len(b["target"]), model.horizon, model.act_dim,
                                generator=g).to(device)
            msg = message_of(model, b) if message_of is not None else None
            ctx = _ctx(model, b)
            nominal = model.sample_chunk(ctx, msg, noise)
            r = model.residual_prefix(nominal, b["step_prop"],
                                      b.get("step_wrench"), msg)
            exe = torch.clamp(nominal[:, :e] + r, -1.0, 1.0)
            tot += torch.nn.functional.mse_loss(
                exe, b["target"][:, :e]).item() * len(b["target"])
            n += len(b["target"])
    model.train()
    return tot / max(n, 1.0)


_valid_action_mse = action_mse                       # older call sites


def train_b0(spec, tr, va, seed, epochs, device, log,
             photometric=None, vision_weights=None) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = PL.build("B0", spec, photometric=photometric,
                     vision_weights=vision_weights).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_FLOW, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    n = len(tr["target"])
    bank = SEL.SnapshotBank(epochs)
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for sel in _batches(n, BATCH, rng):
            b = _batch(tr, sel, device)
            loss = model.flow_loss(_ctx(model, b), b["target"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.detach().item() * len(sel)
        sched.step()
        vl = _valid_flow(model, va, device)
        #: ``action_mse`` samples a chunk per draw per batch, which with vision
        #: costs more than the epoch that produced it.  Selection only ever
        #: reads it at grid points, so that is where it is paid for.
        am = (action_mse(model, va, device) if (ep + 1) in bank.grid else None)
        if bank.observe(model, ep + 1, val_loss=vl, action_mse=am,
                        train_loss=tot / n):
            log(f"      s{seed}/B0 epoch {ep + 1}/{epochs} train={tot / n:.5f} "
                f"val={vl:.5f} act_mse={am:.5f} [snapshot]")
    return {"model": model, "bank": bank, "seconds": round(time.time() - t0, 1),
            "n_trainable": int(sum(p.numel() for p in params)),
            "n_params": model.n_params()}


# --------------------------------------------------------------------------- #
# stage B1: the bounded fast corrector on a frozen B0
# --------------------------------------------------------------------------- #


@torch.no_grad()
def cache_nominals(model, data, draws, device, batch=64, seed=NOISE_SEED):
    n = len(data["target"])
    out = torch.zeros(draws, n, model.horizon, model.act_dim)
    for d in range(draws):
        g = torch.Generator(device="cpu").manual_seed(seed + 1000 * d)
        noise = torch.randn(n, model.horizon, model.act_dim, generator=g)
        for i in range(0, n, batch):
            sel = slice(i, i + batch)
            b = _batch(data, sel, device)
            out[d, sel] = model.sample_chunk(_ctx(model, b),
                                             None, noise[sel].to(device)).cpu()
    return out


@torch.no_grad()
def measured_residual_demand(b0, tr, spec, device, batch=64,
                             n_draws: int = N_EVAL_DRAWS,
                             seed: int = NOISE_SEED) -> Dict[str, Any]:
    """What the corrector would actually have to supply, given a trained ``B0``.

    ``|target - nominal|`` over the executed prefix, where ``nominal`` is B0's
    own sampled chunk. This is the real quantity; the pre-training
    ``vs_chunk_mean`` proxy in the loader stands in for it before B0 exists and
    underestimated it badly on the 0909 recording — a ceiling set from the proxy
    left the corrector pinned at 60 % saturation for the whole of B1.
    """
    b0.eval()
    e = b0.exec_horizon
    out = []
    for d in range(n_draws):
        g = torch.Generator(device="cpu").manual_seed(seed + 1000 * d)
        for i in range(0, len(tr["target"]), batch):
            sel = slice(i, i + batch)
            b = _batch(tr, sel, device)
            noise = torch.randn(len(b["target"]), b0.horizon, b0.act_dim,
                                generator=g).to(device)
            nom = b0.sample_chunk(_ctx(b0, b), None, noise)
            out.append((b["target"][:, :e] - nom[:, :e]).abs().cpu().numpy())
    b0.train()
    a = np.concatenate(out).reshape(-1, spec.act_dim)
    return {f"p{q}": [round(float(v), 5) for v in np.percentile(a, q, axis=0)]
            for q in (50, 90, 95, 99, 100)}


def _fast_loss(model, b, nominal):
    e = model.exec_horizon
    r = model.residual_prefix(nominal, b["step_prop"], b.get("step_wrench"))
    return torch.nn.functional.mse_loss(
        torch.clamp(nominal[:, :e] + r, -1.0, 1.0), b["target"][:, :e])


@torch.no_grad()
def _valid_fast(model, va, nom_va, device, batch=64) -> Tuple[float, float]:
    """Validation action loss, and how often the corrector sits on its ceiling.

    Saturation is the number that says whether the authority is right, and it is
    only knowable after training: a corrector clipping on most steps has been
    given less authority than the task needs, one that never approaches its
    ceiling has been given more than it uses.  dap logs the analogous rate for
    the surprise clip and found it moved 213x between training and rollout, which
    is why it is logged permanently here rather than checked once.
    """
    model.eval()
    tot = n = sat = 0.0
    lim = model.fast.max_residual
    writable = (lim > 0)
    for d in range(len(nom_va)):
        for i in range(0, len(va["target"]), batch):
            sel = slice(i, i + batch)
            b = _batch(va, sel, device)
            nom = nom_va[d, sel].to(device)
            tot += _fast_loss(model, b, nom).item() * len(b["target"])
            r = model.residual_prefix(nom, b["step_prop"], b.get("step_wrench"))
            if bool(writable.any()):
                near = (r[..., writable].abs()
                        >= 0.99 * lim[writable]).float().mean().item()
                sat += near * len(b["target"])
            n += len(b["target"])
    model.train()
    return tot / max(n, 1.0), sat / max(n, 1.0)


def train_b1(b0, spec, tr, va, seed, epochs, device, log) -> Dict[str, Any]:
    torch.manual_seed(seed + 500)
    #: ``model.fast = ...`` after construction does **not** replace the
    #: corrector: ``nn.Module.__setattr__`` intercepts Module values, so it
    #: registers a second submodule while the one ``residual_prefix`` calls stays
    #: at its initialisation, and the optimiser then trains a copy nothing uses.
    #: dap shipped a whole ``IMG_B1`` result from a randomly initialised
    #: corrector that way.  Build it once, in the constructor, and never reassign.
    model = PL.build("B1", spec, core=b0, freeze_base=True).to(device)
    for p in model.fast.parameters():
        p.requires_grad_(True)
    model.assert_authority(spec)

    log(f"      s{seed}/B1 caching {N_NOMINAL_DRAWS} nominal draws")
    nom_tr = cache_nominals(model, tr, N_NOMINAL_DRAWS, device)
    nom_va = cache_nominals(model, va, N_NOMINAL_DRAWS_VALID, device)
    spread = float((nom_tr[0] - nom_tr[1]).abs().mean().item())
    log(f"      s{seed}/B1 mean |draw0 - draw1| = {spread:.4f}")

    params = list(model.fast.parameters())
    assert params, "the corrector has no trainable parameters"
    opt = torch.optim.AdamW(params, lr=LR_FAST, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed + 500)
    n = len(tr["target"])
    bank = SEL.SnapshotBank(epochs, parent_keys=list(b0.state_dict()))
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for sel in _batches(n, BATCH, rng):
            b = _batch(tr, sel, device)
            d = int(rng.integers(N_NOMINAL_DRAWS))
            loss = _fast_loss(model, b, nom_tr[d, sel].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.detach().item() * len(sel)
        sched.step()
        vl, sat = _valid_fast(model, va, nom_va, device)
        #: ``vl`` is B1's own objective (cached nominals, no message); the
        #: comparable number is ``am``, computed exactly as B0's and D2's are.
        am = (action_mse(model, va, device) if (ep + 1) in bank.grid else None)
        if bank.observe(model, ep + 1, val_loss=vl, action_mse=am,
                        train_loss=tot / n, saturation=sat):
            log(f"      s{seed}/B1 epoch {ep + 1}/{epochs} train={tot / n:.5f} "
                f"val={vl:.5f} act_mse={am:.5f} saturated={sat:.1%} [snapshot]")
    _, sat = _valid_fast(model, va, nom_va, device)
    return {"model": model, "bank": bank, "nominal_draw_spread": spread,
            "saturation": sat,
            "seconds": round(time.time() - t0, 1),
            "n_trainable": int(sum(p.numel() for p in params)),
            "n_params": model.n_params()}


# --------------------------------------------------------------------------- #
# stage D2: the upward message on a frozen B1
# --------------------------------------------------------------------------- #


def _message(model, b):
    return model.conditioning({"prop2": b["prop2"], "msg_valid": b["msg_valid"],
                               "hist_S": b["hist_S"], "hist_U": b["hist_U"]})


def _cond_losses(model, b, noise):
    msg = _message(model, b)
    ctx = _ctx(model, b)
    flow = model.flow_loss(ctx, b["target"], msg)
    nominal = model.sample_chunk(ctx, msg, noise)
    r = model.residual_prefix(nominal, b["step_prop"], b.get("step_wrench"), msg)
    e = model.exec_horizon
    act = torch.nn.functional.mse_loss(
        torch.clamp(nominal[:, :e] + r, -1.0, 1.0), b["target"][:, :e])
    return flow, act


@torch.no_grad()
def _valid_cond(model, va, noise_va, device, batch=64) -> float:
    model.eval()
    tot = n = 0.0
    for i in range(0, len(va["target"]), batch):
        sel = slice(i, i + batch)
        b = _batch(va, sel, device)
        flow, act = _cond_losses(model, b, noise_va[sel].to(device))
        tot += (flow + W_ACTION * act).item() * len(b["target"])
        n += len(b["target"])
    model.train()
    return tot / max(n, 1.0)


def train_d2(b1, spec, tr, va, seed, epochs, device, message_in_dims, log
             ) -> Dict[str, Any]:
    torch.manual_seed(seed + 2000)
    model = PL.build("D2", spec, core=b1, message_in_dims=message_in_dims,
                     freeze_base=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_COND, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed + 2000)
    n = len(tr["target"])
    g = torch.Generator(device="cpu").manual_seed(NOISE_SEED + 7)
    noise_tr = torch.randn(n, model.horizon, model.act_dim, generator=g)
    noise_va = torch.randn(len(va["target"]), model.horizon, model.act_dim,
                           generator=g)
    bank = SEL.SnapshotBank(epochs, parent_keys=list(b1.state_dict()))
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for sel in _batches(n, BATCH, rng):
            b = _batch(tr, sel, device)
            flow, act = _cond_losses(model, b, noise_tr[sel].to(device))
            loss = flow + W_ACTION * act
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.detach().item() * len(sel)
        sched.step()
        vl = _valid_cond(model, va, noise_va, device)
        am = (action_mse(model, va, device, message_of=_message)
              if (ep + 1) in bank.grid else None)
        gn = model.gate_norms()
        if bank.observe(model, ep + 1, val_loss=vl, action_mse=am,
                        train_loss=tot / n, **gn):
            log(f"      s{seed}/D2 epoch {ep + 1}/{epochs} train={tot / n:.5f} "
                f"val={vl:.5f} act_mse={am:.5f} gate={gn['gate']:.4f} "
                f"rgate={gn['reconcile_gate']:.4f} [snapshot]")
    return {"model": model, "bank": bank, "gate_norms": model.gate_norms(),
            "seconds": round(time.time() - t0, 1),
            "n_trainable": int(sum(p.numel() for p in params)),
            "n_params": model.n_params()}


# --------------------------------------------------------------------------- #
# exact-null check
# --------------------------------------------------------------------------- #


@torch.no_grad()
def check_exact_null(d2: PL.DrimPolicy, b: Dict[str, torch.Tensor],
                     atol: float = 1e-5) -> Dict[str, float]:
    """``D2`` with a null message must reproduce ``B1`` exactly.

    Structural, not an initialisation property, so it is checkable at any point
    in training and is checked *after* training rather than before — the whole
    value of the identity is that it survives the message path learning
    something.  A failure here means the gated route was edited into an ordinary
    residual and every D2-minus-B1 number is meaningless.
    """
    d2.eval()
    ctx = _ctx(d2, b)
    noise = torch.zeros(len(b["target"]), d2.horizon, d2.act_dim,
                        device=ctx.device)
    zero = torch.zeros(len(b["target"]), d2.cond_dim, device=ctx.device)
    with_null = d2.sample_chunk(ctx, zero, noise)
    dv = d2.dv
    try:
        d2.dv = None                                # the B1 path, same weights
        without = d2.sample_chunk(ctx, None, noise)
    finally:
        #: Restored in ``finally`` because a diagnostic that leaves the model
        #: without its message path on failure turns a check into a corruption.
        d2.dv = dv
        d2.train()
    err = float((with_null - without).abs().max().item())
    return {"max_abs_diff": err, "passed": bool(err <= atol), "atol": atol}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run(zarr_path: Any, out_dir: str, seed: int = 0,
        epochs: Optional[Dict[str, int]] = None, device: str = "cpu",
        stages: Sequence[str] = STAGES, val_ratio: float = 0.2,
        cameras: Sequence[str] = (), n_arms: int = 2,
        layout: Optional[F.PackedLayout] = None,
        fast_frac: Any = "auto",
        require: Sequence[str] = (), action_key: str = "action",
        action_mode: str = "absolute", exo_key: Optional[str] = None,
        exclude: Sequence[str] = (),
        act_scale_per_arm: Optional[Sequence[float]] = None,
        act_per_arm: Optional[int] = None,
        roi: Optional[Dict[str, Any]] = None,
        photometric: Optional[Dict[str, float]] = None,
        vision_weights: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        horizons: Optional[Dict[str, Any]] = None,
        estimator: str = SEL.DEFAULT_ESTIMATOR, keep_grid: bool = True,
        diagnose: bool = True, diag_steps: int = 32, diag_starts: int = 48,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    unknown = [s for s in stages if s not in STAGES]
    assert not unknown, f"unknown stage(s) {unknown}; have {STAGES}"
    #: Each stage nests on the trained parent held in memory, so a run cannot
    #: start in the middle. Saying so here beats a KeyError three stages later.
    for child, parent in (("B1", "B0"), ("D2", "B1"), ("D2", "dyn")):
        assert not (child in stages and parent not in stages), (
            f"stage {child} nests on {parent}, which is not in {list(stages)}")
    #: (superseded by BUDGETS below; kept as the floor when nothing is passed)
    #: ``dyn`` is short on purpose.  Swept on the 0909 recordings, held-out
    #: skill against the no-change baseline runs +67.5 / +68.5 / +65.3 / +57.6 /
    #: +40.1 % at 10 / 30 / 60 / 100 / 200 epochs — monotonically worse past ~30.
    #: It is the one stage with no snapshot grid and no early stopping, so a long
    #: budget is not merely wasted, it is spent: the last weights are the ones
    #: that feed D2's message, and dap lost a round to a dynamics that was worse
    #: than doing nothing.
    epochs = {**BUDGETS["standard"], **(epochs or {})}
    os.makedirs(out_dir, exist_ok=True)
    if log is print:
        log = _tee(out_dir)
    log(f"[out]  {os.path.abspath(out_dir)}")
    if image_size is not None:
        log(f"[data] camera frames resized to {image_size[0]}x{image_size[1]}")
    log(f"[data] {zarr_path}")
    tr, va, norm, eps, spec = load_split(
        zarr_path, cameras=cameras, n_arms=n_arms, layout=layout,
        val_ratio=val_ratio, seed=seed, require=require,
        action_key=action_key, image_size=image_size, exo_key=exo_key,
        act_scale_per_arm=act_scale_per_arm, act_per_arm=act_per_arm, roi=roi,
        exclude=exclude,
        spec_kw=dict(action_mode=action_mode, **(horizons or {})))

    res = eps.resolution
    log(f"[fields] usable: {', '.join(res.usable) or '(none)'}")
    if exclude:
        log(f"[fields] excluded by configuration: {', '.join(exclude)}"
            f"   (--keep puts any of them back)")
    for st in res.missing():
        log(f"[fields] MISSING {st.arm}.{st.name}: {st.reason}")
    act = eps.action_report(spec.act_channels)
    for i, path in enumerate(eps.zarr_paths):
        n = sum(1 for s_ in eps.episode_source if s_ == i)
        log(f"[data] store {i}: {os.path.basename(path)} ({n} episodes)")
    log(f"[data] {len(eps)} episodes, lengths {eps.lengths()}; "
        f"prop_dim={spec.prop_dim} wrench_dim={spec.wrench_dim} "
        f"act_dim={spec.act_dim}")
    log(f"[data] {len(tr['target'])} train chunks / {len(va['target'])} valid; "
        f"{int(tr['msg_valid'].sum())} with a full message window")
    log(f"[data] {max(len(tr['target']) // BATCH, 1)} optimizer steps per epoch "
        f"at batch {BATCH}; budgets " + ", ".join(f"{k}={v}" for k, v in epochs.items()))
    if act["dead"]:
        log(f"[data] action channels never nonzero: {act['dead']} "
            f"(active: {act['active']})")
    demand = action_residual_demand(tr, spec)
    trivial = trivial_action_baselines(va, spec)
    if trivial:
        log(f"[baseline] executed-prefix error with no policy: "
            f"predict-zero {trivial['predict_zero']:.5f}, "
            f"repeat-previous {trivial['repeat_first_action']:.5f}"
            f"   <- action_mse must beat the second one")

    auto_frac = (fast_frac == "auto")
    if fast_frac is not None:
        if auto_frac:
            #: A provisional ceiling from the pre-training proxy, replaced by the
            #: measured one as soon as B0 exists (see below).  Kept only so the
            #: spec is well-formed if B0 is skipped.
            fast_frac = fast_frac_from_demand(demand, act["active"])
        #: Only channels the demonstrations actually move get authority.  A
        #: ceiling on a channel that is identically zero in the data is
        #: authority over something the corrector can never have learned, and
        #: it is free to write there at rollout; a zero ceiling makes the
        #: channel structurally silent instead.
        spec = DrimSpec(**{**spec.to_dict(),
                           "fast_limits": fast_limits_from_fraction(
                               fast_frac, spec.act_scale or
                               tuple([1.0] * spec.act_dim),
                               active=act["active"])})
        summary_frac = (float(fast_frac) if np.isscalar(fast_frac)
                        else [float(v) for v in fast_frac])
        if not any(spec.fast_limits):
            raise ValueError(
                "no action channel is ever nonzero in this dataset, so every "
                "ceiling is 0 and the corrector cannot write anything; B1 and "
                "D2 would train a no-op. Check the recording.")
    else:
        summary_frac = None

    if demand:
        for i in act["active"]:
            ceil = ("" if spec.fast_limits is None
                    else f" | ceiling={spec.fast_limits[i]:.4f}")
            log(f"[authority] ch{i} residual demand (fraction of full command): "
                f"vs_chunk_mean p95={demand['vs_chunk_mean']['p95'][i]:.4f} "
                f"max={demand['vs_chunk_mean']['p100'][i]:.4f} | "
                f"step_to_step p95={demand['step_to_step']['p95'][i]:.4f}"
                f"{ceil}")
        if spec.fast_limits is not None:
            tight = [i for i in act["active"]
                     if spec.fast_limits[i] < demand["step_to_step"]["p95"][i]]
            if tight:
                log(f"[authority] ceiling is below the p95 step-to-step demand on "
                    f"channels {tight}: the corrector will clip on ordinary "
                    f"motion, not just on surprises")

    summary: Dict[str, Any] = {
        "spec": spec.to_dict(), "seed": seed, "zarr": zarr_path,
        "episodes": len(eps), "episode_lengths": eps.lengths(),
        "stores": list(eps.zarr_paths), "episode_source": list(eps.episode_source),
        "n_train_chunks": int(len(tr["target"])),
        "n_valid_chunks": int(len(va["target"])),
        #: what was left out on purpose, so a run says which ablation it is
        "excluded": list(exclude),
        "fields": {"usable": list(res.usable),
                   "missing": [{"arm": s.arm, "field": s.name,
                                "reason": s.reason} for s in res.missing()],
                   "notes": [{"arm": s.arm, "field": s.name, "note": s.note}
                             for s in res.statuses if s.note]},
        #: The crop is a runtime flag, so a run that did not record it cannot be
        #: deployed without guessing — and a deployment that guesses a different
        #: box feeds the encoder a different scene than it was trained on.
        "roi": ({k: list(v) for k, v in roi.items()} if roi else None),
        "action_channels": act,
        "residual_demand": demand, "fast_frac": summary_frac,
        "photometric": photometric,
        "trivial_action_baselines": trivial,
        "dt": getattr(eps, "dt_stats", None),
        "normaliser": norm.state_dict()}

    frozen = None
    if "dyn" in stages or "D2" in stages:
        log(f"[stage] dyn  (state = {spec.dyn_dim}d: prop {spec.prop_dim} + "
            f"contact {spec.wrench_dim})")
        mods = F.modalities(res, spec.dyn_fields)
        dyn = train_dynamics(spec, tr, va, seed, epochs["dyn"], device, log,
                             mods, estimator=estimator)
        frozen = dyn.pop("dynamics")
        summary["dyn"] = {k: v for k, v in dyn.items() if k not in ("mods", "norm")}
        tr = attach_su(tr, frozen, device)
        va = attach_su(va, frozen, device)
        torch.save({"state_dict": frozen.model.state_dict(),
                    "norm": frozen.norm.state_dict(),
                    "mods": mods, "spec": spec.to_dict(),
                    "report": summary["dyn"]},
                   os.path.join(out_dir, f"dynamics_seed{seed}.pt"))

    t_tr, t_va = _to_torch(tr, device), _to_torch(va, device)
    models: Dict[str, PL.DrimPolicy] = {}

    def _save(stage: str, r: Dict[str, Any]) -> Dict[str, Any]:
        sel = SEL.save_selected(
            os.path.join(out_dir, f"{stage}_seed{seed}.pt"), r["bank"], stage,
            spec.to_dict(), estimator=estimator, keep_grid=keep_grid,
            extra={k: v for k, v in r.items() if k != "bank"})
        out = {k: v for k, v in r.items() if k != "bank"}
        out["selection"] = sel
        #: The per-epoch curve also travels inside the checkpoint, but a
        #: checkpoint is 100+ MB and a plot should not require loading one.
        out["curve"] = r["bank"].curve
        out["snapshot_epochs"] = list(r["bank"].grid)
        picks = sel.get("would_pick") or {}
        log(f"      {stage} selected epoch(s) {sel['selected_epochs']} "
            f"by {estimator}"
            + ("" if sel.get("criteria_agree")
               else "   other criteria would pick "
                    + ", ".join(f"{k}={v}" for k, v in picks.items()
                                if v not in sel["selected_epochs"])))
        if "overfit_warning" in sel:
            log(f"      WARNING {sel['overfit_warning']}")
        return out

    jitter = (PhotometricJitter(**photometric)
              if (photometric and spec.is_image) else None)
    if jitter is not None:
        log(f"[augment] {jitter.extra_repr()}")
    if spec.is_image:
        log(f"[vision] resnet18 weights={vision_weights or 'random init'}"
            + ("" if vision_weights else
               "   <- trained from scratch on "
               f"{len(tr['target'])} chunks"))

    if "B0" in stages:
        log("[stage] B0")
        r = train_b0(spec, t_tr, t_va, seed, epochs["B0"], device, log,
                     photometric=jitter, vision_weights=vision_weights)
        models["B0"] = r.pop("model")
        summary["B0"] = _save("B0", r)

    if "B1" in stages and auto_frac and "B0" in models:
        #: Now that B0 exists, the ceiling comes from the residual it actually
        #: leaves rather than from a proxy computed before any model was trained.
        md = measured_residual_demand(models["B0"], t_tr, spec, device)
        raw = max(md["p95"][i] for i in act["active"])
        #: The ceiling is *declared*, not measured. What B0 leaves behind says
        #: how much the corrector would like; what it may have is a safety
        #: question about a robot working next to a person, and the measurement
        #: cannot answer it. 0.15 of full command is ~5.2 mm of pose delta on
        #: the x axis, about 30 % of the operator's own median nudge (17.8 mm)
        #: and under 1 mm of extra end-effector motion per replan interval --
        #: a correction rather than a second policy. dap's two tasks bracket
        #: it: cap declared 0.023 in these units and drawer 0.23.
        frac = float(min(max(raw, 0.02), FAST_FRAC_CAP))
        if raw > FAST_FRAC_CAP:
            log(f"[authority] measured demand {raw:.3f} exceeds the declared "
                f"cap {FAST_FRAC_CAP:.2f}; B0 is leaving the fast level more "
                f"than a reflex should have. Read it as B0 underfit, not as a "
                f"reason to raise the cap.")
        if raw >= 0.5:
            #: The demand is measured against B0's own samples, so an
            #: undertrained B0 inflates it without limit — a --quick run hits
            #: this every time.  Read it as "B0 has not converged", not as "the
            #: corrector needs half the command range".
            log(f"[authority] WARNING measured demand {raw:.3f} hit the 0.5 "
                f"clamp; B0 is probably undertrained, so this ceiling says more "
                f"about B0 than about the task")
        summary["measured_residual_demand"] = md
        log(f"[authority] measured from B0: p95 {md['p95']}  p99 {md['p99']}")
        log(f"[authority] ceiling {fast_frac:.4f} (proxy) -> {frac:.4f} (measured)")
        spec = IrumSpec(**{**spec.to_dict(),
                           "fast_limits": fast_limits_from_fraction(
                               frac, spec.act_scale or
                               tuple([1.0] * spec.act_dim),
                               active=act["active"])})
        summary["fast_frac"] = frac
        summary["spec"] = spec.to_dict()

    if "B1" in stages:
        log("[stage] B1")
        r = train_b1(models["B0"], spec, t_tr, t_va, seed, epochs["B1"], device, log)
        models["B1"] = r.pop("model")
        summary["B1"] = _save("B1", r)

    if "D2" in stages:
        log("[stage] D2")
        dd = summary["dyn"]["delta_dim"]
        r = train_d2(models["B1"], spec, t_tr, t_va, seed, epochs["D2"], device,
                     (dd, dd), log)
        models["D2"] = r.pop("model")
        b = _batch(t_va, slice(0, min(8, len(va["target"]))), device)
        null = check_exact_null(models["D2"], b)
        log(f"      exact-null check: max|D2(m=0) - B1| = "
            f"{null['max_abs_diff']:.3e}  {'PASS' if null['passed'] else 'FAIL'}")
        summary["D2"] = {**_save("D2", r), "exact_null": null}

    #: Everything below runs on the *trained* models and never feeds back into
    #: them: it is what to read before the policy reaches the robot, not another
    #: selector.
    if diagnose and frozen is not None and models:
        log("[stage] diagnostics")
        last = models.get("D2") or models.get("B1") or models.get("B0")
        mods = F.modalities(res, spec.dyn_fields)
        va_eps = sorted({int(i) for i in va["episode_index"]})
        #: the episodes D2's message was actually conditioned on, which is what
        #: the held-out surprise has to be compared against
        tr_eps = sorted({int(i) for i in tr["episode_index"]})
        diag: Dict[str, Any] = {}
        try:
            diag["divergence"] = DG.divergence(
                last, frozen, mods, eps, norm, spec, va_eps, device=device,
                n_steps=diag_steps, n_starts=diag_starts, seed=seed)
            diag["surprise_shift"] = DG.surprise_shift(
                frozen, mods, eps, norm, spec, tr_eps, va_eps, device=device)
        except Exception as exc:                       # pragma: no cover
            log(f"      diagnostics failed: {type(exc).__name__}: {exc}")
            diag["error"] = f"{type(exc).__name__}: {exc}"
        b = _batch(t_va, slice(0, min(32, len(va["target"]))), device)
        if "D2" in models:
            diag["message_reliance"] = DG.message_reliance(
                models["D2"], b, _message)
        if spec.is_image:
            diag["illumination"] = DG.illumination_sensitivity(
                last, b, _message if "D2" in models else None)
            il = diag["illumination"]
            if il:
                log(f"      lighting shift: brighter {il['brighter']:.4f} "
                    f"darker {il['darker']:.4f} warm {il['warm']:.4f} "
                    f"cool {il['cool']:.4f}  vs sampling spread "
                    f"{il['sampling_spread']:.4f} "
                    f"({il['worst_over_spread']:.2f}x)")
        summary["diagnostics"] = diag
        d = diag.get("divergence") or {}
        for k, v in (d.get("at") or {}).items():
            log(f"      divergence {k:8s} policy {v['policy_median']:.3f}  "
                f"replay {v['replay_median']:.3f}  gap {v['gap_median']:+.3f}")
        if "compounding_factor" in d:
            log(f"      compounding x{d['compounding_factor']:.1f} over "
                f"{d['n_steps']} steps; action divergence "
                f"{d['action_divergence_median']:.3f}")
        sh = diag.get("surprise_shift") or {}
        if "p99_ratio" in sh:
            log(f"      surprise p99 x{sh['p99_ratio']:.2f}, clip rate x"
                f"{sh['clip_rate_ratio']:.1f}  (demonstration -> rolled)")
        mr = diag.get("message_reliance") or {}
        if mr:
            log(f"      message reliance: relative shift "
                f"{mr['relative_shift']:.3f} (reliance, not value)")

    verdicts = DG.detectors(summary)
    if verdicts:
        log("[verdict]")
        for v in verdicts:
            log(f"      {v}")
        summary["verdicts"] = verdicts

    path = os.path.join(out_dir, f"summary_seed{seed}.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"[done] summary  {path}")
    log(f"[done] log      {os.path.join(out_dir, 'train.log')}")
    return summary


#: Packed-state layouts for datasets that do not write one array per field.
LAYOUTS = {"packed": DRESS.PACKED_STATE, "none": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", "--zarr", dest="data", required=True,
                    help="a .zarr store, a directory containing several, or a "
                         "comma-separated list of either")
    ap.add_argument("--out", default=None,
                    help="run directory; defaults to "
                         "data/outputs/drim_<timestamp>")
    ap.add_argument("--budget", default="standard", choices=sorted(BUDGETS),
                    help="epoch budget preset. B0/B1 are light and D2 heavy on "
                         "purpose: the first two are the baseline machinery, "
                         "the third is what is being studied, and every stage "
                         "is selected from a grid so a generous budget costs "
                         "time rather than accuracy.")
    ap.add_argument("--quick", action="store_true",
                    help="short budgets for a plumbing check, not a result")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--n-arms", type=int, default=2)
    ap.add_argument("--profile", default="dressing", choices=("dressing", "none"),
                    help="task defaults. 'dressing' applies everything in "
                         "drim/dressing.py -- the packed-state layout, the "
                         "14.3 Hz horizons, the joystick command scale, the "
                         "delta-pose action and the zigzag primitive as a "
                         "dynamics input. 'none' leaves the core defaults, "
                         "which know about no particular robot.")
    ap.add_argument("--roi", default="custom", choices=sorted(DRESS.ROIS),
                    help="field of view to crop to before resizing. custom43 "
                         "(default) is the hand-drawn box at the encoder's 3:4 "
                         "aspect; custom is the same box undistorted-by-nothing "
                         "and so resized anisotropically; wide / mid / hand are "
                         "placed by motion energy; full is no crop.")
    ap.add_argument("--cameras", default=None,
                    help="comma-separated zarr keys; empty for a state-only "
                         "run. Defaults to the profile's cameras.")
    ap.add_argument("--image-size", default=None,
                    help="resize camera frames on load, e.g. 240x320, or "
                         "'native' to keep the recorded resolution. The encoder "
                         "crops ~90%% of whatever it is given, so a frame much "
                         "larger than this is mostly discarded, not used.")
    ap.add_argument("--layout", default="profile",
                    choices=sorted(LAYOUTS) + ["profile"],
                    help="packed-state layout; 'none' expects one array per field")
    ap.add_argument("--action-key", default="action",
                    help="which recorded array is the action the policy "
                         "predicts (e.g. zigzag_action for the scripted "
                         "commanded velocity)")
    ap.add_argument("--action-mode", default=None,
                    choices=("delta_ee_pos", "absolute"),
                    help="predict the pose command relative to the observed "
                         "end-effector pose (default) or as recorded. The "
                         "absolute target is nearly the observed pose on this "
                         "rig, so predicting it is close to copying an input.")
    ap.add_argument("--exo-key", default=None,
                    help="a command injected at both teleop and inference that "
                         "the policy does not predict; it conditions the "
                         "dynamics. Empty to disable.")
    ap.add_argument("--keep", default="",
                    help="comma-separated fields to put back into the "
                         "observation that the profile excludes, e.g. "
                         "--keep q,dq. The run logs what it excluded either "
                         "way, so an ablation is a flag and not an edit.")
    ap.add_argument("--require", default="",
                    help="comma-separated fields that must resolve, e.g. "
                         "'wrench' to refuse a dataset without a contact signal")
    ap.add_argument("--epochs-dyn", type=int, default=None)
    ap.add_argument("--epochs-b0", type=int, default=None)
    ap.add_argument("--epochs-b1", type=int, default=None)
    ap.add_argument("--epochs-d2", type=int, default=None)
    ap.add_argument("--vision-weights", default=None,
                    help="resnet18 initialisation. None (default, matching "
                         "dap) trains it from scratch; IMAGENET1K_V1 starts "
                         "from the pretrained weights, which with a few "
                         "hundred chunks is likely the larger effect.")
    ap.add_argument("--no-diagnose", action="store_true",
                    help="skip the pre-deployment diagnostics (trajectory "
                         "divergence through the dynamics, surprise shift, "
                         "message reliance)")
    ap.add_argument("--diag-steps", type=int, default=32)
    ap.add_argument("--diag-starts", type=int, default=48)
    ap.add_argument("--estimator", default=SEL.DEFAULT_ESTIMATOR,
                    choices=list(SEL.ESTIMATORS),
                    help="offline criterion the deployed checkpoint is chosen "
                         "by: action_mse (default; the executed-prefix error, "
                         "the same metric for every stage), val_loss, or "
                         "divergence (roll each candidate through the dynamics "
                         "-- the only one that sees error compound). last1 / "
                         "last3 / last5 select on nothing and are kept only to "
                         "reproduce earlier runs. Every criterion's pick is "
                         "reported whichever is used.")
    ap.add_argument("--no-keep-grid", action="store_true",
                    help="do not store the epoch grid in the checkpoint; saves "
                         "disk, but the selection can no longer be re-derived")
    ap.add_argument("--fast-frac", default=str(FAST_FRAC_DEFAULT),
                    help="the fast level's authority as a fraction of full "
                         "command (LIN_SCALE 0.02 m/s, ANG_SCALE 0.05 rad/s). "
                         f"{FAST_FRAC_DEFAULT} by default -- a declared reflex "
                         "authority, ~5 mm of pose delta, about 30%% of the "
                         "operator's median nudge. 'auto' measures it from a "
                         "trained B0 instead, capped at "
                         f"{FAST_FRAC_CAP}. "
                         "Channels the data never moves get 0. dap's own arms "
                         "sit at 0.016 (cap) and 0.16 (drawer).")
    a = ap.parse_args(argv)
    prof = DRESS.profile(roi=a.roi) if a.profile == "dressing" else {}
    if a.cameras is not None:
        prof["cameras"] = tuple(c for c in a.cameras.split(",") if c)
    if a.action_mode is not None:
        prof["action_mode"] = a.action_mode
    if a.exo_key is not None:
        prof["exo_key"] = (a.exo_key or None)
    if a.image_size is not None:
        prof["image_size"] = (None if a.image_size == "native"
                              else tuple(int(v) for v in
                                         a.image_size.lower().split("x")))
    if a.layout != "profile":
        prof["layout"] = LAYOUTS[a.layout]
    ep = dict(BUDGETS[a.budget])
    for k, v in (("dyn", a.epochs_dyn), ("B0", a.epochs_b0),
                 ("B1", a.epochs_b1), ("D2", a.epochs_d2)):
        if v is not None:
            ep[k] = v
    out = a.out or os.path.join("data", "outputs", run_name(prof, ep, a))
    if a.quick:
        #: Enough to exercise every stage and the exact-null check; far too few
        #: to mean anything, which is the point of a separate flag rather than a
        #: quietly small default.
        ep = {"dyn": 10, "B0": 6, "B1": 4, "D2": 4}
    run([p for p in a.data.split(",") if p], out, seed=a.seed,
        device=a.device,
        stages=tuple(a.stages.split(",")), val_ratio=a.val_ratio,
        cameras=prof.pop("cameras", ()),
        n_arms=a.n_arms, layout=prof.pop("layout", None),
        fast_frac=_parse_frac(a.fast_frac, prof.pop("fast_frac", None)),
        require=tuple(r for r in a.require.split(",") if r),
        action_key=a.action_key,
        action_mode=prof.pop("action_mode", "absolute"),
        exclude=_ablate(prof.pop("exclude", ()), a.keep),
        exo_key=prof.pop("exo_key", None),
        image_size=prof.pop("image_size", None),
        act_scale_per_arm=prof.pop("act_scale_per_arm", None),
        act_per_arm=prof.pop("act_per_arm", None),
        roi=prof.pop("roi", None),
        photometric=prof.pop("photometric", None),
        vision_weights=a.vision_weights,
        horizons=prof,
        estimator=a.estimator, keep_grid=not a.no_keep_grid, epochs=ep,
        diagnose=not a.no_diagnose, diag_steps=a.diag_steps,
        diag_starts=a.diag_starts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
