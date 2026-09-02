"""Staged IRUM training: ``B0 -> B1 -> D2``, offline, no environment.

Budgets, losses, batch order, cached nominal draws and the snapshot grid follow
``cap_constraint_benchmark/liftoff_v6_image/train_lo6.py``.  Two structural
differences, both forced and both marked at the site:

* selection is :mod:`diffusion_policy.irum.selection` — an epoch grid reported by
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
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_policy.irum import dynamics as DY
from diffusion_policy.irum import policy as PL
from diffusion_policy.irum import selection as SEL
from diffusion_policy.irum import fields as F
from diffusion_policy.irum.dataset import (ChunkNormaliser, load_split,
                                           action_residual_demand)
from diffusion_policy.irum.spec import (IrumSpec, DRESSING_HORIZONS,
                                        DRESSING_CAMERAS,
                                        fast_limits_from_fraction)

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


def _ctx(model: PL.IrumPolicy, b: Dict[str, torch.Tensor]) -> torch.Tensor:
    rgb = ({c: b[f"rgb2_{c}"] for c in model.spec.cameras}
           if model.spec.is_image else None)
    return model.context(rgb, b["prop2"], b.get("wrench2"))


# --------------------------------------------------------------------------- #
# stage: dynamics (the source of D2's S and U)
# --------------------------------------------------------------------------- #


def train_dynamics(spec: IrumSpec, tr, va, seed: int, epochs: int, device: str,
                   log: Callable[[str], None], mods) -> Dict[str, Any]:
    """Fit the delta dynamics whose residual becomes the message.

    Reported against the **no-change baseline**, always.  dap ran a whole round
    on a dynamics model that was 155 % *worse* than predicting no change while
    its NLL looked fine — NLL can be driven down by shrinking sigma while the
    point predictions stay useless, and the message then encodes model error
    rather than physical evidence.  If ``skill`` below is negative the message is
    noise and D2 is not worth training yet.
    """
    torch.manual_seed(seed + 3000)
    assert DY.state_dim(mods) == spec.prop_dim, (
        f"modality table covers {DY.state_dim(mods)} channels, spec declares "
        f"{spec.prop_dim}")

    def flat(d, key):
        a = d[key]
        return a.reshape(-1, a.shape[-1])

    def prep(d):
        y = flat(d, "hist_y")
        yn = flat(d, "hist_y_next")
        a = flat(d, "hist_a")
        dt = flat(d, "hist_dt")
        keep = np.repeat(d["msg_valid"] > 0, d["hist_y"].shape[1])
        return y[keep], yn[keep], a[keep], dt[keep]

    ytr, yntr, atr, dttr = prep(tr)
    yva, ynva, ava, dtva = prep(va)
    assert len(ytr), (
        "no decision step has a full message window behind it; lower "
        "spec.message_window or record longer episodes")
    dtr = DY.state_delta(ytr, yntr, mods)
    norm = DY.Normaliser(y=ytr, d=dtr)
    baseline = DY.no_change_baseline(norm.d(dtr))

    model = DY.DeltaDynamics(mods, spec.act_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_DYN,
                            weight_decay=WEIGHT_DECAY)
    ty = torch.as_tensor(norm.y(ytr), device=device)
    ta = torch.as_tensor(atr, dtype=torch.float32, device=device)
    tdt = torch.as_tensor(dttr, dtype=torch.float32, device=device)
    td = torch.as_tensor(norm.d(dtr), device=device)
    rng = np.random.default_rng(seed + 3000)
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for sel in _batches(len(ty), BATCH, rng):
            i = torch.as_tensor(sel, device=device)
            mu, ls = model(ty[i], ta[i], tdt[i])
            loss = DY.gaussian_nll(mu, ls, td[i])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(sel)
        if (ep + 1) % max(epochs // 5, 1) == 0:
            log(f"      dyn epoch {ep + 1}/{epochs} nll={tot / len(ty):.5f}")

    model.eval()
    frozen = DY.FrozenDynamics(model, norm).to(device)
    dva = DY.state_delta(yva, ynva, mods)
    with torch.no_grad():
        t_ava = torch.as_tensor(ava, dtype=torch.float32, device=device)
        t_dtva = torch.as_tensor(dtva, dtype=torch.float32, device=device)
        mu, _ = frozen(torch.as_tensor(yva, device=device), t_ava, t_dtva)
        mse = float(((torch.as_tensor(norm.d(dva), device=device) - mu) ** 2)
                    .mean().item())
        ch = DY.surprise(frozen, torch.as_tensor(yva, device=device),
                         torch.as_tensor(ynva, device=device), t_ava, t_dtva)
    base_va = DY.no_change_baseline(norm.d(dva))
    skill = (base_va - mse) / max(base_va, 1e-12)
    log(f"      dyn  mse={mse:.6f}  no-change baseline={base_va:.6f}  "
        f"skill={skill:+.1%}"
        + ("" if skill > 0 else "   <-- WORSE THAN DOING NOTHING"))
    return {"dynamics": frozen, "mods": mods, "norm": norm,
            "use_dt": bool(model.use_dt),
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
    with torch.no_grad():
        ch = DY.surprise(frozen, y, yn, a, dt)
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


@torch.no_grad()
def _valid_action_mse(model, va, device, batch=64, seed=NOISE_SEED,
                      message_of=None) -> float:
    """Offline MSE of the **executed prefix** against the demonstrated action.

    Logged for every stage including ``B0``, where the executed action is the
    nominal.  This is the closest offline stand-in for what the robot receives,
    which flow loss is not — but it is a diagnostic, never a selector; see
    :mod:`diffusion_policy.irum.selection` for why.
    """
    model.eval()
    e = model.exec_horizon
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot = n = 0.0
    for i in range(0, len(va["target"]), batch):
        sel = slice(i, i + batch)
        b = _batch(va, sel, device)
        noise = torch.randn(len(b["target"]), model.horizon, model.act_dim,
                            generator=g).to(device)
        msg = message_of(model, b) if message_of is not None else None
        ctx = _ctx(model, b)
        nominal = model.sample_chunk(ctx, msg, noise)
        r = model.residual_prefix(nominal, b["step_prop"], b.get("step_wrench"),
                                  msg)
        exe = torch.clamp(nominal[:, :e] + r, -1.0, 1.0)
        tot += torch.nn.functional.mse_loss(
            exe, b["target"][:, :e]).item() * len(b["target"])
        n += len(b["target"])
    model.train()
    return tot / max(n, 1.0)


def train_b0(spec, tr, va, seed, epochs, device, log) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = PL.build("B0", spec).to(device)
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
        am = _valid_action_mse(model, va, device)
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
        if bank.observe(model, ep + 1, val_loss=vl, action_mse=vl,
                        train_loss=tot / n, saturation=sat):
            log(f"      s{seed}/B1 epoch {ep + 1}/{epochs} train={tot / n:.5f} "
                f"val={vl:.5f} saturated={sat:.1%} [snapshot]")
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
        am = _valid_action_mse(model, va, device, message_of=_message)
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
def check_exact_null(d2: PL.IrumPolicy, b: Dict[str, torch.Tensor],
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
        cameras: Sequence[str] = DRESSING_CAMERAS, n_arms: int = 2,
        layout: Optional[F.PackedLayout] = None,
        fast_frac: Optional[float] = None,
        require: Sequence[str] = (),
        estimator: str = f"last{SEL.LAST_K}", keep_grid: bool = True,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    unknown = [s for s in stages if s not in STAGES]
    assert not unknown, f"unknown stage(s) {unknown}; have {STAGES}"
    #: Each stage nests on the trained parent held in memory, so a run cannot
    #: start in the middle. Saying so here beats a KeyError three stages later.
    for child, parent in (("B1", "B0"), ("D2", "B1"), ("D2", "dyn")):
        assert not (child in stages and parent not in stages), (
            f"stage {child} nests on {parent}, which is not in {list(stages)}")
    epochs = {"dyn": 100, "B0": 60, "B1": 40, "D2": 40, **(epochs or {})}
    os.makedirs(out_dir, exist_ok=True)
    log(f"[data] {zarr_path}")
    tr, va, norm, eps, spec = load_split(
        zarr_path, cameras=cameras, n_arms=n_arms, layout=layout,
        val_ratio=val_ratio, seed=seed, require=require,
        spec_kw=dict(**DRESSING_HORIZONS))
    if fast_frac is not None:
        #: Only channels the demonstrations actually move get authority.  A
        #: ceiling on a channel that is identically zero in the data is
        #: authority over something the corrector can never have learned, and
        #: it is free to write there at rollout; a zero ceiling makes the
        #: channel structurally silent instead.
        act = eps.action_report(spec.act_channels)
        spec = IrumSpec(**{**spec.to_dict(),
                           "fast_limits": fast_limits_from_fraction(
                               fast_frac, spec.act_scale or
                               tuple([1.0] * spec.act_dim),
                               active=act["active"])})
        if not any(spec.fast_limits):
            raise ValueError(
                "no action channel is ever nonzero in this dataset, so every "
                "ceiling is 0 and the corrector cannot write anything; B1 and "
                "D2 would train a no-op. Check the recording.")

    res = eps.resolution
    log(f"[fields] usable: {', '.join(res.usable) or '(none)'}")
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
    if act["dead"]:
        log(f"[data] action channels never nonzero: {act['dead']} "
            f"(active: {act['active']})")
    demand = action_residual_demand(tr, spec)
    if demand:
        for i in act["active"]:
            log(f"[authority] ch{i} residual demand (fraction of full command): "
                f"vs_chunk_mean p95={demand['vs_chunk_mean']['p95'][i]:.4f} "
                f"max={demand['vs_chunk_mean']['p100'][i]:.4f} | "
                f"step_to_step p95={demand['step_to_step']['p95'][i]:.4f}")
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
        "fields": {"usable": list(res.usable),
                   "missing": [{"arm": s.arm, "field": s.name,
                                "reason": s.reason} for s in res.missing()],
                   "notes": [{"arm": s.arm, "field": s.name, "note": s.note}
                             for s in res.statuses if s.note]},
        "action_channels": act,
        "residual_demand": demand,
        "dt": getattr(eps, "dt_stats", None),
        "normaliser": norm.state_dict()}

    frozen = None
    if "dyn" in stages or "D2" in stages:
        log("[stage] dyn")
        mods = F.modalities(res, spec.prop_fields)
        dyn = train_dynamics(spec, tr, va, seed, epochs["dyn"], device, log, mods)
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
    models: Dict[str, PL.IrumPolicy] = {}

    def _save(stage: str, r: Dict[str, Any]) -> Dict[str, Any]:
        sel = SEL.save_selected(
            os.path.join(out_dir, f"{stage}_seed{seed}.pt"), r["bank"], stage,
            spec.to_dict(), estimator=estimator, keep_grid=keep_grid,
            extra={k: v for k, v in r.items() if k != "bank"})
        out = {k: v for k, v in r.items() if k != "bank"}
        out["selection"] = sel
        return out

    if "B0" in stages:
        log("[stage] B0")
        r = train_b0(spec, t_tr, t_va, seed, epochs["B0"], device, log)
        models["B0"] = r.pop("model")
        summary["B0"] = _save("B0", r)

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

    path = os.path.join(out_dir, f"summary_seed{seed}.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"[done] {path}")
    return summary


#: Packed-state layouts for datasets that do not write one array per field.
LAYOUTS = {"smoke": F.SMOKE_LAYOUT, "none": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", "--zarr", dest="data", required=True,
                    help="a .zarr store, a directory containing several, or a "
                         "comma-separated list of either")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--n-arms", type=int, default=2)
    ap.add_argument("--cameras", default=",".join(DRESSING_CAMERAS),
                    help="comma-separated zarr keys; empty for a state-only run")
    ap.add_argument("--layout", default="smoke", choices=sorted(LAYOUTS),
                    help="packed-state layout; 'none' expects one array per field")
    ap.add_argument("--require", default="",
                    help="comma-separated fields that must resolve, e.g. "
                         "'wrench' to refuse a dataset without a contact signal")
    ap.add_argument("--epochs-dyn", type=int, default=100)
    ap.add_argument("--epochs-b0", type=int, default=60)
    ap.add_argument("--epochs-b1", type=int, default=40)
    ap.add_argument("--epochs-d2", type=int, default=40)
    ap.add_argument("--estimator", default=f"last{SEL.LAST_K}",
                    help="which grid epochs the saved checkpoint averages: "
                         "last3 (default), last5, last1 (final snapshot only). "
                         "Score-ranked estimators are refused — there is no "
                         "rollout to rank with.")
    ap.add_argument("--no-keep-grid", action="store_true",
                    help="do not store the epoch grid in the checkpoint; saves "
                         "disk, but the selection can no longer be re-derived")
    ap.add_argument("--fast-frac", type=float, default=None,
                    help="the fast level's authority as a fraction of full "
                         "command (LIN_SCALE 0.02 m/s, ANG_SCALE 0.05 rad/s); "
                         "required for B1 and D2. Channels the data never moves "
                         "get 0. dap's own arms sit at 0.016 (cap) and 0.16 "
                         "(drawer) of full command.")
    a = ap.parse_args(argv)
    run([p for p in a.data.split(",") if p], a.out, seed=a.seed,
        device=a.device,
        stages=tuple(a.stages.split(",")), val_ratio=a.val_ratio,
        cameras=tuple(c for c in a.cameras.split(",") if c),
        n_arms=a.n_arms, layout=LAYOUTS[a.layout], fast_frac=a.fast_frac,
        require=tuple(r for r in a.require.split(",") if r),
        estimator=a.estimator, keep_grid=not a.no_keep_grid,
        epochs={"dyn": a.epochs_dyn, "B0": a.epochs_b0,
                "B1": a.epochs_b1, "D2": a.epochs_d2})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
