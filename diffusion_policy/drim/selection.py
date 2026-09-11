"""Choosing one checkpoint to put on the robot, with no rollout to rank by.

This is the **real-world** track: a run ends with exactly one set of weights
that gets driven on hardware. That constraint is what shapes everything here,
and it is stronger than the one dap works under.

**Why dap's ``last-k`` does not carry over.** In dap, ``last3`` averages the
*rollout success* of the final three snapshots. That is an estimator of how well
a training recipe does, computed by testing three checkpoints. Here nothing can
be tested: there is no ``EnvRunner``, and on hardware you get one deployment,
not three. So "the last three" names no measurement — it is just "roughly the
end of training", and this repository's own 0909 run shows what that costs: D2's
held-out loss bottoms at epoch 29 and rises steadily to epoch 150, so a tail
average deploys the overfitted model on purpose.

**So the checkpoint has to be selected offline, and selected well.** The
criteria available, in increasing fidelity to what the robot will do:

``val_loss``     the stage's own training objective on held-out episodes.
                 Cheapest, and furthest from deployment — for the slow policy it
                 is a flow-matching loss, not an action error.
``action_mse``   error of the **executed prefix** against the demonstrated
                 action, averaged over several noise draws. The same quantity
                 for every stage, which is what makes ``D2 - B1`` a comparison
                 rather than two different rulers.
``divergence``   the policy rolled forward through the learned dynamics, scored
                 on how far the state drifts from the demonstration. The only
                 criterion here that sees error *compound*, which is the failure
                 mode single-step metrics are blind to. Costs a short rollout per
                 candidate, so it is opt-in.

**Selection noise is the thing to control, not to avoid by refusing to select.**
An argmin over a grid of noisy estimates is biased low and, on a plateau, picks
near-randomly within it — dap measured epoch-to-epoch spreads of 0.011-0.040 in
success. The answer is to smooth the *criterion* before taking the argmin
(``smooth``: rank each epoch by the mean of its neighbours) rather than to
average the *weights* and hope. Smoothing keeps the result a real, single,
deployable checkpoint.

Whatever criterion is chosen, every other one's pick is recorded beside it, so a
disagreement is visible as data. On the 0909 run all three stages disagreed
between ``val_loss`` and the tail, which is exactly the signal that the tail was
the wrong place to look.
"""

from __future__ import annotations

import inspect as _inspect
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


#: ``torch.load`` gained ``weights_only`` in 1.13 and flipped its default to
#: ``True`` in 2.6.  These checkpoints carry a :class:`~diffusion_policy.drim.
#: spec.DrimSpec` and other plain objects alongside the tensors, so the unpickler
#: has to stay permissive -- and the cluster's usable env is torch 1.12, which
#: rejects the keyword outright.  Asking the signature is the only thing that
#: works on both.
_WEIGHTS_ONLY = "weights_only" in _inspect.signature(torch.load).parameters


def torch_load(path: str, map_location: str = "cpu") -> Any:
    """``torch.load`` with full unpickling, on any torch back to 1.12."""
    if _WEIGHTS_ONLY:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)

#: Snapshot period, in epochs.  dap's grid is never truncated to the tail — two
#: of nine arm-seeds in its image round peaked at epoch 90 of 150, and where the
#: plateau starts differs by seed and is not knowable in advance.
SNAPSHOT_EVERY = 10
LAST_K = 3
#: Criteria that pick one checkpoint offline.  ``bestval`` is kept as an alias
#: for ``val_loss`` because earlier runs recorded it under that name.
CRITERIA = ("action_mse", "val_loss", "divergence")
#: Everything selectable.  ``last1`` is the final snapshot; ``last3``/``last5``
#: weight-average the tail and are kept only to reproduce an earlier run — they
#: select on nothing, which on a rising tail means deploying the overfit.
ESTIMATORS = CRITERIA + ("bestval", "last1", "last3", "last5", "all")
DEFAULT_ESTIMATOR = "action_mse"
#: grid points either side of a candidate whose criterion is averaged into it
SMOOTH = 1
#: Nothing here ranks by a score; recorded so a checkpoint says so on its face.
SELECTION_RULE = "offline_criterion"


def snapshot_grid(epochs: int, every: int = SNAPSHOT_EVERY) -> Tuple[int, ...]:
    """1-based epochs at which a snapshot is taken; the last epoch always is."""
    g = [e for e in range(every, epochs + 1, every)]
    if not g or g[-1] != epochs:
        g.append(epochs)
    return tuple(g)


def state_minus_parent(model, parent_keys: Optional[Sequence[str]]
                       ) -> Dict[str, torch.Tensor]:
    """Everything the parent this stage freezes cannot restore.

    Not ``requires_grad``-filtered, which is the trap dap fell into.  A
    ``requires_grad`` filter loses three kinds of tensor, each of which reloads
    without error and then scores wrongly:

    * buffers (``slow.obs_mean``, ``fast.max_residual``) — no ``requires_grad``
      filter sees them, and a policy is unusable or silently mis-scaled without;
    * ``requires_grad=False`` Parameters (the vision encoder's dummy variable);
    * modules frozen but **freshly initialised**, so no parent holds them either
      (``dv`` in a conditioned stage) — these come back as random weights.

    ``None`` means no parent, so the snapshot is whole.
    """
    sd = model.state_dict()
    if parent_keys is None:
        return {k: v.detach().cpu().clone() for k, v in sd.items()}
    keep = set(sd) - set(parent_keys)
    return {k: sd[k].detach().cpu().clone() for k in sorted(keep)}


class SnapshotBank:
    """Accumulates snapshots at the pre-declared grid epochs.

    Held in memory during training and written once at the end, so a crashed run
    leaves no half-written grid.  ``val_loss`` is carried for the log and the
    curve; no selector here consults it.
    """

    def __init__(self, epochs: int, every: int = SNAPSHOT_EVERY,
                 parent_keys: Optional[Sequence[str]] = None) -> None:
        self.epochs = int(epochs)
        self.grid = snapshot_grid(epochs, every)
        self.parent_keys = None if parent_keys is None else list(parent_keys)
        self.states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.curve: List[Dict[str, Any]] = []

    def observe(self, model, epoch: int, **metrics) -> bool:
        """Record epoch ``epoch`` (1-based); snapshot if it is a grid point."""
        row = {"epoch": int(epoch)}
        row.update({k: float(v) for k, v in metrics.items() if v is not None})
        self.curve.append(row)
        if epoch in self.grid:
            self.states[int(epoch)] = state_minus_parent(model, self.parent_keys)
            return True
        return False

    @property
    def latest(self) -> Optional[int]:
        return max(self.states) if self.states else None

    def summary(self) -> Dict[str, Any]:
        return {"snapshot_epochs": list(self.grid),
                "snapshots_taken": sorted(self.states),
                "curve": self.curve,
                "selection": {"rule": SELECTION_RULE,
                              "snapshot_every": SNAPSHOT_EVERY,
                              "estimators": list(ESTIMATORS),
                              "default": DEFAULT_ESTIMATOR,
                              "rollout_available": False,
                              "val_loss_used_for_selection": False,
                              "action_mse_used_for_selection": False}}


def _criterion_key(estimator: str) -> str:
    return "val_loss" if estimator == "bestval" else estimator


def smoothed(curve: Sequence[Dict[str, Any]], key: str, grid: Sequence[int],
             smooth: int = SMOOTH) -> Dict[int, float]:
    """Each grid epoch scored by the mean of ``key`` over its neighbours.

    The argmin of a raw noisy curve is biased low and, on a plateau, arbitrary.
    Averaging the *criterion* over a small window damps that while still naming
    a single real checkpoint — unlike averaging the weights, which names one
    that was never evaluated.
    """
    g = sorted(int(e) for e in grid)
    have = {int(r["epoch"]): float(r[key]) for r in curve if key in r}
    scored: Dict[int, float] = {}
    for i, e in enumerate(g):
        lo, hi = max(0, i - smooth), min(len(g), i + smooth + 1)
        vals = [have[x] for x in g[lo:hi] if x in have]
        if vals:
            scored[e] = float(np.mean(vals))
    return scored


def select_epochs(epochs: Sequence[int], estimator: str = DEFAULT_ESTIMATOR,
                  curve: Optional[Sequence[Dict[str, Any]]] = None,
                  key: str = "val_loss", smooth: int = SMOOTH,
                  scores: Optional[Dict[int, float]] = None) -> List[int]:
    """Which grid epochs the saved checkpoint is built from.

    A criterion returns **one** epoch — the checkpoint that would be deployed.
    ``last{k}`` returns k, and the caller then weight-averages them; that path
    exists to reproduce an earlier run, not because it selects anything.

    ``scores`` supplies an externally computed criterion (``divergence``, which
    needs a rollout per candidate and so cannot be read off the training curve).
    Lower is better for every criterion here.
    """
    eps = sorted(int(e) for e in epochs)
    if estimator == "all":
        return eps
    if estimator.startswith("last"):
        return eps[-int(estimator[4:] or LAST_K):]
    if estimator == "divergence":
        if not scores:
            raise ValueError(
                "the divergence criterion needs a score per candidate epoch; "
                "it is computed by rolling each one out through the dynamics")
        return [min(scores, key=lambda e: scores[e])]
    k = _criterion_key(estimator)
    if k not in CRITERIA and estimator != "bestval":
        raise ValueError(
            f"unknown estimator {estimator!r}; have {ESTIMATORS}. Estimators "
            f"that rank by rollout success are unavailable — there is no "
            f"environment to roll out in.")
    if not curve:
        raise ValueError(f"the {estimator!r} criterion needs the training curve")
    sc = smoothed(curve, k, eps, smooth)
    if not sc:
        raise ValueError(f"no grid epoch carries {k!r}")
    return [min(sc, key=lambda e: sc[e])]


def overfit_warning(bank: "SnapshotBank", key: str = "val_loss",
                    n: int = 5) -> Optional[str]:
    """Whether the tail ``last-k`` averages is still going the wrong way.

    ``last-k`` assumes a plateau.  A validation loss **rising** across the final
    grid points means the run is past its useful budget, the tail is not a
    plateau, and averaging it averages the overfitting.  That is the one
    situation where validation loss should override the estimator — not by
    selecting the argmin, but by telling you the budget was wrong.
    """
    s = tail_slope(bank.curve, key, n=n, grid=bank.grid)
    if s is None or s <= 0:
        return None
    return (f"{key} is rising at {s:+.5f} per 10 epochs over the last {n} grid "
            f"points: the tail is not a plateau, so averaging it averages the "
            f"overfitting. Shorten the budget rather than switching estimator.")


def average_states(bank: SnapshotBank, epochs: Sequence[int]
                   ) -> Dict[str, torch.Tensor]:
    """Parameter-average the named grid points — the ``last-k`` checkpoint.

    Averaging weights is what dap's ``last3`` reports, and it is sound here for
    the same reason it is sound there: the points are consecutive grid epochs of
    one run on a plateau, so they sit in one basin.  It is *not* sound across
    seeds or across stages, and this function does not let you do either — it
    only ever sees one bank.

    Integer and boolean buffers are taken from the last epoch rather than
    averaged; a rounded mean of a step counter is not a step counter.
    """
    eps = [int(e) for e in epochs]
    missing = [e for e in eps if e not in bank.states]
    assert not missing, f"epochs {missing} are not in the grid {sorted(bank.states)}"
    ref = bank.states[eps[-1]]
    out: Dict[str, torch.Tensor] = {}
    for k, v in ref.items():
        if not torch.is_floating_point(v):
            out[k] = v.clone()
            continue
        acc = torch.zeros_like(v, dtype=torch.float64)
        for e in eps:
            acc += bank.states[e][k].to(torch.float64)
        out[k] = (acc / len(eps)).to(v.dtype)
    return out


def tail_slope(curve: Sequence[Dict[str, Any]], key: str, n: int = 5,
               grid: Optional[Sequence[int]] = None,
               every: int = SNAPSHOT_EVERY) -> Optional[float]:
    """Least-squares slope of ``key`` over the last ``n`` grid points, per 10 epochs.

    The check last-K depends on.  A tail that is still moving means the budget
    was too short, and the answer is a longer budget — not a different estimator.

    ``grid`` is the bank's own snapshot epochs.  Selecting by ``epoch % every``
    instead silently drops the final point whenever the budget is not a multiple
    of the period — exactly the point the tail is being measured at.
    """
    keep = set(grid) if grid is not None else None
    pts = [(r["epoch"], r[key]) for r in curve if key in r
           and (r["epoch"] in keep if keep is not None
                else r["epoch"] % every == 0)]
    if len(pts) < max(n, 2):
        return None
    x = np.array([p[0] for p in pts[-n:]], dtype=np.float64)
    y = np.array([p[1] for p in pts[-n:]], dtype=np.float64)
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0] * 10.0)


def selection_note(bank: SnapshotBank, estimator: str = DEFAULT_ESTIMATOR,
                   diagnostics: Sequence[str] = ("val_loss", "action_mse"),
                   smooth: int = SMOOTH,
                   scores: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    """What a checkpoint records about how it was chosen — and how it was not.

    The historical keys (``best_epoch`` and the minimum of each diagnostic) are
    kept and clearly marked unused, so the number the old protocol *would* have
    picked stays visible on the record next to the one that was.
    """
    eps = select_epochs(sorted(bank.states), estimator, curve=bank.curve,
                        smooth=smooth, scores=scores)
    note: Dict[str, Any] = {
        "rule": SELECTION_RULE, "estimator": estimator,
        "selected_epochs": eps, "grid": list(bank.grid),
        "rollout_available": False,
    }
    #: What the ordinary selector would have picked, recorded next to what was,
    #: so the disagreement is visible every run instead of being a matter of
    #: opinion.
    #: Every other criterion's pick, recorded beside the one that was used, so
    #: a disagreement is data rather than an opinion.
    note["would_pick"] = {}
    for alt in ("val_loss", "action_mse", "last1"):
        try:
            note["would_pick"][alt] = select_epochs(
                sorted(bank.states), alt, curve=bank.curve, smooth=smooth)[0]
        except (ValueError, KeyError):
            pass
    if scores:
        note["would_pick"]["divergence"] = min(scores, key=lambda e: scores[e])
        note["divergence_scores"] = {int(k): float(v) for k, v in scores.items()}
    note["criteria_agree"] = len(set(note["would_pick"].values())) == 1
    w = overfit_warning(bank)
    if w:
        note["overfit_warning"] = w
    for key in diagnostics:
        vals = [(r[key], r["epoch"]) for r in bank.curve if key in r]
        if not vals:
            continue
        lo, lo_ep = min(vals)
        note[f"min_{key}"] = float(lo)
        note[f"min_{key}_epoch"] = int(lo_ep)
        note[f"final_{key}"] = float(vals[-1][0])
        note[f"{key}_used_for_selection"] = False
        s = tail_slope(bank.curve, key, grid=bank.grid)
        if s is not None:
            note[f"{key}_tail_slope_per_10ep"] = s
    return note


def save_selected(path: str, bank: SnapshotBank, variant: str, spec_dict: Dict[str, Any],
                  estimator: str = DEFAULT_ESTIMATOR,
                  extra: Optional[Dict[str, Any]] = None,
                  keep_grid: bool = True, smooth: int = SMOOTH,
                  scores: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    """Write the ``last-k`` checkpoint, with the grid it was built from beside it.

    ``keep_grid=False`` drops the snapshots and keeps only the averaged weights.
    It saves a lot of disk on a stage with no parent — ``B0`` snapshots are whole
    models — at the cost of being unable to re-derive the selection later, which
    is the one thing keeping the grid buys.
    """
    eps = select_epochs(sorted(bank.states), estimator, curve=bank.curve,
                        smooth=smooth, scores=scores)
    payload = {
        "variant": variant,
        "spec": spec_dict,
        "state_dict": average_states(bank, eps),
        "trainable_only": bank.parent_keys is not None,
        "parent_keys": bank.parent_keys,
        "selection": selection_note(bank, estimator, smooth=smooth,
                                    scores=scores),
        "summary": bank.summary(),
    }
    if keep_grid:
        payload["grid_states"] = {int(e): bank.states[e]
                                  for e in sorted(bank.states)}
    if extra:
        payload["train"] = extra
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    return payload["selection"]


def load_selected(path: str, model, strict_parent: bool = True,
                  map_location: str = "cpu") -> Dict[str, Any]:
    """Load a saved ``last-k`` checkpoint into a model already nested on its parent.

    A trainable-only checkpoint is loaded with ``strict=False`` by necessity, so
    the missing keys are checked *here* against what the parent was supposed to
    supply.  dap's rule: unexpected keys are always an error; missing keys are an
    error unless the parent already holds every one of them.
    """
    ck = torch_load(path, map_location=map_location)
    res = model.load_state_dict(ck["state_dict"], strict=False)
    assert not res.unexpected_keys, f"checkpoint has unknown keys: {res.unexpected_keys[:8]}"
    if strict_parent and ck.get("trainable_only"):
        parent = set(ck.get("parent_keys") or ())
        orphan = [k for k in res.missing_keys if k not in parent]
        assert not orphan, (
            f"{len(orphan)} keys are in neither the checkpoint nor its parent, "
            f"so they are freshly initialised: {orphan[:8]}")
    return ck
