"""Checkpoint selection with no environment to roll out in.

dap's protocol (``benchmarks/common/selection.py``, and the reasoning in
``results/CHECKPOINT_SELECTION_NOTES.md``) is:

    training writes snapshots on an epoch grid; a **separate** stage rolls every
    snapshot out on a fixed bank of start states, ranks by success, keeps the top
    K plus the latest, and reports ``best`` / ``last3`` / ``last5``.  Validation
    loss is logged and never selected on.

That protocol exists because the selector turned out to be a first-order
experimental choice rather than a formality.  On the one round that kept a grid,
moving from "restore the lowest-val-loss epoch" to a last-3 average shifted arms
by -0.044 to **+0.130** success and turned the round's headline from -0.151 to
-0.053 with a CI covering zero — larger than nearly every effect that project
reports.  So "just use val loss" is not a neutral fallback.

**What does not survive the port.**  The dressing task here has no ``EnvRunner``
— the real rig is a physical robot — so the ranking stage is unavailable and
nothing can be scored by success.

**What does survive.**  Read dap's :func:`select_epochs` closely and the
estimators split in two:

    ``best`` / ``top-k``    rank by rollout success        — unavailable here
    ``last-k`` / ``all``    **rank nothing**               — pure epoch position

``last-k`` never consults a score, which is why it survives the absence of a
rollout, and it is the estimator that beat val-loss selection in dap's own
measurement.

**One substitution, stated plainly because it is not a like-for-like port.**  In
dap, ``last3`` is a *reporting* estimator: it averages the rollout **metrics** of
the final three snapshots, while the checkpoint actually promoted is chosen by
rollout top-1.  Here there is no metric to average and no ranking to promote by,
so ``last-k`` instead names a **weight average** of those snapshots — one
deployable checkpoint, constructed without consulting any score.  That is
standard practice (SWA / model soups) and it is sound in this setting for
reasons worth checking rather than assuming: the points are consecutive grid
epochs of one run under a cosine schedule, so they sit in one basin; the
normalisation layers are GroupNorm, so there are no batch statistics to
invalidate; and the non-parameter buffers that travel in the state dict
(``obs_mean``/``obs_std``, ``fast.max_residual``) are constants, so averaging
returns them unchanged.  Averaging across **seeds** or **stages** is not sound
and :func:`average_states` cannot do it — it only ever sees one bank.

``--estimator last1`` is the conservative alternative: the final snapshot alone,
no averaging.

So the offline protocol is the online one with the ranking stage deleted:

    ==========================  =============================================
    training budget             fixed in advance, never tuned on the result
    snapshot period             every ``ROLLOUT_EVERY`` epochs, whole run
    reported                    ``last3`` (primary), ``last5``, ``all``
    validation loss             logged, never selected on
    offline action MSE          logged, never selected on
    stages                      the same rule at B0, B1 and D2
    handoff                     the checkpoint frozen into the next stage is
                                the one the table reports
    ==========================  =============================================

**Why not just select on the offline action MSE.**  It is tempting — it is the
deployed quantity, unlike flow loss — and it is computed here and reported.  But
dap measured offline loss ordering these variants across a 3 % spread against a
0.44 spread in success, i.e. it barely orders them at all, and selecting on a
metric that does not order is how you get a selector whose variance exceeds the
effect.  It is a diagnostic here for the same reason val loss is.

**What last-K assumes, stated so it can be checked.**  That the run has
plateaued: averaging the tail is only sound where the tail is flat.  dap found
3 of 9 arm-seeds still climbing at 150 epochs and recorded it as an observation
rather than a gate.  :func:`tail_slope` computes the same number here — check it
before believing a last-K number, and extend the budget rather than moving the
estimator if the tail is still rising.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

#: Snapshot period, in epochs.  dap's grid is never truncated to the tail — two
#: of nine arm-seeds in its image round peaked at epoch 90 of 150, and where the
#: plateau starts differs by seed and is not knowable in advance.
SNAPSHOT_EVERY = 10
LAST_K = 3
ESTIMATORS = ("last3", "last5", "last1", "bestval", "all")
#: Nothing here ranks by a score; recorded so a checkpoint says so on its face.
SELECTION_RULE = "offline_last_k"


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
                              "last_k": LAST_K,
                              "rollout_available": False,
                              "val_loss_used_for_selection": False,
                              "action_mse_used_for_selection": False}}


def select_epochs(epochs: Sequence[int], estimator: str = f"last{LAST_K}",
                  curve: Optional[Sequence[Dict[str, Any]]] = None,
                  key: str = "val_loss") -> List[int]:
    """Which grid epochs an estimator reports.

    ``last-k`` and ``all`` rank nothing, which is why they survive the absence of
    a rollout.

    ``bestval`` is the ordinary thing — restore the epoch with the lowest
    validation loss — and it is offered rather than forbidden, because the case
    against it is quantitative, not a matter of principle, and it is the case
    the rest of this repository follows.  What it costs is **selection noise**:
    an argmin over a grid of noisy estimates is biased low and, on a plateau,
    picks largely at random within it.  dap measured the epoch-to-epoch spread
    at 0.011-0.040 success with the grid maximum carrying +0.023-0.041 of upward
    bias, and found last-3 the better estimator of rollout success.  With a
    validation set of two or three episodes that noise is larger here, not
    smaller.

    The honest summary: on a plateau the two agree to within the noise and
    ``last-k`` has lower variance; off a plateau ``last-k`` is averaging a tail
    that should not be averaged, and validation loss is the thing that tells you
    so.  Use :func:`overfit_warning` to find out which case you are in rather
    than assuming.

    A rollout-ranked estimator still raises — that one is unavailable, not merely
    discouraged.
    """
    eps = sorted(int(e) for e in epochs)
    if estimator == "all":
        return eps
    if estimator.startswith("last"):
        return eps[-int(estimator[4:] or LAST_K):]
    if estimator == "bestval":
        if not curve:
            raise ValueError("bestval needs the training curve")
        scored = [(r[key], int(r["epoch"])) for r in curve
                  if key in r and int(r["epoch"]) in set(eps)]
        if not scored:
            raise ValueError(f"no grid epoch carries {key!r}")
        return [min(scored)[1]]
    raise ValueError(
        f"estimator {estimator!r} ranks checkpoints by rollout success, which "
        f"is unavailable without an EnvRunner; use one of {ESTIMATORS}")


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


def selection_note(bank: SnapshotBank, estimator: str = f"last{LAST_K}",
                   diagnostics: Sequence[str] = ("val_loss", "action_mse")
                   ) -> Dict[str, Any]:
    """What a checkpoint records about how it was chosen — and how it was not.

    The historical keys (``best_epoch`` and the minimum of each diagnostic) are
    kept and clearly marked unused, so the number the old protocol *would* have
    picked stays visible on the record next to the one that was.
    """
    eps = select_epochs(sorted(bank.states), estimator, curve=bank.curve)
    note: Dict[str, Any] = {
        "rule": SELECTION_RULE, "estimator": estimator,
        "selected_epochs": eps, "grid": list(bank.grid),
        "rollout_available": False,
    }
    #: What the ordinary selector would have picked, recorded next to what was,
    #: so the disagreement is visible every run instead of being a matter of
    #: opinion.
    try:
        note["bestval_epoch"] = select_epochs(
            sorted(bank.states), "bestval", curve=bank.curve)[0]
        note["agrees_with_bestval"] = note["bestval_epoch"] in eps
    except ValueError:
        pass
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
                  estimator: str = f"last{LAST_K}", extra: Optional[Dict[str, Any]] = None,
                  keep_grid: bool = True) -> Dict[str, Any]:
    """Write the ``last-k`` checkpoint, with the grid it was built from beside it.

    ``keep_grid=False`` drops the snapshots and keeps only the averaged weights.
    It saves a lot of disk on a stage with no parent — ``B0`` snapshots are whole
    models — at the cost of being unable to re-derive the selection later, which
    is the one thing keeping the grid buys.
    """
    eps = select_epochs(sorted(bank.states), estimator, curve=bank.curve)
    payload = {
        "variant": variant,
        "spec": spec_dict,
        "state_dict": average_states(bank, eps),
        "trainable_only": bank.parent_keys is not None,
        "parent_keys": bank.parent_keys,
        "selection": selection_note(bank, estimator),
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
    ck = torch.load(path, map_location=map_location, weights_only=False)
    res = model.load_state_dict(ck["state_dict"], strict=False)
    assert not res.unexpected_keys, f"checkpoint has unknown keys: {res.unexpected_keys[:8]}"
    if strict_parent and ck.get("trainable_only"):
        parent = set(ck.get("parent_keys") or ())
        orphan = [k for k in res.missing_keys if k not in parent]
        assert not orphan, (
            f"{len(orphan)} keys are in neither the checkpoint nor its parent, "
            f"so they are freshly initialised: {orphan[:8]}")
    return ck
