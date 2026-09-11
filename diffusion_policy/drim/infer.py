"""Run a trained DRIM chain on the real rig.

Standalone by construction: this module reads a run directory and nothing else
— no zarr, no dataset, no training code path.  That is the same property
:mod:`~diffusion_policy.drim.train` has in the other direction (it cannot
import an environment), and it is what makes "the robot ran the checkpoint, not
a re-derived copy of it" checkable rather than asserted.

The loop the rig drives::

    pol = DrimRunner.load("~/drim_runs/<tag>", device="cuda:0")
    pol.reset()
    while running:
        obs = read_the_robot()                    # see ``Observation``
        target = pol.step(obs)                    # absolute EE pose target, m
        send_to_robot(target)

Four things this module exists to get right, each of which produced a wrong
number during development before it was caught:

* **The policy predicts a delta, the robot takes an absolute pose.**  The
  network's output is ``action - ee_pos`` in metres; :meth:`DrimRunner.step`
  adds the *measured* pose back.  Skipping that is what drove the offline
  rollout to 1e19 and then NaN.
* **The zigzag is an input, never an output.**  It is injected at teleop and at
  inference alike; the dynamics is conditioned on it and refuses to run without
  it.  Pass it every step.
* **The message needs history.**  ``S``/``U`` come from the last
  ``message_window`` transitions through the dynamics.  Before that window
  fills, the message is structurally null and D2 *is* B1 — which is correct
  behaviour, not a degraded mode, and :attr:`DrimRunner.message_ready` says so.
* **Frames are cropped the way training cropped them.**  The ROI is a property
  of the run, carried in the spec, and applied here by the same code path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_policy.drim import dressing as DRESS
from diffusion_policy.drim import dynamics as DY
from diffusion_policy.drim import policy as PL
from diffusion_policy.drim import selection as SEL
from diffusion_policy.drim.dataset import ChunkNormaliser
from diffusion_policy.drim.spec import DrimSpec

STAGES = ("B0", "B1", "D2")


@dataclass
class Observation:
    """One timestep as the rig measures it.

    Field widths are checked against the spec on the first call, so a wiring
    mistake stops the loop rather than being normalised into plausible-looking
    nonsense.  ``dt`` is seconds since the previous observation; the recordings
    ran at 14.3 Hz and the dynamics was fitted with ``dt`` as an input, so a
    stalled frame is handled correctly if you report it honestly.
    """

    q: np.ndarray                 # (7,)  joint positions
    dq: np.ndarray                # (7,)  joint velocities
    ee_pos: np.ndarray            # (3,)  end-effector position, metres
    ee_quat: np.ndarray           # (4,)  end-effector orientation
    ee_twist: np.ndarray          # (6,)  linear then angular velocity
    wrench: np.ndarray            # (6,)  estimated external wrench at the EE
    images: Dict[str, np.ndarray]  # {camera: (H, W, 3) uint8}, full frames
    zigzag_action: np.ndarray     # (6,)  the primitive, as commanded
    dt: float = 0.07

    def prop(self, spec: DrimSpec) -> np.ndarray:
        parts = {"q": self.q, "dq": self.dq, "ee_pos": self.ee_pos,
                 "ee_quat": self.ee_quat, "ee_twist": self.ee_twist}
        return np.concatenate([np.asarray(parts[f], np.float32).ravel()
                               for f in spec.prop_fields])

    def wrench_vec(self, spec: DrimSpec) -> np.ndarray:
        parts = {"wrench": self.wrench}
        if not spec.wrench_fields:
            return np.zeros(0, np.float32)
        return np.concatenate([np.asarray(parts[f], np.float32).ravel()
                               for f in spec.wrench_fields])


def crop_resize(img: np.ndarray, roi: Optional[Sequence[int]],
                size: Tuple[int, int]) -> np.ndarray:
    """``(H, W, 3) -> (size[0], size[1], 3)``, cropping exactly as training did.

    ``roi`` is ``(y0, x0, h, w)`` — row first, the same order
    :class:`~diffusion_policy.drim.dataset.DrimEpisodes` slices with, and the
    same order :mod:`~diffusion_policy.drim.roi_tool` writes.  ``INTER_AREA``
    for the same reason the loader uses it: the default bilinear aliases when
    downscaling, and aliasing looks like texture a policy can key on.

    A frame arriving at a different resolution than the recording is a wiring
    error, not something to silently rescale the box for, so it is refused.
    """
    import cv2
    if roi is not None:
        y0, x0, h, w = (int(v) for v in roi)
        assert (0 <= y0 and 0 <= x0 and y0 + h <= img.shape[0]
                and x0 + w <= img.shape[1]), (
            f"ROI (y={y0}, x={x0}, h={h}, w={w}) does not fit a "
            f"{img.shape[0]}x{img.shape[1]} frame; the camera is not producing "
            f"what the run was trained on")
        img = img[y0:y0 + h, x0:x0 + w]
    if img.shape[:2] != tuple(size):
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _roi_from_log(run_dir: str) -> str:
    """Recover ``--roi`` from a run that predates recording it in the summary."""
    path = os.path.join(run_dir, "train.log")
    names = set()
    if os.path.exists(path):
        with open(path) as fh:
            for ln in fh:
                if "--roi" in ln:
                    bits = ln.split()
                    names.add(bits[bits.index("--roi") + 1])
    assert len(names) == 1, (
        f"cannot tell which crop {run_dir} was trained with "
        f"({sorted(names) or 'no --roi in train.log'}); pass roi= explicitly. "
        f"Deploying the wrong box gives the encoder a different scene than it "
        f"was trained on, and nothing downstream will say so.")
    name = names.pop()
    assert name in DRESS.ROIS, f"train.log names an unknown roi {name!r}"
    return name


@dataclass
class DrimRunner:
    """A loaded chain plus the history the message needs."""

    model: Any
    dyn: Any
    norm: ChunkNormaliser
    spec: DrimSpec
    roi: Dict[str, Tuple[int, int, int, int]]
    device: str = "cpu"
    stage: str = "D2"
    _hist: list = field(default_factory=list)
    _frames: list = field(default_factory=list)
    _chunk: Optional[torch.Tensor] = None
    _feats: Optional[tuple] = None
    _msg: Optional[torch.Tensor] = None
    _k: int = 0
    _prev: Optional[tuple] = None
    #: Unmasked target from the network's current step.  Useful for live
    #: ablations that hold the robot on a fixed centroid while still showing
    #: what the policy would have requested.
    last_policy_target: Optional[np.ndarray] = None
    last_replanned: bool = False
    last_chunk_index: int = 0

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, run_dir: str, seed: int = 0, device: str = "cpu",
             upto: str = "D2", roi: Optional[str] = None) -> "DrimRunner":
        """Rebuild the chain from a run directory.

        ``upto`` stops the stack early — ``B0`` is the slow policy alone, ``B1``
        adds the bounded corrector, ``D2`` adds the message.  Each stage after
        B0 is trainable-only and nests on the one before, so they load in order;
        a D2 file alone is corrector and message machinery with no backbone.
        """
        run_dir = os.path.expanduser(run_dir)
        with open(os.path.join(run_dir, f"summary_seed{seed}.json")) as fh:
            summary = json.load(fh)
        spec = DrimSpec.from_dict(summary["spec"])
        norm = ChunkNormaliser.from_state(summary["normaliser"])

        dck = SEL.torch_load(os.path.join(run_dir, f"dynamics_seed{seed}.pt"),
                             map_location=device)
        mods = [(n, tuple(sl), w, k) for n, sl, w, k in dck["mods"]]
        m = DY.DeltaDynamics(mods, spec.act_dim, exo_dim=spec.exo_dim)
        m.load_state_dict(dck["state_dict"])
        m.eval()
        frozen = DY.FrozenDynamics(
            m, DY.Normaliser.from_state(dck["norm"])).to(device)

        model = parent = None
        for st in STAGES[:STAGES.index(upto) + 1]:
            kw = ({} if st != "D2"
                  else {"message_in_dims": (DY.delta_dim(mods),
                                            DY.delta_dim(mods))})
            model = PL.build(st, spec, core=parent, **kw).to(device)
            SEL.load_selected(os.path.join(run_dir, f"{st}_seed{seed}.pt"),
                              model, strict_parent=(st != "B0"),
                              map_location=device)
            parent = model
        model.eval()

        #: The crop has to be the one training used, and a run is only
        #: self-describing about it if it recorded it.  Older runs did not, so
        #: the command line in ``train.log`` is the fallback — read, not
        #: guessed — and an unreadable one is refused rather than defaulted,
        #: because a wrong box silently feeds the encoder a different scene.
        if roi is not None:
            boxes = DRESS.ROIS[roi]
        elif summary.get("roi"):
            boxes = {k: tuple(v) for k, v in summary["roi"].items()}
        else:
            boxes = DRESS.ROIS[_roi_from_log(run_dir)]
        return cls(model=model, dyn=frozen, norm=norm, spec=spec,
                   roi=dict(boxes or {}), device=device, stage=upto)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear the history.  Call before every episode.

        The message window does not carry across episodes: transitions from a
        previous trial are not evidence about this one.
        """
        self._hist.clear(); self._frames.clear()
        self._chunk = self._feats = self._msg = None
        self._k = 0
        self._prev = None
        self.last_policy_target = None
        self.last_replanned = False
        self.last_chunk_index = 0

    @property
    def message_ready(self) -> bool:
        """Whether a full causal window exists behind the current step.

        Until it does the message is structurally null, and with the exact-null
        identity that means D2 is *exactly* B1 — the same weights producing the
        same actions, not a fallback.
        """
        return len(self._hist) >= self.spec.message_window

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, obs: Observation,
             target_override: Optional[np.ndarray] = None,
             target_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
             ) -> np.ndarray:
        """One control step.  Returns the **absolute** EE position target, in metres.

        Replans every ``exec_horizon`` steps and applies the bounded corrector
        on every step in between, which is the same schedule the rollout and the
        training-time executed prefix use.
        """
        sp = self.spec
        prop = obs.prop(sp)
        wr = obs.wrench_vec(sp)
        assert prop.shape[0] == sp.prop_dim, (
            f"proprioception is {prop.shape[0]} wide, spec says {sp.prop_dim}; "
            f"expected {list(sp.prop_fields)}")
        assert wr.shape[0] == sp.wrench_dim, (
            f"wrench is {wr.shape[0]} wide, spec says {sp.wrench_dim}")
        exo = np.asarray(obs.zigzag_action, np.float32).ravel()
        assert exo.shape[0] == sp.exo_dim, (
            f"the primitive is {exo.shape[0]} wide, spec says {sp.exo_dim}; it "
            f"is a conditioning input and the dynamics refuses to run without it")

        frame = None
        if sp.is_image:
            miss = [c for c in sp.cameras if c not in obs.images]
            assert not miss, f"missing camera(s) {miss}; need {list(sp.cameras)}"
            frame = {c: crop_resize(obs.images[c], self.roi.get(c),
                                    (sp.image_shape[1], sp.image_shape[2]))
                     for c in sp.cameras}

        self._push(prop, wr, frame)
        dyn_state = np.concatenate([prop, wr]) if sp.wrench_dim else prop

        self.last_replanned = (self._k % sp.exec_horizon == 0)
        if self.last_replanned:
            ctx = self._context()
            self._msg = self._message()
            self._chunk = self.model.sample_chunk(ctx, self._msg, None)
            self._feats = self.model.step_features(self._chunk)
        idx = self._k % sp.exec_horizon
        self.last_chunk_index = idx

        #: ``[1, D]``, the shape the corrector takes one step at a time — the
        #: normaliser is fitted on ``[N, E, D]`` and broadcasts over the batch
        #: axis, so the row must not be given a second one.
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        spn = t(self.norm.apply_vec("step_prop", prop[None]))
        swn = (t(self.norm.apply_vec("step_wrench", wr[None]))
               if sp.wrench_dim else None)
        r = self.model.residual_step(self._feats, idx, spn, swn, self._msg)
        a_norm = torch.clamp(self._chunk[:, idx] + r, -1.0, 1.0)
        delta = self.norm.invert_vec("target", a_norm.cpu().numpy())[0]

        #: the network's output is ``action - ee_pos``; the robot takes an
        #: absolute pose, so the *measured* pose goes back on
        target = (delta + np.asarray(obs.ee_pos, np.float32)
                  if sp.action_mode == "delta_ee_pos" else delta)
        self.last_policy_target = target.astype(np.float32, copy=True)
        assert target_override is None or target_transform is None, (
            "use either target_override or target_transform, not both")
        if target_transform is not None:
            target = np.asarray(target_transform(target.copy()), np.float32)
            assert target.shape == (sp.act_dim,), (
                f"target transform returned {target.shape}, expected {(sp.act_dim,)}")
        if target_override is not None:
            target = np.asarray(target_override, np.float32)
            assert target.shape == (sp.act_dim,), (
                f"target override is {target.shape}, expected {(sp.act_dim,)}")

        self._advance(dyn_state, target, exo, float(obs.dt))
        self._k += 1
        return target.astype(np.float32)

    def replace_pending_action(self, action: np.ndarray) -> None:
        """Replace this step's action before it enters the next history row.

        Most deployments execute the predicted centroid directly, which is why
        :meth:`step` records ``target`` by default.  A centroid-state servo,
        however, deliberately sends an *integrated* centroid that can differ
        from a noisy instantaneous prediction.  Training's ``hist_a`` is that
        integrated collector centroid, so the servo must replace the pending
        history action with its controller state before the next observation
        closes the transition.
        """
        if self._prev is None:
            raise RuntimeError("call replace_pending_action only after step()")
        a = np.asarray(action, np.float32)
        if a.shape != (self.spec.act_dim,):
            raise ValueError(f"action is {a.shape}, expected {(self.spec.act_dim,)}")
        y, _, exo, dt = self._prev
        self._prev = (y, a.copy(), exo, dt)

    def replace_pending_exo(self, exo: np.ndarray) -> None:
        """Replace this step's executed exogenous command before history use.

        The transition staged at observation ``t`` is closed when observation
        ``t+1`` arrives, so it must carry the complete command actually sent at
        ``t``.  A live controller only knows that command after it combines the
        policy base velocity and zigzag primitive.
        """
        if self._prev is None:
            raise RuntimeError("call replace_pending_exo only after step()")
        x = np.asarray(exo, np.float32)
        if x.shape != (self.spec.exo_dim,):
            raise ValueError(f"exo is {x.shape}, expected {(self.spec.exo_dim,)}")
        y, action, _, dt = self._prev
        self._prev = (y, action, x.copy(), dt)

    # ------------------------------------------------------------------ inner
    def _push(self, prop, wr, frame) -> None:
        self._frames.append((prop, wr, frame))
        keep = max(-min(self.spec.slow_offsets), 1) + 1
        del self._frames[:-keep]

    def _obs_window(self):
        """The ``n_obs_steps`` frames the slow level reads, oldest first."""
        out = []
        for off in self.spec.slow_offsets:
            i = len(self._frames) - 1 + off
            out.append(self._frames[max(i, 0)])
        return out

    def _context(self) -> torch.Tensor:
        sp = self.spec
        win = self._obs_window()
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        prop2 = t(self.norm.apply_vec(
            "prop2", np.stack([w[0] for w in win])[None]))
        w2 = (t(self.norm.apply_vec(
            "wrench2", np.stack([w[1] for w in win])[None]))
            if sp.wrench_dim else None)
        rgb = None
        if sp.is_image:
            rgb = {c: torch.as_tensor(
                np.stack([w[2][c] for w in win])[None], device=self.device)
                for c in sp.cameras}
        return self.model.context(rgb, prop2, w2)

    def _message(self) -> Optional[torch.Tensor]:
        sp = self.spec
        if self.model.cond_dim == 0:
            return None
        W = sp.message_window
        valid = float(len(self._hist) >= W)
        if not valid:
            #: structurally null, which by the exact-null identity makes this
            #: step exactly B1 rather than an approximation of it
            z = torch.zeros(1, W, self.dyn.model.delta_dim, device=self.device)
            return self.model.conditioning(
                {"prop2": torch.zeros(1, 1, device=self.device),
                 "hist_S": z, "hist_U": z,
                 "msg_valid": torch.zeros(1, device=self.device)})
        h = self._hist[-W:]
        f = lambda i, d: torch.as_tensor(
            np.stack([x[i] for x in h])[None].astype(np.float32),
            device=self.device)
        ch = DY.surprise(self.dyn, f(0, 0), f(1, 0), f(2, 0), f(3, 0),
                         f(4, 0) if sp.exo_dim else None)
        return self.model.conditioning(
            {"prop2": torch.zeros(1, 1, device=self.device),
             "hist_S": ch["S"], "hist_U": ch["U"],
             "msg_valid": torch.ones(1, device=self.device)})

    def _advance(self, y: np.ndarray, action: np.ndarray, exo: np.ndarray,
                 dt: float) -> None:
        """Close one transition once the *next* state is known.

        ``S`` is ``(what happened) - (what the model expected)``, so a
        transition is only complete when the following observation arrives.  The
        action stored is the absolute pose target actually sent, because that is
        what ``hist_a`` carried in training.
        """
        if self._prev is not None:
            y0, a0, x0, _ = self._prev
            # Dataset ``dt[t]`` is timestamp[t + 1] - timestamp[t], i.e. the
            # duration of precisely the transition being closed here.  At live
            # time that duration is only known when the next observation has
            # arrived, so it is the *current* ``dt``, not the one staged with
            # the previous state/action.
            self._hist.append((y0, y, a0, np.float32([dt]), x0))
            del self._hist[:-self.spec.message_window]
        self._prev = (y, action, exo, dt)
