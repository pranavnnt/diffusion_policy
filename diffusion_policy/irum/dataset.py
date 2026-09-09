"""Chunk assembly for IRUM, from a ``ReplayBuffer`` zarr.

dap's loaders (``liftoff_v4/loader_lo4.py``, ``liftoff_v6_image/loader_lo6.py``)
cut fixed-length episodes into a fixed number of chunks at fixed boundaries,
because both of its environments emit exactly 128 control steps.  Demonstrations
do not: the dressing episodes are whatever length the operator recorded.  So the
chunk grid here is a **sliding decision step** over each episode, and the two
things dap got from the fixed grid are recovered explicitly:

* the slow frames are ``slow_offsets`` relative to the decision step, clamped at
  the episode start exactly as ``dataset_lo4.slow_input`` clamps;
* ``msg_valid`` is 1 only where a full causal message window fits behind the
  decision step, which is dap's "1 from chunk 4 onward" generalised.

The message window is **rolling**, not the fixed event window cap uses.  Cap's
32 steps are anchored to a scripted event; a teleoperated dressing episode has no
such anchor, and 32 steps is longer than the entire recorded episode anyway.

Splitting is by **episode**, never by chunk.  Chunks from one episode overlap in
both observations and actions, so a chunk-level split leaks the validation set
into training and every offline number becomes optimistic.
"""

from __future__ import annotations

import warnings
import os
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.irum import fields as F
from diffusion_policy.irum.spec import IrumSpec, from_resolution


class ChunkNormaliser:
    """Per-array statistics, fitted on the train split only.

    Proprioception is standardised (mean/std); actions are scaled to ``[-1, 1]``
    by their range, because the flow sampler clamps its output to that interval
    and a target outside it would be unreachable by construction.

    RGB is deliberately absent: it is scaled by 1/255 and handed to the vision
    encoder, whose own ImageNet normalisation is what the pretrained backbone
    expects.  Standardising here as well applies two normalisations and makes the
    frozen weights meaningless.
    """

    STD_KEYS = ("prop2", "step_prop", "wrench2", "step_wrench")
    RANGE_KEYS = ("target",)

    def __init__(self, data: Optional[Dict[str, np.ndarray]] = None,
                 state: Optional[Dict[str, Any]] = None,
                 act_scale: Optional[Sequence[float]] = None):
        self.stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if state is not None:
            self.stats = {k: (np.asarray(v[0], np.float32),
                              np.asarray(v[1], np.float32))
                          for k, v in state.items()}
            return
        assert data is not None
        for k in self.STD_KEYS:
            if k not in data or data[k].shape[-1] == 0:
                continue
            flat = np.asarray(data[k], np.float64).reshape(-1, data[k].shape[-1])
            self.stats[k] = (flat.mean(0).astype(np.float32),
                             np.maximum(flat.std(0), 1e-3).astype(np.float32))
        for k in self.RANGE_KEYS:
            if k not in data:
                continue
            width = data[k].shape[-1]
            if act_scale is not None:
                #: The rig's own declared command scale, not the observed range.
                #: The observed range of a handful of episodes is not the range
                #: the robot can be commanded over, and normalising by it makes
                #: "1.0" mean something different in every dataset — which in
                #: turn makes a fast-level ceiling expressed in normalised units
                #: incomparable across runs.
                self.stats[k] = (np.zeros(width, np.float32),
                                 np.asarray(act_scale, np.float32))
                continue
            flat = np.asarray(data[k], np.float64).reshape(-1, width)
            lo, hi = flat.min(0), flat.max(0)
            centre = ((hi + lo) / 2.0).astype(np.float32)
            #: A channel that never moves would divide by ~0 and turn float
            #: noise into a full-scale target; the floor keeps it at zero.
            half = np.maximum((hi - lo) / 2.0, 1e-4).astype(np.float32)
            self.stats[k] = (centre, half)

    def apply_vec(self, key: str, a: np.ndarray) -> np.ndarray:
        if key not in self.stats:
            return np.asarray(a, np.float32)
        m, s = self.stats[key]
        return ((np.asarray(a, np.float32) - m) / s).astype(np.float32)

    def invert_vec(self, key: str, a: np.ndarray) -> np.ndarray:
        if key not in self.stats:
            return np.asarray(a, np.float32)
        m, s = self.stats[key]
        return (np.asarray(a, np.float32) * s + m).astype(np.float32)

    def apply(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(data)
        for k in list(self.STD_KEYS) + list(self.RANGE_KEYS):
            if k in out and out[k].size:
                out[k] = self.apply_vec(k, out[k])
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {k: [v[0].tolist(), v[1].tolist()] for k, v in self.stats.items()}

    @classmethod
    def from_state(cls, s: Dict[str, Any]) -> "ChunkNormaliser":
        return cls(state=s)


def discover_zarrs(path: Any) -> List[str]:
    """One zarr, a directory of them, or an explicit list — all end up as a list.

    A dressing session produces one zarr per recording, so ``data/`` will hold
    several. Sorted, so the episode order a run sees is the same on every
    machine and a seeded split is reproducible.
    """
    if isinstance(path, (list, tuple)):
        out: List[str] = []
        for p in path:
            out.extend(discover_zarrs(p))
        return out
    p = pathlib.Path(str(path))
    if (p / ".zgroup").exists():                  # a zarr store itself
        return [str(p)]
    if p.is_dir():
        found = sorted(str(c) for c in p.iterdir()
                       if c.is_dir() and (c / ".zgroup").exists())
        if not found:
            raise FileNotFoundError(f"{p} contains no zarr stores")
        return found
    raise FileNotFoundError(f"{p} is neither a zarr store nor a directory")


class IrumEpisodes:
    """Episodes read out of a ``ReplayBuffer`` zarr and resolved against the schema.

    Resolution happens **once, over the whole buffer**, so the presence report
    describes the dataset rather than whichever episode happened to be looked at
    first — an arm that drops out for one episode and not another would
    otherwise change the state width halfway through.
    """

    def __init__(self, zarr_path: Any, cameras: Sequence[str] = (),
                 n_arms: int = 2, layout: Optional[F.PackedLayout] = None,
                 action_key: str = "action", time_key: str = "timestamp",
                 require: Sequence[str] = (), warn: bool = True,
                 image_size: Optional[Tuple[int, int]] = None):
        self.zarr_paths = discover_zarrs(zarr_path)
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.n_declared = int(n_arms)
        self.action_key = action_key

        #: Resolve every store separately, then keep only what **all** of them
        #: agree on.  Two recordings that measured different fields cannot be
        #: concatenated into one state vector, and taking the union would leave
        #: the missing half zero-filled — the exact thing the schema exists to
        #: prevent.
        buffers, resolutions = [], []
        for path in self.zarr_paths:
            rb = ReplayBuffer.create_from_path(path, mode="r")
            src = {k: rb[k] for k in rb.keys()}
            resolutions.append(F.resolve(src, n_arms=n_arms, layout=layout,
                                         require=require, warn=warn))
            buffers.append(rb)

        arms = tuple(sorted(set.intersection(
            *[set(r.arms) for r in resolutions])))
        usable = tuple(n for n in
                       (f.name for f in F.ARM_FIELDS)
                       if all(n in r.usable for r in resolutions)
                       and all(f"{F.arm_name(a)}_{n}" in r.arrays
                               for r in resolutions for a in arms))
        if warn and len(self.zarr_paths) > 1:
            for path, r in zip(self.zarr_paths, resolutions):
                lost = tuple(n for n in r.usable if n not in usable)
                lost_arms = tuple(a for a in r.arms if a not in arms)
                if lost or lost_arms:
                    warnings.warn(
                        f"{os.path.basename(path)} carries fields {lost} and "
                        f"arms {lost_arms} that other stores do not; they are "
                        f"dropped so every store contributes the same channels.",
                        stacklevel=3)
        assert usable, (
            "no field is usable in every store; the stores do not share a "
            "state vector:\n" + "\n".join(
                f"  {os.path.basename(p)}: {r.usable}"
                for p, r in zip(self.zarr_paths, resolutions)))

        #: one Resolution standing for the whole dataset, used to build the spec
        self.resolution = F.Resolution(
            arrays={}, statuses=[st for r in resolutions for st in r.statuses],
            arms=arms, usable=usable)
        self.prop_names = self.resolution.prop_names()
        self.wrench_names = self.resolution.wrench_names()
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.episode_source: List[int] = []

        for src_i, (path, rb, r) in enumerate(
                zip(self.zarr_paths, buffers, resolutions)):
            r = F.Resolution(arrays=r.arrays, statuses=r.statuses, arms=arms,
                             usable=usable)
            self._add_store(rb, r, src_i, action_key, time_key, path)

        assert self.episodes, f"{self.zarr_paths} hold no episodes"
        self._report_rate()

    def _add_store(self, rb, res, src_i: int, action_key: str, time_key: str,
                   path: str) -> None:
        keys = list(rb.keys())
        prop_all = res.stack(self.prop_names)
        wrench_all = res.stack(self.wrench_names)
        act_all = np.asarray(rb[action_key], np.float32)
        time_all = (np.asarray(rb[time_key], np.float64)
                    if time_key in keys else None)
        if self.episodes:
            got, want = act_all.shape[-1], self.episodes[0]["action"].shape[-1]
            assert got == want, (
                f"{os.path.basename(path)} records a {got}-wide action, other "
                f"stores record {want}")

        ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        for a, b in zip(starts, ends):
            ep = {"prop": prop_all[a:b], "wrench": wrench_all[a:b],
                  "action": act_all[a:b]}
            #: ``dt`` is carried because this rig does not run at a fixed rate —
            #: the smoke episode varies from 0.05 s to 0.75 s between steps. A
            #: state *change* over a 15x-varying interval is not comparable
            #: across steps, so the dynamics is conditioned on it rather than
            #: pretending the grid is uniform. The final step repeats its
            #: predecessor, having no successor to be measured against.
            if time_all is not None:
                d = np.diff(time_all[a:b])
                ep["dt"] = np.concatenate([d, d[-1:]]).astype(np.float32)[:, None] \
                    if len(d) else np.zeros((b - a, 1), np.float32)
            else:
                ep["dt"] = np.ones((b - a, 1), np.float32)
            for cam in self.cameras:
                ep[cam] = self._frames(rb[cam][a:b])
            self.episodes.append(ep)
            self.episode_source.append(src_i)

    def _frames(self, raw) -> np.ndarray:
        """Camera frames at the size the policy declares, resized on load.

        Not only a memory measure, though it is that too: the vision encoder
        takes a ``crop_shape`` fraction of whatever it is handed, so feeding it a
        480x640 frame with a 216x288 crop would show it 45% of the scene and call
        the rest augmentation.  Resizing first, then cropping ~90%, is what the
        crop was sized for.  ``image_size=None`` keeps the native resolution.
        """
        a = np.asarray(raw)
        if self.image_size is None or a.shape[1:3] == tuple(self.image_size):
            return a
        import cv2

        h, w = self.image_size
        out = np.empty((len(a), h, w, a.shape[3]), dtype=a.dtype)
        for i in range(len(a)):
            #: INTER_AREA is the right filter for downscaling; the default
            #: bilinear aliases, and aliasing on a wrist camera looks like
            #: texture the policy can key on.
            out[i] = cv2.resize(a[i], (w, h), interpolation=cv2.INTER_AREA)
        return out

    def native_image_size(self) -> Optional[Tuple[int, int]]:
        for ep in self.episodes:
            for cam in self.cameras:
                return tuple(ep[cam].shape[1:3])
        return None

    def _report_rate(self) -> None:
        """A rate this irregular breaks the slow/fast separation, so it is said out loud."""
        dt = np.concatenate([e["dt"][:-1, 0] for e in self.episodes
                             if len(e["dt"]) > 1])
        if not len(dt) or dt.max() <= 0:
            return
        self.dt_stats = {"mean": float(dt.mean()), "min": float(dt.min()),
                         "max": float(dt.max()), "hz": float(1.0 / dt.mean())}
        if dt.max() > 3.0 * dt.min():
            warnings.warn(
                f"control period varies {dt.min():.3f}-{dt.max():.3f} s "
                f"(mean {dt.mean():.3f}, ~{1 / dt.mean():.1f} Hz). IRUM assumes a "
                f"slow level replanning every exec_horizon steps and a fast level "
                f"correcting every step; at this jitter those are not separated in "
                f"time. The dynamics is conditioned on dt, but the horizons mean "
                f"different durations at different points in an episode.",
                stacklevel=3)

    def __len__(self) -> int:
        return len(self.episodes)

    def lengths(self) -> List[int]:
        return [len(e["prop"]) for e in self.episodes]

    def action_report(self, channels: Sequence[int] = ()) -> Dict[str, Any]:
        """Which action channels ever move, indexed as the policy sees them.

        ``channels`` is the spec's channel selection, so the returned indices
        address the **trimmed** action the policy commands.  Reporting recorded
        indices instead would be right only while the live arm happens to be the
        first one — with arm1 dropped, recorded channel 7 is trimmed channel 1,
        and a ceiling built from the wrong indexing silences the wrong axis.

        A channel that is identically zero across the dataset cannot be learned,
        so giving the corrector authority over it is authority over something it
        never saw — free to write at rollout, unconstrained by any data.
        """
        a = np.concatenate([e["action"] for e in self.episodes])
        chans = list(channels) if len(channels) else list(range(a.shape[-1]))
        moves = (a[:, chans] != 0).any(0)
        return {"n_channels": len(chans), "recorded_channels": chans,
                "active": [int(i) for i in np.nonzero(moves)[0]],
                "dead": [int(i) for i in np.nonzero(~moves)[0]]}


def decision_steps(n: int, spec: IrumSpec) -> np.ndarray:
    """Steps at which the slow level replans and a full target chunk exists.

    Stride is ``exec_horizon``: that is the rate the slow level actually replans
    at, and a stride-1 grid would train it on ``exec_horizon`` times as many
    near-duplicate windows without adding information.
    """
    last = n - spec.pred_horizon
    if last < 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, last + 1, spec.exec_horizon, dtype=np.int64)


def build_chunks(eps: IrumEpisodes, spec: IrumSpec, indices: Sequence[int]
                 ) -> Dict[str, np.ndarray]:
    """Flatten the named episodes into one array per field.

    ``episode_index`` and ``decision_step`` are carried through so any later
    diagnostic can group rows back the way they came, which is what makes an
    episode-level split checkable after the fact rather than merely intended.
    """
    H, E, W = spec.pred_horizon, spec.exec_horizon, spec.message_window
    prev_off = spec.slow_offsets[0]
    chans = (list(spec.act_channels) if spec.act_channels
             else list(range(eps.episodes[0]["action"].shape[-1])))

    out: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("prop2", "wrench2", "step_prop", "step_wrench", "target",
                        "hist_y", "hist_y_next", "hist_a", "hist_dt",
                        "msg_valid", "episode_index", "decision_step")}
    for cam in spec.cameras:
        out[f"rgb2_{cam}"] = []

    for j in indices:
        ep = eps.episodes[j]
        y, w, a, dt = ep["prop"], ep["wrench"], ep["action"], ep["dt"]
        n = len(y)
        for s in decision_steps(n, spec):
            frames = [max(s + prev_off, 0), s]
            out["prop2"].append(y[frames])
            out["wrench2"].append(w[frames])
            out["step_prop"].append(y[s:s + E])
            out["step_wrench"].append(w[s:s + E])
            out["target"].append(a[s:s + H][:, chans])
            for cam in spec.cameras:
                out[f"rgb2_{cam}"].append(ep[cam][frames])

            #: Transitions ``t`` in ``[s-W, s)`` use ``(y_t, a_t, y_{t+1})``.
            #: Before a full window exists the slot is zero **and** masked; the
            #: mask is what matters, the zeros just keep the array rectangular.
            valid = int(s >= W)
            if valid:
                out["hist_y"].append(y[s - W:s])
                out["hist_y_next"].append(y[s - W + 1:s + 1])
                out["hist_a"].append(a[s - W:s][:, chans])
                out["hist_dt"].append(dt[s - W:s])
            else:
                out["hist_y"].append(np.zeros((W, y.shape[-1]), np.float32))
                out["hist_y_next"].append(np.zeros((W, y.shape[-1]), np.float32))
                out["hist_a"].append(np.zeros((W, len(chans)), np.float32))
                out["hist_dt"].append(np.zeros((W, 1), np.float32))
            out["msg_valid"].append(np.float32(valid))
            out["episode_index"].append(np.int32(j))
            out["decision_step"].append(np.int32(s))

    return {k: (np.stack(v) if np.ndim(v[0]) else np.asarray(v))
            for k, v in out.items() if v}


def short_episodes(eps: "IrumEpisodes", spec: IrumSpec) -> List[int]:
    """Episodes too short to yield a single chunk.

    A recording aborted after a couple of steps contributes nothing but still
    counts as an episode — so it can be drawn into the validation split, where
    it silently shrinks the held-out set to nothing.  Naming them is cheaper
    than wondering why validation has fewer chunks than the ratio implies.
    """
    return [i for i, n in enumerate(eps.lengths())
            if len(decision_steps(n, spec)) == 0]


def action_residual_demand(chunks: Dict[str, np.ndarray], spec: IrumSpec
                          ) -> Dict[str, Any]:
    """How much per-step correction a chunk-level plan cannot express.

    The quantity the fast level exists to supply, measured before any model is
    trained, so the ceiling can be set from the data rather than guessed and
    then discovered to be binding.  Two bounds, because neither alone is the
    answer:

    ``vs_chunk_mean``
        ``|a_t - mean(a_chunk)|`` over the executed prefix — what a plan that
        committed to *one* action for the whole chunk would have to be corrected
        by.  This **over**-estimates: the real slow level predicts a whole chunk
        and can already express within-chunk variation.
    ``step_to_step``
        ``|a_t - a_{t-1}|`` — the variation a smoothly-sampled plan cannot
        track, which is closer to what actually lands on the corrector.

    Everything is in normalised action units, so a percentile reads directly as
    a fraction of full command and can be compared with ``fast_limits``.
    """
    tgt = np.asarray(chunks["target"], np.float64)[:, :spec.exec_horizon]
    if tgt.size == 0:
        return {}
    dev = np.abs(tgt - tgt.mean(1, keepdims=True))
    step = np.abs(np.diff(tgt, axis=1)) if tgt.shape[1] > 1 else np.zeros_like(tgt)

    def pct(a):
        flat = a.reshape(-1, a.shape[-1])
        return {f"p{q}": [round(float(v), 5) for v in np.percentile(flat, q, axis=0)]
                for q in (50, 90, 95, 99, 100)}

    return {"vs_chunk_mean": pct(dev), "step_to_step": pct(step),
            "note": "normalised action units; 1.0 = full command"}


def episode_split(n_episodes: int, val_ratio: float = 0.2, seed: int = 42,
                  sources: Optional[Sequence[int]] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Split **by episode**, stratified by recording session.

    Chunks from one episode overlap in both observations and actions, so a
    chunk-level split leaks the validation set into training and every offline
    number becomes optimistic.

    ``sources`` names which store each episode came from, and the split is taken
    within each store rather than over the pool.  Sessions differ in lighting,
    garment placement and operator, so an unstratified draw can put a whole
    session in validation and turn "held out" into "a different setup" — which
    is a harder question than the one being asked, and it moves with the seed.

    With one episode, everything is train and validation is empty; inventing one
    is left undone rather than done silently.
    """
    rng = np.random.default_rng(seed)
    if n_episodes <= 1:
        return np.arange(n_episodes), np.zeros(0, dtype=np.int64)
    src = (np.asarray(sources) if sources is not None
           else np.zeros(n_episodes, dtype=np.int64))
    val: List[int] = []
    for s in np.unique(src):
        idx = np.nonzero(src == s)[0]
        order = rng.permutation(idx)
        k = int(round(len(idx) * val_ratio))
        #: A store with a single episode contributes none: holding it out would
        #: remove that session from training entirely.
        k = min(k, len(idx) - 1)
        val.extend(order[:k].tolist())
    #: Guarantee a non-empty validation set once more than one episode exists,
    #: even when every store is a singleton and rounding took nothing.
    if not val:
        val = [int(rng.permutation(n_episodes)[0])]
    val_idx = np.sort(np.asarray(val, dtype=np.int64))
    train_idx = np.sort(np.setdiff1d(np.arange(n_episodes), val_idx))
    return train_idx, val_idx


def load_split(zarr_path: str, cameras: Sequence[str] = (), n_arms: int = 2,
               layout: Optional[F.PackedLayout] = None, val_ratio: float = 0.2,
               seed: int = 42, require: Sequence[str] = (),
               spec_kw: Optional[Dict[str, Any]] = None, warn_scale: bool = True,
               image_size: Optional[Tuple[int, int]] = None, **kw
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
                          ChunkNormaliser, "IrumEpisodes", IrumSpec]:
    """Resolve the dataset, derive the spec from it, and cut it into chunks."""
    eps = IrumEpisodes(zarr_path, cameras=cameras, n_arms=n_arms, layout=layout,
                       require=require, image_size=image_size, **kw)
    spec_kw = dict(spec_kw or {})
    native = eps.native_image_size()
    if native is not None:
        #: The declared input shape follows the frames actually produced, so a
        #: shape_meta that disagrees with the tensors cannot reach the encoder.
        spec_kw["image_shape"] = (3, native[0], native[1])
        #: ~90% of the frame, matching the ratio the 240x320 default used.
        spec_kw["crop_shape"] = (int(round(native[0] * 0.9)),
                                 int(round(native[1] * 0.9)))
    spec = from_resolution(eps.resolution, cameras=cameras,
                           act_width=eps.episodes[0]["action"].shape[-1],
                           n_declared=eps.n_declared, **spec_kw)
    if spec.act_scale is None and warn_scale:
        warnings.warn(
            f"actions are {spec.act_dim}-wide, which is not the 3-linear + "
            f"3-angular layout the declared command scale describes, so they "
            f"are normalised by their observed range instead. A ceiling "
            f"expressed as a fraction of that is a fraction of *this dataset's* "
            f"range, not of full command, and is not comparable across "
            f"recordings.", stacklevel=2)
    n_rec = eps.episodes[0]["action"].shape[-1]
    if len(spec.act_channels) < n_rec:
        dropped = [c for c in range(n_rec) if c not in spec.act_channels]
        warnings.warn(
            f"action channels {dropped} belong to arms the state vector does "
            f"not cover and are dropped from the target; the policy commands "
            f"{list(spec.act_channels)}.", stacklevel=2)
    empty = short_episodes(eps, spec)
    if empty and warn_scale:
        warnings.warn(
            f"episode(s) {empty} are shorter than pred_horizon="
            f"{spec.pred_horizon} and yield no chunks: "
            f"{[eps.lengths()[i] for i in empty]} steps. They are excluded from "
            f"the split so they cannot occupy a validation slot.", stacklevel=2)
    keep = [i for i in range(len(eps)) if i not in set(empty)]
    tr_idx, va_idx = episode_split(
        len(keep), val_ratio, seed,
        sources=[eps.episode_source[i] for i in keep])
    tr_idx = np.asarray([keep[i] for i in tr_idx], dtype=np.int64)
    va_idx = np.asarray([keep[i] for i in va_idx], dtype=np.int64)
    tr = build_chunks(eps, spec, tr_idx)
    norm = ChunkNormaliser(tr, act_scale=spec.act_scale)
    tr = norm.apply(tr)
    if len(va_idx):
        va = norm.apply(build_chunks(eps, spec, va_idx))
    else:
        warnings.warn(
            f"{len(eps)} episode(s) across {len(eps.zarr_paths)} store(s): no "
            "held-out episode exists, so the validation arrays are the training "
            "ones and every offline number below is a training number.",
            stacklevel=2)
        va = tr
    return tr, va, norm, eps, spec
