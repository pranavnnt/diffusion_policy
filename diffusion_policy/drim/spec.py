"""Dimensions, horizons and windows for one DRIM run.

The dap benchmarks (`cap_constraint_benchmark`, `drawer_constraint_benchmark`)
bake their dimensions into module-level constants — ``OBS_DIM = 59``,
``MESSAGE_WINDOW = 32`` — because each of them is one frozen task.  Porting the
method here means those numbers become inputs: the dressing rig publishes a
28-dim state and a 12-dim action, the sim publishes a 37-dim state and 2-3 dim
action, and neither is the cap layout.

Everything that was a constant over there is a field here, and every field is
recorded in the checkpoint, so a rollout cannot silently feed a policy the stack
it was not trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DrimSpec:
    """One task's shape contract.

    ``wrench_dim`` may be 0.  The dap stacks always carried a wrench because
    both of their environments published a force/torque reading; the real
    dressing rig publishes joint positions and an end-effector pose and nothing
    else, so the channel is genuinely absent rather than merely unused.  Keeping
    it as a width rather than deleting the argument is what lets a force-equipped
    dataset — the dressing sim, which has ``force`` at ``state[7:11]``, or a
    future real rig with an F/T sensor — drop in without touching the models.
    """

    #: proprioceptive channels the slow *and* fast levels read
    prop_dim: int
    #: action width
    act_dim: int
    #: force/torque channels, 0 when the rig publishes none
    wrench_dim: int = 0
    #: camera keys; empty means a state-only track
    cameras: Tuple[str, ...] = ()
    image_shape: Tuple[int, int, int] = (3, 240, 320)
    crop_shape: Tuple[int, int] = (216, 288)

    #: chunk the slow level predicts
    pred_horizon: int = 8
    #: prefix of that chunk the environment actually receives
    exec_horizon: int = 4
    #: slow frames stacked as context, as offsets from the decision step
    slow_offsets: Tuple[int, int] = (-1, 0)

    #: causal window the upward message reads, in control steps
    message_window: int = 8
    message_dim: int = 16
    #: per-channel ceiling on the fast correction, in normalized action units
    fast_limits: Optional[Tuple[float, ...]] = None
    #: Action channels the policy commands, as indices into the recorded action.
    #: Empty means "all of them".  The rig records 6 channels per arm whether or
    #: not that arm is live, and asking a policy to command an arm its
    #: observation does not cover is worse than not commanding it: the channel
    #: cannot be predicted from anything visible, so it becomes noise the flow
    #: field spends capacity fitting.
    act_channels: Tuple[int, ...] = ()
    #: How the action is parameterised.
    #:
    #: ``"absolute"``      predict the command as recorded.
    #: ``"delta_ee_pos"``  predict it **relative to the end-effector pose at the
    #:                     step it executes on**.  For this rig the recorded
    #:                     ``action`` is an absolute pose target that the
    #:                     controller already tracks to within 8 mm, so its
    #:                     absolute value is nearly the observed pose and
    #:                     predicting it is close to copying an input: holding
    #:                     the previous action scores 0.0002 against a
    #:                     predict-zero of 0.184.  The operator's actual
    #:                     contribution is the *nudge* away from where the arm
    #:                     is, and on that the copycat only reaches 0.117
    #:                     against 0.243.  Same command either way -- the pose
    #:                     is added back at execution, where it is observed.
    action_mode: str = "absolute"
    #: Width of a known exogenous command that is injected at both teleop and
    #: inference and is *not* predicted (this rig's zigzag primitive).  It
    #: conditions the dynamics, because a model that cannot see a driving input
    #: it could have seen puts that input's effect into the surprise channel.
    exo_dim: int = 0
    #: Command scale per action channel in physical units (m/s, rad/s), used to
    #: normalise actions to [-1, 1] and to express the fast level's authority as
    #: a fraction of full command.  ``None`` falls back to the observed range,
    #: which on a small dataset is not the command range.
    act_scale: Optional[Tuple[float, ...]] = None

    #: width of the learned visual latent per camera (resnet18 after global pool)
    vision_dim: int = 512

    #: Which declared fields (``drim.fields.ARM_FIELDS``) the widths above were
    #: built from, and over how many arms.  Recorded so a checkpoint cannot be
    #: loaded against a dataset that measured a different set: the widths can
    #: match by coincidence while the channels mean different things.
    #: Empty means the spec was built by hand rather than resolved from data.
    n_arms: int = 1
    #: 0-based indices of the arms the state vector covers, in order
    arm_ids: Tuple[int, ...] = ()
    prop_fields: Tuple[str, ...] = ()
    wrench_fields: Tuple[str, ...] = ()

    @property
    def dyn_fields(self) -> Tuple[str, ...]:
        """What the delta dynamics models, and so what its surprise can be about.

        Proprioception **and** the contact channels.  Splitting them is right for
        the policy's inputs — the corrector needs a per-step wrench of its own —
        but wrong for the dynamics: cap's model predicts the wrench along with
        everything else, and a surprise computed over a state that excludes force
        cannot say "it pushed back harder than expected", only "the arm did not
        go where the command implied".  For dressing that is the difference
        between a contact signal and a tracking error.
        """
        return tuple(self.prop_fields) + tuple(self.wrench_fields)

    @property
    def dyn_dim(self) -> int:
        return self.prop_dim + self.wrench_dim

    def __post_init__(self) -> None:
        assert self.pred_horizon >= self.exec_horizon > 0
        assert self.message_window > 0
        assert len(self.slow_offsets) == 2 and self.slow_offsets[1] == 0
        if self.fast_limits is not None:
            assert len(self.fast_limits) == self.act_dim, (
                f"{len(self.fast_limits)} ceilings for {self.act_dim} channels")
            assert all(v >= 0 for v in self.fast_limits)
        assert self.action_mode in ("absolute", "delta_ee_pos",
                                    "delta_action"), self.action_mode
        if self.act_scale is not None:
            assert len(self.act_scale) == self.act_dim, (
                f"{len(self.act_scale)} scales for {self.act_dim} channels")
            assert all(v > 0 for v in self.act_scale)

    # -- derived widths ----------------------------------------------------

    @property
    def n_obs_steps(self) -> int:
        return len(self.slow_offsets)

    @property
    def frame_dim(self) -> int:
        """Width of one slow frame: visual latents + proprioception + wrench."""
        return (self.vision_dim * len(self.cameras) + self.prop_dim
                + self.wrench_dim)

    @property
    def context_dim(self) -> int:
        return self.frame_dim * self.n_obs_steps

    @property
    def is_image(self) -> bool:
        return len(self.cameras) > 0

    def limits(self) -> np.ndarray:
        """The fast level's per-channel authority.

        ``fast_limits=None`` is not a default authority — it is a refusal to
        invent one.  dap derives cap's ceiling from ``fast_v09``'s declared
        reflex influence and drawer's from its scripted 8 mm reaction; both are
        numbers the *environment* published, and neither transfers to a
        different robot.  A run on new hardware has to state its own.
        """
        if self.fast_limits is None:
            raise ValueError(
                "fast_limits is unset: the corrector's authority must come from "
                "the rig's own declared reflex ceiling, expressed in normalized "
                "action units, not from a default carried over from cap/drawer")
        return np.asarray(self.fast_limits, dtype=np.float32)

    def assert_schema(self, other: "DrimSpec") -> None:
        """Refuse a spec whose channels are not the ones these weights saw.

        Width equality is not enough.  Dropping ``gripper_pos`` and
        ``gripper_vel`` (1+1) while gaining nothing else keeps every tensor
        shape valid and silently shifts every channel after them, which loads
        without complaint and scores wrongly.
        """
        if not self.prop_fields or not other.prop_fields:
            return                      # hand-built spec; nothing to compare
        assert (self.arm_ids, self.prop_fields, self.wrench_fields) == (
            other.arm_ids, other.prop_fields, other.wrench_fields), (
            f"field mismatch: checkpoint saw {other.n_arms}x"
            f"{other.prop_fields}+{other.wrench_fields}, this dataset offers "
            f"{self.n_arms}x{self.prop_fields}+{self.wrench_fields}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DrimSpec":
        d = dict(d)
        for k in ("cameras", "image_shape", "crop_shape", "slow_offsets",
                  "prop_fields", "wrench_fields", "arm_ids", "act_channels",
                  "act_scale"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        if d.get("fast_limits") is not None:
            d["fast_limits"] = tuple(float(v) for v in d["fast_limits"])
        return cls(**d)


def fast_limits_from_fraction(frac, act_scale: Sequence[float],
                              active: Sequence[int] = ()) -> Tuple[float, ...]:
    """Ceilings as a fraction of full command authority.

    This is dap's construction with the numerator left to the caller.  Cap
    computes ``MAX_TRANS_INFLUENCE_M / D`` — the displacement its environment
    declares the optional reflex may cause, divided by the displacement one unit
    of action delivers — and drawer computes ``8 mm / 0.05 = 0.16`` the same way.
    Both are "physical authority / action scale"; only the numerator is
    task-specific, and neither of theirs transfers to a different robot.

    Because actions here are already normalised by ``act_scale``, one normalised
    unit *is* full command, so the ratio reduces to ``frac`` directly.
    ``active`` names the channels the corrector may write; every other channel
    gets 0, which :class:`~diffusion_policy.drim.nets.FastCorrector` makes
    structurally silent rather than merely discouraged.

    ``frac`` may be one number for every channel, or one per channel.  The
    per-channel form is what a rig with axes of genuinely different travel
    needs: a single fraction is either too tight on the axis that moves or far
    wider than the others can justify, and a ceiling wider than a channel's own
    motion is authority the corrector was never shown how to use.
    """
    n = len(act_scale)
    fr = ([float(frac)] * n if np.isscalar(frac)
          else [float(v) for v in frac])
    assert len(fr) == n, f"{len(fr)} ceilings for {n} channels"
    for v in fr:
        assert 0.0 <= v <= 1.0, f"a fraction of full command, got {v}"
    keep = set(active) if len(active) else set(range(n))
    return tuple(fr[i] if i in keep else 0.0 for i in range(n))


def fast_frac_from_demand(demand: Dict[str, Any], active: Sequence[int],
                          lo: float = 0.02, hi: float = 0.5) -> float:
    """A ceiling measured from the data instead of carried over from cap/drawer.

    dap's numerator is a physical authority its *environment* declared.  Nothing
    here declares one, so the next most defensible source is the correction the
    demonstrations actually require: ``vs_chunk_mean`` p95 over the channels that
    move, which is what a plan committing to one action per chunk would have to
    be corrected by.

    Deliberately the **larger** of the two demand measures.  Too loose costs
    unused authority the ablation can take back; too tight clips on ordinary
    motion, which looks like a policy that cannot track and is much harder to
    read.  This is a hyperparameter set from training data, which is ordinary —
    it is not selection, and the derived value is logged and stored in the spec.
    """
    p95 = demand.get("vs_chunk_mean", {}).get("p95")
    if not p95 or not len(active):
        return lo
    return float(min(max(max(p95[i] for i in active), lo), hi))


def from_resolution(res, cameras: Sequence[str] = (),
                    act_width: Optional[int] = None,
                    n_declared: Optional[int] = None,
                    act_range: Optional[Sequence[float]] = None,
                    act_scale_per_arm: Optional[Sequence[float]] = None,
                    act_per_arm: Optional[int] = None,
                    **kw) -> DrimSpec:
    """Build a spec from what a dataset actually measured.

    The widths are a consequence of the resolution, never an argument: a spec
    whose ``prop_dim`` disagreed with the arrays it is fed would fail deep inside
    the observation encoder, where the shape error names a matmul rather than a
    missing sensor.
    """
    from diffusion_policy.drim import fields as F

    prop = res.prop_names()
    wrench = res.wrench_names()
    chans = kw.pop("act_channels", None)
    #: How many action channels belong to one arm is a property of the
    #: recording, not a constant: the bimanual teleop writes 6 per arm, the
    #: single-arm zigzag script writes 3 in total.  Deriving it from the
    #: recorded width is what lets one spec serve both; assuming 6 indexes past
    #: the end of a 3-wide action.
    per_arm = int(act_per_arm or 0)
    if act_width is not None and n_declared and per_arm:
        if act_width % n_declared == 0:
            per_arm = act_width // n_declared
        else:
            per_arm = act_width          # not divisible: treat as one block
    if chans is None:
        chans = (tuple(a * per_arm + c for a in res.arms for c in range(per_arm))
                 if per_arm else tuple(range(act_width or 0)))
        if act_width is not None:
            chans = tuple(c for c in chans if c < act_width)
    scale = kw.pop("act_scale", None)
    if scale is None and act_scale_per_arm and per_arm == len(act_scale_per_arm):
        cand = tuple(act_scale_per_arm[c % per_arm] for c in chans)
        #: A declared command scale is usable only if the recording fits inside
        #: it: the sampler clamps to [-1, 1], so a target beyond the scale is
        #: unreachable for any parameter values, and a scripted controller is
        #: under no obligation to stay within what a joystick can command.
        if act_range is None or not any(r > c for r, c in zip(act_range, cand)):
            scale = cand
    kw.pop("act_dim", None)          # a consequence of the channels, not an input
    return DrimSpec(
        prop_dim=res.width(prop), wrench_dim=res.width(wrench),
        act_dim=len(chans), act_channels=tuple(chans),
        act_scale=(tuple(scale) if scale is not None else None),
        n_arms=res.n_arms, arm_ids=tuple(res.arms),
        prop_fields=prop, wrench_fields=wrench,
        cameras=tuple(cameras), **kw)


#: Defaults for the dressing rig, independent of which fields a given recording
#: happens to carry.
#:
#: Task-specific defaults — horizons, cameras, command scales, packed-state
#: layouts — live in :mod:`diffusion_policy.drim.dressing`. Nothing in this
#: module knows about a particular robot.
