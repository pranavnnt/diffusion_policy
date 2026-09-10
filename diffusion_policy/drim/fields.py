"""What a dressing episode is supposed to contain, and what to do when it isn't.

The collector is being extended to record the rig's full per-arm state.  Until
that lands, and afterwards whenever an arm is disconnected or a topic drops, a
dataset will be **missing fields it declares**.  Silently zero-filling them is
the failure this module exists to prevent: a constant channel trains fine, costs
nothing visible, and quietly removes the signal a whole stage depends on.

So the canonical schema is written down once, here, and every dataset is
resolved against it and reported on.  A missing field is **excluded from the
state vector**, not zero-filled — the width of what the policy reads then always
equals the width of what was actually measured, and the spec records which
fields it was built from so a checkpoint cannot be loaded against a different
set without saying so.

Field resolution accepts two dataset layouts:

* **one array per field**, ``data/arm1_q``, ``data/arm2_wrench``, … — what the
  extended collector should write, since a missing field is then simply an
  absent array;
* **one packed ``state`` array** plus a declared per-arm layout — what
  ``image_trials_4.zarr`` is, and what any earlier recording will be.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Field:
    """One per-arm quantity the rig can publish.

    ``group`` decides where the channel lands in the policy's inputs:
    ``"prop"`` joins the slow context and the fast corrector's per-step
    observation; ``"wrench"`` is handed to the corrector as its own per-step
    argument, mirroring dap's split (the slow context carries two frames, the
    corrector needs a wrench at every executed step).
    """

    name: str
    dim: int
    kind: str            # "linear" | "quat"
    group: str           # "prop" | "wrench"
    aliases: Tuple[str, ...] = ()
    #: what degrades when this field is absent, in one line, shown in the warning
    needed_for: str = ""
    #: fields carrying the same physical quantity in another form.  When two of
    #: them resolve, the one declared **earlier** in :data:`ARM_FIELDS` wins and
    #: the other is dropped with a note — keeping both would count the same
    #: measurement twice and quietly widen the state vector.
    conflicts: Tuple[str, ...] = ()


#: The baseline the collector is being extended to record, per arm.  Order is
#: the concatenation order of the state vector and is part of the contract:
#: changing it invalidates every existing checkpoint.
ARM_FIELDS: Tuple[Field, ...] = (
    Field("q", 7, "linear", "prop",
          ("joint", "joint_pos", "joint_position", "joint_positions"),
          "the slow policy's basic configuration input"),
    Field("dq", 7, "linear", "prop",
          ("joint_vel", "joint_velocity", "joint_velocities"),
          "the corrector's only view of how fast the arm is already moving"),
    Field("ee_pos", 3, "linear", "prop", ("eef_pos", "ee_position"),
          "the slow policy's task-space position"),
    Field("ee_quat", 4, "quat", "prop", ("eef_quat", "ee_rot", "ee_orientation"),
          "task-space orientation, and the rotation relating the wrench frames"),
    Field("ee_lin_vel", 3, "linear", "prop", ("eef_lin_vel", "ee_linear_velocity"),
          "task-space velocity the corrector reacts to",
          conflicts=("ee_twist",)),
    Field("ee_ang_vel", 3, "linear", "prop", ("eef_ang_vel", "ee_angular_velocity"),
          "task-space angular velocity", conflicts=("ee_twist",)),
    #: The same six numbers as ``ee_lin_vel`` + ``ee_ang_vel``, as one array.
    #: Declared after them so that a dataset carrying both keeps the split pair
    #: and drops this; a dataset carrying only this keeps it.
    Field("ee_twist", 6, "linear", "prop", ("eef_twist",),
          "task-space linear and angular velocity, as one array",
          conflicts=("ee_lin_vel", "ee_ang_vel")),
    Field("gripper_pos", 1, "linear", "prop", ("gripper_position",),
          "whether the cloth is still held; a slip is invisible without it"),
    Field("gripper_vel", 1, "linear", "prop", ("gripper_velocity",),
          "gripper motion, distinguishing a commanded open from a slip"),
    Field("tau_J", 7, "linear", "prop", ("joint_torque", "tau", "effort"),
          "joint-space contact, including drag on the forearm that a 6-D "
          "end-effector wrench cannot represent"),
    Field("wrench", 6, "linear", "wrench",
          ("wrench_ee_base", "wrench_base", "wrench_ee_ee", "wrench_ee"),
          "the fast corrector's contact signal, and the physical content of "
          "D2's surprise channel; without it S is a kinematic tracking error",
          conflicts=("ee_force",)),
    #: Force without torque.  Half a wrench, and the half that carries most of
    #: the contact signal for dressing — but a policy cannot see a moment about
    #: the end-effector with it, so it is the fallback, not the target.
    Field("ee_force", 3, "linear", "wrench", ("force", "ee_force_base"),
          "the contact signal, when the full 6-D wrench is not recorded",
          conflicts=("wrench",)),
)

FIELDS_BY_NAME: Dict[str, Field] = {f.name: f for f in ARM_FIELDS}
_ORDER: Dict[str, int] = {f.name: i for i, f in enumerate(ARM_FIELDS)}


def _order(name: str) -> int:
    return _ORDER[name]


PROP_FIELDS: Tuple[str, ...] = tuple(f.name for f in ARM_FIELDS if f.group == "prop")
WRENCH_FIELDS: Tuple[str, ...] = tuple(f.name for f in ARM_FIELDS if f.group == "wrench")


def arm_name(i: int) -> str:
    """The rig names its arms ``arm1``/``arm2``; indices here are 0-based."""
    return f"arm{i + 1}"


@dataclass(frozen=True)
class PackedLayout:
    """Where each field sits inside a concatenated per-arm ``state`` block.

    Only needed for datasets that pack everything into one array.  ``stride`` is
    the per-arm block width, so arm ``i``'s field occupies
    ``state[:, i*stride + lo : i*stride + hi]``.
    """

    stride: int
    slices: Dict[str, Tuple[int, int]]
    key: str = "state"

    def validate(self) -> None:
        for name, (lo, hi) in self.slices.items():
            f = FIELDS_BY_NAME.get(name)
            assert f is not None, f"{name!r} is not a declared field"
            assert hi - lo == f.dim, (
                f"{name} occupies {hi - lo} channels, the schema declares {f.dim}")
            assert 0 <= lo < hi <= self.stride, (
                f"{name} slice ({lo},{hi}) escapes the {self.stride}-wide arm block")


#: ``demo_for_testing/image_trials_4.zarr``: 28 = 2 arms x (7 joint, 3 eef
#: position, 4 eef quaternion).  Everything else in ``ARM_FIELDS`` is absent, and
#: :func:`resolve` says so rather than inventing it.
SMOKE_LAYOUT = PackedLayout(
    stride=14,
    slices={"q": (0, 7), "ee_pos": (7, 10), "ee_quat": (10, 14)},
)


@dataclass
class FieldStatus:
    name: str
    arm: str
    present: bool
    reason: str = ""          # why it is unusable, when present is False
    constant: bool = False    # present but never changes
    note: str = ""


@dataclass
class Resolution:
    """The per-field arrays that survived, plus a full account of what did not."""

    arrays: Dict[str, np.ndarray]           # "arm1_q" -> [T, dim]
    statuses: List[FieldStatus]
    #: 0-based indices of the arms that carry any usable field at all.  An arm
    #: the collector could not reach is recorded as zeros, and dropping it is
    #: the honest reading — the alternative, dropping every field that arm lacks,
    #: would empty the state vector because it lacks all of them.
    arms: Tuple[int, ...] = ()
    #: fields usable on **every live** arm — the only ones that enter the state
    #: vector, because a field present on one live arm and absent on another has
    #: no well-defined width
    usable: Tuple[str, ...] = ()

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def missing(self) -> List[FieldStatus]:
        return [s for s in self.statuses if not s.present]

    def prop_names(self) -> Tuple[str, ...]:
        return tuple(n for n in PROP_FIELDS if n in self.usable)

    def wrench_names(self) -> Tuple[str, ...]:
        return tuple(n for n in WRENCH_FIELDS if n in self.usable)

    def width(self, names: Sequence[str]) -> int:
        return self.n_arms * sum(FIELDS_BY_NAME[n].dim for n in names)

    def stack(self, names: Sequence[str]) -> np.ndarray:
        """Concatenate ``names`` across arms, in schema order: ``[T, width]``."""
        if not names:
            n = len(next(iter(self.arrays.values()))) if self.arrays else 0
            return np.zeros((n, 0), np.float32)
        parts = [self.arrays[f"{arm_name(a)}_{n}"]
                 for a in self.arms for n in names]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def report(self) -> str:
        rows = []
        for s in self.statuses:
            mark = "ok " if s.present else "-- "
            extra = s.note or s.reason
            rows.append(f"  {mark}{s.arm}.{s.name}" + (f"   {extra}" if extra else ""))
        return "\n".join(rows)


def _lookup(source: Dict[str, Any], arm: int, f: Field, n_arms: int
            ) -> Optional[np.ndarray]:
    """One array per field, under any of the spellings the field declares."""
    a = arm_name(arm)
    names = (f.name,) + f.aliases
    candidates = [f"{a}_{n}" for n in names] + [f"{n}_{a}" for n in names]
    if n_arms == 1:
        candidates += list(names)
    for c in candidates:
        if c in source:
            return np.asarray(source[c], dtype=np.float32)
    return None


def _from_packed(source: Dict[str, Any], arm: int, f: Field,
                 layout: PackedLayout) -> Optional[np.ndarray]:
    if layout is None or f.name not in layout.slices or layout.key not in source:
        return None
    lo, hi = layout.slices[f.name]
    o = arm * layout.stride
    return np.asarray(source[layout.key], dtype=np.float32)[:, o + lo:o + hi]


def _check(a: np.ndarray, f: Field, arm: str) -> FieldStatus:
    """Present-but-unusable is as important as absent, and looks identical downstream.

    An arm the collector could not reach is recorded as ``zeros(14)``, so its
    quaternion has norm 0 rather than 1 — a value that is not a rotation at all.
    Feeding it to the quaternion delta produces a well-defined zero and no error,
    which is exactly why it has to be caught here.
    """
    if f.kind == "quat":
        norms = np.linalg.norm(a, axis=-1)
        if not np.all(np.isfinite(norms)) or np.any(norms < 0.5):
            return FieldStatus(f.name, arm, False,
                               reason=f"quaternion norm {norms.min():.3f}, not a rotation")
        if not np.allclose(norms, 1.0, atol=1e-2):
            return FieldStatus(f.name, arm, True, constant=False,
                               note=f"quaternion norms off unit by "
                                    f"{np.abs(norms - 1).max():.3f}")
    if not np.all(np.isfinite(a)):
        return FieldStatus(f.name, arm, False, reason="contains NaN or inf")
    if np.all(a == 0.0):
        return FieldStatus(f.name, arm, False, reason="identically zero")
    if float(a.std(0).max()) == 0.0:
        return FieldStatus(f.name, arm, True, constant=True, note="constant")
    return FieldStatus(f.name, arm, True)


def resolve(source: Dict[str, Any], n_arms: int = 2,
            layout: Optional[PackedLayout] = None,
            require: Sequence[str] = (),
            warn: bool = True) -> Resolution:
    """Match a dataset against :data:`ARM_FIELDS` and account for every field.

    ``require`` names fields that must resolve; anything listed there and absent
    raises instead of warning, which is how a training run declares "this stage
    is meaningless without a wrench" rather than discovering it in the numbers.
    """
    if layout is not None:
        layout.validate()
    arrays: Dict[str, np.ndarray] = {}
    statuses: List[FieldStatus] = []
    for a in range(n_arms):
        an = arm_name(a)
        for f in ARM_FIELDS:
            raw = _lookup(source, a, f, n_arms)
            if raw is None:
                raw = _from_packed(source, a, f, layout)
            if raw is None:
                statuses.append(FieldStatus(f.name, an, False, reason="not recorded"))
                continue
            if raw.ndim == 1:
                raw = raw[:, None]
            if raw.shape[-1] != f.dim:
                statuses.append(FieldStatus(
                    f.name, an, False,
                    reason=f"{raw.shape[-1]} channels, schema declares {f.dim}"))
                continue
            st = _check(raw, f, an)
            statuses.append(st)
            if st.present:
                arrays[f"{an}_{f.name}"] = raw

    #: A field usable on one arm and not another has no well-defined width, so
    #: it is dropped from both rather than making the two arms asymmetric.
    #: An arm with nothing usable is dropped whole, before the across-arms rule
    #: runs.  Applying that rule first would let one disconnected arm delete
    #: every field from the live one.
    live = tuple(a for a in range(n_arms)
                 if any(k.startswith(f"{arm_name(a)}_") for k in arrays))
    dead = [arm_name(a) for a in range(n_arms) if a not in live]
    usable = tuple(f.name for f in ARM_FIELDS
                   if all(f"{arm_name(a)}_{f.name}" in arrays for a in live))
    dropped_dupes: List[Tuple[str, str]] = []
    for f in ARM_FIELDS:
        if f.name not in usable:
            continue
        for other in f.conflicts:
            #: schema order decides; only a *later* field yields to an earlier one
            if other in usable and _order(other) < _order(f.name):
                usable = tuple(n for n in usable if n != f.name)
                dropped_dupes.append((f.name, other))
                break
    arrays = {k: v for k, v in arrays.items() if k.split("_", 1)[1] in usable}
    res = Resolution(arrays=arrays, statuses=statuses, arms=live, usable=usable)
    if dropped_dupes and warn:
        warnings.warn(
            "dropped as duplicates of a field already present: "
            + ", ".join(f"{a} (same quantity as {b})" for a, b in dropped_dupes),
            stacklevel=3)
    if dead and warn:
        warnings.warn(
            f"{', '.join(dead)} carries no usable field and is dropped; the "
            f"state vector covers {', '.join(arm_name(a) for a in live) or 'nothing'}. "
            f"Its action channels are still in the target, so the policy is being "
            f"asked to command an arm it cannot observe.", stacklevel=3)

    bad = [n for n in require if n not in usable]
    if bad:
        raise ValueError(
            f"required field(s) {bad} are not usable in this dataset:\n"
            + res.report())
    if warn:
        _warn(res)
    return res


def _warn(res: Resolution) -> None:
    missing = res.missing()
    if not missing:
        return
    by_name: Dict[str, List[FieldStatus]] = {}
    for s in missing:
        by_name.setdefault(s.name, []).append(s)
    lines = [f"dataset is missing {len(by_name)} of {len(ARM_FIELDS)} declared "
             f"per-arm fields; they are excluded from the state vector, not "
             f"zero-filled:"]
    for name, sts in by_name.items():
        f = FIELDS_BY_NAME[name]
        arms = ", ".join(s.arm for s in sts)
        why = sts[0].reason
        lines.append(f"  - {name} ({f.dim}d) on {arms}: {why}")
        if f.needed_for:
            lines.append(f"      needed for: {f.needed_for}")
    degraded = [n for n in ("wrench", "dq", "ee_lin_vel") if n not in res.usable]
    if degraded:
        lines.append("  => the fast corrector reads position only; D2's surprise "
                     "is a kinematic tracking error, not a contact signal.")
    warnings.warn("\n".join(lines), stacklevel=3)


def modalities(res: Resolution, names: Sequence[str]
               ) -> List[Tuple[str, Tuple[int, int], int, str]]:
    """Modality table for the delta dynamics over ``res.stack(names)``.

    Quaternions get three delta channels, not four: the delta of a rotation is a
    rotation vector, and subtracting quaternions componentwise would put a jump
    at the double cover into every surprise that crosses it.
    """
    out, off = [], 0
    for a in res.arms:
        for n in names:
            f = FIELDS_BY_NAME[n]
            w = 3 if f.kind == "quat" else f.dim
            out.append((f"{arm_name(a)}_{n}", (off, off + f.dim), w, f.kind))
            off += f.dim
    return out
