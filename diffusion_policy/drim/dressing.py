"""Everything that is true of **this rig and this task**, and not of DRIM.

DRIM is the staged chain in :mod:`~diffusion_policy.drim.policy` — a frozen slow
chunk policy, a bounded fast corrector, and an upward message built from a
dynamics residual, with the exact-null identity that makes ``D2 - B1`` a
measurable claim. None of that is in this file.

What *is* in this file is the set of choices the real dressing rig forced, each
of which would be wrong somewhere else. They are collected here so the boundary
is visible: if a number or a reparameterisation below looks arbitrary, that is
because it is a property of a bimanual Franka running a scripted zigzag under
joystick supervision, not of the method.

**The two substantive ones.**

``action_mode = "delta_ee_pos"``
    The recorded ``action`` is an absolute end-effector pose target, and the
    controller tracks it to within 8 mm — so its absolute value is nearly the
    observed pose, and predicting it is close to copying an input. Measured on
    ``zigzag_bed_0909_clean``: holding the previous action scores 0.00024
    against a predict-zero of 0.184, beating a trained ``B0`` (~0.017) by ~90x.
    Predicting the delta puts the copycat at 0.117 against 0.243. The command
    sent to the robot is unchanged — the observed pose is added back at
    execution — so this is a reparameterisation, not a different controller.

    It also makes trajectory divergence mean something. With an absolute target
    the commanded pose does not depend on where the arm actually is, so drift
    self-corrects and compounding never shows. With a delta it accumulates,
    which is the thing worth measuring on a task that is hard at the trajectory
    level and easy step to step.

``exo_key = "zigzag_action"``
    A scripted zigzag velocity is injected during teleop *and* during inference
    and is never predicted. The arm's measured motion is almost entirely that
    primitive — ``corr(ee_twist, zigzag_action)`` is 0.78 / 0.996 / 0.988 per
    axis against 0.006 / -0.038 / 0.026 for the operator's delta — so a dynamics
    model not given it must infer its phase from state, and whatever it cannot
    infer lands in the surprise channel that is the entire content of D2's
    message. It is known at training and inference alike, so conditioning on it
    is not privileged information. Feeding it raised held-out dynamics skill
    from +68.5 % to +71.0 %.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from diffusion_policy.drim import fields as F

#: One arm's action: 3 linear m/s then 3 angular rad/s.
ACT_PER_ARM: int = 6

#: The teleop's full-deflection command, from
#: ``dressing_policies/data_collection/single_joystick_teleop.py``
#: (``LIN_SCALE = 0.02``, ``ANG_SCALE = 0.05``).  Used to normalise actions so
#: that one unit means "full command" and a fast-level ceiling expressed as a
#: fraction of it is comparable across recordings.
#:
#: A *scripted* controller is under no obligation to stay inside what the
#: joystick can command — the zigzag runs to 0.08 m/s — so the loader falls back
#: to the observed range whenever a recording exceeds this.
ARM_ACT_SCALE: Tuple[float, ...] = (0.02, 0.02, 0.02, 0.05, 0.05, 0.05)

#: ``demo_for_testing/image_trials_4.zarr`` and the ``zigzag_bed_*`` recordings
#: pack per-arm state into one array: 7 joint positions, 3 eef position, 4 eef
#: quaternion.
PACKED_STATE = F.PackedLayout(
    stride=14, slices={"q": (0, 7), "ee_pos": (7, 10), "ee_quat": (10, 14)})

#: Horizons sized for the rig's measured 14.3 Hz, which the 0909 recordings hold
#: to within 0.079 s.  ``pred16/exec8`` is 1.1 s predicted and 0.56 s executed;
#: ``message_window=32`` is 2.24 s, one full zigzag cycle (the commanded
#: velocity flips sign every 14 steps, so a half-cycle window would make the
#: message's content depend on which half it landed in).
HORIZONS: Dict[str, Any] = dict(pred_horizon=16, exec_horizon=8,
                                message_window=32)

#: Frames are resized to this on load.  The encoder crops ~90 % of what it is
#: handed, so a 480x640 frame would show it 45 % of the scene and call the rest
#: augmentation; and two cameras at full resolution is 16.6 GB before chunking.
IMAGE_SIZE: Tuple[int, int] = (240, 320)

#: Where the task actually happens in each frame, as ``(y0, x0, h, w)`` in the
#: native 480x640, applied **before** the resize.
#:
#: Measured from per-pixel temporal motion energy over ``zigzag_bed_0910``
#: (2026-09-10).  The action is off-centre in both views — the energy centroid
#: is at (179, 434) for the front camera and (256, 185) for the back, against a
#: frame centre of (240, 320) — so a *centred* crop is the wrong shape here. A
#: 70 % box placed on the action captures 88.5 % / 85.2 % of the motion energy,
#: which matches or beats a 90 % centred crop (87.7 % / 86.3 %) while spending
#: ~1.6x more of the output pixels on the cloth.
#:
#: This is field-of-view selection and is fixed. The encoder's ``crop_shape``
#: random crop still runs inside it, as augmentation.
ROI: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (16, 192, 336, 448),
    "image_bed_back": (128, 0, 336, 448),
}

#: The bed-front camera faces a window.  Between-episode brightness std is 6.7
#: against 2.5 within one, and the colour shifts with it (R sits ~10 below G/B,
#: by a margin that tracks the level).  Five episodes from one session already
#: show that; different times of day would show much more, and with every
#: episode a success nothing in the data discourages keying on it.
PHOTOMETRIC = dict(brightness=0.3, contrast=0.3, saturation=0.3,
                   channel_gain=0.12)

CAMERAS_0909: Tuple[str, ...] = ("image_bed_front", "image_bed_back")
CAMERAS_SMOKE: Tuple[str, ...] = ("image_arm1", "image_bed_front")


def arm_act_channels(arm_ids: Sequence[int], per_arm: int = ACT_PER_ARM
                     ) -> Tuple[int, ...]:
    """The recorded action channels belonging to the given arms."""
    return tuple(a * per_arm + c for a in arm_ids for c in range(per_arm))


def profile(cameras: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Defaults a dressing run starts from.  ``--profile none`` skips all of it."""
    return {
        "layout": PACKED_STATE,
        "cameras": tuple(CAMERAS_0909 if cameras is None else cameras),
        "image_size": IMAGE_SIZE,
        "roi": ROI,
        "photometric": PHOTOMETRIC,
        "action_mode": "delta_ee_pos",
        "exo_key": "zigzag_action",
        "act_scale_per_arm": ARM_ACT_SCALE,
        "act_per_arm": ACT_PER_ARM,
        **HORIZONS,
    }
