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

#: One arm's action: 3 channels, the end-effector position target.  The
#: ``zigzag_bed_*`` recordings write ``data/action`` as ``(T, 3)`` — there is no
#: angular half, and with ``action_mode="delta_ee_pos"`` the quantity the policy
#: predicts is ``action - ee_pos``, a pose *offset* in metres.
#:
#: This was 6 while the bimanual teleop was the reference recording.  Six was
#: not merely unused on a 3-wide action, it was silently corrosive: the spec
#: derives ``per_arm`` from the recorded width and gets 3, the declared scale
#: below is then 6 long, the two disagree, and the loader drops the declared
#: scale entirely and normalises by whatever range this particular recording
#: happened to cover.  A ceiling expressed as a fraction of that is a fraction
#: of one dataset, comparable with nothing.
ACT_PER_ARM: int = 3

#: What one normalised unit of action means, in metres of pose offset.
#:
#: Not the teleop's ``LIN_SCALE = 0.02``: that is a *velocity* deflection, and
#: the recorded offset reaches 0.039 m because the target leads the measured
#: pose by the controller's standing tracking error.  A scale the recording
#: exceeds is rejected by :func:`~diffusion_policy.drim.spec.from_resolution`
#: — the sampler clamps to [-1, 1], so a target beyond the scale is unreachable
#: for any parameter values — and 0.02 is exceeded on x by 1.96x.
#:
#: Measured over all 119 episodes (94,611 steps), ``|action - ee_pos|`` reaches
#: x 0.0392, y 0.0196, z 0.0072 m.  These are those maxima with ~1.5-2x
#: headroom, rounded: they are declared constants of the rig, not statistics of
#: this drop, so a later recording is still comparable as long as it fits.
ARM_ACT_SCALE: Tuple[float, ...] = (0.06, 0.03, 0.015)

#: The fast level's per-channel ceiling, as a fraction of ``ARM_ACT_SCALE``.
#:
#: One number for all three axes cannot be right here: the axes differ by more
#: than an order of magnitude.  Per-step demand (``|a_t - a_{t-1}|`` over the
#: executed prefix, p99, all 119 episodes) is 5.93 mm on x but 0.29 mm on y and
#: 0.54 mm on z — so a single 0.15 both clips x on ordinary motion and hands y
#: fifty times the authority its own motion justifies.
#:
#: Set from that demand with room for the reactive case, which the
#: demonstrations cannot show: every episode is an unperturbed success, so
#: sizing purely by demonstrated demand would leave the corrector unable to
#: answer a disturbance it was built for.  Each ceiling sits above the largest
#: within-chunk deviation observed on its channel and under
#: ``train.FAST_FRAC_CAP`` (0.30), which is the project's own bound on how much
#: a reflex may ever be given.
#:
#:     channel   scale     ceiling            step p99    within-chunk max
#:     x         0.06 m    0.25  = 15.0 mm    5.93 mm     18.4 mm
#:     y         0.03 m    0.05  =  1.5 mm    0.29 mm      0.97 mm
#:     z         0.015 m   0.10  =  1.5 mm    0.54 mm      1.26 mm
ARM_FAST_FRAC: Tuple[float, ...] = (0.25, 0.05, 0.10)

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
#: Placed by per-pixel temporal motion energy over ``zigzag_bed_0910``
#: (2026-09-10).  The action is off-centre in both views — the energy centroid
#: is at (179, 434) for the front camera and (256, 185) for the back, against a
#: frame centre of (240, 320) — so a *centred* crop is the wrong shape here.
#:
#: Three candidates, because the trade is real and this data cannot settle it:
#: a tighter box spends more of the output on the cloth and less on the arm and
#: bed that give it context.
#:
#: =========  ==============  ===================  ==========================
#: box        motion energy   native px / out px   
#: =========  ==============  ===================  ==========================
#: ``WIDE``   89 % / 86 %     1.96 (downsampled)   most context
#: ``MID``    77 % / 72 %     **1.00 (no resample)**  240x320 exactly as recorded
#: ``HAND``   41 % / 48 %     0.49 (upsampled)     cloth largest, context gone
#: =========  ==============  ===================  ==========================
#:
#: ``MID`` is worth noticing: it is exactly the network's input size, so the
#: frames reach the encoder with no interpolation at all.
#:
#: None of these is an illumination fix. Measured inside the boxes, the
#: *between-episode* brightness spread on the front camera is 6.2 / 6.6 / 6.6
#: against 6.7 for the full frame — unchanged. What falls is d', and only
#: because a tighter box moves more (within-episode std 3.0 / 4.5 / 5.9), which
#: masks the illumination cue rather than removing it. The hand is also the
#: *brightest* part of the scene (187 against 161 overall): it sits by the
#: window, so it is the region most exposed to a change in the daylight.
ROI_WIDE: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (16, 192, 336, 448),
    "image_bed_back": (128, 0, 336, 448),
}
ROI_MID: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (48, 320, 240, 320),
    "image_bed_back": (152, 0, 240, 320),
}
ROI_HAND: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (40, 352, 168, 224),
    "image_bed_back": (152, 56, 168, 224),
}
#: Drawn by hand on 2026-09-10 with :mod:`~diffusion_policy.drim.roi_tool`,
#: which is why it keeps less motion energy than the measured boxes and is the
#: better default anyway: a person can see that the arm and the bed edge are
#: what make the cloth's position readable, and motion energy cannot.
#:
#: **These are not 3:4.** The encoder input is 240x320, so the front box
#: (318x302, nearly square) is squashed vertically by 0.755 and stretched
#: horizontally by 1.060 — anisotropic by 1.40x — and the back box by 1.17x.
#: A consistent distortion is learnable, but the two cameras get *different*
#: ones, so the same object is a different shape in each. ``*_43`` below are
#: the same boxes forced to 3:4 about the same centre, kept for comparison.
ROI_CUSTOM: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (32, 181, 318, 302),   # energy 49%, px/out 1.25, d' 2.1
    "image_bed_back": (179, 119, 281, 319),   # energy 46%, px/out 1.17, d' 0.5
}
#: ``ROI_CUSTOM`` grown to the encoder's aspect: no distortion, and it happens
#: to keep more of the action (63 % / 52 % against 49 % / 46 %) because the
#: growth is sideways, which is where the arm is.
ROI_CUSTOM_43: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (32, 120, 318, 424),   # energy 63%, px/out 1.76, d' 2.2
    "image_bed_back": (180, 92, 280, 374),    # energy 52%, px/out 1.36, d' 0.5
}
ROIS: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "wide": ROI_WIDE, "mid": ROI_MID, "hand": ROI_HAND,
    "custom": ROI_CUSTOM, "custom43": ROI_CUSTOM_43, "full": {},
}
#: The default: the hand-drawn box, at the encoder's aspect. Chosen over the
#: measured ones because a person could see that the arm and the bed edge are
#: what make the cloth's position readable, and over the raw hand-drawn box
#: because that one is resized anisotropically, by a different factor in each
#: camera.
ROI = ROI_CUSTOM_43

#: The bed-front camera faces a window.  Between-episode brightness std is 6.7
#: against 2.5 within one, and the colour shifts with it (R sits ~10 below G/B,
#: by a margin that tracks the level).  Cropping does not fix this: measured
#: inside every ROI candidate the between-episode spread is unchanged at ~6.2-6.6,
#: and the hand is the *brightest* part of the scene because it sits by the
#: window.  Five episodes from one session already show it, and with every
#: episode a success nothing in the data discourages keying on it.
PHOTOMETRIC = dict(brightness=0.3, contrast=0.3, saturation=0.3,
                   channel_gain=0.12)

CAMERAS_0909: Tuple[str, ...] = ("image_bed_front", "image_bed_back")
CAMERAS_SMOKE: Tuple[str, ...] = ("image_arm1", "image_bed_front")


def arm_act_channels(arm_ids: Sequence[int], per_arm: int = ACT_PER_ARM
                     ) -> Tuple[int, ...]:
    """The recorded action channels belonging to the given arms."""
    return tuple(a * per_arm + c for a in arm_ids for c in range(per_arm))


def profile(cameras: Optional[Sequence[str]] = None,
            roi: str = "custom43") -> Dict[str, Any]:
    """Defaults a dressing run starts from.  ``--profile none`` skips all of it."""
    if roi not in ROIS:
        raise KeyError(f"unknown roi {roi!r}; have {sorted(ROIS)}")
    return {
        "layout": PACKED_STATE,
        "cameras": tuple(CAMERAS_0909 if cameras is None else cameras),
        "image_size": IMAGE_SIZE,
        "roi": ROIS[roi],
        "photometric": PHOTOMETRIC,
        "action_mode": "delta_ee_pos",
        "exo_key": "zigzag_action",
        "act_scale_per_arm": ARM_ACT_SCALE,
        "act_per_arm": ACT_PER_ARM,
        "fast_frac": ARM_FAST_FRAC,
        **HORIZONS,
    }
