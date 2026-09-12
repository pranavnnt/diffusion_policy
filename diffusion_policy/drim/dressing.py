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

#: One arm's action: 3 channels.  ``data/action`` is the operator's **centroid**
#: — the point the zigzag oscillates around, integrated from joystick velocity
#: by the collector — not an end-effector pose target.  There is no angular
#: half.
#:
#: This was 6 while the bimanual teleop was the reference recording.  Six was
#: not merely unused on a 3-wide action, it was silently corrosive: the spec
#: derives ``per_arm`` from the recorded width and gets 3, the declared scale
#: below is then 6 long, the two disagree, and the loader drops the declared
#: scale entirely and normalises by whatever range this particular recording
#: happened to cover.  A ceiling expressed as a fraction of that is a fraction
#: of one dataset, comparable with nothing.
ACT_PER_ARM: int = 3

#: What one normalised unit means, in metres of centroid displacement.
#:
#: The target is ``action[t+i] - action[t]`` (:data:`action_mode` below), whose
#: largest excursion over a 16-step chunk, measured across all 119 episodes, is
#: x 10.07, y 11.25, z 10.00 mm.  0.015 m covers every channel with ~1.4x
#: headroom and is one number because the three axes now genuinely agree —
#: under the previous target they spanned 39 / 20 / 7 mm and could not.
#:
#: Not the teleop's ``LIN_SCALE = 0.02``: that is a velocity deflection, and a
#: scale the recording exceeds is rejected outright by
#: :func:`~diffusion_policy.drim.spec.from_resolution`, since ``sample_chunk``
#: clamps to [-1, 1] and a target beyond the scale is then unreachable for any
#: parameter values.
ARM_ACT_SCALE: Tuple[float, ...] = (0.015, 0.015, 0.015)

#: The fast level's per-channel ceiling, as a fraction of ``ARM_ACT_SCALE``.
#:
#: Per-step demand over the executed prefix is 0.0444 at p99 on **every**
#: channel, and the largest within-chunk deviation is 0.156 / 0.217 / 0.156.
#: 0.25 sits above all of those and under ``train.FAST_FRAC_CAP`` (0.30), the
#: project's own bound on how much a reflex may ever be given.
#:
#: The per-channel form is kept although the three values are equal: the
#: asymmetry that forced it (5.93 mm of step demand on x against 0.29 mm on y)
#: belonged to ``delta_ee_pos``, whose x channel carried the zigzag excursion
#: through the subtracted pose.  This target does not, so the axes agree.
#:
#: They stay deliberately looser than demonstrated demand, because the
#: demonstrations cannot show the reactive case: every episode is an
#: unperturbed success, so sizing purely by what was asked for would leave the
#: corrector unable to answer the disturbance it exists for.
ARM_FAST_FRAC: Tuple[float, ...] = (0.25, 0.25, 0.25)

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
#:
#: Was (240, 320).  Three reasons it moved, in order of how much they matter:
#:
#: * **Every axis downsamples.**  At 240x320 the front box went 310 -> 240
#:   vertically but 189 -> 320 horizontally: real rows discarded while 69 % more
#:   columns were invented by interpolation, in the same frame.  Manufactured
#:   pixels cost compute and carry nothing, and the rows they were traded for do
#:   not come back.  The current boxes are 205x142 and 188x229, so the input has
#:   to satisfy ``H <= 188`` and ``W <= 142`` for both axes of both cameras to
#:   shrink; 160x128 is the largest multiple of 32 that does.
#: * **Cost.**  73 % fewer pixels than 240x320.  Measured on an RTX 4090,
#:   resnet18 on the training batch shape drops from 49.0 ms a step to about
#:   13 ms, which is most of a five-hour run.  Inference barely moves: the
#:   replan is dominated by flow sampling, not by vision, so this is not a fix
#:   for the control-rate margin.
#: * 192 and 160 are both divisible by 32, the resnet's total stride.  240 was
#:   not, so the last stage never landed on a whole number.
#:
#: What it costs: px/out rises to 1.42 / 2.10, so a third to a half of the
#: native pixels inside the box are dropped rather than seen.  Dropping is not
#: the same failure as inventing — it is a deliberate resolution choice, where
#: an upsampled axis spends compute on interpolation and returns nothing.
#:
#: It does **not** fix the distortion.  The two boxes disagree with each other
#: (front 1.44 tall, back 0.82 wide), so every single input size squashes one
#: of them; only redrawing the boxes to a shared aspect removes that.
IMAGE_SIZE: Tuple[int, int] = (160, 128)

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
ROI_CUSTOM_0910: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (32, 181, 318, 302),   # energy 49%, px/out 1.25, d' 2.1
    "image_bed_back": (179, 119, 281, 319),   # energy 46%, px/out 1.17, d' 0.5
}
#: ``ROI_CUSTOM_0910`` grown to 3:4 about the same centre, to match the 240x320
#: the encoder took at the time.  Kept only so the trained runs stay readable:
#: at the current 192x160 its 0.75 aspect is the *worst* of the set (1.60x, and
#: px/out 4.4 — three quarters of the pixels inside the box discarded), so it is
#: no longer a candidate for anything.
ROI_CUSTOM_43: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (32, 120, 318, 424),   # energy 63%, px/out 1.76, d' 2.2
    "image_bed_back": (180, 92, 280, 374),    # energy 52%, px/out 1.36, d' 0.5
}
#: Drawn by hand on 2026-09-11 over ``zigzag_bed_0910_4``, after the curtains
#: went up, and this is the default.
#:
#: It is much tighter than anything before it — front 310x189 against the old
#: 318x424 — and the whole point of that is the third number.  ``d'`` on the
#: front camera falls from **2.2 to 0.78**: between-episode brightness spread
#: inside the box (1.43) is now *below* the within-episode spread (1.83), so
#: identifying which demonstration a frame came from by illumination alone is no
#: longer easy.  Earlier boxes could not do this — measured inside every
#: candidate the between-episode spread stayed at 6.2-6.6, because the hand sits
#: by the window and was the brightest thing in frame.  Excluding the window
#: side is what moved it.
#:
#: The cost is motion energy: 50 % / 35 % against 63 % / 52 %.  Half the
#: movement in the scene is now outside the box.  Whether what was dropped was
#: the task or the background is not something ``energy`` can distinguish, and
#: this is the trade that was accepted knowingly.
#:
#: It is also the only box in this module whose every axis *downsamples* at
#: 192x160 (0.62/0.85 front, 0.91/0.65 back).  The 3:4 boxes above upsample
#: horizontally — inventing columns by interpolation in the same frame where
#: rows are being thrown away — which costs compute and returns nothing.
ROI_CUSTOM_0911: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (34, 310, 310, 189),   # energy 50%, px/out 1.91, d' 0.78
    "image_bed_back": (197, 148, 212, 245),   # energy 35%, px/out 1.69, d' 0.60
}
#: Drawn by hand on 2026-09-11 over ``zigzag_bed_0911_1`` — the first recording
#: made at the 192x160 input, and the first made entirely behind the curtains.
#: This is the default.
#:
#: The front box finally lands on ``px/out`` **0.95**: one native pixel per
#: output pixel, so the resize neither discards rows nor invents columns. Every
#: earlier box was above 1.2 and some above 4, which is a straight loss —
#: pixels read off disk, carried through the crop, and then averaged away.
#:
#: ``d'`` reads 0.20 / 0.32 here, but that number is not comparable with the
#: ones above it: it was measured on a recording whose between-episode
#: brightness spread is **0.18 levels** against the 9.80 of the sunlit data the
#: earlier boxes were scored on. Any box scores well on data with no
#: illumination variation to key on. The box may well also be better — it
#: excludes the window side — but this measurement cannot separate the two, and
#: the comparison was not re-run.
#:
#: The cost is again motion energy, 44 % / 25 %, and the front box is now tight
#: enough that the mannequin's arm is largely outside it. Whether the gripper
#: and the top of the garment are enough to act on is a judgement the videos
#: were rendered to support, not something ``energy`` answers.
ROI_CUSTOM_0911_2: Dict[str, Tuple[int, int, int, int]] = {
    "image_bed_front": (31, 310, 205, 142),   # energy 44%, px/out 0.95, d' 0.20
    "image_bed_back": (214, 159, 188, 229),   # energy 25%, px/out 1.40, d' 0.32
}
#: The name the rest of the module reaches for.
ROI_CUSTOM = ROI_CUSTOM_0911_2
ROIS: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "wide": ROI_WIDE, "mid": ROI_MID, "hand": ROI_HAND,
    "custom": ROI_CUSTOM, "custom0910": ROI_CUSTOM_0910,
    "custom0911": ROI_CUSTOM_0911, "custom43": ROI_CUSTOM_43, "full": {},
}
#: The default: the 2026-09-11 hand-drawn box.  Chosen over the measured ones
#: because a person can see that the arm and the bed edge are what make the
#: cloth's position readable and motion energy cannot, and over its own
#: predecessors because it is the one that actually removed the illumination
#: shortcut (d' 2.2 -> 0.78) rather than only moving the box around.
ROI = ROI_CUSTOM_0911_2

#: Photometric augmentation, and the brightness range is **asymmetric on
#: purpose**.
#:
#: Measured inside the current ROI, per episode: the 0910 recordings sit at
#: 100.8 (bed-front) and 97.8 (bed-back) while the curtained 0911 conditions
#: the robot now runs under sit at 91.2 and 64.1. Training on 0910 and
#: deploying at 0911 therefore asks for a multiplier down to **x0.66** on the
#: back camera, and never more than x1.11. The old symmetric 0.3 reached only
#: x0.70, so inference was outside the training distribution; widening it
#: symmetrically to x[0.5, 1.5] would cover the gap but also stretch the
#: brightest 0910 episodes (155.8) to 234 and clip.
#:
#: This range belongs to *this* pairing and should not be carried forward. A
#: run trained on 0911 has no such gap — those recordings vary by 0.81 / 1.34
#: between episodes against 0910's 10.37 — so its augmentation is protecting
#: against future drift, not closing a measured shift, and wants re-deciding
#: rather than inheriting.
#:
#: The bed-front camera faces a window, which is what the per-channel gain is
#: for: brightness and contrast move all three channels together, daylight
#: moves them apart. In 0910 the red channel sits ~10 below green and blue by a
#: margin that tracks the level.
PHOTOMETRIC = dict(brightness=(0.55, 1.25), contrast=0.3, saturation=0.3,
                   channel_gain=0.12)

#: Channels left out of the observation although the rig records them.
#:
#: Rotation first, because it is not a judgement call: within an episode the
#: end-effector turns by at most **0.19 deg** (median 0.04) across all 119
#: episodes, and ``ee_twist``'s angular half is *identically zero* on every one
#: of the 94,611 steps.  ``ee_quat`` does differ between episodes, by up to
#: 19.8 deg, which makes it worse than useless: a channel that is constant
#: within a trial and distinct across trials is an episode fingerprint, the
#: same shape of shortcut as the illumination one.
#:
#: Then joint space.  The arm is 7-DoF, the wrist never turns, and
#: ``elbow_lock_enabled`` is true on 100 % of steps with
#: ``commanded_elbow_velocity`` identically zero — so the only thing free to
#: move is the 3-D end-effector position, and ``q`` is a re-encoding of it.
#: Measured on a held-out third: a linear probe of the target reads 0.3565 from
#: ``ee_pos + ee_lin_vel``, 0.3753 once all fourteen ``q``/``dq`` channels are
#: added.  Fourteen dimensions buy 0.019, and on their own they reach 0.0874.
#: ``dq`` is also the noisiest thing recorded — its value at the nearest
#: neighbour 4.4 mm away differs by more than its own standard deviation.
#:
#: This is a *default*, not a finding about what a policy could use. Restore any
#: of it with ``--keep`` and the run will say what changed.  Two reasons to
#: expect ``dq`` back: the probe is linear while the policy is not, and under
#: contact — which no demonstration here contains — ``dq`` is the only channel
#: that can show the arm failing to move as commanded, since ``tau_J`` is not
#: recorded.
EXCLUDE: Tuple[str, ...] = ("q", "dq", "ee_quat", "ee_twist", "ee_ang_vel")

CAMERAS_0909: Tuple[str, ...] = ("image_bed_front", "image_bed_back")
CAMERAS_SMOKE: Tuple[str, ...] = ("image_arm1", "image_bed_front")


def arm_act_channels(arm_ids: Sequence[int], per_arm: int = ACT_PER_ARM
                     ) -> Tuple[int, ...]:
    """The recorded action channels belonging to the given arms."""
    return tuple(a * per_arm + c for a in arm_ids for c in range(per_arm))


def profile(cameras: Optional[Sequence[str]] = None,
            roi: str = "custom") -> Dict[str, Any]:
    """Defaults a dressing run starts from.  ``--profile none`` skips all of it."""
    if roi not in ROIS:
        raise KeyError(f"unknown roi {roi!r}; have {sorted(ROIS)}")
    return {
        "layout": PACKED_STATE,
        "cameras": tuple(CAMERAS_0909 if cameras is None else cameras),
        "image_size": IMAGE_SIZE,
        "roi": ROIS[roi],
        "photometric": PHOTOMETRIC,
        "action_mode": "delta_action",
        "exo_key": "zigzag_action",
        "exclude": EXCLUDE,
        "act_scale_per_arm": ARM_ACT_SCALE,
        "act_per_arm": ACT_PER_ARM,
        "fast_frac": ARM_FAST_FRAC,
        **HORIZONS,
    }
