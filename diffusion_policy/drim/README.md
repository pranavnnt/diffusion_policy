# DRIM in the dressing stack

A port of the `IMG_B0` / `IMG_B1` / `IMG_D2` chain from `~/dap`
(`empriselab/reactive-policy`) into this repository, so it can be trained on
dressing demonstrations instead of a simulator.

## What the three stages are

Staged, each freezing the last, so every contrast isolates one addition:

| stage | what it adds | trains |
|---|---|---|
| `B0` | the slow policy alone: vision + proprioception (+ wrench) → action chunk, rectified flow over `ConditionalUnet1D` | vision + flow field |
| `B1` | a bounded per-channel **fast corrector** on a frozen `B0`, supervised directly on the executed action | corrector only |
| `D2` | the **upward message** on a frozen `B1` — `D2_SURPRISE_ONLY`: the message reads two dynamics channels, `S` (standardised surprise) and `U` (predicted log-sigma), over a causal window, and nothing else | message encoder, `dv`, reconciler, two gates |

`D2` needs a fourth thing trained first: the **delta dynamics** whose residual
*is* the surprise. That is stage `dyn`.

The claim `D2 − B1` is only meaningful because of the exact-null identity:

```
v(x, t, ctx, m) = v_base(x, t, ctx) + gate * (dv(x,t,ctx,m) − dv(x,t,ctx,0))
```

At `m = 0` the bracket is identically zero *for any parameter values*, so `D2`
with a null message **is** `B1`, bit for bit — not merely at initialisation.
`tests/test_drim.py` checks this after perturbing the message path, which is the
only version of the check that means anything.

## What is DRIM, and what is this rig

`dressing.py` holds every choice that is true of the real dressing setup and
not of the method. If a constant or a reparameterisation elsewhere looks
arbitrary, it should be in there; if it is in there, it is a property of a
bimanual Franka running a scripted zigzag under joystick supervision.

| | |
|---|---|
| **DRIM** (`policy.py`, `nets.py`, `selection.py`, `diagnose.py`, `dynamics.py`) | the staged B0/B1/D2 chain, the exact-null identity, the bounded corrector, the S/U message, offline selection, the pre-deployment checks. Knows about no robot. |
| **this rig** (`dressing.py`, and `ARM_FIELDS` in `fields.py`) | the packed-state layout, the 14.3 Hz horizons, the joystick command scale, the per-arm field schema, and the two adaptations below. |

`--profile dressing` (default) applies them; `--profile none` leaves the core
defaults.

### The two that change what is learned

**`action_mode = "delta_ee_pos"`.** The recorded `action` is an absolute
end-effector pose target that the controller tracks to within 8 mm, so its
absolute value is nearly the observed pose and predicting it is close to copying
an input: holding the previous action scores 0.00024 against a predict-zero of
0.184, beating a trained `B0` (~0.017) by ~90x. Predicting the delta puts the
copycat at 0.117 against 0.243. The command sent to the robot is unchanged — the
observed pose is added back at execution.

It is also what makes trajectory divergence measurable. With an absolute target
the commanded pose does not depend on where the arm is, so drift self-corrects
and compounding never shows; with a delta it accumulates. On a task that is easy
step-to-step and hard over a trajectory, that is the difference between
measuring something and measuring nothing.

**`exo_key = "zigzag_action"`.** A scripted zigzag velocity is injected during
teleop *and* inference and is never predicted. The arm's measured motion is
almost entirely that primitive — `corr(ee_twist, zigzag_action)` is
0.78 / 0.996 / 0.988 per axis against 0.006 / -0.038 / 0.026 for the operator's
delta — so a dynamics model not given it must infer its phase from state, and
whatever it cannot infer lands in the surprise channel that is the entire
content of D2's message. It is known at training and inference alike, so
conditioning on it is not privileged information. Held-out dynamics skill went
+68.5 % to +71.0 %.

## Running it

```bash
conda activate maniskill3          # this machine has no `robodiff` env
cd ~/diffusion_policy

# plumbing check: every stage, tiny budgets, output under data/outputs/
python -m diffusion_policy.drim.train --data <path> --quick

# a real run
python -m diffusion_policy.drim.train --data <path> --fast-frac 0.15 \
  --out data/outputs/irum_v1

pytest tests/test_drim.py          # 36 invariants, no data or GPU needed
```

### Real-robot inference

`real_robot.py` is the live adapter for a trained run.  It reads the same
arm1 state and direct RealSense streams as `direct_realsense_joystick_collector`
and sends the same Cartesian **velocity** command API.  Do not run the
collector at the same time: both processes claim the two RealSense devices and
would command arm1.

Start the usual infrastructure first (the NUC panda server remains the socket
endpoint for `FrankaRobotClient`):

```bash
roscore
# in the NUC session: start panda_server
cd ~/workspace/bimanual_ws/src/bimanual_dressing/scripts
./run_teleop_background.sh --direct-realsense
```

Then, in an environment sourced with `devel/setup.bash`, first make an
observation-only pass; it loads the checkpoint, checks all 27 proprioception
and six wrench channels, and prints the target/delta without sending motion:

```bash
cd ~/workspace/bimanual_ws/src/bimanual_dressing/src/diffusion_policy
python -m diffusion_policy.drim.real_robot \
  --run /home/dressing/DATASET/drim_runs/roi-custom43_vis-scratch_b-standard_s0
```

Only after that passes, add `--execute`.  It requires a terminal confirmation;
Ctrl-C or a robot-server error sends `stop` immediately.

The deployed run is `delta_ee_pos`: `DrimRunner.step` returns
`measured_ee_pos + predicted_delta`, which is the predicted **zigzag-free
centroid**, not an EE pose target.  The adapter's default
`--execution-mode centroid_servo` keeps its own centroid state, sends bounded
`gain * (predicted_centroid - maintained_centroid)` base velocity, integrates
that velocity into the maintained centroid, and then adds the separately
scripted zigzag velocity around it. This preserves the collector's
`centroid += joystick_velocity * dt` state structure without differentiating
the policy's noisy per-step centroid predictions.  `centroid_velocity` is kept
only as a diagnostic legacy mode; `position_p` directly tracks the measured EE
and is also an experiment, not the default.
The D2 transition history likewise receives this executed, integrated centroid
(and the complete preceding velocity command), exactly matching the recorded
`action` and `zigzag_action` fields.
The dataset field named `zigzag_action` is the complete Cartesian velocity
actually sent by the collector (centroid-motion velocity plus zigzag), despite
its name.  The adapter therefore gives DRIM the previously executed complete
command, while the zigzag term itself is never predicted.  `--no-zigzag` is an
explicit ablation; do not use it for the checkpoint above, whose dynamics was
trained with that conditioning input.

To verify the physical primitive before attributing a failure to the policy,
use `--zero-policy-output --execute`.  It holds the initial centroid for the
runner history, masks all policy motion, and sends **only** the 0.07 m/s,
0.05 m-amplitude position-reversing X zigzag.  It still prints `raw_target`,
the unmasked policy result.  On the collector data, subtracting centroid
velocity from the recorded full command recovers `-0.07` and `+0.07` m/s on X
in equal proportions; this is the reference this ablation reproduces.

`--policy-axis-mask 0 1 1` is the narrower version for this task: it gives the
zigzag primitive sole ownership of X while preserving learned centroid motion
on Y/Z.  A masked axis holds its initial centroid and contributes exactly zero
policy velocity, so it cannot quietly oppose the primitive through the P
tracker.

### Verify the physical Y direction first

Before using visual intuition to diagnose the policy, establish which physical
direction arm1's base-frame Y represents on this particular rig.  The separate
probe has no policy, camera, centroid, or zigzag component; it calls the same
`FrankaRobotClient.set_ee_velocity([vx, vy, vz])` API as collection and live
inference and permits only `[0, +Y, 0]` and `[0, -Y, 0]` commands:

```bash
python -m diffusion_policy.drim.axis_probe --execute
```

Keep the E-stop accessible.  The script starts at 2 mm/s (hard capped at
5 mm/s), has a 250 ms keyboard watchdog, and calls `stop` on space, `q`,
Ctrl-C, server error, or exit.  In its terminal, briefly hold `w` for `+Y`,
then `s` for `-Y`; it displays the measured `ee_pos` at the same time.  This
is a coordinate calibration only—do not run it together with the collector or
policy process.

### Where the output goes

Everything from a run lands in one directory — `--out`, or
`data/outputs/irum_<timestamp>` relative to the working directory if you do not
pass one. The last two lines of every run print the path.

| file | |
|---|---|
| `train.log` | the full console output, **including the warnings** — the field report, the control-rate report and the "no held-out episode" notice arrive through `warnings` and are captured here too. Appended and line-buffered, so a killed run keeps what it had |
| `summary_seed0.json` | everything structured: the resolved spec, which fields were usable and which were missing and why, the dt statistics, the residual demand and the derived `fast_frac`, and **per-epoch curves** for every stage |
| `B0_seed0.pt` … | the selected checkpoint per stage, with the epoch grid it was averaged from (`--no-keep-grid` drops the grid) |
| `dynamics_seed0.pt` | the delta dynamics feeding D2's message |

The per-epoch curve is in the JSON as well as in the checkpoint, so plotting a
run does not mean loading a 100+ MB `.pt`.

**Where the data path goes: `--data`.** Point it at a directory and every
`*.zarr` store inside is loaded, sorted by name, episodes concatenated:

```bash
--data /path/to/data                     # every .zarr inside
--data /path/to/a.zarr,/path/to/b.zarr   # an explicit list
```

Each store is resolved against the schema separately and only the fields **all
of them** agree on are kept, with a warning naming what any one store
contributed that the others could not. Two recordings that measured different
things cannot be concatenated into one state vector, and taking the union would
zero-fill the difference.

The other flags worth knowing:

| flag | default | |
|---|---|---|
| `--fast-frac` | `auto` | the fast level's authority as a fraction of full command; `auto` derives it from the correction the demonstrations require |
| `--layout` | `smoke` | packed-state layout. `none` expects one array per field |
| `--n-arms` | 2 | how many arms to look for; dead ones are dropped |
| `--require` | — | e.g. `wrench` — refuse a dataset without a contact signal |
| `--stages` | all | `dyn,B0,B1,D2`; a stage cannot start without its parent |
| `--val-ratio` | 0.2 | held-out **episodes**, stratified by store |
| `--estimator` | `last3` | `last3` / `last5` / `last1` / `bestval`; rollout-ranked estimators are refused |
| `--no-keep-grid` | off | drop the epoch grid from the checkpoint to save disk |
| `--epochs-{dyn,b0,b1,d2}` | 100/60/40/40 | fixed in advance, never tuned on the result |

`pytest tests/test_drim.py` runs the invariants — 42 of them, no data or GPU
needed.

## What changed in the port, and why

dap's benchmarks are two frozen tasks with their dimensions as module constants.
Everything that was a constant there is a field of `DrimSpec` here.

| dap | here | why |
|---|---|---|
| `OBS_DIM = 59`, `ACT_DIM = 7`, `WRENCH_DIM = 6` | `DrimSpec` fields | the rig publishes a 28-dim state and a 12-dim action |
| wrench always present | `wrench_dim` may be **0** | the real rig has no force channel at all (see below) |
| fixed 128-step episodes cut into 16 fixed chunks | sliding decision step, stride `exec_horizon` | demonstrations are whatever length the operator recorded |
| fixed 32-step **event** window anchored to a scripted event | **rolling** causal window, `msg_valid` where it fits | a teleop episode has no such anchor, and 32 > the whole episode |
| `pred16 / exec8`, 32-step message window | same, once the rig held a fixed 14.3 Hz | the earlier `pred8/exec4` + 8-step window was sized for a jittery ~6 Hz recording |
| rank the epoch grid by rollout success | **`last-k`**, which ranks nothing | there is no `EnvRunner`; see below |
| `MAX_RESIDUAL` derived from the env's declared reflex | `fast_limits`, no default | no number from cap or drawer transfers to this robot |

Unchanged and deliberately so: the flow objective and sampler, the corrector's
input set and chunk features, the bottlenecked message encoder, the gated
`dv`/reconciler route and both gates, the delta-dynamics network with its
modality heads and log-sigma clamp, `CLIP = 3.0`, and the staged
freeze/nest/handoff discipline. The vision front end is this repository's own
`MultiImageObsEncoder` + `get_resnet('resnet18')` — which is what dap imported
too, so it is shared code rather than two implementations that can drift.

## Checkpoint selection without a rollout

dap snapshots on an epoch grid, rolls **every** snapshot out on a fixed bank of
start states, ranks by success, and reports `best` / `last3` / `last5`.
Validation loss is logged and never selected on. That is not fussiness: on the
one round that kept a grid, moving from lowest-val-loss restoration to a last-3
average shifted arms by −0.044 to **+0.130** success and turned the round's
headline from −0.151 to −0.053 with a CI covering zero — a larger move than
nearly every effect that project reports.

The ranking stage cannot survive the port: the real rig is a physical robot. But
dap's estimators split cleanly in two:

* `best` / `top-k` — rank by rollout success → **unavailable**
* `last-k` / `all` — **rank nothing**, pure epoch position → **available**

**One substitution, stated because it is not like-for-like.** In dap, `last3` is
a *reporting* estimator — it averages the rollout **metrics** of the final three
snapshots, while the promoted checkpoint is chosen by rollout top-1. Here there
is no metric to average and no ranking to promote by, so `last-k` instead names
a **weight average** of those snapshots: one deployable checkpoint, built
without consulting any score. That is standard practice (SWA / model soups), and
it is sound here for reasons worth checking rather than assuming — the points are
consecutive grid epochs of one run under a cosine schedule so they sit in one
basin; normalisation is GroupNorm so there are no batch statistics to
invalidate; and the constant buffers that travel in the state dict
(`obs_mean`/`obs_std`, `fast.max_residual`) come back unchanged. Averaging
across seeds or stages is **not** sound, and `average_states` cannot do it — it
only ever sees one bank. `--estimator last1` is the conservative alternative.

Two quantities are logged every epoch and **neither is a selector**:

* `val_loss` — flow / action loss on held-out episodes;
* `action_mse` — offline MSE of the *executed prefix* against the demonstrated
  action, i.e. the closest offline stand-in for what the robot receives.

`action_mse` is the tempting selector and is not the default: dap measured
offline loss ordering these variants across a 3 % spread against a 0.44 spread
in success — it barely orders them, and a selector that does not order has
variance exceeding the effect it is choosing between.

### Why not just select on validation loss?

It is on offer — `--estimator bestval` — because the case against it is
quantitative rather than a matter of principle, and it is what the rest of this
repository does.

What it costs is **selection noise**. An argmin over a grid of noisy estimates
is biased low and, on a plateau, picks close to at random within it. dap measured
the epoch-to-epoch spread at 0.011–0.040 success with the grid maximum carrying
+0.023–0.041 of upward bias, and found last-3 the better estimator of rollout
success. With a validation set of two or three episodes, that noise is larger
here, not smaller. Weight averaging over the tail reduces it.

The honest summary is that the two disagree less than the argument suggests: **on
a plateau they agree to within the noise and `last-k` has lower variance; off a
plateau `last-k` is averaging a tail that should not be averaged, and validation
loss is exactly what tells you so.** So validation earns its keep by deciding
whether the tail is trustworthy, not by picking the epoch — `selection.tail_slope`
and `overfit_warning` do that, and a rising validation loss across the final grid
points is reported as "shorten the budget", not "switch estimator".

Every run logs what `bestval` would have chosen next to what was chosen, so the
disagreement is visible as data rather than assumed:

```
B0 last3 -> epochs [40, 50, 60]; lowest val_loss was epoch 30 (DIFFERS)
```

If they routinely differ by a lot, that is evidence the run has not plateaued —
look at the budget before the estimator.

What `last-k` assumes is that the run has plateaued. `selection.tail_slope`
computes the slope of the last five grid points so the assumption is checked
rather than hoped for; if the tail is still moving, **extend the budget** rather
than switching estimator.

### The validation set

Splitting is by **episode** and stratified by store. Chunks from one episode
overlap in both observations and actions, so a chunk-level split leaks; and
sessions differ in lighting, garment placement and operator, so an unstratified
draw can put a whole session in validation and turn "held out" into "a different
setup" — a harder question than the one being asked, and one that moves with the
seed. A store with a single episode contributes none, since holding it out would
remove that session from training entirely.

**How much to hold out.** Because nothing selects on it, a small validation set
does not carry the usual risk — it is a diagnostic, not a chooser. What matters
is the number of held-out **episodes**, not the ratio: 20 % of 10 episodes is 2,
which is thin but readable. Keep `--val-ratio 0.2` while the dataset is small
and let it fall toward 0.1 once 10 % is five or more episodes. The one thing the
validation set genuinely earns its keep for is catching a dynamics model that
does not generalise — `skill` below zero on held-out episodes is the signal that
`D2` is being fed noise.

## The data schema

`fields.py` declares, once, what a dressing episode is supposed to contain per
arm — the set the collector is being extended to record:

| field | dim | | field | dim |
|---|---|---|---|---|
| `q` joint position | 7 | | `gripper_pos` | 1 |
| `dq` joint velocity | 7 | | `gripper_vel` | 1 |
| `ee_pos` | 3 | | `tau_J` joint torque | 7 |
| `ee_quat` | 4 | | `wrench` | 6 |
| `ee_lin_vel` | 3 | | | |
| `ee_ang_vel` | 3 | | **total** | **42/arm** |

`wrench` is the only member of the `wrench` group: it is handed to the fast
corrector as its own per-step argument, because the slow context carries two
frames while the corrector needs a contact reading at every executed step. That
is dap's split, not a new one. Only one frame of it needs recording —
`wrench_ee_ee = Rᵀ · wrench_ee_base` and `R` comes from `ee_quat`, so the two
are inter-convertible.

`tau_J` is **not** a substitute for `wrench`. It is measured joint torque, with
gravity, inertia and friction included; the external wrench the rig publishes is
computed on-line from `tau_ext_hat_filtered` through a damped-least-squares
Jacobian inverse, and neither `tau_ext_hat_filtered` nor the Jacobian survives
into a recording. Both are worth having — a 6-D end-effector wrench cannot
represent cloth dragging on the forearm, and 7 joint torques can.

**A missing field is excluded from the state vector, never zero-filled.** A
constant channel trains without complaint, costs nothing visible, and removes
the signal a stage depends on. So the widths always equal what was actually
measured, `DrimSpec` records which fields it was built from, and
`assert_schema` refuses a checkpoint whose channels are not the ones its weights
saw — width equality is not enough, since dropping `gripper_pos`+`gripper_vel`
keeps every tensor shape valid while shifting every channel after them.

Two dataset layouts resolve:

* **one array per field** — `data/arm1_q`, `data/arm2_wrench`, … (aliases
  accepted: `joint_positions`, `effort`, `wrench_ee_base`, …). What the extended
  collector should write, since a missing field is then simply an absent array.
* **one packed `state` array** plus a declared `PackedLayout`. `SMOKE_LAYOUT`
  is `image_trials_4.zarr`: stride 14, `q`/`ee_pos`/`ee_quat`.

Everything absent is warned about once, by name, with what it was needed for.
`--require wrench` turns that warning into a refusal, which is how a run
declares "this stage is meaningless without a contact signal" instead of
discovering it in the numbers.

Two present-but-unusable cases are caught, because downstream they look
identical to a healthy field: a channel that is identically zero, and a
quaternion whose norm is not 1 (a disconnected arm is recorded as `zeros(14)`,
so its "quaternion" is not a rotation at all). An arm with **nothing** usable is
dropped whole rather than deleting its fields from the live arm.

## Horizons

Sized for the rig's measured **14.3 Hz** — the 0909 recordings hold that to
within 0.079 s, so a step is a fixed duration rather than a hope.

* `pred_horizon=16` / `exec_horizon=8` — cap's, and at this rate 1.1 s predicted
  and 0.56 s executed.
* `message_window=32` = 2.24 s, chosen to cover **one full zigzag cycle**: the
  commanded velocity flips sign every 14 steps in these recordings, so 28 steps
  plus margin is the shortest window in which the message sees a whole period
  rather than half of one. A half-cycle window would make the message's content
  depend on which half it happened to land in.

The earlier `pred8/exec4` with an 8-step window was sized for the smoke
recording's jittery ~6 Hz, where 16 steps would have spanned most of a
manoeuvre.

## What the dynamics models

The delta dynamics predicts **proprioception and the contact channels**, not
proprioception alone (`DrimSpec.dyn_fields`). Splitting `prop` from `wrench` is
right for the policy's inputs — the corrector needs a per-step contact reading of
its own — but wrong for the dynamics: cap's model predicts the wrench along with
everything else, and a surprise computed over a state that excludes force cannot
say *"it pushed back harder than expected"*, only *"the arm did not go where the
command implied"*. For dressing that is the difference between a contact signal
and a tracking error, and `S` is the whole content of D2's message.

The combined state is built by one `res.stack(prop + wrench)` call, never by
concatenating two stacks: `stack` interleaves arm-major, so the two orderings
differ as soon as there is more than one arm, and the modality table follows the
former.

## Variable control rate

The rig runs at ~6 Hz and the period varies 0.05–0.75 s — a factor of 15. A
state *change* is not comparable across steps under that, so `dt` is carried
through the loader and the delta dynamics is **conditioned on it**; without that
the residual it leaves behind, which is exactly the surprise channel, would be
dominated by how long the step happened to take rather than by what the arm ran
into. The loader warns when the jitter exceeds 3×.

Conditioning does not repair everything. `pred_horizon` and `message_window` are
counted in steps, so they mean different durations at different points in an
episode, and DRIM's slow/fast separation — a slow level replanning every
`exec_horizon` steps, a fast level correcting every step — is not separated in
time at this rate. That is a data-collection property, not something the model
can fix.

## The fast level's authority

`fast_limits` is a per-channel ceiling on the correction, in normalised action
units, and it has no default. dap's construction is

```
ceiling = (physical authority the reflex is allowed) / (physical units per unit action)
```

drawer: `8 mm / 0.05 = 0.16`. cap: `MAX_TRANS_INFLUENCE_M / D = 0.016`. The
recipe is portable; **the numerator is not** — both are numbers those
*environments published* about their own scripted reflex, and they differ by 10x
from each other.

Here the denominator is known exactly. The teleop commands Cartesian velocity at
`LIN_SCALE = 0.02` m/s and `ANG_SCALE = 0.05` rad/s at full stick deflection
(`single_joystick_teleop.py`), and `ChunkNormaliser` scales actions by those
rather than by the observed range — so one normalised unit *is* full command,
and a ceiling reads directly as a fraction of it. `--fast-frac 0.05` means "the
corrector may add up to 5 % of full command".

What remains is choosing the fraction, and dap's answer to exactly this question
was **not to choose**: `authority_lo5.FAST_ARMS` pre-registers five arms
(`V4`, `SCALED`, `MASKED`, `CONTRACT`, `SCALAR`) and the ablation picks. That
structure is available here — `fast_limits` is a per-channel vector, so an arm
is one vector — and it is the honest route given that neither 0.016 nor 0.16
transfers. A reasonable pre-registered set is ~0.05 / 0.15 / 0.30 of full
command, bracketing cap's 1.6 % and drawer's 16 %.

Channels the demonstrations never move get **0**, not the fraction.
`FastCorrector` emits `limits * tanh(...)`, so a zero ceiling makes a channel
identically silent for any parameter values rather than merely discouraged —
the same structural guarantee the message's exact-null route has. Without it the
corrector holds authority over axes no data ever constrained, and is free to
write there at rollout.

## Before it reaches the robot

`diagnose.py`, run at the end of training unless `--no-diagnose`. Two different
questions, deliberately not mixed.

### Does error compound? (`divergence`)

Single-step action MSE cannot answer that: it is measured at demonstration
states, and what breaks a behaviour-cloned policy is visiting its own. So the
policy is rolled forward **through the learned dynamics** — replanning every
`exec_horizon` steps as it would on the robot, with the fast level correcting
against the *rolled* proprioception — and the rolled state is compared with the
demonstration.

```
divergence step1    policy 0.702  replay 0.705  gap -0.004
divergence step8    policy 2.241  replay 2.200  gap +0.041
divergence step24   policy 3.433  replay 3.328  gap +0.105
compounding x4.9 over 24 steps
```

Two honesty requirements, enforced in code rather than left to the reader:

* **`replay` is the floor.** The same rollout driven by the *demonstrated*
  actions. Whatever it diverges by is the dynamics model's own error, not the
  policy's. Only the gap above it is evidence about the policy — without this
  row a bad model and a bad policy look identical.
* **Vision is teacher-forced.** The dynamics models low-dimensional state, not
  pixels, so each replan sees the demonstration's frames. Real drift would move
  the images too, so every number here is **optimistic**.

### Is anything broken? (`detectors`)

None of these predicts success. Each catches a specific way a dap round went
wrong, and the run prints them as verdicts:

```
OK   dynamics skill +67.5% vs no-change baseline
WARN corrector saturation 0.0%  <-- never reaches its ceiling; authority unused
OK   exact-null identity (max |D2(m=0) - B1| = 0.0e+00)
WARN surprise p99 shift x28.9  <-- D2 conditions on values it never saw in training
OK   state divergence grows x4.9 over 24 steps
```

`surprise_shift` is dap's most expensive lesson measured here: `S` is fitted on
demonstration transitions and read at states the policy visits. On drawer its
p99 went 2.66 → 7.07 and its clip rate up 213x. `message_reliance` is reported
as **reliance, not value** — zeroing an input the model trained with is out of
distribution, and dap measured 99 % reliance on a message worth *negative*
success.

### What offline metrics cannot do

Predict how well it will work. dap's own note: *"offline loss barely orders these
variants (a 3 % spread against a 0.44 spread in success)"*. The dynamic range of
the proxy is a fifteenth of the thing being predicted, and there is a documented
case where an offline probe pointed the opposite way. These are failure
detectors; ranking arms still needs rollouts.

## Open decisions — read before trusting a number

**1. The authority fraction is not chosen.** See above: the recipe is settled,
the number is not, and the intended route is an ablation over pre-registered
arms rather than a pick.

**2. The control rate is not the model's to fix.** dt-conditioning makes the
dynamics comparable across steps; it does not give the slow and fast levels a
separation in time that a jittery ~6 Hz recording does not have.

**3. `B0` grid snapshots are whole models.** Purely a disk note: `B1` and `D2`
store only what their frozen parent cannot supply, while `B0` — being the start
of the chain — has no parent to diff against. Nothing to fix; budget the space.

## Smoke-test result (2026-09-02)

`maniskill3` env, RTX 4090, one seed, short budgets, on
`image_trials_4.zarr`. Evidence that the plumbing is connected — not a result:

```
arm2 dropped (no usable field); state covers arm1
usable fields: q, ee_pos, ee_quat   (7 of 10 missing)
prop_dim=14 wrench_dim=0 act_dim=6; delta_dim=13
dt 0.053-0.754 s (~1.3-19 Hz)
action channels never nonzero: [0,2,3,4,5]   active: [1]
[authority] ch1 residual demand: vs_chunk_mean p95=0.100 | step_to_step p95=0.076
dyn   skill=+15.7%   B0 val=0.996   B1 val=0.510   D2 val=1.493
exact-null check: max|D2(m=0) - B1| = 0.000e+00  PASS
```

The dynamics beating the no-change baseline (`skill > 0`) is the one number
worth watching from the start: dap ran a whole round on a dynamics model 155 %
*worse* than predicting no change while its NLL looked healthy, and the message
it fed encoded model error rather than physical evidence. `train_dynamics`
reports it every run and shouts when it is negative.
