# Running a trained DRIM chain on the rig

What the robot side has to get right, and the numbers behind each claim. Every
figure here was measured on `roi-custom43_vis-scratch_b-standard_s0` — 119
episodes, 94,611 steps, trained 2026-09-11.

```python
from diffusion_policy.drim.infer import DrimRunner, Observation

pol = DrimRunner.load("~/drim_runs/roi-custom43_vis-scratch_b-standard_s0",
                      device="cuda:0")
pol.reset()                       # once per episode, never mid-episode
while running:
    obs = Observation(q=..., dq=..., ee_pos=..., ee_quat=..., ee_twist=...,
                      wrench=..., images={...}, zigzag_action=..., dt=...)
    target = pol.step(obs)        # (3,) ABSOLUTE EE position target, metres
    send_pose_target(target)
```

## The four things that are easy to get wrong

### 1. The output is an absolute pose, because the runner already added it back

The network predicts `action - ee_pos` — the operator's contribution, a pose
*offset* of a few centimetres. `step()` adds the **measured** pose back and
returns an absolute target, which is what the controller takes.

This is not a detail. Feeding the bare delta to something expecting an absolute
pose is centimetres where half a metre belongs; the offline rollout did exactly
that and compounded to 1e19 by step 16 and NaN by step 32, while the
demonstrated-action floor beside it stayed flat at 1.0. If you ever bypass
`DrimRunner` and call the model directly, this is the step to reproduce.

The action is **3-dim, position only**. There is no orientation channel: the
recordings write `data/action` as `(T, 3)`. Orientation comes from wherever it
came from during teleop, not from this policy.

### 2. The zigzag is an input, every step, forever

`zigzag_action` (6-dim) is injected at teleop and at inference alike. The
dynamics is conditioned on it and **refuses to run without it** rather than
quietly guessing. It is never predicted — if you find yourself reading it out of
the policy, something is wired backwards.

### 3. The message needs 32 transitions of history

`S` (surprise) and `U` (log-sigma) come from the last `message_window = 32`
transitions through the learned dynamics. Until that window fills, the message
is **structurally null**, and by the exact-null identity
`v = v_base + gate*(dv(x,t,ctx,m) - dv(x,t,ctx,0))` that makes D2 *exactly* B1 —
the same weights producing the same actions, not a degraded fallback.

`pol.message_ready` tells you which regime you are in. At 14.3 Hz the window
fills after about 2.2 s. Verified on a recorded episode: `message_ready` first
true at step 32.

`reset()` clears it. Call it between episodes — transitions from the previous
trial are not evidence about this one.

### 4. The crop must be the one training used

`custom43`: `image_bed_front (32, 120, 318, 424)`, `image_bed_back
(180, 92, 280, 374)`, as **`(y0, x0, h, w)` — row first**, then resized to
240x320 with `INTER_AREA`. `DrimRunner` applies this itself; you pass full
frames.

Runs from 2026-09-11 and later record the box in `summary_seed0.json`. Older
ones are recovered from the `--roi` on the command line in `train.log`, and a
run where that is ambiguous is **refused** rather than defaulted.

Source frames must be 480x640. A different resolution is a wiring error, not
something to rescale the box for, and is rejected.

## The observation

| field | width | note |
|---|---|---|
| `q` | 7 | joint positions |
| `dq` | 7 | joint velocities |
| `ee_pos` | 3 | metres |
| `ee_quat` | 4 | same convention as `data/state[:, 10:14]` |
| `ee_twist` | 6 | linear then angular |
| `wrench` | 6 | estimated external wrench at the EE |
| `images` | 2 cameras | `image_bed_front`, `image_bed_back`, full 480x640 uint8 |
| `zigzag_action` | 6 | the primitive, as commanded |
| `dt` | — | seconds since the previous observation |

27 proprioception + 6 wrench. Widths are checked on every call against the
spec, so a mis-wired field stops the loop instead of being normalised into
plausible nonsense.

**`dt` matters.** The dynamics takes it as an input and the recordings ran at
14.3 Hz (mean 0.0700 s, max 0.188 s). Report stalls honestly rather than
passing a nominal constant; the model handles an irregular step correctly if
you tell it the truth.

**Not recorded, therefore not available:** `gripper_pos`, `gripper_vel`,
`tau_J`. The consequence is stated in the run's own field report: the fast
corrector reads position only, and D2's surprise is a *kinematic tracking
error*, not a contact signal. A cloth slip is invisible to this policy.

## Timing

Measured on an RTX 4090, torch 2.10+cu128, the full D2 chain:

| | mean | p95 | worst |
|---|---|---|---|
| replan step (every 8th) | 45.7 ms | 56.8 ms | 66.5 ms |
| steps between | 1.20 ms | 1.31 ms | — |

The control period at 14.3 Hz is **70 ms**, so a replan uses 95 % of it in the
worst case *on a 4090*. On a weaker robot-side GPU it will overrun.

The headroom is there if you need it: the chunk covers `pred_horizon = 16`
steps but only `exec_horizon = 8` are executed, so the next chunk can be
computed asynchronously during the seven cheap steps rather than in the eighth.
`DrimRunner` does not do this — it is a straight loop on purpose, so what it
does is obvious — but nothing in the design prevents it.

## What was actually measured about this checkpoint

| | |
|---|---|
| action MSE, held out | **0.00320** vs **0.02772** for repeat-previous |
| trajectory divergence at 32 steps | 5.81 mm policy vs 5.66 mm replay floor, gap **+0.15 mm** |
| dynamics skill | **+81.8 %** vs predicting no change |
| exact-null identity | `max abs(D2(m=0) - B1) = 0.0e+00` |
| surprise shift, trained-on to held-out | **x1.1** |
| lighting-only action shift | 0.23x the policy's own sampling spread |
| corrector saturation | **0.0 %** |

Read the divergence line carefully: almost all of it is *model* error, not
policy error. The replay floor — the demonstrated actions through the same
dynamics — is 5.66 mm, and the policy sits 0.15 mm above it. Vision is
teacher-forced in that measurement, so it is optimistic about the real loop.

**None of this is a success rate.** There is no `EnvRunner` for this task and no
rollout, so nothing here predicts whether the garment goes on. These numbers say
the machinery is not broken; the robot says whether it works.

## Known limits, before you drive it

- **The corrector never approaches its ceiling** (0.0 % saturation against
  `(0.25, 0.05, 0.10)` of full command = 15.0 / 1.5 / 1.5 mm). The ceiling was
  set for a reactive case the demonstrations cannot show — all 119 episodes are
  unperturbed successes — so it is deliberately looser than demonstrated demand.
  Whether it is *right* is not answerable from this data.
- **Lighting.** The bed-front camera faces a window; between-episode brightness
  spread is 6.7 against 2.5 within one episode. On the five-episode smoke set,
  brightness alone identified which episode a frame came from 51 % of the time
  against a 20 % chance rate — that separability was never re-measured at 119
  episodes, so treat it as a reason for care rather than a current figure. The
  training data mixes curtained and sunlit sessions; keep the curtains roughly
  as they were. What *was* measured on this checkpoint is the line above: a
  lighting-only change moves the action 0.23x as far as the policy's own
  sampling spread.
- **No contact sensing**, per the field report above.
- **`vis-scratch` beat `vis-IMAGENET1K_V1`** (0.00320 vs 0.00425), and the
  ImageNet run's B0 showed rising validation loss. Use the scratch checkpoint.
