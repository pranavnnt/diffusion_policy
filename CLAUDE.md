# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch

**All work happens on `dohyeok/reactive_policy`. Never `main`, never a new branch.**
`~/bimanual_dressing` pins this repo as a submodule at this branch's HEAD, so it
is the integration point. Check `git rev-parse --abbrev-ref HEAD` before editing.

## What this repo is

A fork of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (Chi et
al.) adapted for **robot-assisted dressing**. The upstream benchmark code
(PushT, robomimic, blockpush, kitchen) is still present and documented in
`README.md`; everything dressing-specific is this branch's own.

**There is no `EnvRunner` for any dressing task.** Training is offline behaviour
cloning from zarr replay buffers; `env_runner` instantiation is guarded by
`hasattr(cfg.task, "env_runner")`, and closed-loop evaluation happens outside
this repo (RCareWorld for sim, ROS for the real rig). Everything downstream
follows from that — see "Checkpoint selection" below.

## Two bodies of code — do not confuse them

### `diffusion_policy/irum/` — the active work

A port of the `IMG_B0` / `IMG_B1` / `IMG_D2` chain from `~/dap`
(`empriselab/reactive-policy`), **written fresh for the real-world track**.
`diffusion_policy/irum/README.md` is the authoritative document: what the three
stages are, what changed in the port, and the open decisions.

Staged, each freezing the last, plus a `dyn` stage that trains the delta
dynamics whose residual becomes the message:

| stage | adds | trains |
|---|---|---|
| `B0` | slow policy alone: vision + proprioception → action chunk, rectified flow over `ConditionalUnet1D` | vision + flow field |
| `B1` | bounded per-channel fast corrector on a frozen `B0` | corrector only |
| `D2` | upward message on a frozen `B1`; reads dynamics channels `S` (surprise) and `U` (log-sigma) and nothing else | message encoder, `dv`, reconciler, 2 gates |

The `D2 − B1` claim rests entirely on the exact-null identity
`v = v_base + gate*(dv(x,t,ctx,m) − dv(x,t,ctx,0))`: at `m = 0` the bracket is
identically zero for any parameter values, so `D2` with a null message **is**
`B1`. Never refactor that into an ordinary residual. `tests/test_irum.py` pins
it and three other invariants — run `pytest tests/test_irum.py`.

Every dimension is a field of `IrumSpec` (`irum/spec.py`), never a module
constant. `spec.limits()` deliberately raises rather than inventing a fast-level
authority.

```bash
conda activate maniskill3 && cd ~/diffusion_policy
python -m diffusion_policy.irum.train --data <path> --quick --fast-frac 0.15
```

`--data` takes a `.zarr` store, a directory of them, or a comma-separated list.
`--fast-frac` is the fast level's authority as a fraction of full command and has
no default; see the README's "fast level's authority" section.

### Everything else dressing-related — pre-existing, treat as frozen

`dataset/dressing_sim_dataset.py`, `dataset/dressing_real_dataset.py`,
`dataset/head_dressing.py`, the `diffusion_policy/dressing/` transform package
(`sim2real_transforms.py`, `real_transforms.py`, `head_transforms.py`,
`keys.py`), the task configs below, and the `.sub` scripts are **earlier work
not written in this line of development**. They still run, and their config
entrypoints are:

| config | task | obs / action |
|---|---|---|
| `train_diffusion_unet_lowdim_workspace` | `dressing_sim` | 11 / 2 |
| `train_diffusion_unet_real_lowdim_workspace_left` | `left_arm` | 17 / 2 |
| `train_diffusion_unet_real_lowdim_workspace_right` | `right_arm` | 17 / 2 |
| `train_diffusion_unet_real_lowdim_workspace_head` | `head` | 23 / 6 |

Do not extend or refactor these as part of IRUM work, and do not assume their
17-dim sim/real-aligned observation space applies to the IRUM track — it does
not. The sim line is developed separately; this checkout's purpose is the real
track.

## Data

The **only dataset on this machine** is
`~/dressing_policies/demo_for_testing/image_trials_4.zarr` — the real-world
smoke dataset, and the target the IRUM track runs against. Schema:

| array | shape | meaning |
|---|---|---|
| `data/state` | `(T, 28)` | per arm × 2: 7 joint positions, 3 eef position, 4 eef quaternion |
| `data/action` | `(T, 12)` | per arm × 2: 3 linear + 3 angular Cartesian velocity command |
| `data/image_arm1` | `(T, 240, 320, 3)` uint8 | wrist/arm camera |
| `data/image_bed_front` | `(T, 240, 320, 3)` uint8 | bed-front camera |
| `data/timestamp`, `data/image_*_timestamp` | `(T,)` | ~6 Hz, irregular (dt 0.05–0.75 s) |

Recorded by `~/dressing_policies/data_collection/image_joystick_collector.py`
over ROS from two Franka arms (`FrankaRobotClient`, `robot_name` `arm1`/`arm2`).

**What the rig publishes but the zarr does not record** (`bimanual_dressing`
`control/server.py`, read back through `control/client.py`): joint velocity
`dq`, measured joint torque `tau_J`, EE twist `ee_lin_vel`/`ee_ang_vel`,
estimated external wrench `wrench_ee_ee` / `wrench_ee_base` (6-D, derived from
`tau_ext_hat_filtered` through a damped-least-squares Jacobian inverse — there
is no F/T sensor), `dls_manipulability`, elbow state, gripper position. Adding
any of these is a change to `_arm_state` in the collector, not a hardware
change.

Every other zarr path in the task configs points at `/home/dressing/...` (the
robot machine) or `/scratch/pnt8/...` (the cluster) and is not present here.

## Checkpoint selection

There is no rollout, so **success-ranked selection is unavailable** and
`irum/selection.py` refuses it rather than falling back silently. The protocol
is dap's with the ranking stage deleted: snapshot on an epoch grid, report the
parameter average of the last K grid points (`last-k` ranks nothing, so it
survives). `val_loss` and `action_mse` are logged every epoch and **neither is a
selector** — dap measured that swapping val-loss restoration for a last-3
average moved arms by −0.044 to +0.130 success, larger than most effects that
project reports. `selection.tail_slope` checks the plateau assumption `last-k`
depends on.

The pre-existing configs still use `checkpoint.topk.monitor_key: val_loss`;
that is the older convention, not the one the IRUM track follows.

## Environment

There is **no `robodiff` conda env on this machine** despite what
`conda_environment.yaml` and the `.sub` scripts assume. `maniskill3` has torch
2.10 + zarr + torchvision and is what the IRUM smoke test runs under. It does
not have this repo installed, so prefix the path:

Run from the repo root, where `python -m` puts the local package first on
`sys.path` — no `PYTHONPATH` needed. Avoid `conda run -n maniskill3`, which has
been observed to kill long jobs (exit 137) where the same command run after
`conda activate` succeeds.

`HYDRA_FULL_ERROR=1` when debugging config/instantiation failures. Outputs go to
`data/outputs/<name>_<task_name>.<hydra.job.num>_<timestamp>/`; wandb project
for the fork is `first_arm_dressing`.

## Architecture (upstream parts, unchanged)

Read the "Codebase Tutorial" section of `README.md` for the Task/Method split,
the obs/action dict interface, the `To|Ta|T` horizon terminology, and
`ReplayBuffer`/`SequenceSampler`.

For a Hydra run: `train.py` → `hydra.utils.get_class(cfg._target_)` → a
`Workspace` in `diffusion_policy/workspace/`, which instantiates
`cfg.task.dataset` and `cfg.policy` and owns the whole loop in `run()`.
`BaseWorkspace.save_checkpoint` snapshots every attribute that is an
nn.Module/optimizer, so training state must live as workspace attributes.
`training.resume=True` auto-loads `checkpoints/latest.ckpt` from the run dir —
re-running with the same `name` silently resumes instead of starting fresh.

`${eval:'...'}` runs arbitrary Python in configs (resolver registered in
`train.py`); used for `pad_before`/`pad_after`.

The IRUM track does **not** go through Hydra or `BaseWorkspace`; it is a
standalone `python -m diffusion_policy.irum.train` with its own argparse, which
is deliberate — no environment can be imported from it, which is what makes "no
on-policy data was used" structurally checkable.

## Related repositories

Four sibling checkouts under `/home/dlee`:

- `~/dressing_policies` (empriselab/dressing_policies) — data collection
  (RCareWorld sim, ROS real teleop) and `preprocessing/`. Holds the smoke zarr.
- `~/bimanual_dressing` (empriselab/bimanual_dressing) — ROS catkin package for
  the dual-Franka rig; `control/server.py` is the authority on what the robot
  publishes. Pins this repo and `dressing_policies` as submodules at `src/`.
  Develop in the standalone checkouts, never in `src/`.
- `~/dap` (empriselab/reactive-policy) — the ManiSkill/robosuite perturbation
  line the IRUM algorithms come from. Reference for method details
  (`cap_constraint_benchmark/liftoff_v6_image/`, `benchmarks/common/selection.py`,
  `results/CHECKPOINT_SELECTION_NOTES.md`); its tasks and configs do not carry
  over. Has its own `CLAUDE.md`.
