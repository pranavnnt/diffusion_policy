# Running DRIM on the Cornell unicorn cluster

Written locally, because Claude Code cannot run on the login node. Everything
that needs a decision is a flag here rather than something to work out there.

## One-time setup

```bash
ssh dl2342@unicorn-login-03.coecis.cornell.edu

# the repo
git clone <this repo> ~/diffusion_policy
cd ~/diffusion_policy && git checkout dohyeok/reactive_policy

# the env already exists: ~/.conda/envs/robodiff
~/.conda/envs/robodiff/bin/python -c "import torch, zarr, torchvision; print(torch.__version__)"
```

If `robodiff` is missing anything, the training entry point needs only torch,
torchvision, zarr, numpy and opencv — no Hydra, no simulator.

## The data

Already at `/share/bhattacharjee/dlee_data/drim/datasets/` — four zarr stores,
119 episodes, 94,611 steps, ~96 GB. **The originals**, not a preprocessed copy,
so the field of view is still a flag rather than something baked in.

`/share` is NFS and shared with the whole group. Training re-reads the dataset
every epoch, so the job stages it to the node's local `/scratch` first. `rsync
-au` makes that a no-op when the node already has it, which is what makes a
repeated array on one node cheap.

## Submitting

```bash
cd ~/diffusion_policy
sbatch drim_unicorn.sub                          # the ROI x vision sweep, 10 jobs
sbatch --array=0 drim_unicorn.sub                # just custom43, from scratch
sbatch --export=ALL,BUDGET=long,SEED=1 drim_unicorn.sub
squeue -u $USER
tail -f slurm/drim_*.out
```

The array is five fields of view against two vision initialisations:

| index | ROI | vision |
|---|---|---|
| 0–4 | custom43, custom, mid, wide, hand | random init |
| 5–9 | the same five | `IMAGENET1K_V1` |

`custom43` is the hand-drawn box at the encoder's 3:4 aspect and the default.
The vision arm matters: `get_resnet` defaults to `weights=None`, so the encoder
is otherwise trained from scratch, and with this much data the pretrained start
may be the larger effect of the two.

Results land in `/share/bhattacharjee/dlee_data/drim/runs/<tag>/`, one directory
per arm, each with `train.log`, `summary_seed0.json` (spec, per-epoch curves,
selection, diagnostics) and the selected checkpoints.

## What to read in the output

Every run ends with a verdict block. It does not say how well the policy will
do — no offline metric here does — it says whether anything is broken:

```
OK   dynamics skill +61.9% vs no-change baseline
OK   corrector saturation 0.1%
OK   exact-null identity (max |D2(m=0) - B1| = 0.0e+00)
FAIL best action_mse 0.13586 vs copycat 0.08373
OK   lighting-only action shift is 0.15x the policy's own sampling spread
```

The `FAIL` line is the one to watch first: holding the previous action is the
bar a learned policy has to clear to be doing anything at all, and on the small
early recordings nothing cleared it. If it still fails with the full dataset,
nothing downstream of it means anything.

`exact-null` failing would mean the gated route has been broken and every
`D2 - B1` number is meaningless. `dynamics skill` below zero means the message
is carrying model error rather than physical evidence.

## Re-checking a checkpoint without retraining

```bash
~/.conda/envs/robodiff/bin/python -m diffusion_policy.drim.diagnose \
  --run /share/bhattacharjee/dlee_data/drim/runs/<tag> \
  --data /scratch/$USER/drim/datasets --device cuda:0
```

Use this before driving a checkpoint on the robot.
