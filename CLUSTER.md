# Running DRIM on the Cornell unicorn cluster

Written locally, because Claude Code cannot run on the login node. Everything
that needs a decision is a flag here rather than something to work out there.

## One-time setup

```bash
ssh dl2342@unicorn-login-03.coecis.cornell.edu

# the repo
git clone <this repo> ~/diffusion_policy
cd ~/diffusion_policy && git checkout dohyeok/reactive_policy

# the env
~/.conda/envs/dp_bw/bin/python -c "import torch, zarr, numcodecs, torchvision, cv2; print(torch.__version__, numcodecs.__version__)"
```

Use **`dp_bw`**: py 3.10.19, torch 2.10.0+cu128, torchvision 0.25.0, zarr
2.18.3, numcodecs 0.13.1, numpy 1.26.4, scipy 1.15.2, cv2 4.12.0 — the same
stack the tests pass under locally.

`~/.conda/envs/robodiff` **cannot run this**, for a reason that is not a
preference:

* numcodecs 0.10.2. The zarr stores were written by numcodecs 0.13.1, whose
  Zstd codec records a `checksum` field in the compressor config. Older Zstd
  rejects it — `TypeError: __init__() got an unexpected keyword argument
  'checksum'` — before a single frame is read. `checksum` arrived in numcodecs
  0.13.0, which requires Python >= 3.10, and `robodiff` is Python 3.9. So no
  numcodecs installable in that env can read this data. It is not fixable by
  upgrading a package.
* torch 1.12.1, which predates `torch.load(..., weights_only=)`. That one *is*
  worked around — `selection.torch_load` asks the signature — so checkpoints
  load on any torch back to 1.12.

`dp_blackwell` (torch 2.5.1 / cu124) would also work. pytorch3d is present in
both but the DRIM import graph never loads it; it is only reached through
`model/common/rotation_transformer.py`, which nothing here imports.

The training entry point needs only torch, torchvision, zarr, numpy, opencv and
scipy — no Hydra, no simulator, and deliberately no way to import an
environment. `pytest` is in neither env; it lives in `~/.pylibs_test` so that
neither env is modified:

```bash
cd ~/diffusion_policy
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$HOME/.pylibs_test:$HOME/diffusion_policy \
  ~/.pylibs_test/bin/pytest tests/test_drim.py -q      # 82 passed
```

## The data

Already at `/share/bhattacharjee/dlee_data/drim/datasets/` — four zarr stores,
119 episodes, 64,501 chunks, 102.4 GB, verified against the local copy
array-by-array (shapes, chunk counts, byte totals, episode counts all match).
**The originals**, not a preprocessed copy, so the field of view is still a
flag rather than something baked in.

`/share` is NFS and shared with the whole group. Training re-reads the dataset
every epoch, so the job stages it to the node's local `/scratch` first. `rsync
-au` makes that a no-op when the node already has it, which is what makes a
repeated array on one node cheap.

## Submitting

```bash
cd ~/diffusion_policy
sbatch --array=0 --export=ALL,SMOKE=1 -t 4:00:00 drim_unicorn.sub   # plumbing
sbatch drim_unicorn.sub                          # both vision inits, 2 jobs
sbatch --export=ALL,BUDGET=long,SEED=1 drim_unicorn.sub
squeue -u $USER
tail -f slurm/drim_*.out
```

**The field of view is settled and no longer swept.** `custom43` — the
hand-drawn box grown to the encoder's aspect, resized to the 240x320 the
encoder takes — is the one to use. It stays a `--roi` flag, so changing it
later costs a resubmission and no new data, but the array does not spend GPUs
on it.

What is left is the vision initialisation:

| index | vision |
|---|---|
| 0 | random init |
| 1 | `IMAGENET1K_V1` |

`get_resnet` defaults to `weights=None`, so the encoder is otherwise trained
from scratch, and with 119 episodes the pretrained start is plausibly the
larger effect of the two.

`SMOKE=1` adds `--quick` (10/6/4/4 epochs) **and** narrows the data to one
store (`zigzag_bed_0910.zarr`, 5 episodes, 4.5 GB) rather than all 102 GB. The
data has already been verified array-by-array, so a smoke has nothing to learn
from the other three stores and every reason to finish in minutes — it is for
plumbing, and a plumbing check is only useful if it is cheap enough to repeat.
It names the store rather than the directory, because a node that has already
staged a full run keeps all four on `/scratch`.

Staging the full 102 GB measured **690 s (11.5 min) at ~145 MB/s**, with 3.5 T
free on `/scratch`. That is cheap enough that no downscaled copy is worth
making, and `rsync -au` makes a repeat on the same node free.

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
~/.conda/envs/dp_bw/bin/python -m diffusion_policy.drim.diagnose \
  --run /share/bhattacharjee/dlee_data/drim/runs/<tag> \
  --data /scratch/$USER/drim/datasets --device cuda:0
```

Use this before driving a checkpoint on the robot.
