# Efficient and Reliable Teleoperation through Real2Sim2Real Shared Autonomy

## Links

- **Code:** https://github.com/shuosha/Residual_Copilot
- **Paper:** https://arxiv.org/abs/2603.22787

---

## Table of Contents

- [Installation](#installation)
- [Quick Demo](#quick-demo)
- [Inference](#inference)
  - [Evaluating Pilot and Copilot Policies](#evaluating-pilot-and-copilot-policies)
  - [Visualizing Residual Corrections](#visualizing-residual-corrections)
  - [Real-world Deployment](#real-world-deployment)
- [Training](#training)
  - [Training Residual Copilot](#training-residual-copilot)
  - [Training Guided Diffusion Copilot](#training-guided-diffusion-copilot)
  - [Adding New Tasks and Models](#adding-new-tasks-and-models)
- [Three Blocks: SpaceMouse Teleop Workflow](#three-blocks-spacemouse-teleop-workflow)
- [RandomBlock: Randomized Block/Bin Task](#randomblock-randomized-blockbin-task)
- [HuggingFace Collection](#huggingface-collection)

## Installation

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 12.8 (matches IsaacSim 5.1.0 / PyTorch 2.7.0 cu128; for other versions see the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))

### Setup

```bash
git clone --recurse-submodules https://github.com/shuosha/Residual_Copilot.git
cd Residual_Copilot
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
UV_HTTP_TIMEOUT=300 uv sync   # Isaac Sim wheels are large; extended timeout prevents download failures
```

To verify that all environments are registered correctly:
```bash
python scripts/list_envs.py   # should list XArm-{GearMesh,NutThread,PegInsert}-{Residual,GuidedDiffusion}
```

> **First run:** Isaac Sim will prompt you to accept the EULA and will download/cache simulator assets, which may take several minutes.

> **Note:** Isaac Sim can exhaust static TLS slots, causing `cannot allocate memory in static TLS block`. Preload `libgomp` before running any script:
> ```bash
> export LD_PRELOAD=/lib/x86_64-linux-gnu/libgomp.so.1
> ```
> The provided shell scripts handle this automatically.

## Quick Demo

Generate side-by-side comparison videos of the **Residual Copilot** vs. an unassisted **Replay** baseline:

```bash
bash scripts/demo_side_by_side.sh NutThread              # launches with Isaac Sim viewer
bash scripts/demo_side_by_side.sh GearMesh --headless   # no viewer (for remote machines or no display)
bash scripts/demo_side_by_side.sh PegInsert --no-clean  # keep intermediate files
```

Output: `logs/demos/<task>/demo_*.mp4` — annotated videos with action overlays (red = base action, pink = residual, blue = net).

## Inference

### Evaluating Pilot and Copilot Policies

```bash
# Pilot + copilot with recording
python scripts/play.py \
  --task NutThread --pilot kNNPilot --copilot ResidualCopilot \
  --num_envs 16 --record

# Pilot only
python scripts/play.py --task NutThread --pilot kNNPilot --num_envs 16
```

**Arguments:** `--task` (`GearMesh` / `GearMeshIntent` / `PegInsert` / `NutThread` / `ThreeBlocks`), `--pilot` (see below), `--copilot` (optional), `--num_envs` (default 1), `--record` (save to `logs/rollouts/`), `--no_rand` (disable domain randomization), `--vis_obs` (show held/target visualization markers), `--checkpoint` (override the HF copilot download with a local path; applies to RL-Games copilots `ResidualBC` / `ResidualCopilot` as well as guided-diffusion copilots `GuidedDiffusionBC` / `GuidedDiffusionExpert` / `DiSCo`).

`GearMeshIntent` is a gear-mesh variant where each demo episode targets one of three gear slots (small/medium/large); `ThreeBlocks` is a multi-goal block-sorting task (pick up to 3 blocks into a bin) primarily driven by `SpaceMousePilot` for live shared-autonomy teleop.

**Pilots:**

| Name | Description |
|------|-------------|
| `kNNPilot` | kNN Pilot |
| `BCPilot` | Teleop BC policy |
| `ExpertPilot` | Expert BC policy |
| `NoisyPilot` | Expert BC + mistakeful behaviors (noisy actions) |
| `LaggyPilot` | Expert BC + laggy behaviors (repeat actions) |
| `ReplayPilot` | Replay recorded episodes |
| `SpaceMousePilot` | Live human teleop via a 3Dconnexion SpaceMouse (real HID device required) |

**Copilots:**

| Name | Description |
|------|-------------|
| `ResidualCopilot` | Residual RL trained with Residual Copilot |
| `ResidualBC` | Residual RL trained with BC  pilot |
| `GuidedDiffusionBC` | Guided Diffusion trained on teleop data |
| `GuidedDiffusionExpert` | Guided Diffusion trained on expert data |
| `DiSCo` | Guided-diffusion copilot with DiSCo sampling (seed/inpaint toward the pilot's reference action) + beta-blend; used for live shared-autonomy assistance, e.g. with `SpaceMousePilot` on `ThreeBlocks` |

All checkpoints are auto-downloaded from HuggingFace on first use. To evaluate a **local** RL-Games copilot checkpoint instead, pass `--checkpoint <path/to/FactoryXarm.pth>`:

```bash
python scripts/play.py \
  --task GearMeshIntent --pilot NoisyPilot --copilot ResidualCopilot \
  --checkpoint logs/rl_games/FactoryXarm/<run_name>/nn/FactoryXarm.pth \
  --num_envs 15 --no_rand --record
```

Combined with `--no_rand`, envs are run in deterministic order (one episode per env), so `--num_envs N` evaluates exactly `N` ordered episodes.

For guided-diffusion copilots (`GuidedDiffusionBC`/`GuidedDiffusionExpert`/`DiSCo`), `--checkpoint` points at a LeRobot `pretrained_model` directory instead of an RL-Games `.pth`:

```bash
# DiSCo copilot assisting a live SpaceMouse pilot, from a locally-trained checkpoint
python scripts/play.py \
  --task ThreeBlocks --pilot SpaceMousePilot --copilot DiSCo --num_envs 1 \
  --checkpoint outputs/train/threeblocks_expert_50_bc_expert/checkpoints/060000/pretrained_model
```

### Visualizing Residual Corrections

**Per-episode and collage videos** from recorded rollouts:

```bash
# <recording_dir> is created by `play.py --record` under logs/rollouts/,
# named eval_<task>_with_<copilot>_and_<pilot> (e.g. eval_GearMesh_with_ResidualCopilot_and_kNNPilot)

# Annotated single-episode videos
python scripts/vis/to_videos.py \
  logs/rollouts/<recording_dir> \
  --single --annotate

# Collage grid
python scripts/vis/to_videos.py \
  logs/rollouts/<recording_dir> \
  --collage --annotate --cols 4 --scale 0.5
```

When `--annotate` is enabled, action arrows are drawn per frame: **red** = base action, **pink** = residual, **blue** = net action.

### Real-world Deployment

See [Residual_Copilot_Deployment](https://github.com/shuosha/Residual_Copilot_Deployment) for real-world setup, hardware configuration, and deployment instructions.

## Training

### Training Residual Copilot

Train the RL residual copilot using PPO with a pilot model:

```bash
python scripts/train.py \
  --task XArm-GearMesh-Residual \
  --pilot kNNPilot \
  --num_envs 128 \
  --headless
```

**Arguments:** `--task` (full gym ID, e.g. `XArm-GearMesh-Residual`), `--pilot` (pilot model), `--num_envs` (default 128), `--checkpoint` (resume), `--distributed` (multi-GPU), `--track` (W&B logging), `--wandb_project_name` (W&B project name, defaults to task config name), `--wandb_name` (W&B experiment name, defaults to log directory). Logs saved to `logs/rl_games/`.

### Training Guided Diffusion Copilot

Diffusion policies are trained with [LeRobot](https://github.com/huggingface/lerobot). Training data can either come from successful rollout data of an existing policy or real-world data collected using the [deployment repo](https://github.com/shuosha/Residual_Copilot_Deployment).

**1. Collect data:**

```bash
python scripts/collect_data.py \
  --task GearMesh --pilot kNNPilot \
  --num_envs 16 --num_episodes 500 \
  --output_dir logs/data/gearmesh_knn_500 --headless
```

**2. Augment (optional):**

```bash
python scripts/augment_data.py \
  --in logs/data/gearmesh_train.npy --target-total 2000 \
  --pos-aug 0.02 --rot-aug-deg 5
```

**Visualize training data** with fingertip position heatmaps. You can point to either an `.npy` file or a data root directory:

```bash
# From .npy file
python scripts/vis/plot_data.py path/to/camera_image.jpg \
  --npy-path logs/data/gearmesh_train.npy \
  --out logs/vis/gearmesh_heatmap.png

# From data root directory
python scripts/vis/plot_data.py path/to/camera_image.jpg \
  --data-root logs/data/gearmesh_knn_500 \
  --out logs/vis/gearmesh_heatmap.png
```

**3. Train DiffusionPolicy:**

```bash
bash scripts/train_bc.sh <dataset_path> <job_name>
```

The `dataset_path` can be a HuggingFace dataset ID or a locally collected dataset. Ensure the observation and action spaces are consistent with the target task, as defined in [`source/pilot_models/config/state_dp_cfg.json`](source/pilot_models/config/state_dp_cfg.json).

**Available HF datasets:** Expert — `shashuo0104/0126_gearmesh_expert_2000`, `shashuo0104/0129_peginsert_expert_2000`, `shashuo0104/0129_nutthread_expert_2000`. Augmented Teleop — `shashuo0104/0121_gearmesh_teleop_aug_2000`, `shashuo0104/0121_peginsert_teleop_aug_2000`, `shashuo0104/0121_nutthread_teleop_aug_2000`.

### Adding New Tasks and Models

#### New Assembly Task

1. **Create USD assets** for the held and fixed objects (local paths or uploaded to HuggingFace)
2. **Define asset configs** in `assembly_tasks_cfg.py` — subclass `HeldAssetCfg` and `FixedAssetCfg`
3. **Define task config** as `AssemblyTask` subclass — set data paths, rewards, success threshold
4. **Create env config** in `xarm_env_cfg.py` — subclass `XArmEnvCfg`
5. **Register** in `__init__.py` with `gym.register()`
6. **Verify** with `python scripts/list_envs.py`

#### New Pilot Model

1. Implement in `source/pilot_models/` with `get_actions(episode_idx, pos, quat, grip)` (returns an 8D pos+quat+gripper base action) and `clear(env_ids)` methods for retrieval-style pilots (see `knn_pilot.py`), or `act(obs)` / `reset()` for learned policies (see `bc_pilot.py`)
2. Register in `_init_pilot()` in `xarm_env.py`
3. Add mapping in `source/utils/constants.py` (`PILOT_NAME_MAP`)

## Three Blocks: SpaceMouse Teleop Workflow

`ThreeBlocks` is a multi-goal block-sorting task (pick up to 3 blocks into a bin) driven live by a physical [SpaceMouse](https://3dconnexion.com/) via `SpaceMousePilot`, rather than pre-recorded teleop/expert `.npy` data. This is the end-to-end loop for collecting demos, converting them to the kNN dataset format, and training a residual copilot on top.

The bin is kinematic (`kinematic_enabled=True` in `ThreeBlocks.fixed_asset`) — a placement target that is never pushed by the arm or by dropped blocks.

**1. Drive the task by hand** (sanity-check the SpaceMouse connection and controls):

```bash
python scripts/play.py --task ThreeBlocks --pilot SpaceMousePilot --num_envs 1
```

**2. Record demos:**

```bash
python scripts/play.py --task ThreeBlocks --pilot SpaceMousePilot --num_envs 1 --record
```

**3. Append the recording to your dataset** (`convert_demos.py` reads `logs/rollouts/<dir>`, keeps only successful episodes per `meta/stats.json`, and writes/appends to a single `.npy`):

```bash
python scripts/convert_demos.py \
  --rollout_dir logs/rollouts/eval_ThreeBlocks_with_SpaceMousePilot \
  --output logs/data/threeblocks_seq_demos.npy --append
```

**4. Verify the output loads in kNN format:**

```bash
python -c "import numpy as np; print('episodes:', len(np.load('logs/data/threeblocks_seq_demos.npy', allow_pickle=True).item()))"
```

**5. Test the kNN pilot** against the newly appended demos:

```bash
python scripts/play.py --task ThreeBlocks --pilot kNNPilot --num_envs 1
```

**6. Train the residual copilot** (long-running — disable auto-suspend first if training on a laptop/workstation):

```bash
# disable auto-suspend (long run)
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'

# train
python scripts/train.py --task XArm-ThreeBlocks-Residual --pilot kNNPilot --num_envs 128 --headless
```

**7. Run the trained copilot live**, assisting the SpaceMouse pilot:

```bash
python scripts/play.py \
  --task ThreeBlocks --pilot SpaceMousePilot --copilot ResidualCopilot --num_envs 1 \
  --checkpoint logs/rl_games/FactoryXarm/none/nn/last_FactoryXarm_ep_400_rew_-109.26714.pth
```

**8. Alternative: run a DiSCo guided-diffusion copilot live**, from a BC checkpoint trained on the recorded demos (see [Training Guided Diffusion Copilot](#training-guided-diffusion-copilot) to train one against `threeblocks_seq_demos.npy`):

```bash
python scripts/play.py \
  --task ThreeBlocks --pilot SpaceMousePilot --copilot DiSCo --num_envs 1 \
  --checkpoint outputs/train/threeblocks_expert_50_bc_expert/checkpoints/060000/pretrained_model
```

## RandomBlock: Randomized Block/Bin Task

`RandomBlock` is the base task for the VLA shared-autonomy study. It reuses the
`ThreeBlocks` scene but changes two things:

- **Randomized block positions.** The three block xy positions are resampled
  uniformly over a rectangle every reset (per-env), so a policy cannot memorize
  one trajectory and has to localize the blocks from observation.
- **Single-target, language-conditioned pick.** Each episode names one block
  colour — `"pick up the {color} block and put it in the bin"` — and the other
  two are distractors. **Success = the instructed block is in the bin and no
  distractor is.** The instruction string is on
  `env.unwrapped.instructions[env_id]`, printed at reset for small runs, and
  saved into `meta/stats.json` by `--record`.

> **Colour balance is not guaranteed.** The target advances round-robin *within a
> process*, but `target_block_idx` is seeded randomly at env construction — so
> when you collect one episode per launch (the normal teleop workflow) the colour
> is effectively random each time. A 100-demo run came out 41/34/25. Check the
> balance partway through and keep going if one colour is lagging.

The **bin is not randomized**: it is pinned to a constant pose (`bin_pos`, default
`(0.54, 0.0, 0.0025)`) every episode. The bin geometry is kinematic (inherited
from `ThreeBlocks.fixed_asset`) so it never moves under gravity or contact.

Registered environments:

| Gym id | Action space | Use |
|--------|--------------|-----|
| `XArm-RandomBlock-Residual` | 7D residual | pilot + residual-RL copilot stack (kNN / SpaceMouse / BC) |
| `XArm-RandomBlock-GuidedDiffusion` | 8D absolute | absolute-action policies (guided diffusion; SmolVLA integration) |

**Randomization knobs** — `RandomBlock` in `source/xarm_assembly_env/assembly_tasks_cfg.py`:

| Field | Default | Meaning |
|-------|---------|---------|
| `randomize_positions` | `True` | master switch for block-position randomization |
| `block_x_range` | `(0.30, 0.44)` | x-range each block is sampled from |
| `block_y_range` | `(-0.15, 0.15)` | y-range each block is sampled from |
| `block_min_separation` | `0.06` | min centre-to-centre distance between blocks (rejection-resampled) |
| `bin_clearance` | `0.11` | min distance a block may spawn from the bin centre |
| `block_spawn_z` | `0.012` | block spawn height — the resting centre height of a 3 cm cube, so blocks spawn at rest rather than dropping |
| `bin_pos` | `(0.54, 0.0, 0.0025)` | fixed bin pose — never resampled; the bin is kinematic |
| `start_eef_xy` | `(0.36, 0.0)` | starting fingertip xy (over the block region) |
| `start_eef_z` | `0.18` | starting fingertip height above the table (arm starts fairly upright) |
| `single_target` | `True` | one instructed block per episode; the rest are distractors |
| `target_colors` | `("red", "green", "blue")` | colour name per block index (A/B/C) |
| `allowed_target_idx` | `(0, 1, 2)` | which colours episodes may target — restrict (e.g. `(0, 1)`) to hold a colour out for the challenge eval set |
| `instruction_template` | `"pick up the {color} block and put it in the bin"` | instruction string format |

Block sampling happens in `XArmEnv._sample_block_layout()` (uniform over the
rectangle, rejection-resampled for separation + bin clearance). To build the
"challenging task" held-out variation, subclass `RandomBlock` with disjoint (or
wider) ranges and register a new cfg the same way `XArmRandomBlockCfg` is
registered.

**1. Drive the task by hand** (visual sanity-check that the blocks land in new
positions each episode):

```bash
python scripts/play.py --task RandomBlock --pilot SpaceMousePilot --num_envs 1
```

> Block positions resample on every reset (and per-env), but the RNG is seeded.
> `play.py` now draws a fresh random seed each run and prints it — pass
> `--seed <int>` to reproduce a specific layout. With a fixed seed, episode 0 is
> identical every run; let the run continue past the first episode to see the
> positions change.

**2. Record demos.** `--record` enables the front camera and writes RGB frames +
robot state per timestep, plus the per-episode instruction into `meta/stats.json`.
The episode ends on success.

```bash
python scripts/play.py --task RandomBlock --pilot SpaceMousePilot --num_envs 1 --record
```

> **One episode per launch.** With `--num_envs 1` the recorder stops once that
> env's episode ends, so collecting N demos means N launches (steps 2+3 repeated).
>
> **Two instructions get printed — drive the first.** The second is printed by the
> auto-reset that fires as the episode ends, and belongs to an episode you never
> drive. The saved label is taken from the correct one either way.
>
> Set `SM_DEBUG=1` to re-enable the per-step SpaceMouse axis readout (off by
> default — at 15 Hz it prints ~1400 lines per episode and scrolls the instruction
> off screen).

**2b. Controlling *what* gets collected.** Because the target colour is drawn
independently every launch, the dataset's colour balance is left to chance and
drifts — the first 100 demos came out 46 green / 33 blue / 21 red, and eval
success tracked that ordering. Two flags make a batch deliberate:

```bash
# top up an under-represented colour
python scripts/play.py --task RandomBlock --pilot SpaceMousePilot --num_envs 1 --record \
  --target_color red

# same scene, different instruction: re-record one layout under each colour
python scripts/play.py --task RandomBlock --pilot SpaceMousePilot --num_envs 1 --record \
  --layout_seed 7 --target_color red
python scripts/play.py --task RandomBlock --pilot SpaceMousePilot --num_envs 1 --record \
  --layout_seed 7 --target_color green
python scripts/play.py --task RandomBlock --pilot SpaceMousePilot --num_envs 1 --record \
  --layout_seed 7 --target_color blue
```

`--layout_seed` pins the block layout for the whole run (the sampler is re-seeded
with it on every reset), so the three commands above differ *only* in the
instruction. That matters: with every layout seen exactly once, the layout alone
predicts the demonstrated trajectory and the instruction is redundant — the
policy can score well on training data without ever reading it. Repeating a
layout across colours removes that shortcut.

**3. Append the recording to a dataset:**

```bash
python scripts/convert_demos.py \
  --rollout_dir logs/rollouts/eval_RandomBlock_with_SpaceMousePilot \
  --output logs/data/randomblock_demos.npy --append
```

Answer `y` to the overwrite prompt on the next `--record` run — that demo is
already in the `.npy`, and the rollout dir is only a staging area. Re-running the
conversion on an unchanged rollout dir is detected by trajectory fingerprint and
skipped, so a stray repeat won't duplicate an episode.

This writes three things:

- `randomblock_demos.npy` — the kNN-format trajectories. Actions are clamped to
  the executable workspace, and the terminal timestep is dropped (on the step
  that reports `done` the env has already reset, so that sample belongs to the
  next episode).
- `randomblock_demos_frames/episode_NNNNN/` — the RGB frames, **copied** out of
  the rollout dir. This matters: `play.py --record` wipes and reuses the same
  rollout directory every run, so anything that merely pointed at it would go
  stale as soon as the next demo was recorded.
- `randomblock_demos.npy.tasks.json` — a sidecar holding each episode's `task`
  string, `frames` path, `steps` / `n_frames`, `block_xy`, `target_idx` and a
  trajectory `fingerprint` (re-running the conversion on an already-converted
  recording is skipped rather than duplicated). It is kept out of the `.npy`
  because the kNN loader tensorizes every key in an episode dict.

**4. Verify the collection.** Do this after the first ~3 demos (to confirm the
pipeline) and again before training. It checks the failure modes that are
otherwise silent:

```bash
python - <<'PY'
import numpy as np, json, collections, os
d = np.load('logs/data/randomblock_demos.npy', allow_pickle=True).item()
m = json.load(open('logs/data/randomblock_demos.npy.tasks.json'))
print(f"episodes {len(d)} / meta {len(m)}")
print("colours :", dict(collections.Counter(v['task'].split()[3] for v in m.values())))
bad = collections.defaultdict(list)
seen = {}
for k in sorted(d):
    e, n = m[str(k)], len(d[k]['obs.gripper'])
    if e.get('steps') != n or e.get('n_frames') != n: bad['count'].append(k)
    fd = e.get('frames')
    if not fd or not os.path.isdir(fd): bad['no_frames'].append(k)
    elif len([f for f in os.listdir(fd) if f.endswith(('.jpg', '.png'))]) != n: bad['count'].append(k)
    p = d[k]['obs.fingertip_pos'][-1]           # must NOT be the reset start pose
    if abs(p[0] - 0.36) < 1e-3 and abs(p[1]) < 1e-3 and abs(p[2] - 0.18) < 1e-3: bad['terminal'].append(k)
    if e.get('fingerprint') in seen: bad['dup'].append(k)
    seen[e.get('fingerprint')] = k
print({k: v for k, v in bad.items()} or "all checks pass")
ag = np.concatenate([d[k]['action.gripper'] for k in sorted(d)])
ap = np.concatenate([d[k]['action.fingertip_pos'] for k in sorted(d)])
print(f"action.grip [{ag.min()}, {ag.max()}] (want [0,1]);  pos z floor {ap[:,2].min():.3f} (want 0.000)")
PY
```

Also confirm each episode's `task` matches the colour you actually drove — that
is the one check no script can make for you, and it is the check that catches a
label desync.

**5. Build the SmolVLA dataset** (images + per-episode `task` string):

```bash
python scripts/npy_to_lerobot.py \
  --input logs/data/randomblock_demos.npy \
  --repo_id local/randomblock_vla \
  --root logs/lerobot/randomblock_vla \
  --images --overwrite
```

`--images` reads the sidecar, attaches `observation.images.front` from the copied
frames, and uses the per-episode instruction as the `task` (it also drops
`observation.environment_state` — the VLA grounds spatial relations from pixels +
language). Drop `--images` (and pass `--task "..."`) for a state-only
DiffusionPolicy dataset.

- `--root <dir>` keeps the dataset local; omit it to use the HF cache, add
  `--push_to_hub` (with a real `<hf_user>/…` repo id) to upload.
- **`--overwrite`** deletes `--root` first. `LeRobotDataset.create()` refuses to
  write into an existing directory, so without it any failed run blocks every
  retry with `FileExistsError`.
- `--image_dtype video` (default) needs ffmpeg; use `--image_dtype image` to
  store PNGs instead — larger on disk, trains the same.

Sizing guidance from the SmolVLA paper: ~50 demos per variation was enough and 25
was not. **~100 total** across the three colours is a reasonable target; collect
more if eval comes back weak on a particular colour.

**6. Finetune SmolVLA** on that dataset (LeRobot's built-in trainer, no custom script):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/randomblock_vla \
  --dataset.root=logs/lerobot/randomblock_vla \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/rb_smolvla --job_name=rb_smolvla \
  --policy.device=cuda
```

- **`--rename_map`** is required. `smolvla_base` declares three cameras
  (`camera1/2/3`); our dataset has one (`front`). Validation passes when the
  dataset's visual keys are a *subset* of the policy's, so mapping `front →
  camera1` is enough — SmolVLA then trains on that single camera and ignores the
  other two (missing keys are only zero-padded up to `config.empty_cameras`,
  which is `0`). Without it: `Feature mismatch between dataset/environment and
  policy config`.
  → The finetuned checkpoint therefore expects **`observation.images.camera1`** at
  inference. `eval_smolvla.py` reads the key off the checkpoint config
  automatically, so this needs no extra flag at eval time.
- **`--dataset.video_backend=pyav`** avoids `torchcodec`, whose default decoder
  links FFmpeg's shared libraries (`libavutil.so.*`) — those are usually absent
  even when the `ffmpeg` *binary* is present, so dataset encoding succeeds and
  then training dies at the first batch with
  `RuntimeError: Could not load libtorchcodec`. `pyav` ships its own FFmpeg.
  (The other way out is rebuilding the dataset with `--image_dtype image`, which
  skips video decoding entirely at the cost of disk.)
- **`--policy.push_to_hub=false`** is required for local training — it defaults to
  true, and the config validator then rejects the run with
  `'policy.repo_id' argument missing`. To push instead, pass
  `--policy.repo_id=<hf_user>/rb_smolvla`.
- Drop `--dataset.root` if the dataset was pushed to the Hub.
- Lower `--batch_size` (e.g. 32) if it OOMs.
- SmolVLA's SmolVLM processor needs a package that isn't pulled in by default:
  `uv pip install num2words` (otherwise it fails with
  `ImportError: Package num2words is required to run SmolVLM processor`).

~20k steps ≈ 4 h on one A100; the RTX 5090 is comparable. Checkpoints land in
`outputs/train/rb_smolvla/checkpoints/`.

**7. Evaluate** — N fresh episodes, no human, SmolVLA driving:

```bash
python scripts/eval_smolvla.py \
  --checkpoint outputs/train/rb_smolvla/checkpoints/last/pretrained_model \
  --num_episodes 30
```

Feeds the front-camera RGB + 14D state + instruction to SmolVLA each step, runs
its 8D absolute action through `XArm-RandomBlock-GuidedDiffusion`, and prints the
overall + per-colour success rate. That number is the step-1 result.

---

*Not the VLA path:* `python scripts/train.py --task XArm-RandomBlock-Residual --pilot kNNPilot --num_envs 128 --headless`
trains the original state-based residual-RL copilot. Unrelated to the SmolVLA
study — kept only for the residual-copilot framework.

## HuggingFace Collection

All data, models, and assets are hosted as a [HuggingFace collection](https://huggingface.co/collections/shashuo0104/residual-copilot), auto-downloaded on first use.

| Repo | Contents |
|------|----------|
| [`residual_copilot_assets`](https://huggingface.co/datasets/shashuo0104/residual_copilot_assets) | Robot USD/URDF, object meshes, camera params |
| [`residual_copilot_models`](https://huggingface.co/shashuo0104/residual_copilot_models) | BC pilots, RL copilot checkpoints, DP baselines |
| [`residual_copilot_data`](https://huggingface.co/datasets/shashuo0104/residual_copilot_data) | Teleoperation trajectories |

See each HuggingFace repo for the detailed file structure.
