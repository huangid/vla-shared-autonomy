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

> Measured results for every training round — dataset sizes, configs, success rates,
> failure modes and what they imply — are kept in **[docs/RESULTS.md](docs/RESULTS.md)**.
> This section is the how-to; that file is the record.

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

> **`OSError: open failed` from `spacemouse_pilot.py`** means the HID device could
> not be opened. In order of likelihood: the SpaceMouse is unplugged (`lsusb -d 256f:`
> should list it — the pilot opens vendor `0x256f`, product `0xc635`); a crashed
> earlier run still holds it, since the device is exclusive-open (`pgrep -af play.py`,
> then kill it); or the hidraw node is root-only (`ls -l /dev/hidraw*` should show
> `crw-rw-rw-` for the SpaceMouse — `/etc/udev/rules.d/99-spacemouse.rules` grants
> this via `SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"`).

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

**2c. Batch collection.** Run this **once** — it performs all 150
record-and-convert cycles itself (50 layouts x 3 colours), taking the dataset from
100 to **250 demos**. Your only job is to drive each episode with the SpaceMouse as
it appears; when the episode ends the process exits and the next one launches
automatically.

```bash
cd ~/vla-shared-autonomy
scripts/collect_randomblock.sh 1 50
```

What the script handles, so a long session can't silently go wrong:

- **Failed demos are re-recorded.** A missed grasp times out after 60 s and is
  dropped by `convert_demos.py`. The script checks the dataset actually grew and,
  if not, records the same layout/colour again — so no layout ends up missing a
  colour. After 3 misses in a row it asks: retry, skip, or quit.
- **Ctrl-C stops cleanly** after the current step and prints the exact resume
  command, e.g. `scripts/collect_randomblock.sh 12 50 blue`. It never closes your
  terminal (a pasted `for` loop with `exit` in it does).
- **The dataset save cannot be interrupted.** `convert_demos.py` writes the `.npy`
  and its `.tasks.json` sidecar to temp files and swaps both in with Ctrl-C ignored.
  A half-written save would truncate the dataset or leave the sidecar out of step
  with it, which silently mislabels every later episode's instruction.
- **Pre-flight and backup.** It refuses to start if the SpaceMouse is unplugged or
  another `play.py` holds it, and copies the `.npy` + sidecar into
  `logs/data/backups/<timestamp>/` before recording anything.
- **It doesn't depend on `play.py`'s exit code**, only on whether the demo was saved.

Budget ~4.5–5 h for 150 demos: Isaac restarts on every episode (~40–60 s), which
is unavoidable with one-episode-per-launch. To split it across sittings:

| Sitting | Command | New demos |
|---|---|---|
| 1 | `scripts/collect_randomblock.sh 1 17` | 51 |
| 2 | `scripts/collect_randomblock.sh 18 34` | 51 |
| 3 | `scripts/collect_randomblock.sh 35 50` | 48 |

Each sitting appends to the same dataset. Do not reuse a range: the same
`layout_seed` reproduces the same scene. If you stop early, use the resume command
the script prints. With the first 100 demos at 46 green / 33 blue / 21 red, the full
batch ends at 96 / 83 / 71.

> **This does not narrow where blocks spawn.** Within one layout the three
> episodes target three *different* blocks, so 50 layouts x 3 colours yields 150
> distinct reach targets — as many as 150 fully random demos. Binned 5x5 over the
> spawn rectangle, seeds 1–50 leave no empty cell (x 0.300–0.438, y -0.148–0.149).
> What repeats is the *scene*, which is the point: the same image now maps to three
> different trajectories, so only the instruction can resolve them.

Check progress at any time from another terminal:

```bash
python3 -c "
import json,collections
m=json.load(open('logs/data/randomblock_demos.npy.tasks.json'))
c=collections.Counter(v['task'].split()[3] for v in m.values())
lay=set(tuple(round(x,3) for b in v['block_xy'] for x in b) for v in m.values())
print('episodes:',len(m),' colours:',dict(c),' unique layouts:',len(lay))
"
```

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
  --repo_id local/randomblock_vla_250 \
  --root logs/lerobot/randomblock_vla_250 \
  --images --overwrite
```

Give each dataset build its own `--repo_id` / `--root` (here `_250` for the 250-demo
set) rather than overwriting the previous one, so older checkpoints stay reproducible.
Sanity-check the result: `logs/lerobot/randomblock_vla_250/meta/info.json` should
report `total_episodes: 250`.

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

Sizing: the SmolVLA paper found ~50 demos per variation enough and 25 not. On this
task **100 demos was not enough** — with blocks, distractors and target all
randomized, the model scored 0/20. The 250-demo set (100 random layouts + 50
layouts x 3 colours from step 2c) reached 11/20. See the results table in step 7.

**6. Finetune SmolVLA** on that dataset (LeRobot's built-in trainer, no custom script):

```bash
# keep the machine awake for a multi-hour run
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/randomblock_vla_250 \
  --dataset.root=logs/lerobot/randomblock_vla_250 \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.6,1.4]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.04,0.04]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}}}' \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=64 --steps=30000 --save_freq=2500 \
  --output_dir=outputs/train/rb_smolvla_v4 --job_name=rb_smolvla_v4 \
  --policy.device=cuda 2>&1 | tee /tmp/train_v4.log
```

This is the exact configuration that produced `rb_smolvla_v4` (and, apart from the
dataset and step count, `rb_smolvla_v3`).

- **`--dataset.image_transforms.*`** turns on photometric augmentation (brightness,
  contrast, saturation, small hue shift, sharpness). LeRobot's default transform set
  also includes a random affine; it is deliberately left out because shifting the
  image moves the blocks relative to the arm, which corrupts the very positions the
  policy must reach. The CLI only accepts `tfs` as a whole JSON dict —
  `--dataset.image_transforms.tfs.affine.weight=0` is rejected.
- **`--steps=30000`** scales with data: v3's best checkpoint was ~67 epochs over 100
  demos, and the same epoch count over 250 demos (24k frames) is ~25k steps.
- **`--save_freq=2500`** keeps intermediate checkpoints. The last one is not
  necessarily the best (v3 peaked at 10k of 20k), so step 7 sweeps several.
- **`--output_dir` must not exist** unless resuming; `lerobot-train` refuses to write
  into an existing run.
- Each checkpoint is ~1.1 GB; 12 checkpoints need ~13 GB free.

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

Measured speed on this workstation: ~0.65 s/step, so 20k steps ≈ 3.5 h and 30k ≈
5.5 h. The log line shows where the time goes — `data_s` (video decode) exceeds
`updt_s` (GPU), so the GPU is mostly waiting. Adding `--num_workers=12` (default 4;
the machine has 20 cores) should shorten it without changing the result.
Checkpoints land in `outputs/train/rb_smolvla_v4/checkpoints/`.

**7. Evaluate** — N fresh episodes, no human, SmolVLA driving:

```bash
python scripts/eval_smolvla.py \
  --checkpoint outputs/train/rb_smolvla_v4/checkpoints/025000/pretrained_model \
  --num_episodes 30 --n_action_steps 5 --max_steps 150 --debug_grounding
```

Feeds the front-camera RGB + 14D state + instruction to SmolVLA each step, runs
its 8D absolute action through `XArm-RandomBlock-GuidedDiffusion`, and prints a
summary: overall and per-colour success, failure modes, closest fingertip approach,
and a target-vs-aimed confusion matrix. Layouts are fresh random ones, not the
training scenes. Opens the sim window; add `--headless` to run without it.

- **`--n_action_steps 5`** matters most. SmolVLA predicts 50 steps per call and by
  default executes all of them before looking again — 3.3 s blind, more than half an
  episode, so it cannot correct a grasp that is drifting off. Re-planning every 5
  steps needs no retraining.
- **`--max_steps 150`** (10 s) ends a stuck episode instead of waiting out the env's
  60 s timeout. Successful demos average ~96 steps.
- **`--debug_grounding`** records which block the first predicted chunk aims at, for
  the confusion matrix. A strong diagonal means the instruction is being followed.
- **`--seed N`** fixes the episode sequence. Use the same seed for every checkpoint
  you compare; omit it for new random layouts each run.

**Pick the checkpoint by sweeping**, all on the same episodes:

```bash
for ck in v3/010000 v4/015000 v4/020000 v4/025000 v4/030000; do
  run=${ck%%/*}; step=${ck##*/}
  python scripts/eval_smolvla.py \
    --checkpoint outputs/train/rb_smolvla_$run/checkpoints/$step/pretrained_model \
    --num_episodes 20 --seed 123 --debug_grounding --max_steps 150 --n_action_steps 5 \
    --headless 2>&1 | tee /tmp/eval_${run}_${step}.log
done
```

Measured results for every round — success rates, per-colour splits, failure modes,
grounding and what they imply — live in **[docs/RESULTS.md](docs/RESULTS.md)**, so
they stay in one place as rounds accumulate. In short: 100 demos scored 0/20, 250
scored 11/20, and 400 scored 10–11/20 (no further gain). Block choice is solved
(90–95% grounding, no distractor ever binned); every remaining failure is "never
lifted the target", i.e. the grasp — which is the partial competence the
shared-autonomy correction study builds on.

> The "closest approach" figures are slightly unreliable for *successful* episodes:
> the env auto-resets inside the step that ends an episode, so that final reading
> compares the reset fingertip against the next episode's blocks. Success rates,
> failure modes and grounding are unaffected.

**8. Second round: 400 demos → v5.** Same pipeline as steps 2c–7, with these
changes:

```bash
# collect 150 more on NEW layouts (1-50 are already recorded)
scripts/collect_randomblock.sh 51 100

# build a separate dataset
python scripts/npy_to_lerobot.py \
  --input logs/data/randomblock_demos.npy \
  --repo_id local/randomblock_vla_400 \
  --root logs/lerobot/randomblock_vla_400 \
  --images --overwrite
```

Train with the step 6 command, changing only these arguments:

```bash
  --dataset.repo_id=local/randomblock_vla_400 \
  --dataset.root=logs/lerobot/randomblock_vla_400 \
  --batch_size=64 --steps=40000 --save_freq=5000 --num_workers=12 \
  --output_dir=outputs/train/rb_smolvla_v5 --job_name=rb_smolvla_v5 \
  --policy.device=cuda 2>&1 | tee /tmp/train_v5.log
```

- **`--steps=40000`** keeps the same amount of training per demo: v4 peaked at ~66
  epochs, which over 400 demos (37k frames) is ~40k steps.
- **`--save_freq=5000`** halves the checkpoint count (8 x ~1.3 GB) to fit on disk.
- **`--num_workers=12`** speeds up data loading only; it does not change the result.

When training prints `End of training`, **watch the model** — opens the sim window,
new random layouts every run, Ctrl-C to stop:

```bash
python scripts/eval_smolvla.py \
  --checkpoint outputs/train/rb_smolvla_v5/checkpoints/040000/pretrained_model \
  --num_episodes 5 --n_action_steps 5 --max_steps 150
```

Keep `--n_action_steps 5` even just to watch: without it the robot executes 50 steps
before looking again and misses far more than it really would.

Then **score it** against v4, one checkpoint at a time:

```bash
python scripts/eval_smolvla.py \
  --checkpoint outputs/train/rb_smolvla_v5/checkpoints/040000/pretrained_model \
  --num_episodes 20 --seed 123 --n_action_steps 5 --max_steps 150 --debug_grounding
```

Keep **`--num_episodes 20 --seed 123`**: those are the episodes in the step 7
results table, so the score is directly comparable to v4 @ 25k's 11/20. Then swap
`040000` for `020000`, `025000`, `030000` or `035000` — the final checkpoint is not
necessarily the best. To run them all unattended instead, use the step 7 sweep loop
with `for ck in v4/025000 v5/020000 v5/025000 v5/030000 v5/035000 v5/040000`.

**9. Shared autonomy: record human corrections.** The study's core step. SmolVLA and
the SpaceMouse both act on the *same* observation each timestep; the robot executes
the blend, and all three actions are logged:

```
a_R    = SmolVLA (the base policy from step 7/8)
a_H    = you, on the SpaceMouse
a_exec = alpha * a_R + (1 - alpha) * a_H      <- what actually runs
```

```bash
python scripts/shared_autonomy.py \
  --checkpoint outputs/train/rb_smolvla_v5/checkpoints/025000/pretrained_model \
  --num_episodes 5 --alpha 0.5
```

Unlike `play.py`, this runs **several episodes per launch** — no 60 s Isaac restart
between them. Recording is on by default (`logs/rollouts/shared_autonomy_<task>/`,
`--no_record` to drive without saving).

- **`--alpha`** (0.5): weight on the *robot*. 1.0 = robot only, 0.0 = human only.
- **Idle gating (default).** An untouched SpaceMouse reports zero, so blending it in
  unconditionally would drag the arm toward standing still and fill the log with
  `a_H = hold` samples that are not corrections. While you are not touching it the
  robot runs on `a_R` alone; blending starts the moment you push. `--blend_always`
  disables this.
- **Gripper** does not average meaningfully, so open/close follows whoever is in
  charge: you for `--grip_hold_steps` (30) after a button press, the policy otherwise.
  Both raw values are logged either way.
- **`--n_action_steps 5`** matches the evaluation protocol; the checkpoint's own
  default (50) is 3.3 s open-loop and cannot react to a correction.
- Both actions are computed from `o_t` *before* the step, so `(o_t, a_H, a_R, a_exec)`
  carries no one-step lag.

Per episode it prints how many steps you intervened on and the median / p90 of
`d_t = ||a_H - a_R||` — that distribution is how the disagreement threshold `tau`
gets chosen, once real data exists.

**Build both comparison datasets from the same recording** — identical observations,
differing only on the selected corrective steps:

```bash
# repeat per session dir; --append merges them
for M in human blend; do
  python scripts/convert_demos.py --rollout_dir logs/rollouts/shared_autonomy_s2 \
    --output logs/data/corrections_$M.npy --append --action_prefix $M --angle_threshold 0
done

python scripts/npy_to_lerobot.py --input logs/data/corrections_human.npy \
  --repo_id local/corrections_human --root logs/lerobot/corrections_human --images --overwrite
```

`--action_prefix human` / `blend` label a step with `a_H` / `a_exec` when it is
corrective (human intervening, direction disagreement >= `--angle_threshold`) and with
the *policy's own action* otherwise — the same fallback in both, so the two datasets
differ by exactly the label under test. Two details this protects against:

- Taking `base_action` wholesale would train on "hold this pose" for the ~80% of steps
  the human never touched the SpaceMouse — i.e. teach the policy to stop moving.
- Taking `exec_action` wholesale would leave D_blend differing from D_human on
  interventions the threshold excluded, confounding the comparison at any `tau > 0`.

Then finetune two copies from the *same* checkpoint with identical settings — same
`--seed`, a reduced `--policy.optimizer_lr` (2.5e-5; full rate overwrites a competent
policy on a 4k-frame set) — and evaluate both plus the untouched base with the step 7
protocol.

Measured on the first 50-episode session: 4,144 steps, 721 corrective (17%),
correction bursts of median 3 steps clustered in the approach phase, direction
disagreement median 69 deg. `tau` = 0 / 45 / 90 deg selects 721 / 498 / 218 steps.

> Not yet exercised against Isaac hardware-in-the-loop. The blend math and the
> two-dataset conversion are unit-tested; the env/SpaceMouse path is not. Do one
> `--num_episodes 1` run first and check that `d_t` moves when you push the mouse.

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
