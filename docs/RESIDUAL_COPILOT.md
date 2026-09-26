# Residual Copilot — project overview

The original project this repository was built for: **"Efficient and Reliable
Teleoperation through Real2Sim2Real Shared Autonomy"** ([paper](https://arxiv.org/abs/2603.22787),
[upstream code](https://github.com/shuosha/Residual_Copilot)). The VLA shared-autonomy
thesis work ([docs/OVERVIEW.md](OVERVIEW.md)) is a later addition built on the same
simulator and control stack; this file describes what was already here.

---

## 1. The idea

A human teleoperating a robot through contact-rich assembly is slow and error-prone:
small misalignments jam parts, and force feedback is poor through a 6-DoF mouse. Full
autonomy is not available either — the policies are not reliable enough.

**Shared autonomy via a residual copilot.** A *pilot* (the human, or a policy standing
in for one) proposes an action; a *copilot* trained with RL outputs a small correction
on top; the sum is executed:

```
a_exec = a_pilot + a_residual
```

The copilot never drives on its own. It learns only the correction that turns a
roughly-right human command into one that actually completes the insertion, which is a
far easier learning problem than the whole task and keeps the human in charge.

**Real2Sim2Real.** Train in Isaac Sim against randomized dynamics, then deploy the same
copilot on hardware (see
[Residual_Copilot_Deployment](https://github.com/shuosha/Residual_Copilot_Deployment)).
Domain randomization over the admittance-control parameters is the main transfer
mechanism.

---

## 2. Control stack

```
obs (11D)  →  Pilot (kNN or BC/DiffusionPolicy)  →  base_action (8D)
                      ↓
        [+noise or +lag augmentation, if pilot_type != "none"]
                      ↓
obs (35D policy / 53D critic)  →  RL residual policy  →  residual_action (7D)
                      ↓
      _apply_residual()  →  admittance control  →  IK (Pinocchio / SAPIEN)
                      ↓
                 joint targets  →  Isaac Sim physics
```

- **Actions are absolute end-effector targets** — position (3), quaternion (4),
  gripper (1). The residual policy emits 7D (no gripper); guided-diffusion copilots
  emit the full 8D and skip the residual step (`XArmEnvGuidedDiffusion`).
- **Admittance control** (`source/utils/control.py`, `adm_ctrl_task_space`) turns a
  target pose into a compliant motion given measured contact force — the arm yields
  instead of jamming. Stiffness/mass terms `Kx, Kr, mx, mr` are what domain
  randomization perturbs.
- **IK** via Pinocchio maps the Cartesian target to joint targets.

### Observation spaces

| Space | Dim | Contents |
|---|---|---|
| Policy obs | 35 | fingertip pos/quat/gripper, relative held- and base-object positions, EE linear/angular velocity, the pilot's base action, previous actions — all **noisy**, matching the data distribution |
| Critic state | 53 | the above plus ground-truth joint positions and held/base object poses |

Asymmetric actor-critic: the critic sees privileged simulator state, the actor only
what a real robot could measure.

---

## 3. Tasks

Registered in `source/xarm_assembly_env/__init__.py`, configured in
`assembly_tasks_cfg.py`:

| Task | Description |
|---|---|
| `GearMesh` | Place a gear onto a shaft so its teeth mesh |
| `GearMeshIntent` | Gear mesh where each episode targets one of three slots (small / medium / large) — multi-goal |
| `PegInsert` | Peg-in-hole insertion |
| `NutThread` | Thread a nut onto a bolt — the tightest tolerance |
| `ThreeBlocks` | Multi-goal block sorting: pick up to 3 blocks into a bin; built for live SpaceMouse shared autonomy |
| `RandomBlock` | *Added for the VLA study* — `ThreeBlocks` with randomized block positions, a fixed kinematic bin, and a single language-specified target |

Each variant exists as `XArm-<Task>-Residual` (7D residual actions) and
`XArm-<Task>-GuidedDiffusion` (8D absolute actions).

Assets (USD/URDF), demo data and pretrained checkpoints are fetched from the
HuggingFace repo `shashuo0104/residual_copilot_data` via `resolve_hf()`.

---

## 4. Pilots and copilots

### Pilots — what proposes the base action

| Pilot | Source |
|---|---|
| `kNNPilot` | K=10 nearest neighbours in the demo set, interpolated, with variable-horizon queuing (1–15 steps). Config: `config/knn_cfg.json` |
| `BCPilot` | LeRobot DiffusionPolicy trained on teleop (`bc_teleop`) or expert (`bc_expert`) data |
| `ExpertPilot` | Expert BC policy |
| `NoisyPilot` | Expert BC plus injected action noise — a stand-in for an imprecise human |
| `LaggyPilot` | Expert BC with repeated actions — a stand-in for a slow human |
| `ReplayPilot` | Replays recorded episodes |
| `SpaceMousePilot` | A real human on a 3Dconnexion SpaceMouse (HID device required) |

The noisy/laggy variants matter: they let the copilot be trained and evaluated
against reproducible "bad human" behaviour instead of requiring a person in the loop.

### Copilots — what corrects it

| Copilot | Mechanism |
|---|---|
| `ResidualCopilot` | PPO residual policy trained with the kNN pilot |
| `ResidualBC` | PPO residual policy trained with the BC pilot |
| `GuidedDiffusionBC` / `GuidedDiffusionExpert` | Diffusion policy that takes the pilot's action as a reference during denoising and outputs a complete 8D action |
| `DiSCo` | Guided diffusion with **DiSCo sampling** (Wang et al., HRI '26): forward-diffuse the pilot's current action to seed the reverse process, inpaint the known part for the first `k_ip` reverse steps, then β-blend policy and user actions — position and gripper linearly, orientation by SLERP. When the user is idle, β is forced to 1 so the arm holds still rather than drifting on its own |

`DiSCo` is the one used for live assistance with a human on the SpaceMouse.

---

## 5. Reward and training

Residual PPO (`agents/rl_games_ppo_cfg.yaml`): γ = 0.995, τ = 0.95, lr 1e-4
(adaptive), horizon 128, minibatch 512, and an LSTM(1024, 2 layers) in front of an
MLP [512, 128, 64].

Reward terms (`assembly_tasks_cfg.py`), weights as configured:

| Term | Scale | Purpose |
|---|---|---|
| `task_success` | 30.0 | Completing the assembly |
| `termination` | 50.0 | Penalty for terminating without success |
| `action_norm` | 0.3 | Keep the residual small — stay a *correction*, not a takeover |
| `action_smoothing` | 0.1 | Penalise jerk |
| `tilt_penalty` | 1.0 | Keep the held part upright |
| `force_penalty` | 0.2 | Discourage jamming |
| `xy_aligned` | 0.05 | Shaping toward alignment |

The `action_norm` penalty is what keeps the system *shared* autonomy: without it the
copilot learns to override the human entirely.

```bash
python scripts/train.py --task XArm-GearMesh-Residual --pilot kNNPilot \
  --num_envs 128 --headless
```

128 parallel environments; logs to `logs/rl_games/`. `--distributed` for multi-GPU,
`--track` for W&B.

---

## 6. Domain randomization

Enabled by `DomainRandCfg` (`xarm_env_cfg.py`), resampled per episode:

- **Controller:** `Kx ∈ [190, 210]`, `Kr ∈ [95, 105]`, `mx ∈ [0.119, 0.131]`,
  `mr ∈ [0.0143, 0.0158]`
- **Observation noise:** ±2 mm on fixed- and held-asset positions
- **Data augmentation:** ±2 cm in xy, ±2 mm in z on demo replay

Randomizing the admittance parameters rather than only visuals is the deliberate
choice here: for contact-rich assembly the sim-to-real gap is dominated by contact
dynamics, not appearance.

---

## 7. Evaluation and visualization

```bash
# pilot + copilot, recording rollouts
python scripts/play.py --task NutThread --pilot kNNPilot --copilot ResidualCopilot \
  --num_envs 16 --record

# annotated videos: red = base action, pink = residual, blue = net
python scripts/vis/to_videos.py logs/rollouts/<dir> --single --annotate
python scripts/vis/to_videos.py logs/rollouts/<dir> --collage --annotate --cols 4
```

With `--no_rand`, environments run in deterministic order (one episode each), so
`--num_envs N` evaluates exactly N ordered episodes. The arrow overlays are the
quickest way to see whether the copilot is nudging or taking over.

---

## 8. Relationship to the VLA study

The thesis work reuses the simulator, control stack, recording format and
`SpaceMousePilot`, and adds:

- `RandomBlock` — randomized positions, fixed kinematic bin, language-specified target
- `SmolVLA_Pilot` — a VLA in place of kNN/BC as the action source
- `scripts/shared_autonomy.py` — VLA and human blended by a scalar α, with full
  `(o_t, a_H, a_R, a_exec)` logging
- The demo/eval tooling: `convert_demos.py`, `npy_to_lerobot.py`, `eval_smolvla.py`,
  `probe_smolvla.py`, `collect_randomblock.sh`, `merge_npy_datasets.py`

The conceptual difference: the residual copilot **corrects the human**, learning
`a_residual` on top of a human command. The VLA study inverts this — the **human
corrects the policy**, and the question is which signal that correction should leave
behind for training.

---

## 9. Note on CLAUDE.md

`CLAUDE.md` describes the PPO config as "lr = 5e-4 (adaptive), 32-step horizon, 2-layer
MLP [32, 32]". The committed `agents/rl_games_ppo_cfg.yaml` says lr 1e-4, horizon 128,
LSTM(1024) + MLP [512, 128, 64]. The YAML is what runs; treat the CLAUDE.md figures as
stale.
