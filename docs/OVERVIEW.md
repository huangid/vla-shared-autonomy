# Interactive Learning for VLA Shared Autonomy — project overview

Undergraduate thesis project. This file explains *what* the project asks, *why* each
piece exists, and *where* things stand. The [README](../README.md) holds the commands;
[RESULTS.md](RESULTS.md) holds the measurements.

---

## 1. The research question

A vision-language-action model (VLA) and a human operate the same robot arm at once.
Each timestep both propose an action and the robot executes a blend:

```
a_exec = alpha * a_R + (1 - alpha) * a_H
```

where `a_R` is the VLA's action, `a_H` the human's, and `alpha` the authority split
(0.5 here). When the two disagree — `d_t = || a_H - a_R ||` above a threshold `tau` —
the human is correcting the robot.

**The question:** to improve the VLA from those episodes, should it be finetuned on
the *blended* action it actually executed (`a_exec`), or on the *raw human* action
(`a_H`)?

**The hypothesis:** `a_H` is the stronger learning signal. `a_exec` is contaminated —
it is half the policy's own erroneous action, so training on it partly teaches the
error back.

Answering this needs a policy that is **partially competent**: good enough to attempt
the task, bad enough to need correcting. Building that policy is most of the work
below.

---

## 2. The task: `RandomBlock`

Three 3 cm blocks (red, green, blue) spawn at random xy positions in a 14 x 30 cm
rectangle in front of an xArm; a bin sits at a fixed position and is kinematic
(immovable). Each episode issues one instruction — *"pick up the {colour} block and
put it in the bin"* — and the other two blocks are distractors that must stay out of
the bin. Success = the named block binned, neither distractor binned.

Why this task:

- **Language matters.** Three colours means the instruction, not the scene alone,
  determines the goal.
- **It cannot be memorised.** Positions are continuous and resampled every episode.
- **It is hard in the right place.** A 3 cm block needs roughly 1.5 cm of accuracy to
  grasp, so the difficulty lives in precision — which is exactly what human
  corrections address.

Defined in `source/xarm_assembly_env/assembly_tasks_cfg.py` (`RandomBlock`) on top of
`XArmEnv` / `XArmEnvGuidedDiffusion`. Actions are 8D absolute end-effector targets
(pos3 + quat4 + gripper1), routed through admittance control and Pinocchio IK.

---

## 3. Pipeline

```
   teleop demos            base policy              corrections             comparison
 ┌───────────────┐      ┌──────────────┐        ┌────────────────┐      ┌──────────────┐
 │ play.py       │      │ lerobot-train│        │shared_autonomy │      │ two finetunes│
 │ + SpaceMouse  │─────▶│ SmolVLA      │───────▶│ VLA + human    │─────▶│ D_human vs   │
 │ 400 episodes  │      │ v5 @ 25k     │        │ 50 episodes    │      │ D_blend      │
 └───────────────┘      └──────────────┘        └────────────────┘      └──────────────┘
   convert_demos.py       eval_smolvla.py         convert_demos.py        eval_smolvla.py
   npy_to_lerobot.py                              --action_prefix         (paired, n=200)
```

### Scripts

| Script | Role |
|---|---|
| `scripts/play.py` | Drive a task by hand (SpaceMouse) and record rollouts. `--layout_seed` / `--target_color` make a batch deliberate. |
| `scripts/collect_randomblock.sh` | Batch demo collection: re-records failed demos, safe Ctrl-C, backups. |
| `scripts/convert_demos.py` | Rollouts -> kNN-format `.npy` + `.tasks.json` sidecar. `--action_prefix` selects which action stream becomes the BC target. |
| `scripts/npy_to_lerobot.py` | `.npy` -> LeRobot dataset with images and per-episode instruction. |
| `scripts/merge_npy_datasets.py` | Mix correction episodes with a seeded slice of base demos. |
| `scripts/shared_autonomy.py` | Run SmolVLA and the human together; execute the blend; log `(o_t, a_H, a_R, a_exec)`. |
| `scripts/eval_smolvla.py` | Autonomous evaluation: success, per-colour, failure modes, closest approach, language grounding. |
| `scripts/probe_smolvla.py` | Offline diagnostics on training frames (accuracy vs baselines, grounding, input sensitivity). |
| `source/pilot_models/smolvla_pilot.py` | Inference wrapper around a finetuned SmolVLA checkpoint. |
| `source/pilot_models/spacemouse_pilot.py` | Live human teleop via 3Dconnexion SpaceMouse. |

### Evaluation protocol

`--num_episodes N --seed 123 --n_action_steps 5 --max_steps 150 --debug_grounding`.
The seed fixes the episode sequence so every model faces identical layouts, which
makes comparisons **paired** (McNemar exact test). `--n_action_steps 5` matters: the
checkpoint default of 50 runs 3.3 s open-loop and cannot react to a drifting grasp.

Reported per run: overall and per-colour success, failure-mode breakdown, closest
fingertip-to-target approach (the precision measure), and first-chunk grounding — does
the policy's first plan aim at the *named* block (chance 33%).

---

## 4. Where things stand

### Base policy: built, partially competent (as intended)

| Round | Demos | Success (paired eval) | Grounding |
|---|---|---|---|
| v3 | 100 | 0/20 | 60% |
| v4 | 250 | 11/20 | 90% |
| **v5 @ 25k** | 400 | **104/200 = 52%** | 76% |

- 100 demos was not enough; 250 was. 250 -> 400 changed nothing, so data stopped
  being the constraint.
- Block *selection* is solved (76% grounding, strong confusion diagonal).
- Nearly every failure is "never lifted the target" — the grasp, not the choice. This
  is the partial competence the study needs.

### Correction rounds: the hypothesis is refuted

Three rounds, all finetuned from `v5/025000`, all evaluated on the same 200 episodes
(paired McNemar). Full numbers in [RESULTS.md](RESULTS.md) sections 9-13.

| Round | alpha | Corrections | D_human | D_blend | Label gap |
|---|---|---|---|---|---|
| 1 (corrections only) | 0.5 | 50 eps | 37.5% | 41.0% | 4.3 cm |
| 2 (+ 100 base demos) | 0.5 | 50 eps | 49.0% | 44.5% | 4.3 cm |
| 3 (+ 100 base demos) | 0.5 | 100 eps | 47.5% | 49.0% | 6.2 cm |
| **3 (+ 100 base demos)** | **0.8** | **100 eps** | **34.0%** | **53.0%** | **10.9 cm** |

Base policy: 52.0%.

**The answer is `a_exec`, not `a_H`** — the opposite of the hypothesis, at p = 0.00018
on a pre-registered primary test (section 11 was committed before the data existed).
Training on the raw human action is also significantly *worse than leaving the policy
alone* (p = 0.0011), while the blend matches it (p = 0.93).

Three further findings:

1. **Dose-response.** The effect is absent at alpha = 0.5 (p = 0.83, labels 6.2 cm
   apart) and large at alpha = 0.8 (labels 10.9 cm apart). That is why rounds 1 and 2
   found nothing: both ran at alpha = 0.5.
2. **The mechanism is off-policy mismatch.** On disagreeing steps, `D_human` asks the
   arm to travel 14.1 cm against `D_blend`'s 3.7 cm and the base demos' 8.7 cm. With
   only 20% authority the operator pushes hard, so `a_H` becomes an out-of-distribution
   target that never produced the observations it is paired with. `a_exec` is
   self-consistent by construction.
3. **`a_H` is not "pure human intent" in an absolute-target action space.** Its
   magnitude is an artefact of alpha. Future work using raw human actions should
   rescale to the policy's action distribution, or use a relative/velocity action space.

A control arm (`ballast_only`: the same 100 base demos, same recipe, **no**
corrections) scores 48.0% and is indistinguishable from the base (p = 0.47). Against
that control, no arm improves (best p = 0.40) and only `a_H` at alpha = 0.8 harms
(p = 0.0051) — so the damage comes from the labels, not from the finetuning recipe.

Separately, round 1 established that **training recipe matters more than label
choice**: corrections-only finetuning cost 11-15 points, and adding base-demo ballast
repaired it (p = 0.027). No arm in any round improved on the base policy.

## 5. Things that were wrong, and how they were caught

Recorded because they shaped the results, and because each was invisible in the
training loss.

- **Visualization markers in the training images.** `--vis_obs` renders a marker on
  the target block; recording with it gives the policy an answer key it will never see
  at evaluation. Now off by default, with a loud warning if combined with `--record`.
- **Stale camera on the first frame of every episode.** Isaac updates sensors inside
  `env.step()`, so the first observation showed the *previous* scene. The policy
  committed a 3.3 s open-loop chunk to a layout that no longer existed.
- **Instruction redundant in the training data.** With every layout appearing exactly
  once, the image alone predicted the trajectory and the prompt carried no
  information. Fixed by recording each layout under all three colours.
- **The human gripper channel.** The SpaceMouse reports a *latched* open/closed state,
  not an intent; a driver who steers position and leaves grasping to the policy emits
  "open" on every step. `D_human` therefore labelled 21% of corrective steps "hold the
  gripper OPEN" at the moments the policy was closing — training the model not to
  grasp. Caught because that model reached grasping range more often than any other
  yet converted worst.
- **The disagreement metric.** `|| a_H - a_R ||` was *larger* while the human was idle
  than while correcting: the human's action is the current pose plus a small increment,
  the policy's a waypoint 5-13 cm ahead, so the norm mostly measured the policy's
  lookahead. Replaced by the angle between the two intended displacements.
- **Confounded dataset fallback.** An early build gave `D_blend` the executed action on
  *every* step, so at `tau > 0` the two datasets differed on more than the selected
  steps. Both now share the same fallback and differ only on the selection.

---

## 6. Open questions

1. **Answered:** `a_exec` beats `a_H`, and only when authority is skewed enough for the
   two to diverge. See section 4.
2. **Why does neither label improve on the base?** Non-corrective steps are labelled
   with the policy's own action, so most of the finetuning is self-distillation. Only
   the 5-12% corrective steps carry new information.
3. **Would aggregating the successful shared-autonomy trajectories help?** The 200
   episodes succeeded 200/200; labelling every step `exec_action` and pooling with the
   400 demos is the untested, most promising route to higher success.
4. **Stale corrections.** All corrections were recorded against `v5/025000`; a
   finetuned policy errs differently (the DAgger argument for iterating collection).
5. **Does grasp failure cluster by table position?** `eval_smolvla.py` does not log
   block positions, so this remains untested. Correction directions were roughly
   isotropic (34% along the camera ray, 29% lateral, 38% vertical), which argues
   against monocular depth error and toward descent height / gripper timing.

## 7. Environment

Isaac Sim 5.1 / Isaac Lab 2.3.2, SmolVLA (450M) via LeRobot, `uv` for dependencies
(not pip), Python 3.11. Running any environment requires the Isaac runtime; plain
`import residual_copilot` fails outside it on USD imports. A physical 3Dconnexion
SpaceMouse is required for demo collection and shared autonomy.

Not version controlled: `logs/rollouts/` (recordings), `logs/data/` and
`logs/lerobot/` (datasets), `outputs/train/` (checkpoints, ~1.3 GB each). Those exist
only on the workstation — the 50 correction episodes in particular are unbacked-up and
irreplaceable without redoing the teleop.
