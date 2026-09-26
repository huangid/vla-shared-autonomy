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

### Correction round 1: both finetunes made it worse

50 shared-autonomy episodes (50/50 successful with human help), 4,144 steps, 721
corrective (17%), corrections clustered in the approach phase.

| Model | Success (200 paired episodes) | vs base |
|---|---|---|
| base v5 @ 25k | 104/200 = 52.0% | — |
| `ft_blend` | 82/200 = 41.0% | p = 0.032 |
| `ft_human` | 75/200 = 37.5% | p = 0.0046 |

Two conclusions: **finetuning on corrections alone significantly degrades the
policy**, and **the label choice makes no measurable difference** (p = 0.53). The
research question is therefore *not yet answered* — both arms were damaged by a shared
cause, so there is nothing to compare. Most likely catastrophic forgetting: 47 epochs
over 4k correction frames against a policy built from 37k.

### Round 2: in progress

Both arms rebuilt as corrections + 100 base demos (13,273 frames; differing labels
5.4% instead of 17%), identical base slice in both, 5,000 steps. The open risk is the
reverse of round 1: dilute the contrast too far and both models simply match the base.

---

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

1. **Does base-demo ballast fix the degradation?** Round 2 answers this.
2. **Is `a_H` better than `a_exec`?** The original question, still unanswered.
3. **How much correction data is needed?** 721 corrective samples may be too few to
   move a 450M-parameter policy regardless of labelling.
4. **Stale corrections.** Corrections were recorded against v5 @ 25k and describe
   *that* policy's mistakes. Once a finetune changes the policy, they go stale — the
   DAgger argument for iterating collection against the current best model.
5. **Do failures cluster by table position?** `eval_smolvla.py` does not log block
   positions, so this is untested; it would say whether the spawn range is too wide
   for the data budget.

---

## 7. Environment

Isaac Sim 5.1 / Isaac Lab 2.3.2, SmolVLA (450M) via LeRobot, `uv` for dependencies
(not pip), Python 3.11. Running any environment requires the Isaac runtime; plain
`import residual_copilot` fails outside it on USD imports. A physical 3Dconnexion
SpaceMouse is required for demo collection and shared autonomy.

Not version controlled: `logs/rollouts/` (recordings), `logs/data/` and
`logs/lerobot/` (datasets), `outputs/train/` (checkpoints, ~1.3 GB each). Those exist
only on the workstation — the 50 correction episodes in particular are unbacked-up and
irreplaceable without redoing the teleop.
