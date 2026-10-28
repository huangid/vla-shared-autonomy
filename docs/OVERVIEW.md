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
(0.5 or 0.8 in the experiments). When the two disagree — `d_t = || a_H - a_R ||` above a threshold `tau` —
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

**Two scenes, one per branch.** The study was first run on `single-camera`: a fixed
front camera, the bin straight ahead at (0.54, 0), blocks in a 14 x 30 cm rectangle.
It was then repeated on `two-camera` (this branch, the repo default): a gripper-mounted
wrist camera plus a square-on side camera, the bin at the robot's right (0.37, -0.18),
blocks in a 16 x 16 cm square. Success rates do not compare across the two.

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
The seed fixes the episode sequence, compared with the McNemar exact test. **Until
2026-10-27 only the target-colour sequence was actually shared:** block layouts drew
from the same random stream as SmolVLA's sampling noise and diverged between models
after the first episode or two (RESULTS section 23). The tests stay valid but were
effectively unpaired; layouts now come from their own seeded generator. `--n_action_steps 5` matters: the
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

The numbers above are the single-camera study; its full record is in `RESULTS.md` on
the `single-camera` branch. [RESULTS.md](RESULTS.md) here holds the two-camera
sections 14-17 below.

### Two-camera round: the finding replicates

New scene and cameras, 240 demos -> base `twocam_v1/020000` at **142/200 = 71%**.
100 correction episodes at alpha = 0.8, then the same two arms plus a control
(RESULTS sections 14-15):

| Model | Success |
|---|---|
| control (50 base demos, no corrections) | 146/200 = 73.0% |
| base | 142/200 = 71.0% |
| D_blend (`a_exec`) | 141/200 = 70.5% |
| **D_human (`a_H`)** | **119/200 = 59.5%** |

`a_H` is again the harmful label (vs `a_exec` p = 0.037, vs control p = 0.0086, vs base
p = 0.018), `a_exec` is again harmless, and again nothing beats the base. Declared a
pilot in advance, so it counts as a replication of the pre-registered single-camera
result rather than a test on its own. It also revised the mechanism: here `D_blend`'s
labels sit *further* from the demo distribution than `D_human`'s, so magnitude alone
does not explain the harm. What survives is **self-consistency** — `a_exec` is the
action that produced the next observation; `a_H` was only 20% executed.

### Residual on a frozen base: no effect

A different use of the same corrections, in the style of CR-DAgger: keep the base
frozen and train a small network to predict the adjustment `a_exec - a_R`, added to the
base's action at run time (RESULTS sections 16-17). It learned "about 1 cm lower before
the grasp", and applied it, but changed nothing measurable:

| Run (same 200 episodes) | Success |
|---|---|
| base + state-only residual (primary) | 151/200 = 75.5% (vs base p = 0.35) |
| base + image residual | 155/200 = 77.5% |
| base + zero adjustment (sanity) | 157/200 = 78.5% |

Two plain re-runs of the base both scored 142/200 yet disagreed on 72 episodes: **the
simulator is not repeatable between runs**, episode by episode. The zero-adjustment
run's 157 is unexplained (chance at p = 0.115, or a side effect of the residual code).
Either way the learned adjustment adds nothing — both residuals scored below the
zero-adjustment run — and the label findings above are larger than this noise.

### Weaker base, alpha 0.5: corrections help for the first time

Corrections collected on `twocam_v1/010000` (68%) at alpha = 0.5, then three arms
finetuned from it, each with all 240 demos (RESULTS sections 18-19):

| Model | Success |
|---|---|
| base `010000` | 136/200 = 68.0% |
| control (240 demos only, same training) | 135/200 = 67.5% |
| + corrections, blend label (`a_exec`) | 150/200 = 75.0% |
| **+ corrections, human label (`a_H`)** | **158/200 = 79.0%** |

Extra training alone changed nothing, while both correction arms improved grasping
("never lifted" fell from ~58 to ~38). Human vs control p = 0.012 (secondary test;
p = 0.047 after Bonferroni); the pre-registered primary, blend vs control, gave
p = 0.096. At this alpha the human label is not harmful — the opposite of alpha = 0.8
— which fits the dose-response: the more the two labels diverge, the more `a_H` hurts.
The base turned out barely weaker than `020000` (68% vs 71%), so "weaker base" is not
the explanation; alpha, the larger demo ballast, longer training and operator practice
all changed together. One training seed per arm.

---

### 5k base, alpha 0.5: the gain does not replicate

The same procedure with fresh corrections on `twocam_v1/005000` (50%), pre-registered
with the human arm vs control as the primary test (RESULTS sections 20-21):

| Model | Success |
|---|---|
| base `005000` | 100/200 = 50.0% |
| **control (240 demos only)** | **119/200 = 59.5%** |
| + corrections, blend label | 113/200 = 56.5% |
| + corrections, human label | 107/200 = 53.5% |

Extra training on demos helped this undertrained base on its own; the corrections added
nothing (human vs control p = 0.29, the wrong direction). Pooled over the 10k and 5k
rounds, corrections vs control is null (p = 0.46 human, 0.54 blend). The 10k gain is
therefore not yet a reliable finding; a training-seed replication of those arms is
next.

---

### 10k seed replication: the gain survives a second training run

The three `010000` arms retrained with a different seed (RESULTS section 22): control
141, blend 146, **human 155** (seed 1000: 135, 150, 158). The human-label arm reproduces
within a few episodes; its gain over the control shrinks from +23 to +14 but keeps its
direction (pooled over both seeds p = 0.004, post hoc). So on `010000` the corrections
gave a modest, reproducible improvement; on `005000` (a different correction set) they
gave none, and pooled over all three runs the effect is not significant (p = 0.14).
Whether the base or the correction set explains the difference is open.

---

### Paired re-evaluation: the 10k human-label gain does not hold up

After fixing layout pairing (RESULTS section 23), all seven 10k models were
re-evaluated on identical layouts (section 25):

| | seed 1000 | seed 2000 |
|---|---|---|
| base `010000` | 130 | |
| control | 140 | 136 |
| blend label | **148** | **149** |
| human label | 137 | 153 |

The pre-registered test (human vs control, both seeds) gives p = 0.18. The blend arm is
the consistent one (+8 and +13 over its controls; pooled p = 0.026, secondary). So the
earlier "human label +23" was partly an evaluation draw. Overall: corrections gave at
most a small gain on this base, more reliably with the executed (blend) label, and the
label choice still makes no detectable difference at alpha = 0.5.

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

1. **Answered:** `a_exec` beats `a_H`, and only when authority is skewed enough for the
   two to diverge. See section 4.
2. **Why does nothing improve on the base?** Non-corrective steps are labelled
   with the policy's own action, so most of the finetuning is self-distillation. Only
   the 5-20% corrective steps carry new information. A residual on a frozen base, which
   learns *only* from those steps, did not help either (two-camera section 17).
3. **Would aggregating the successful shared-autonomy trajectories help?** The 200
   single-camera episodes succeeded 200/200 and the 100 two-camera ones 100/100;
   labelling every step `exec_action` and pooling with the demos is the untested, most
   promising route to higher success.
4. **Stale corrections.** All corrections were recorded against `v5/025000`; a
   finetuned policy errs differently (the DAgger argument for iterating collection).
5. **Does grasp failure cluster by table position?** `eval_smolvla.py` does not log
   block positions, so this remains untested. Correction directions were roughly
   isotropic (34% along the camera ray, 29% lateral, 38% vertical), which argues
   against monocular depth error and toward descent height / gripper timing.
6. **Training-seed variance** is not characterised: every arm is one training run.

## 7. Environment

Isaac Sim 5.1 / Isaac Lab 2.3.2, SmolVLA (450M) via LeRobot, `uv` for dependencies
(not pip), Python 3.11. Running any environment requires the Isaac runtime; plain
`import residual_copilot` fails outside it on USD imports. A physical 3Dconnexion
SpaceMouse is required for demo collection and shared autonomy.

Not version controlled: `logs/rollouts/` (recordings), `logs/data/` and
`logs/lerobot/` (datasets), `outputs/train/` (checkpoints, ~1.3 GB each),
`outputs/residual/` (residual networks), `docs/eval_logs/` (evaluation logs). Those
exist only on the workstation — the correction episodes in particular
(`logs/rollouts/`) are irreplaceable without redoing the teleop.
