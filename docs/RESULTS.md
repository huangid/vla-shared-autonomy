# RandomBlock / SmolVLA — experiment log

Running record of the base-policy rounds for the shared-autonomy study
("Interactive Learning for VLA Shared Autonomy"). Step numbers refer to the
RandomBlock workflow in the [README](../README.md).

**Task.** `XArm-RandomBlock-*`: three 3 cm blocks (red / green / blue) spawn at
random xy in a 14 x 30 cm rectangle in front of the arm; the bin is fixed and
kinematic. Each episode names one colour — *"pick up the {colour} block and put it
in the bin"* — and the other two are distractors that must stay out of the bin.
Success requires the named block binned and neither distractor binned.

**Current base policy:** `outputs/train/rb_smolvla_v5/checkpoints/025000/pretrained_model`

---

## 1. Headline result

| Round | Demos | Best checkpoint | Success (20 eps, seed 123) | Grounding |
|---|---|---|---|---|
| v3 | 100 | 10k of 20k | **0/20 = 0%** | 60% |
| v4 | 250 | 25k of 30k | **11/20 = 55%** | 90% |
| v5 | 400 | 25k–30k of 40k | **10–11/20 = 50–55%** | 75–95% |

Two findings:

1. **100 demos is not enough for this task; 250 is.** 0% → 55%. The SmolVLA paper's
   "~50 demos per variation" does not transfer here, because target, distractors and
   layout are all randomized, so no scene repeats.
2. **250 → 400 demos changed nothing.** All five v5 checkpoints land at 10–11/20,
   against v4's 11/20. Data is no longer the binding constraint; more teleop of the
   same kind is not worth the hours.

Treat 10 vs 11 out of 20 as identical (±11% standard error at n=20).

---

## 2. Datasets

| Set | Demos | Frames | Composition | Colour balance |
|---|---|---|---|---|
| `randomblock_vla` | 100 | 9,614 | 100 unique random layouts | 46 G / 33 B / 21 R |
| `randomblock_vla_250` | 250 | 24,081 | + 50 layouts x 3 colours (seeds 1–50) | 96 G / 83 B / 71 R |
| `randomblock_vla_400` | 400 | 37,370 | + 50 layouts x 3 colours (seeds 51–100) | 146 G / 133 B / 121 R |

**Why "layout x 3 colours".** In the first 100 demos every layout appeared exactly
once, so the layout alone predicted the demonstrated trajectory and the instruction
was redundant — a policy could fit the data without reading the prompt. Recording one
layout under all three instructions makes the image ambiguous and the instruction the
only disambiguator. `play.py --layout_seed N --target_color C` (step 2b) does this;
`scripts/collect_randomblock.sh START END` runs the batch.

The first 100 demos also drifted to 46/33/21 on colour, because the target colour is
drawn independently on each launch and one launch records one episode. `--target_color`
removes that drift.

Layout seeds 1–50 and 51–100 produce 150 distinct reach targets each, with no overlap
and no empty cell in a 5x5 binning of the spawn rectangle, so the repeated-scene design
costs nothing in positional coverage.

---

## 3. Training configurations

All rounds: `lerobot/smolvla_base`, batch 64, `train_expert_only`, frozen vision
encoder, lr 1e-4, chunk_size 50, photometric augmentation (brightness / contrast /
saturation / hue ±0.04 / sharpness, max 3 per sample; LeRobot's default random affine
deliberately excluded — shifting the image moves the blocks relative to the arm).

| Run | Dataset | Steps | save_freq | Epochs at best ckpt | Wall clock |
|---|---|---|---|---|---|
| `rb_smolvla_v3` | 100 demos | 20,000 | 2,500 | ~67 | ~3.5 h |
| `rb_smolvla_v4` | 250 demos | 30,000 | 2,500 | ~67 | ~5.5 h |
| `rb_smolvla_v5` | 400 demos | 40,000 | 5,000 | ~43–64 | ~8 h |

Step counts were scaled to hold epochs roughly constant. Final training loss: v4
0.021, v5 0.027 — higher on v5 as expected, since 400 varied demos are harder to fit
than 250; it does not indicate a worse policy.

Throughput is data-bound, not GPU-bound (`data_s` ≈ 0.39 s vs `updt_s` ≈ 0.28 s per
step even at `--num_workers=12`): video decode is the bottleneck.

---

## 4. Evaluation protocol

```bash
python scripts/eval_smolvla.py \
  --checkpoint outputs/train/rb_smolvla_v5/checkpoints/025000/pretrained_model \
  --num_episodes 20 --seed 123 --n_action_steps 5 --max_steps 150 --debug_grounding \
  --headless
```

- **20 episodes, `--seed 123`** — the same episode sequence for every checkpoint, so
  numbers are comparable across runs. Layouts are freshly sampled, *not* the training
  scenes.
- **`--n_action_steps 5`** — the checkpoint default is 50 (3.3 s open-loop, longer than
  half an episode), which cannot correct a drifting grasp. This is an inference-time
  setting; no retraining needed.
- **`--max_steps 150`** — 10 s cap; successful demos average ~96 steps.
- **Grounding** = does the first predicted chunk's lowest point land nearest the named
  block (chance 33%). It is measured on the *first* observation, so it separates "wrong
  from the start" from "drifted later".

---

## 5. Full results (20 episodes, seed 123, n_action_steps 5)

| Model | Demos | Success | Per colour (R/G/B) | Within 1.5 cm | Grounding | Median closest |
|---|---|---|---|---|---|---|
| v3 @ 10k | 100 | 0/20 | 0/6, 0/7, 0/7 | 3/20 | 60% | 3.5 cm |
| v4 @ 15k | 250 | 6/20 | — | 10/20 | 75% | 1.6 cm |
| v4 @ 20k | 250 | 7/20 | — | 10/20 | 70% | 1.5 cm |
| **v4 @ 25k** | 250 | **11/20** | 4/6, 5/7, 2/7 | 9/20 | 90% | 1.6 cm |
| v4 @ 30k | 250 | 9/20 | 3/6, 3/7, 3/7 | 11/20 | 90% | 1.4 cm |
| v5 @ 20k | 400 | 10/20 | 2/6, 3/7, 5/7 | 14/20 | 70% | 1.0 cm |
| **v5 @ 25k** | 400 | 10/20 | 1/6, 3/7, 6/7 | 11/20 | **95%** | 1.3 cm |
| v5 @ 30k | 400 | 11/20 | 2/6, 5/7, 4/7 | 10/20 | 75% | 1.4 cm |
| v5 @ 35k | 400 | 10/20 | 2/6, 3/7, 5/7 | 13/20 | 85% | 1.1 cm |
| v5 @ 40k | 400 | 10/20 | 3/6, 4/7, 3/7 | 14/20 | 70% | 1.1 cm |

Per-colour splits at n=6–7 are noise; ignore single-colour swings.

### Failure modes

| Model | SUCCESS | never lifted | binned distractor | grasped, not binned |
|---|---|---|---|---|
| v3 @ 10k | 0 | 17 | 3 | 0 |
| v4 @ 25k | 11 | 9 | 0 | 0 |
| v5 @ 25k | 10 | 10 | 0 | 0 |
| v5 @ 30k | 11 | 9 | 0 | 0 |
| v5 @ 40k | 10 | 10 | 0 | 0 |

### Target vs aimed-at (v5 @ 25k, the cleanest)

```
 target      red  green   blue    n
 red           5      0      1     6
 green         0      7      0     7
 blue          0      0      7     7
```

---

## 6. What the numbers say

**Block selection is solved.** 90–95% grounding at the best checkpoints, a strong
confusion diagonal, and no distractor binned by any v4/v5 checkpoint (v3 did it 3
times). The policy reads the instruction.

**Grasp execution is the ceiling.** Every remaining v4/v5 failure is "never lifted the
target". v5 reaches within 1.5 cm more often than v4 (14/20 vs 9/20) yet succeeds no
more often, which argues the limit is no longer aim precision alone but the grasp
itself — descent height and when the gripper closes.

**More demos will not fix it.** 250 → 400 moved nothing, and the near-miss profile
says the missing ingredient is corrective signal, not more nominal demonstrations —
which is exactly what the shared-autonomy study supplies.

**This is the right base policy for the study.** Step 2 of the plan asks for a policy
that is *partially competent*: it understands the instruction, reaches the right block,
and misses roughly half the grasps. A near-perfect policy would leave nothing for human
corrections to improve.

---

## 7. Caveats

- **n = 20 per checkpoint.** Standard error ≈ 11%. Differences under ~5 episodes are
  not meaningful; the v3-vs-v4 gap (0 vs 11) is.
- **`closest approach` is slightly unreliable on *successful* episodes.** The env
  auto-resets inside the step that ends an episode, so the final reading compares the
  reset fingertip against the *next* episode's blocks. Success rates, failure modes and
  grounding are unaffected.
- **The v4 @ 25k baseline re-run alongside v5 was interrupted** at startup; the v4
  numbers above come from its earlier run under identical settings (same seed, same
  flags).
- **An earlier v3 result of 2/6 was noise.** On the seeded 20-episode protocol v3
  scores 0/20. Small-n evals misled the diagnosis once already.
- **Untested hypothesis:** whether failures cluster at particular table positions (e.g.
  far or edge placements). `eval_smolvla.py` does not log block positions, so the logs
  cannot answer it. Adding that would settle whether the spawn range is too wide for
  the data budget.

---

## 8. Reproducing

```bash
# dataset  (step 5)
python scripts/npy_to_lerobot.py --input logs/data/randomblock_demos.npy \
  --repo_id local/randomblock_vla_400 --root logs/lerobot/randomblock_vla_400 \
  --images --overwrite

# training (step 6/8) — full command in the README
lerobot-train --policy.path=lerobot/smolvla_base ... \
  --batch_size=64 --steps=40000 --save_freq=5000 --num_workers=12 \
  --output_dir=outputs/train/rb_smolvla_v5

# checkpoint sweep (step 7)
for ck in v5/020000 v5/025000 v5/030000 v5/035000 v5/040000; do
  run=${ck%%/*}; step=${ck##*/}
  python scripts/eval_smolvla.py \
    --checkpoint outputs/train/rb_smolvla_$run/checkpoints/$step/pretrained_model \
    --num_episodes 20 --seed 123 --debug_grounding --max_steps 150 --n_action_steps 5 \
    --headless 2>&1 | tee /tmp/eval_${run}_${step}.log
done
```

Raw eval output for each run is in `/tmp/eval_<run>_<step>.log` (not version
controlled; copy anything you need to keep).

---

## 9. Shared autonomy: D_human vs D_blend (round 1)

The study's main comparison. `scripts/shared_autonomy.py` runs SmolVLA and the human
SpaceMouse on the same observation each step, executes `a_exec = alpha*a_R +
(1-alpha)*a_H` (alpha = 0.5, blending gated on human input), and logs
`(o_t, a_H, a_R, a_exec)`.

**Correction sessions** — base policy `rb_smolvla_v5/025000`, 5 sessions of 10:

| | |
|---|---|
| Episodes | 50, **50/50 successful** (the policy alone scores ~52%) |
| Steps | 4,144, of which **721 corrective** (17%) |
| Correction bursts | 179, median 3 steps, clustered in the approach phase |
| Direction disagreement | median 69 deg, p25 45, p90 122 |
| Selection at tau = 0 / 45 / 90 deg | 721 / 498 / 218 steps |

**Datasets.** `D_human` and `D_blend` share every observation and the gripper channel,
differing on exactly the 711 selected corrective steps in position/orientation only:
`a_H` vs `a_exec` (median gap 4.3 cm, p90 8.1 cm). Non-corrective steps take the
policy's own action in both.

**Finetuning.** Both from `rb_smolvla_v5/025000`, identical settings — 3,000 steps
(~47 epochs over 4,094 frames), batch 64, lr 2.5e-5, `--seed 1000`, corrections only,
no base demos mixed in. Final loss: human 0.040, blend 0.036.

### Result: both finetunes are significantly WORSE than the base policy

200 episodes, `--seed 123`, identical episode sequence across all three runs, so the
comparison is paired (McNemar exact test on discordant pairs):

| Model | Success | vs base | Within 1.5 cm | Grounding | Median closest |
|---|---|---|---|---|---|
| **base v5 @ 25k** | **104/200 = 52.0%** | — | 130/200 | 76% | 1.0 cm |
| `ft_blend` @ 3k | 82/200 = 41.0% | **p = 0.032** | 115/200 | 78% | 1.2 cm |
| `ft_human` @ 3k | 75/200 = 37.5% | **p = 0.0046** | 114/200 | 70% | 1.3 cm |

| Pair | Discordant | p (exact) |
|---|---|---|
| base vs human | 64 / 35 | **0.0046** |
| base vs blend | 59 / 37 | **0.032** |
| human vs blend | 42 / 49 | 0.53 |

Two conclusions, both solid at this sample size:

1. **Finetuning on corrections alone degrades the policy** — by 11-15 points, in both
   arms. The damage shows up in the approach: within-grasping-range episodes fall from
   130/200 to ~115/200, and mean closest approach worsens by ~0.25 cm.
2. **The label choice makes no measurable difference** (37.5% vs 41.0%, p = 0.53). The
   study's central question is *not answered* by this round: with both arms degraded by
   a shared cause, there is nothing to compare.

The most likely mechanism is catastrophic forgetting: 47 epochs over 4,094 frames of
corrections-only data, against a policy trained on 37,370 frames. Earlier checkpoints
do not rescue it — at 1k steps (50 episodes) human scored 17/50 and blend 21/50, no
better than at 3k, so this is not simple over-training.

### Bug found and fixed mid-round: the human gripper channel

The first `D_human` build took `base_action.gripper` literally. The SpaceMouse reports
a *latched* open/closed state, not an intent, and a driver who steers position while
leaving grasping to the policy emits `gripper = 0.0` (open) on every step — measured,
that was all 4,144 steps. So `D_human` labelled 154 corrective steps (21% of them)
"hold the gripper OPEN" at the moments the policy was closing: it was training the
model not to grasp.

The symptom was diagnostic: that build reached grasping range more often than any
other model (34/50) while converting worst (17/50). `D_human` now takes the gripper
from the *executed* action — the policy's when the human leaves grasping alone, the
human's during the grip-hold window after a button press. The numbers above are from
the corrected rebuild.

### Label scale, checked and cleared

Suspecting the human's teleop targets were on a smaller scale than the policy's own
action distribution, measured distance-from-fingertip per label:

| Source | Median |
|---|---|
| Base demos (what v5 learned from) | 8.7 cm |
| Policy's own actions (non-corrective steps) | 13.2 cm |
| Human corrections (`D_human`) | 7.6 cm |
| Blended actions (`D_blend`) | 4.9 cm |

The human's corrections sit closer to the original demo distribution than the policy's
own outputs do, so scale mismatch does not explain the degradation. `D_blend` is the
outlier at 4.9 cm — averaging two directions ~69 deg apart shortens the vector — which
is a reason to expect the blend to be the weaker target, though the eval does not
separate them.

## 10. Correction round 2: with base-demo ballast

Round 1 showed corrections-only finetuning degrades the policy, so both arms were
rebuilt as **corrections + 100 base demos** (`merge_npy_datasets.py`, seeded so the
same 100 episodes land in both): 150 episodes, 13,273 frames, differing labels 711 =
**5.4%** of the data (was 17%). 5,000 steps (~24 epochs), otherwise identical settings.

### All five models, 200 identical episodes, paired

| Model | Training data | Success | vs base |
|---|---|---|---|
| **base v5 @ 25k** | 400 demos | **104/200 = 52.0%** | — |
| `mix_human` @ 5k | corrections + 100 demos | 98/200 = 49.0% | p = 0.60 |
| `mix_blend` @ 5k | corrections + 100 demos | 89/200 = 44.5% | p = 0.15 |
| `ft_blend` @ 3k | corrections only | 82/200 = 41.0% | p = 0.032 |
| `ft_human` @ 3k | corrections only | 75/200 = 37.5% | p = 0.0046 |

| Pair | p (McNemar exact) |
|---|---|
| `ft_human` -> `mix_human` | **0.027** — ballast significantly repaired the damage |
| `ft_blend` -> `mix_blend` | 0.53 |
| base vs `mix_human` | 0.60 |
| base vs `mix_blend` | 0.15 |
| **`mix_human` vs `mix_blend`** | **0.41** |

### What this establishes

1. **Base-demo ballast fixes the degradation.** `mix_human` is significantly better
   than `ft_human` (p = 0.027) and statistically indistinguishable from the base. The
   round-1 damage was catastrophic forgetting, as suspected.
2. **Corrections do not improve the policy either.** Neither mixed model beats base;
   the best (`mix_human`, 49.0%) is nominally *below* it. 721 corrective samples at
   5.4% of the training mix move nothing.
3. **The study's question remains unanswered** (p = 0.41), and the direction is *not*
   consistent across rounds: round 1 had blend ahead (41.0% vs 37.5%), round 2 has
   human ahead (49.0% vs 44.5%). Two rounds pointing opposite ways is the signature of
   noise, not a small real effect.

Grounding is unchanged by any of this (base 76%, both mixed arms 74%), as is the
failure profile: "never lifted the target" dominates throughout.

### Why more evaluation will not settle it

At the observed effect (discordant split 52/43), reaching 80% power needs ~874
discordant pairs, i.e. **~1,840 episodes per model — about 20 h of evaluation each**.
The answer is not more eval episodes; it is a larger effect or none at all.

Levers that would increase the effect size, rather than the sample size:

- **Raise alpha.** At alpha = 0.5 the blend sits midway, so `a_exec` and `a_H` differ
  by half the disagreement (median 4.3 cm). At alpha = 0.8 that gap is 1.6x larger,
  making the two labels genuinely different targets rather than near-neighbours.
- **More corrective share.** More correction episodes raises the 5.4% contrast without
  removing the ballast that prevents forgetting.
- **Corrections against the current policy.** These describe v5 @ 25k's mistakes; any
  finetuned policy errs differently (the DAgger argument).

## 11. Round 3 — analysis plan (written BEFORE collection)

Recorded in advance so the reported result cannot be chosen after seeing the numbers.
Rounds 1 and 2 were both null on the label question with the direction flipping
between them, so round 3 changes the two things that govern *effect size* rather than
sample size, and fixes what will be reported.

**Design.** 2 x 2: alpha in {0.5, 0.8} x label in {a_H, a_exec}. 100 correction
episodes per alpha (the existing 50 at alpha = 0.5 plus 50 new; 100 new at alpha =
0.8), each mixed with the same seeded 100 base demos. Training identical to round 2
(5,000 steps, lr 2.5e-5, seed 1000). Evaluation: 200 episodes, `--seed 123`, paired
McNemar exact test — the same protocol and the same episodes as every earlier model.

**Primary comparison (declared in advance):** `mix08_human` vs `mix08_blend`, i.e. the
label choice at alpha = 0.8. Significance at p < 0.05 on this one test.

**Secondary (reported, not used to claim the hypothesis):**
- label choice at alpha = 0.5 with 100 episodes
- the effect of alpha itself
- each arm against the base policy (104/200)

**Rationale for the two changes.** At alpha = 0.5 the executed action is the midpoint,
so `a_exec` and `a_H` differ by only half the disagreement (median 4.3 cm) — the two
labels are near-neighbours. At alpha = 0.8 that gap is 1.6x larger. Doubling the
correction episodes raises the share of training data whose labels differ from 5.4% to
~8.3%. Together they give the largest honest contrast available without changing the
task or the base policy.

**Stopping rule.** Round 3 is the last round. Whatever it shows is what gets reported;
no further rounds will be run in search of a favourable result. A null across three
rounds — at 1.6x label separation and 1.5x corrective share — is the finding.

## 12. Round 3 result: the hypothesis is refuted, with a dose-response

Executed as pre-registered in section 11. Four arms, each `rb_smolvla_v5/025000`
finetuned on 100 correction episodes mixed with the same seeded 100 base demos,
6,500 steps, lr 2.5e-5, seed 1000. 200 paired evaluation episodes, `--seed 123`, the
same episode sequence as every earlier model.

| Model | alpha | Labels | Success | Within 1.5 cm | Grounding |
|---|---|---|---|---|---|
| base v5 @ 25k | — | — | 104/200 = 52.0% | 130/200 | 76% |
| `mix08_blend` | 0.8 | a_exec | **106/200 = 53.0%** | 123/200 | 80% |
| `mix05b_blend` | 0.5 | a_exec | 98/200 = 49.0% | 114/200 | 79% |
| `mix05b_human` | 0.5 | a_H | 95/200 = 47.5% | 116/200 | 78% |
| `mix08_human` | 0.8 | a_H | **68/200 = 34.0%** | 120/200 | 72% |

**Primary test (declared in advance):** `mix08_human` vs `mix08_blend` —
discordant 31/69, **p = 0.00018**. The blend wins decisively.

| Secondary | Result |
|---|---|
| a_H vs a_exec at alpha = 0.5 | 95 vs 98, p = 0.83 — null |
| base vs `mix08_human` | **p = 0.0011** — training on a_H actively harms |
| base vs `mix08_blend` | p = 0.93 — the blend is harmless |
| base vs either alpha = 0.5 arm | p = 0.44 / 0.61 — null |

### The hypothesis was wrong, and the pattern says why

The study predicted a_H would be the stronger signal because a_exec is contaminated by
the policy's own action. The opposite holds: **a_exec is the better label, and a_H is
actively damaging — but only when the two diverge.**

| alpha | Label gap | a_H vs a_exec |
|---|---|---|
| 0.5 | 6.2 cm | null (p = 0.83) |
| 0.8 | 10.9 cm | **a_exec wins (p = 0.00018)** |

That is a dose-response: the effect is absent when the labels are close and large when
they separate, which is what a real mechanism looks like rather than noise. It also
explains rounds 1 and 2 — both ran at alpha = 0.5, where there is nothing to detect.

**The mechanism is off-policy mismatch, visible directly in the labels.** On the steps
where the two disagree:

| | Distance the label asks the arm to travel |
|---|---|
| Base demos (what the policy learned from) | 8.7 cm |
| `D_blend` at alpha = 0.8 | 3.7 cm |
| `D_human` at alpha = 0.8 | **14.1 cm** |

At alpha = 0.8 the human has only 20% authority, so they push hard and `a_H` becomes a
target far beyond anything the robot executed — 14.1 cm, well outside the demo
distribution. Those labels are paired with observations that `a_exec` produced, not
`a_H`, so the policy is taught to command large displacements that never generated the
states it sees. The resulting policy overshoots: it still reaches grasping range about
as often (120/200 vs 123/200) but converts far less, loses grounding (72% vs 80%), and
bins a distractor 9 times against the blend's 4.

`a_exec` has no such problem: it is by construction what produced the next observation,
so it is self-consistent imitation data.

### What this means for the study

1. **The answer is a_exec, not a_H** — the reverse of the hypothesis, at p = 0.00018 on
   a pre-registered test.
2. **The effect only exists when authority is skewed.** At alpha = 0.5, either label is
   equally (un)helpful. The naive intuition "raw human intent is purer" fails precisely
   where it would matter most.
3. **Neither label improves on the base policy.** The best arm matches it (53.0% vs
   52.0%, p = 0.93). Corrections at this scale do not raise success; the finding is
   about which label avoids *harm*.
4. **a_H is not "the human's intent" in an absolute-target action space.** It is the
   human's command given the authority they were granted, so its magnitude is an
   artefact of alpha. Any future use of raw human actions should rescale to the policy's
   action distribution, or use a relative/velocity action space where the human's
   command does not depend on alpha.

### Caveats

- One operator, one task, one base policy. The dose-response is within-study.
- The alpha = 0.5 fresh set had a lower intervention rate (10% of steps) than both the
  alpha = 0.8 set (24%) and round 2's alpha = 0.5 set (17%), so the alpha contrast is
  partly confounded with how much the operator intervened as they grew practised.
- Gripper authority reached the operator only partway through collection (see the fixes
  in section 9); human-held gripper steps are 5-6% in both round-3 conditions.
- The base checkpoint was selected on an underpowered 20-episode evaluation; only
  `v5/025000` was ever characterised at n = 200.

## 13. Control arm: ballast without corrections

Every finetuned arm trained on correction episodes **plus 100 of the 400 base demos**,
so "arm vs base policy" confounded two changes: adding corrections, and finetuning on a
quarter of the original data. `ballast_only` isolates them — the same seeded 100-demo
slice, same base checkpoint, lr and seed, 3,400 steps so those demos get the same ~23.7
passes they received inside the mixed arms (measured 23.71 vs 23.6). No corrections.

**Control: 96/200 = 48.0%**, and it is indistinguishable from the 400-demo base policy
(104/200, p = 0.47) — so finetuning on the 100-demo subset was not itself harmful, and
the earlier base comparisons were sound.

All six models, 200 identical episodes, paired:

| Model | Success | vs control | vs base |
|---|---|---|---|
| `mix08_blend` (alpha 0.8, a_exec) | 106/200 = 53.0% | p = 0.40 | p = 0.93 |
| base v5 @ 25k (400 demos) | 104/200 = 52.0% | p = 0.47 | — |
| `mix05b_blend` (alpha 0.5, a_exec) | 98/200 = 49.0% | p = 0.92 | p = 0.61 |
| **`ballast_only` (control)** | **96/200 = 48.0%** | — | p = 0.47 |
| `mix05b_human` (alpha 0.5, a_H) | 95/200 = 47.5% | p = 1.00 | p = 0.44 |
| `mix08_human` (alpha 0.8, a_H) | 68/200 = 34.0% | **p = 0.0051** | **p = 0.0011** |

With the control in place the secondary claims sharpen:

1. **Corrections never help.** No arm beats finetuning on the same demos *without*
   corrections (best p = 0.40). The corrective signal — 5-12% of the training data —
   adds nothing measurable at this scale.
2. **Only `a_H` at alpha = 0.8 does harm, and it is the labels, not the recipe.**
   `mix08_human` is significantly worse than the control (p = 0.0051) as well as the
   base (p = 0.0011). The control shares its recipe, its demo slice and its step budget,
   and differs only by having no correction episodes — so the damage is attributable to
   the raw-human labels themselves.
3. **Finetuning on a demo subset is harmless.** Control vs base p = 0.47, which
   retrospectively validates the round 1-3 comparisons against the base.

## 14. Two-camera round — analysis plan (written BEFORE the evaluation)

Committed before any evaluation of the two-camera correction arms, so that a later
decision to extend the experiment cannot be mistaken for choosing the sample size that
gave a favourable answer.

**Setting.** New scene and sensors on the `two-camera` branch (formerly `wrist-camera`): a gripper-mounted
wrist camera plus a square-on side camera, no front camera; bin moved to the robot's
right at (0.37, -0.18); square 16 x 16 cm block region. Base policy `twocam_v1/020000`,
240 demos. **Not comparable to the single-camera study** — both the cameras and the
scene differ.

**Collected.** 100 shared-autonomy episodes at alpha = 0.8, 100/100 successful, 7,349
steps of which 1,499 corrective (20%). Mixed with a seeded 50-demo ballast slice the
two arms differ on 1,491 steps = 13.4% of training data, median label gap 11.5 cm —
slightly stronger contrast than the single-camera round that resolved at p = 0.00018
(11.7%, 10.9 cm).

**This round is a PILOT, not the test.** 100 episodes was chosen to get feedback
quickly, not from a power calculation. Its purpose is to confirm the pipeline and
estimate the effect size.

- If an arm wins decisively, that is a result worth reporting, with the caveat that the
  sample size was not pre-specified.
- **If the comparison is null, the pilot's p-value is NOT the answer.** The honest
  response is to collect to a pre-specified N and report that analysis. Reporting a
  null pilot as a finding, or extending and then presenting the combined data as a
  single planned test, would both misstate what was done.

**Secondary.** A control arm (`tc_ballast_only`: the same seeded 50 demos, same recipe,
no corrections, step count matched for equal passes over those demos) is evaluated
alongside, so "does it beat the base" becomes "does it beat finetuning without
corrections".

**Protocol.** 200 episodes, `--seed 123`, `--two_cam`, `--n_action_steps 5`, paired
McNemar exact test — the same protocol as every earlier round, on the new scene.

## 15. Two-camera round — result: the single-camera finding replicates

Run as declared in section 14. Four models, 200 identical episodes (`--seed 123`,
`--two_cam`), paired McNemar exact tests.

**Setup.** Wrist camera (320x240, on the gripper's front face) + square-on side camera
(480x360, 44 deg down); no front camera. Bin at (0.37, -0.18), blocks in a square
16 x 16 cm region (x 0.29-0.45, y -0.06..0.10), spawn rejection 0.12 m from the bin
and 0.05 m from the gripper's start pose. Base: 240 demos (80 layouts x 3 colours),
18,463 frames, 20,000 steps (69 epochs) -> `twocam_v1/020000`. Corrections: 100
episodes at alpha = 0.8 against that checkpoint. Arms: corrections + the same seeded
50-demo ballast, 4,000 steps, lr 2.5e-5, seed 1000. Control: the 50 demos alone, 1,400
steps (23.3 epochs over them, matching the arms).

| Model | Success | Within 1.5 cm | Grounding | Distractor binned |
|---|---|---|---|---|
| `tc_ballast_only` (control) | 146/200 = 73.0% | 160 | 77% | 12 |
| `twocam_v1` (base, 240 demos) | 142/200 = 71.0% | 169 | 83% | 5 |
| `tc_ft_blend` (a_exec) | 141/200 = 70.5% | 164 | 83% | 10 |
| **`tc_ft_human` (a_H)** | **119/200 = 59.5%** | 140 | 72% | 11 |

| Comparison | Discordant | p (exact) |
|---|---|---|
| **a_H vs a_exec** | 40 / 62 | **0.037** |
| a_H vs control | 36 / 63 | **0.0086** |
| a_H vs base | 32 / 55 | **0.018** |
| a_exec vs control | 31 / 36 | 0.63 |
| a_exec vs base | 45 / 46 | 1.00 |
| control vs base | 47 / 43 | 0.75 |

### The same three findings as the single-camera study

1. **Training on the raw human action harms the policy** — 11.5 points below base,
   13.5 below the control, significant against both.
2. **Training on the executed blend is harmless** — indistinguishable from base and
   control.
3. **Corrections do not improve the policy.** No arm beats finetuning without them.

A different scene, a different camera set, a base policy 19 points stronger (71% vs
52%), and a fresh set of corrections give the same ordering. That is the most useful
thing this round shows.

### How much weight it carries

This round was declared a pilot, so its own p = 0.037 is moderate evidence, not a
pre-specified test, and six comparisons were run. Its value is as a **replication** of
the pre-registered single-camera result (section 12, p = 0.00018): same direction, same
arm harmed, same arm harmless, and the two supporting comparisons (a_H vs control, a_H
vs base) agree at p < 0.02. Read together the conclusion is well supported; read alone
this round would not be.

### The mechanism, revised

Section 12 attributed the harm to `a_H` being an out-of-distribution *magnitude*. This
round weakens that account:

| On steps where the labels differ | Single-camera | Two-camera |
|---|---|---|
| Base demos, label travel | 8.7 cm | 11.8 cm |
| `D_human` | 14.1 cm (+62%) | 15.5 cm (+31%) |
| `D_blend` | 3.7 cm (-57%) | 5.1 cm (-57%) |

Here `D_blend` sits *further* from the demo distribution than `D_human` does, yet it is
the harmless one. Magnitude alone therefore does not explain the result. What both
rounds share is **self-consistency**: `a_exec` is by construction the action that
produced the next observation, while `a_H` is a command the robot only 20% followed, so
it is paired with states it never generated. That is the explanation that survives.

### Other observations

- The base reaches 71% here against 52% on the original setup, but the scene is also
  easier (a 40% smaller block region, a closer bin), so the gain cannot be credited to
  the wrist camera alone. The residual failure is unchanged: "never lifted the target".
- Intervention rate was 9% in the first correction session and 13-26% in the other
  nine — the operator calibrating. All 100 episodes were kept; six exceed 50%
  corrective and one runs to 156 steps against a median of 71.
- Base checkpoint `020000` was the final one, chosen without a sweep; `005000`-`015000`
  were never evaluated.
- Each arm is a single training run (seed 1000); training-seed variance is not
  characterised, here or in the single-camera study.

## 16. Residual on a frozen base — analysis plan (written BEFORE any simulator run)

Sections 9-15 finetuned the base on corrections and never beat it. This tries the
alternative: leave `twocam_v1/020000` frozen and learn only the adjustment the
corrections produced, in the style of CR-DAgger. It is position-only residual learning
from SpaceMouse blending, not CR-DAgger itself (no force sensing, no compliant
kinesthetic interface, no force output).

**Data.** The 100 two-camera correction episodes of section 15 (alpha = 0.8), all 7,349
steps. Label per step: `a_exec[:3] - a_R[:3]`, which is exactly zero on the 5,850 steps
without a correction. On the 1,499 corrective steps its norm has median 3.3 cm, p90
4.4 cm, p99 5.1 cm. Orientation is left to the base (executed vs policy orientation
differs by a median of 0.8 deg), and so is the gripper.

**Model.** MLP 3 x 256 on [robot state 14 | base action 8 | base target relative to
fingertip 3], optionally with frozen ImageNet ResNet-18 features of the wrist and side
frames; corrective steps weighted 5x; lr 1e-4, dropout 0.3, weight decay 1e-3. Epoch
count chosen on 20 held-out episodes, then retrained on all 100. `scripts/
residual_extract.py`, `scripts/residual_train.py`, `source/pilot_models/
residual_policy.py`.

**What the offline analysis showed** (held-out episodes; no simulator involved):

| Predictor | Error on held-out corrective steps | Direction cosine |
|---|---|---|
| Predict nothing | 3.26 cm | — |
| One constant vector (the mean correction) | 2.74 cm | +0.49 |
| Images + state + base action (`tc_residual_images`, 8 epochs) | 2.55 cm | +0.55 |
| **State + base action only (`tc_residual_state`, 77 epochs)** | **2.39 cm** | **+0.61** |

- The corrections are mostly one consistent nudge: 89% occur before the grasp with the
  fingertip near the table (median z 2.9 cm), and their mean is (-0.6, -1.4, -1.3) cm —
  down and toward -y. 80% have a negative z component.
- Image features do not help at this data size; they make held-out error worse. The
  state-only network is therefore the configuration selected, even though the original
  design called for images.
- Both networks output ~0.9 cm down near the table and ~0 above 8 cm; the 99th
  percentile of their output is under 4 cm.

**Evaluation (fixed now).** 200 episodes, `--seed 123 --two_cam --n_action_steps 5
--max_steps 150 --debug_grounding`, the same episodes as section 15. Adjustment capped
at 5 cm per step (the p99 of the labels). One run per configuration; nothing is tuned
on simulator results.

- **Sanity:** the residual path with a zero adjustment should reproduce the base's
  142/200. If the simulator is not bitwise repeatable, the number of episodes that flip
  is the run-to-run noise floor and is reported as such.
- **Primary:** base + `tc_residual_state` vs the base (142/200), paired McNemar exact.
- **Secondary:** base + `tc_residual_images` vs the base; each residual vs the control
  (146/200); state vs images.

**Expectation, stated in advance.** The learned adjustment is largely "go about 1 cm
lower at the grasp". If the base's failures are mostly grasps closed too high, this
should convert some of them; if they are lateral misalignments the residual cannot see,
it will not, and it may break grasps that were already succeeding. Either outcome is
reported.

**Not done:** iterating (collecting new corrections with the residual running, then
retraining).

## 17. Residual on a frozen base — result: no effect beyond run-to-run noise

200 episodes each, `--seed 123 --two_cam`, run 2026-10-13 exactly as planned in
section 16. Logs: `docs/eval_logs/eval200_tc_residual_{state,zero,images}.log`.

| Run | Success | Mean adjustment | Never lifted the target | Binned a distractor |
|---|---|---|---|---|
| Base `twocam_v1/020000` (section 15) | 142/200 = 71.0% | — | 53 | 5 |
| **Base + zero adjustment (sanity)** | **157/200 = 78.5%** | 0.00 cm | 33 | 10 |
| Base + `tc_residual_state` (primary) | 151/200 = 75.5% | 0.78 cm, 31% of steps > 1 cm | 42 | 6 |
| Base + `tc_residual_images` | 155/200 = 77.5% | 0.96 cm, 39% of steps > 1 cm | 40 | 5 |
| Control `tc_ballast_only` (section 15) | 146/200 = 73.0% | — | | |

No adjustment ever reached the 5 cm cap.

Paired McNemar exact tests (same 200 episodes):

| Comparison | First only | Second only | p |
|---|---|---|---|
| zero vs base (same model twice) | 47 | 32 | 0.115 |
| **state vs base (primary)** | 41 | 32 | **0.35** |
| images vs base | 42 | 29 | 0.15 |
| state vs control | 40 | 35 | 0.64 |
| images vs control | 42 | 33 | 0.36 |
| state vs zero | 32 | 38 | 0.55 |
| images vs zero | 32 | 34 | 0.90 |
| state vs images | 28 | 32 | 0.70 |

**The sanity check failed, and that is the main finding.** With a zero adjustment the
residual path runs the same frozen base on the same 200 episodes, and nothing it does
reaches the simulator (adding an exact zero to the action; the torch RNG stream is
restored after the network is built). It still disagreed with the original base run
on 79 of 200 episodes and scored 15 episodes higher. The simulator is not repeatable
between runs (GPU physics), so an episode's outcome is close to a coin flip for
roughly 40% of layouts, and two runs of one model can differ by 15/200 (p = 0.115 —
not significant, but large).

Consequences:

- **The primary hypothesis is not supported.** The state residual is not
  distinguishable from the base (p = 0.35), and it scored *below* the zero-adjustment
  run made the same day (151 vs 157). The images residual likewise (155 vs 157).
  Neither residual measurably changes the success rate.
- **The residual does what it learned, just without effect.** It moved the arm a mean
  0.8-1.0 cm, mostly downward (mean vector (-0.3, -0.2, -0.35) cm state, (-0.37, -0.32,
  -0.44) cm images), as the offline analysis predicted. The section 16 expectation —
  "go 1 cm lower at the grasp" converts failed grasps — did not show: "never lifted"
  failures in the residual runs (42, 40) are not lower than in the zero run (33).
- **Pairing is weaker than assumed in sections 12-15.** McNemar remains a valid test
  (it uses only the discordant episodes), but a base re-run alone produces ~80
  discordant episodes, so differences under ~15/200 between single runs should not be
  read as effects. The earlier conclusions survive this: `tc_ft_human` was 23-27
  episodes below every other two-camera arm, and the single-camera round-3 gap was
  19/100 episodes with p = 0.00018. The two-camera base figure itself (142/200) is one
  draw; 142-157 is the range seen for that model.
- Whether all three same-day runs being higher than the earlier ones (151-157 vs
  141-146) is chance or something that changed between sessions is not established. A
  plain base re-run (no `--residual`) would settle it.

**Not done:** iterating the residual (collect corrections with it running, retrain);
more than one run per configuration.

## 18. Next

1. **Re-run the plain base** (no `--residual`, `--seed 123 --two_cam`, 200 episodes)
   to tell whether the same-day runs of section 17 were high by chance or because
   something changed between sessions.
2. **Update [OVERVIEW.md](OVERVIEW.md)** with the run-to-run noise finding.
3. **Optional, for accuracy rather than the research question:** the 200
   shared-autonomy episodes are 200/200 successful, so labelling every step
   `exec_action` and pooling them with the 400 demos is standard DAgger aggregation
   and the most promising route to raising success. Keep it separate from the
   label-comparison experiments.
4. Smaller, unresolved: sweep `--n_action_steps` (fixed at 5 from a 6-episode
   comparison); fix the `closest approach` contamination on successful episodes; log
   block positions in `eval_smolvla.py` to test whether failures cluster by table
   position.
