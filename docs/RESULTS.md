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

## 9. Next

Step 3 of the plan: shared autonomy with logging. Run the SmolVLA policy and the
SpaceMouse pilot simultaneously, execute `a_exec = α·a_R + (1−α)·a_H`, and log
`(o_t, a_H, a_R, a_exec)` per step. Then select corrective samples by disagreement
`d_t = ‖a_H − a_R‖ > τ`, build `D_blend` and `D_human`, and fine-tune from
`rb_smolvla_v5/checkpoints/025000` for the comparison. Not yet implemented.
