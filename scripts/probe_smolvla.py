"""Probe what a finetuned SmolVLA checkpoint learned, offline (no Isaac needed).

Two questions, in order of usefulness:

1. ACCURACY — on frames it was trained on, does the predicted action chunk match
   the demonstrated one? Reported in metres, against two baselines: predicting
   "hold still" (repeat the current pose) and predicting the dataset mean action.
   If the policy cannot beat those on its own training data, it never learned the
   task at all; if it beats them here but fails in sim, the problem is closed-loop
   drift / distribution shift instead.

2. GROUNDING — does the predicted reach location track the *target block*? For
   each episode we compare where the policy aims against where the instructed
   block actually is, versus where the distractors are. A policy that has learned
   colour grounding aims closest to the named block.

A sensitivity section is included last: how much the output moves when only the
prompt / only the image / only the state is swapped. Sensitivity proves an input
is *used*, not that it is used *correctly* — read it after the accuracy numbers.

Usage:
    python scripts/probe_smolvla.py \
        --checkpoint outputs/train/rb_smolvla/checkpoints/last/pretrained_model \
        --dataset_root logs/lerobot/randomblock_vla \
        --repo_id local/randomblock_vla
"""
import argparse
import json

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def predict_chunk(policy, pre, post, image_key, img, state, task, device):
    """Predicted action chunk in REAL units: (n_steps, action_dim).

    NB: dataset frames already arrive as float CHW in [0,1]; live sim frames are
    uint8 HWC, which is why SmolVLA_Pilot normalizes before this point. Keep the
    two paths consistent — a silent mismatch here would invalidate the whole probe.
    """
    policy.reset()
    frame = {image_key: img.to(device), "observation.state": state.to(device), "task": task}
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(frame))       # normalized
        chunk = post(chunk.squeeze(0))                        # -> real units
    policy.reset()
    return chunk.float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo_id", default="local/randomblock_vla")
    p.add_argument("--dataset_root", default="logs/lerobot/randomblock_vla")
    p.add_argument("--tasks_json", default="logs/data/randomblock_demos.npy.tasks.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_episodes", type=int, default=12, help="How many episodes to evaluate.")
    p.add_argument("--frame_idx", type=int, default=0, help="Timestep within each episode.")
    args = p.parse_args()

    device = torch.device(args.device)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    pol_key = sorted(policy.config.image_features)[0]
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root, video_backend="pyav")
    ds_key = sorted(k for k in ds.meta.features if k.startswith("observation.images."))[0]
    H = policy.config.n_action_steps
    print(f"policy image key {pol_key} | dataset image key {ds_key} | n_action_steps {H}")

    ep_from = ds.meta.episodes["dataset_from_index"]
    ep_to = ds.meta.episodes["dataset_to_index"]
    n_eps = min(args.n_episodes, ds.meta.total_episodes)

    # Per-episode block layout + which block is the target (from convert_demos sidecar).
    layout = {}
    try:
        meta = json.load(open(args.tasks_json))
        for k, v in meta.items():
            if "block_xy" in v and "target_idx" in v:
                layout[int(k)] = (np.array(v["block_xy"], dtype=np.float32), int(v["target_idx"]))
    except FileNotFoundError:
        print(f"[warn] {args.tasks_json} not found — skipping the grounding section")

    # ---------- 1. ACCURACY vs demonstrated actions ----------
    errs, hold_errs, mean_errs = [], [], []
    all_actions = torch.stack([ds[i]["action"] for i in range(0, len(ds), 37)])
    mean_action = all_actions.mean(0)

    rows = []
    for ep in range(n_eps):
        f, t = int(ep_from[ep]), int(ep_to[ep])
        i = f + args.frame_idx
        if i + H >= t:
            continue
        s = ds[i]
        gt = torch.stack([ds[i + k]["action"] for k in range(H)])          # (H, 8)
        pred = predict_chunk(policy, pre, post, pol_key, s[ds_key], s["observation.state"], s["task"], device)
        hold = s["observation.state"][:8].unsqueeze(0).repeat(H, 1)        # naive "stay put"

        e = (pred[:, :3] - gt[:, :3]).norm(dim=-1).mean().item()
        eh = (hold[:, :3] - gt[:, :3]).norm(dim=-1).mean().item()
        em = (mean_action[:3].unsqueeze(0) - gt[:, :3]).norm(dim=-1).mean().item()
        errs.append(e); hold_errs.append(eh); mean_errs.append(em)
        rows.append((ep, s["task"].split()[3], e, eh, em, pred))

    # Error as a function of how far ahead in the chunk we look. Decides whether the
    # model is precise enough for a 3 cm block at SHORT horizon (=> closed-loop /
    # config problem) or imprecise even one step ahead (=> underfit, needs data).
    print("\n=== 0. ERROR vs HORIZON (xyz error on training frames, metres) ===")
    hz = [1, 2, 5, 10, 20, 50]
    acc = {h: [] for h in hz}
    for ep in range(n_eps):
        f, t = int(ep_from[ep]), int(ep_to[ep])
        i = f + args.frame_idx
        if i + H >= t:
            continue
        s_ = ds[i]
        gt = torch.stack([ds[i + k]["action"] for k in range(H)])
        pred = predict_chunk(policy, pre, post, pol_key, s_[ds_key], s_["observation.state"], s_["task"], device)
        for h in hz:
            if h <= H:
                acc[h].append((pred[:h, :3] - gt[:h, :3]).norm(dim=-1).mean().item())
    print(f"  {'steps ahead':>12} {'seconds':>9} {'mean xyz err':>14}")
    for h in hz:
        if acc[h]:
            print(f"  {h:>12} {h/15.0:>9.2f} {np.mean(acc[h]):>14.4f}")
    print("  (a 3 cm block needs roughly <0.015 m to grasp reliably)")

    print("\n=== 1. ACCURACY on training frames (xyz error vs demonstrated chunk, metres) ===")
    print(f"{'ep':>4} {'colour':<7} {'policy':>8} {'hold-still':>11} {'mean-action':>12}")
    for ep, c, e, eh, em, _ in rows:
        print(f"{ep:>4} {c:<7} {e:>8.4f} {eh:>11.4f} {em:>12.4f}")
    print(f"{'':>4} {'MEAN':<7} {np.mean(errs):>8.4f} {np.mean(hold_errs):>11.4f} {np.mean(mean_errs):>12.4f}")

    # ---------- 2. GROUNDING: does it aim at the NAMED block? ----------
    if layout:
        print("\n=== 2. GROUNDING (distance from predicted reach point to each block, metres) ===")
        print(f"{'ep':>4} {'colour':<7} {'->target':>9} {'->nearest distractor':>21}  {'closest block is':>16}")
        hit = 0
        tot = 0
        for ep, c, _e, _eh, _em, pred in rows:
            if ep not in layout:
                continue
            blocks, tgt = layout[ep]                                       # (3,2), int
            # "Reach point" = the xy where the chunk dips lowest, i.e. the grasp attempt.
            lo = pred[int(pred[:, 2].argmin()), :2]
            d = np.linalg.norm(blocks - lo.numpy()[None, :], axis=1)       # dist to each block
            order = int(np.argmin(d))
            others = [d[i] for i in range(3) if i != tgt]
            hit += int(order == tgt); tot += 1
            print(f"{ep:>4} {c:<7} {d[tgt]:>9.4f} {min(others):>21.4f}  "
                  f"{'TARGET' if order == tgt else 'distractor':>16}")
        if tot:
            print(f"\n  aims at the named block in {hit}/{tot} episodes "
                  f"({100.0 * hit / tot:.0f}%)  — chance is 33%")

    # ---------- 3. SENSITIVITY (which inputs move the output at all) ----------
    by_task = {}
    for ep in range(ds.meta.total_episodes):
        by_task.setdefault(ds[int(ep_from[ep])]["task"], []).append(ep)
    picks = [(t, by_task[t][0]) for t in sorted(by_task)]
    samp = []
    for task, ep in picks:
        s = ds[int(ep_from[ep]) + args.frame_idx]
        samp.append({"task": task, "ep": ep, "img": s[ds_key], "state": s["observation.state"]})
    base = samp[0]

    def div(chunks):
        return float(np.mean([(chunks[i] - chunks[j]).abs().mean().item()
                              for i in range(len(chunks)) for j in range(i + 1, len(chunks))]))

    lang = [predict_chunk(policy, pre, post, pol_key, base["img"], base["state"], s["task"], device) for s in samp]
    vis = [predict_chunk(policy, pre, post, pol_key, s["img"], base["state"], base["task"], device) for s in samp]
    st = [predict_chunk(policy, pre, post, pol_key, base["img"], s["state"], base["task"], device) for s in samp]
    print("\n=== 3. SENSITIVITY (mean |Δaction| in real units when only one input is swapped) ===")
    print(f"  language {div(lang):.4f} | vision {div(vis):.4f} | state {div(st):.4f}")
    print("  (an input near zero is being ignored; large does not imply *correct*)")


if __name__ == "__main__":
    main()
