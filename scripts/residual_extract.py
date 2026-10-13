"""Build the residual-policy training set from shared-autonomy rollouts (no Isaac needed).

For every recorded step of every episode it stores

    feat_wrist, feat_side   frozen-encoder features of the two camera frames
    state                   14D robot state, exactly as logged
    base_action             a_R, the frozen base policy's 8D action
    resid                   a_exec[:3] - a_R[:3]  (zero when the human was not correcting)
    corrective              whether the human was commanding motion on that step

Features are computed once here because the encoder is frozen: training then takes
seconds instead of re-decoding 14,000 images per epoch.

    python scripts/residual_extract.py \
        --rollouts 'logs/rollouts/tc_sa_s*' --output logs/data/tc_residual.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "pilot_models"))
from residual_policy import FrozenViewEncoder  # noqa: E402

STATE_KEYS = ["obs.fingertip_pos", "obs.fingertip_quat", "obs.gripper",
              "obs.ee_linvel_fd", "obs.ee_angvel_fd"]
ACTION_KEYS = ["policy_action.fingertip_pos", "policy_action.fingertip_quat", "policy_action.gripper"]


def load_frames(rgb_dir, n):
    files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
    if len(files) != n:
        raise RuntimeError(f"{rgb_dir}: {len(files)} frames for {n} recorded steps")
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])


def encode(encoder, frames, device, batch=128):
    out = []
    for i in range(0, len(frames), batch):
        out.append(encoder(torch.from_numpy(frames[i:i + batch]).to(device)).half().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", nargs="+", default=["logs/rollouts/tc_sa_s*"],
                   help="Rollout dirs or globs written by shared_autonomy.py --two_cam.")
    p.add_argument("--output", default="logs/data/tc_residual.npz")
    p.add_argument("--all", action="store_true", help="Keep unsuccessful episodes too.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dirs = []
    for pat in args.rollouts:
        dirs += glob.glob(pat)
    dirs = sorted(set(dirs), key=lambda d: (len(d), d))
    if not dirs:
        raise SystemExit(f"no rollout dirs match {args.rollouts}")

    encoder = FrozenViewEncoder().to(args.device)
    cols = {k: [] for k in ("feat_wrist", "feat_side", "state", "base_action", "resid",
                            "corrective", "episode", "step")}
    names, ep = [], 0
    for d in dirs:
        stats_path = os.path.join(d, "meta", "stats.json")
        stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
        for ep_dir in sorted(glob.glob(os.path.join(d, "episode_*"))):
            name = os.path.basename(ep_dir)
            if not args.all and stats and not stats.get(name, {}).get("success", False):
                continue
            steps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ep_dir, "robot", "*.json")))]
            if not steps:
                continue
            if "policy_action.fingertip_pos" not in steps[0]:
                raise SystemExit(f"{ep_dir} has no policy_action — not a shared-autonomy rollout")
            n = len(steps)
            # shared_autonomy.py records BEFORE stepping, so every logged step (the last
            # one included) pairs a valid observation with the actions computed from it.
            cols["feat_wrist"].append(encode(encoder, load_frames(os.path.join(ep_dir, "camera_wrist", "rgb"), n), args.device))
            cols["feat_side"].append(encode(encoder, load_frames(os.path.join(ep_dir, "camera_side", "rgb"), n), args.device))
            cols["state"].append(np.array([sum((s[k] for k in STATE_KEYS), []) for s in steps], dtype=np.float32))
            a_r = np.array([sum((s[k] for k in ACTION_KEYS), []) for s in steps], dtype=np.float32)
            a_x = np.array([s["exec_action.fingertip_pos"] for s in steps], dtype=np.float32)
            cols["base_action"].append(a_r)
            cols["resid"].append(a_x - a_r[:, 0:3])
            cols["corrective"].append(np.array([bool(s["intervening"]) for s in steps]))
            cols["episode"].append(np.full(n, ep, dtype=np.int32))
            cols["step"].append(np.arange(n, dtype=np.int32))
            names.append(f"{os.path.basename(d)}/{name}")
            ep += 1
        print(f"  {os.path.basename(d)}: {ep} episodes so far", flush=True)

    data = {k: np.concatenate(v) for k, v in cols.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, episode_names=np.array(names), **data)

    r, c = data["resid"], data["corrective"]
    norm = np.linalg.norm(r, axis=1)
    print(f"\nWrote {args.output}: {ep} episodes, {len(r)} steps, {int(c.sum())} corrective "
          f"({100 * c.mean():.0f}%)")
    print(f"feature shape per view: {data['feat_wrist'].shape[1:]}")
    print("label norm on corrective steps (cm): "
          + "  ".join(f"p{q}={np.percentile(norm[c], q) * 100:.1f}" for q in (50, 90, 99)))
    if not np.allclose(norm[~c], 0.0, atol=1e-6):
        print("[warn] non-zero labels on non-corrective steps — check the recording")


if __name__ == "__main__":
    main()
