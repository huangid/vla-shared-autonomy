"""Convert recorded JSON rollouts into the kNN .npy dataset format.

Reads logs/rollouts/<dir>/episode_XXXX/robot/*.json, keeps only successful
episodes (per meta/stats.json), and writes a single .npy dict the kNN pilot
can load.

Recorded RGB frames are **copied** into a persistent store next to the .npy:

    <output_stem>_frames/episode_<npy_idx:05d>/{000000.jpg, ...}

This matters because `play.py --record` wipes and reuses the same rollout
directory on every run, so a sidecar that merely *pointed* at the rollout dir
would go stale as soon as the next demo was recorded.

Alongside the .npy it writes a sidecar "<output>.tasks.json" mapping each npy
episode index to:
  - task:       the natural-language instruction (RandomBlock single-target)
  - frames:     path of the copied per-episode RGB directory
  - steps:      number of timesteps kept (== number of frames copied)
  - block_xy:   all three block spawn positions, when recorded
  - target_idx: index of the instructed block, when recorded
  - source/episode: the originating rollout dir + episode name (provenance only)

The sidecar is kept out of the .npy because the kNN loader tensorizes every key
in an episode dict. npy_to_lerobot.py reads it to attach per-episode
instructions and to locate the RGB frames.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--rollout_dir", type=str, required=True,
                    help="Path to logs/rollouts/<dir>")
parser.add_argument("--output", type=str, required=True,
                    help="Output .npy path")
parser.add_argument("--all", action="store_true",
                    help="Include all episodes, not just successful ones")
parser.add_argument("--append", action="store_true",
                    help="Append to existing output instead of overwriting.")
parser.add_argument("--camera_subdir", type=str, default="camera_0/rgb",
                    help="Per-episode subdir holding the recorded RGB frames.")
parser.add_argument("--no_frames", action="store_true",
                    help="Skip copying RGB frames (state/action only).")
parser.add_argument("--no_clamp_actions", action="store_true",
                    help="Store the raw pilot command instead of clamping it to the executable "
                         "workspace. Off by default: the sim clamps every target, so unclamped "
                         "commands (e.g. z below the table) are targets that were never executed.")
parser.add_argument("--keep_last_step", action="store_true",
                    help="Keep the final recorded timestep. Off by default because DirectRLEnv "
                         "resets inside the step that reports `done` and _get_observations() runs "
                         "after that reset, so the last recorded obs/action/frame actually belong "
                         "to the NEXT episode.")
args = parser.parse_args()

# Must match XArmEnv._apply_residual / XArmEnvGuidedDiffusion._pre_physics_step.
POS_CLAMP_LO = np.array([-0.2, -0.6, 0.0], dtype=np.float32)
POS_CLAMP_HI = np.array([1.0, 0.6, 0.8], dtype=np.float32)
GRIP_CLAMP_LO, GRIP_CLAMP_HI = 0.0, 1.0

# Episodes shorter than this are dropped — a near-empty trajectory is a recording
# glitch, and it would produce a zero/one-frame LeRobot episode downstream.
MIN_STEPS = 5

rollout = Path(args.rollout_dir).resolve()
output = Path(args.output)
frames_root = output.parent / f"{output.stem}_frames"

# Which episodes succeeded, plus instruction / layout metadata — from meta/stats.json.
stats = {}
meta_stats = rollout / "meta" / "stats.json"
if meta_stats.exists():
    stats = json.loads(meta_stats.read_text())

OBS_KEYS = [
    "obs.fingertip_pos", "obs.fingertip_quat", "obs.gripper",
    "obs.fingertip_pos_rel_fixed", "obs.fingertip_pos_rel_held",
    "obs.ee_linvel_fd", "obs.ee_angvel_fd",
]

# base_action.* is the pilot command computed from THIS timestep's observation, so
# (obs_t, base_action_t) is the correctly paired BC sample. (The recording also stores
# action.* = what the env executed during step t, but that derives from base_action_{t-1}
# and is therefore lagged by one — do not use it as the BC target.)
ACT_SRC_KEYS = ["base_action.fingertip_pos", "base_action.fingertip_quat", "base_action.gripper"]
# Target keys the kNN format expects
ACT_DST_KEYS = ["action.fingertip_pos", "action.fingertip_quat", "action.gripper"]
ALL_SRC_KEYS = OBS_KEYS + ACT_SRC_KEYS

data = {}
start_idx = 0
tasks_path = Path(str(args.output) + ".tasks.json")
new_meta = {}
existing_meta = {}
if args.append and Path(args.output).exists():
    existing = np.load(args.output, allow_pickle=True).item()
    data = dict(existing)
    start_idx = max(existing.keys()) + 1 if existing else 0
    print(f"Appending to {len(data)} existing episodes.")
    if tasks_path.exists():
        for k, v in json.loads(tasks_path.read_text()).items():
            existing_meta[int(k)] = {"task": v} if isinstance(v, str) else v  # tolerate old flat format

ep_dirs = sorted([d for d in rollout.iterdir() if d.name.startswith("episode_")])
kept = 0
skipped_duplicate = 0
for ep_dir in ep_dirs:
    ep_name = ep_dir.name
    ep_stats = stats.get(ep_name, {})
    if not args.all and stats and not bool(ep_stats.get("success", False)):
        continue  # skip failed episodes

    robot_dir = ep_dir / "robot"
    step_files = sorted(robot_dir.glob("*.json"))
    if not step_files:
        continue

    buffers = {k: [] for k in ALL_SRC_KEYS}
    kept_step_idx = []  # original timestep index of each kept step, for frame alignment
    for t, sf in enumerate(step_files):
        entry = json.loads(sf.read_text())
        if not all(k in entry for k in ALL_SRC_KEYS):
            continue  # skip incomplete timesteps (e.g. terminal step)
        for k in ALL_SRC_KEYS:
            buffers[k].append(entry[k])
        kept_step_idx.append(t)

    # Drop the terminal sample: DirectRLEnv.step() calls _reset_idx() before
    # _get_observations(), so on the step that reports `done` the recorded
    # observation, action and RGB frame are all from the NEXT episode (they read
    # back as the reset start pose with the gripper open).
    if not args.keep_last_step and len(kept_step_idx) > 1:
        for k in ALL_SRC_KEYS:
            buffers[k].pop()
        kept_step_idx.pop()

    if len(kept_step_idx) < MIN_STEPS:
        print(f"  SKIP {ep_name}: only {len(kept_step_idx)} usable timesteps.")
        continue

    # Build episode dict, renaming base_action.* -> action.*
    ep_dict = {}
    for k in OBS_KEYS:
        ep_dict[k] = np.array(buffers[k], dtype=np.float32)
    for src, dst in zip(ACT_SRC_KEYS, ACT_DST_KEYS):
        ep_dict[dst] = np.array(buffers[src], dtype=np.float32)

    # The env clamps every commanded target, so an unclamped pilot command (e.g. the
    # SpaceMouse driving z below the table) is a target the sim never executed. Training
    # on those teaches the policy to command into the table — and the eval env applies
    # the same clamp, so the two action spaces must agree.
    if not args.no_clamp_actions:
        ep_dict["action.fingertip_pos"] = np.clip(
            ep_dict["action.fingertip_pos"], POS_CLAMP_LO, POS_CLAMP_HI
        )
        ep_dict["action.gripper"] = np.clip(
            ep_dict["action.gripper"], GRIP_CLAMP_LO, GRIP_CLAMP_HI
        )

    # Guard against converting the same recording twice: play.py --record always
    # reuses the same rollout dir and episode name, so (source, episode) can't
    # identify an episode — fingerprint the trajectory instead.
    fingerprint = hashlib.sha1(
        np.ascontiguousarray(ep_dict["obs.fingertip_pos"]).tobytes()
    ).hexdigest()[:16]
    if any(m.get("fingerprint") == fingerprint for m in existing_meta.values()):
        print(f"  SKIP {ep_name}: already in {args.output} (fingerprint {fingerprint}).")
        skipped_duplicate += 1
        continue

    npy_idx = start_idx + kept
    data[npy_idx] = ep_dict

    entry_meta = {
        "source": str(rollout),
        "episode": ep_name,
        "steps": len(kept_step_idx),
        "fingerprint": fingerprint,
    }
    for key in ("instruction", "block_xy", "target_idx"):
        if key in ep_stats:
            entry_meta["task" if key == "instruction" else key] = ep_stats[key]

    # Copy the RGB frames out of the volatile rollout dir into a persistent store,
    # renumbered to match the kept timesteps one-for-one.
    rgb_src = ep_dir / args.camera_subdir
    if not args.no_frames and rgb_src.is_dir():
        src_frames = sorted(rgb_src.glob("*.jpg")) + sorted(rgb_src.glob("*.png"))
        src_frames = sorted(set(src_frames))
        dst_dir = frames_root / f"episode_{npy_idx:05d}"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        for new_t, orig_t in enumerate(kept_step_idx):
            if orig_t >= len(src_frames):
                break
            shutil.copy2(src_frames[orig_t], dst_dir / f"{new_t:06d}{src_frames[orig_t].suffix}")
            n_copied += 1
        entry_meta["frames"] = str(dst_dir)
        entry_meta["n_frames"] = n_copied
        if n_copied != len(kept_step_idx):
            print(f"  WARNING: {ep_name}: copied {n_copied} frames for {len(kept_step_idx)} steps.")

    new_meta[npy_idx] = entry_meta
    kept += 1

if kept == 0 and skipped_duplicate:
    print(f"Nothing to do: all {skipped_duplicate} episode(s) were already converted. "
          f"{args.output} still has {len(data)} episodes.")
elif kept == 0:
    print("WARNING: no episodes kept. Use --all to include unsuccessful ones, "
          "or check that episodes succeeded.")
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data, allow_pickle=True)
    print(f"Added {kept} new episodes. Total now: {len(data)} in {args.output}")
    if not args.no_frames:
        print(f"Copied RGB frames into {frames_root}/")

    all_meta = {**existing_meta, **new_meta}
    tasks_path.write_text(json.dumps({str(k): v for k, v in sorted(all_meta.items())}, indent=2))
    print(f"Wrote per-episode task/provenance for {len(all_meta)} episodes to {tasks_path}")
