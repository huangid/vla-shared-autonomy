"""Convert a kNN-format .npy demo set (see convert_demos.py) into a LeRobot dataset.

Input .npy is a dict: {episode_idx: {key: (T, D) array, ...}} with keys
obs.fingertip_pos, obs.fingertip_quat, obs.gripper, obs.fingertip_pos_rel_fixed,
obs.fingertip_pos_rel_held, obs.ee_linvel_fd, obs.ee_angvel_fd,
action.fingertip_pos, action.fingertip_quat, action.gripper.

Two output modes:

* State-only (default) — 14D observation.state / 6D observation.environment_state
  / 8D action, matching source/pilot_models/config/state_dp_cfg.json and
  xarm_env.py::_get_bc_pilot_action. Works for any task.

      python scripts/npy_to_lerobot.py \
          --input logs/data/threeblocks_seq_demos.npy \
          --repo_id <hf_user>/threeblocks_expert_50 \
          --task "stack three blocks in the bin"

* With images (--images) — additionally attaches observation.images.<name> from
  the recorded RGB frames and a per-episode `task` string. Requires the sidecar
  "<input>.tasks.json" that convert_demos.py writes (it carries each episode's
  instruction plus the rollout dir / episode name the frames live in). This is
  the SmolVLA training format.

      python scripts/npy_to_lerobot.py \
          --input logs/data/randomblock_demos.npy \
          --repo_id <hf_user>/randomblock_vla_100 \
          --images
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 15  # sim.dt (1/120) * decimation (8) in xarm_env_cfg.py -> 15 Hz control rate

STATE_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": {
            "axes": [
                "fingertip_x", "fingertip_y", "fingertip_z",
                "fingertip_qw", "fingertip_qx", "fingertip_qy", "fingertip_qz",
                "gripper",
                "linvel_x", "linvel_y", "linvel_z",
                "angvel_x", "angvel_y", "angvel_z",
            ]
        },
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": [
                "rel_fixed_x", "rel_fixed_y", "rel_fixed_z",
                "rel_held_x", "rel_held_y", "rel_held_z",
            ]
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"axes": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]},
    },
}


def episode_state_action(ep: dict):
    """Return (state (T,14), env_state (T,6), action (T,8)) for one episode."""
    state = np.concatenate([
        ep["obs.fingertip_pos"],
        ep["obs.fingertip_quat"],
        ep["obs.gripper"],
        ep["obs.ee_linvel_fd"],
        ep["obs.ee_angvel_fd"],
    ], axis=-1).astype(np.float32)

    env_state = np.concatenate([
        ep["obs.fingertip_pos_rel_fixed"],
        ep["obs.fingertip_pos_rel_held"],
    ], axis=-1).astype(np.float32)

    action = np.concatenate([
        ep["action.fingertip_pos"],
        ep["action.fingertip_quat"],
        ep["action.gripper"],
    ], axis=-1).astype(np.float32)

    assert state.shape[-1] == 14 and env_state.shape[-1] == 6 and action.shape[-1] == 8, (
        f"unexpected shapes: state={state.shape}, env_state={env_state.shape}, action={action.shape}"
    )
    return state, env_state, action


def frames_dir(meta: dict, camera_subdir: str) -> Path:
    """Where this episode's RGB frames live.

    Prefers the persistent copy convert_demos.py makes ("frames"); falls back to
    the original rollout dir for sidecars written before that existed — note those
    go stale as soon as another demo is recorded into the same rollout dir.
    """
    if meta.get("frames"):
        return Path(meta["frames"])
    return Path(meta["source"]) / meta["episode"] / camera_subdir


def load_episode_images(rgb_dir: Path, n_frames: int) -> np.ndarray:
    """Load the first n_frames RGB frames from rgb_dir as a (n_frames, H, W, 3) uint8 array."""
    files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    files = sorted(set(files))
    if len(files) < n_frames:
        raise FileNotFoundError(
            f"{rgb_dir}: found {len(files)} frames but the episode has {n_frames} state steps."
        )
    if len(files) - n_frames > 2:
        print(f"  WARNING: {rgb_dir.parent.name} has {len(files)} frames vs {n_frames} state steps "
              f"(using first {n_frames}).")
    imgs = [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files[:n_frames]]
    return np.stack(imgs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to kNN-format .npy demo file.")
    parser.add_argument("--repo_id", type=str, required=True, help="Output LeRobot dataset repo id.")
    parser.add_argument("--root", type=str, default=None, help="Local root dir (default: HF cache).")
    parser.add_argument("--task", type=str, default=None,
                        help="Fallback natural-language task, used for episodes without a per-episode "
                             "instruction in the sidecar. Required in state-only mode.")
    parser.add_argument("--images", action="store_true", default=False,
                        help="Attach observation.images.<name> from recorded RGB frames (SmolVLA format). "
                             "Requires the '<input>.tasks.json' sidecar from convert_demos.py.")
    parser.add_argument("--image_name", type=str, default="front",
                        help="Camera name -> feature key observation.images.<image_name>.")
    parser.add_argument("--camera_subdir", type=str, default="camera_0/rgb",
                        help="Per-episode subdir holding the RGB frames.")
    parser.add_argument("--image_dtype", type=str, choices=["video", "image"], default="video")
    parser.add_argument("--robot_type", type=str, default="xarm7")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    args = parser.parse_args()

    demos = np.load(args.input, allow_pickle=True).item()

    sidecar = {}
    sidecar_path = Path(str(args.input) + ".tasks.json")
    if sidecar_path.exists():
        for k, v in json.loads(sidecar_path.read_text()).items():
            sidecar[int(k)] = {"task": v} if isinstance(v, str) else v
    elif args.images:
        raise FileNotFoundError(
            f"--images needs the sidecar {sidecar_path} (written by convert_demos.py) to locate frames."
        )

    features = dict(STATE_FEATURES)
    image_key = f"observation.images.{args.image_name}"

    # In image mode the VLA grounds spatial relations from pixels + language, so the
    # privileged relative-position vector is dropped (it also isn't available at
    # eval time without ground-truth object poses).
    if args.images:
        features.pop("observation.environment_state", None)

    if args.images:
        # Probe frame size from the first episode's first frame.
        probe = sidecar[sorted(demos.keys())[0]]
        probe_dir = frames_dir(probe, args.camera_subdir)
        probe_files = sorted(probe_dir.glob("*.jpg")) + sorted(probe_dir.glob("*.png"))
        if not probe_files:
            raise FileNotFoundError(f"no RGB frames in {probe_dir}")
        h, w = np.asarray(Image.open(sorted(set(probe_files))[0]).convert("RGB")).shape[:2]
        features[image_key] = {
            "dtype": args.image_dtype,
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
        print(f"[INFO] images: {image_key}  {w}x{h}  dtype={args.image_dtype}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=features,
        root=args.root,
        robot_type=args.robot_type,
        use_videos=(args.images and args.image_dtype == "video"),
    )

    for ep_idx in sorted(demos.keys()):
        meta = sidecar.get(ep_idx, {})
        task = meta.get("task") or args.task
        if task is None:
            raise ValueError(
                f"Episode {ep_idx} has no instruction in the sidecar and no --task fallback was given."
            )

        state, env_state, action = episode_state_action(demos[ep_idx])
        n = state.shape[0]

        images = None
        if args.images:
            images = load_episode_images(frames_dir(meta, args.camera_subdir), n)

        for t in range(n):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "task": task,
            }
            if "observation.environment_state" in features:
                frame["observation.environment_state"] = env_state[t]
            if images is not None:
                frame[image_key] = images[t]
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"Saved episode {ep_idx} ({n} steps) — task: {task!r}")

    print(f"Converted {len(demos)} episodes into LeRobot dataset at {dataset.root}")

    if args.push_to_hub:
        dataset.push_to_hub()
        print(f"Pushed to hub: {args.repo_id}")


if __name__ == "__main__":
    main()
