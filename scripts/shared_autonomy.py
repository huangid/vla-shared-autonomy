"""Shared autonomy: SmolVLA and a human SpaceMouse pilot drive the arm together.

Step 3 of the VLA shared-autonomy study. Each timestep both controllers produce a
full 8D absolute action (pos3 + quat4 + gripper1) from the SAME observation:

    a_R   SmolVLA  (the finetuned base policy)
    a_H   human    (SpaceMouse, integrated on the current fingertip pose)
    a_exec = alpha * a_R + (1 - alpha) * a_H        <- what the robot executes

and all three are logged with the observation and camera frame, giving the
``(o_t, a_H, a_R, a_exec)`` tuples the study compares. Because both actions are
computed from o_t *before* the step, the pairing carries no one-step lag.

Recording format is the one `play.py --record` writes, so the existing tools work:

    python scripts/convert_demos.py --rollout_dir logs/rollouts/shared_autonomy_RandomBlock \\
        --output logs/data/corrections.npy --append --action_prefix base_action   # D_human
    python scripts/convert_demos.py --rollout_dir logs/rollouts/shared_autonomy_RandomBlock \\
        --output logs/data/corrections_blend.npy --append --action_prefix exec_action  # D_blend

Two behaviours worth knowing before driving:

* **Idle gating** (default). The SpaceMouse reports zero when untouched, so blending
  it in unconditionally would drag the robot toward standing still and fill the log
  with `a_H = "hold"` samples that are not corrections. While you are not touching
  it the robot runs on `a_R` alone; blending starts the moment you push. Pass
  `--blend_always` to blend every step regardless.
* **Gripper.** Open/close is binary and does not average meaningfully, so it follows
  whoever is in charge: the human for `--grip_hold_steps` after any button press,
  the policy otherwise. Both raw values are logged either way.

Usage:
    python scripts/shared_autonomy.py \\
        --checkpoint outputs/train/rb_smolvla_v5/checkpoints/025000/pretrained_model \\
        --num_episodes 5 --alpha 0.5
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Shared autonomy: SmolVLA + human SpaceMouse.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="SmolVLA checkpoint dir (…/pretrained_model) or HF repo id.")
parser.add_argument("--task", type=str, default="RandomBlock",
                    help="Task name -> gym id XArm-<task>-GuidedDiffusion.")
parser.add_argument("--num_episodes", type=int, default=5,
                    help="Episodes to run in this launch. Unlike play.py this does NOT need "
                         "one Isaac start per episode.")
parser.add_argument("--alpha", type=float, default=0.5,
                    help="Blend weight on the ROBOT action: a_exec = alpha*a_R + (1-alpha)*a_H. "
                         "1.0 = robot only, 0.0 = human only.")
parser.add_argument("--blend_always", action="store_true", default=False,
                    help="Blend on every step, including while the SpaceMouse is untouched. "
                         "Off by default — see the idle-gating note in the module docstring.")
parser.add_argument("--grip_hold_steps", type=int, default=30,
                    help="Steps (15 Hz) the human keeps gripper control after pressing a button.")
parser.add_argument("--max_steps", type=int, default=300,
                    help="Safety cap per episode (~20 s at 15 Hz); the env also times out at 60 s.")
parser.add_argument("--n_action_steps", type=int, default=5,
                    help="Steps of each predicted chunk SmolVLA executes before re-planning. "
                         "5 matches the evaluation protocol; the checkpoint default of 50 is "
                         "3.3 s open-loop and cannot react to a correction.")
parser.add_argument("--policy_device", type=str, default="cuda",
                    help="Device for the SmolVLA policy (NOT --device: AppLauncher owns that).")
parser.add_argument("--image_key", type=str, default=None,
                    help="Image feature key; defaults to the checkpoint's own config.")
parser.add_argument("--seed", type=int, default=None,
                    help="Env seed. Omit for a fresh random seed each run.")
parser.add_argument("--no_record", action="store_true", default=False,
                    help="Drive without writing a rollout (for trying the feel of a blend).")
parser.add_argument("--rollout_dir", type=str, default=None,
                    help="Where to record. Default: logs/rollouts/shared_autonomy_<task>.")
parser.add_argument("--yes", "-y", action="store_true", default=False,
                    help="Overwrite an existing rollout dir without prompting.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True

import random as _random
if args_cli.seed is None:
    args_cli.seed = _random.randint(0, 2**31 - 1)
print(f"[INFO] Environment seed: {args_cli.seed}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config

import residual_copilot  # noqa: F401
from residual_copilot.pilot_models.smolvla_pilot import SmolVLA_Pilot
from residual_copilot.pilot_models.spacemouse_pilot import SpaceMousePilot

TASK_ID = f"XArm-{args_cli.task}-GuidedDiffusion"


def resolve_image_key(pilot):
    """Which image key this checkpoint expects (camera1 after --rename_map training)."""
    if args_cli.image_key:
        return args_cli.image_key
    keys = sorted(pilot.policy.config.image_features)
    if not keys:
        raise RuntimeError("checkpoint declares no image features")
    return keys[0]


def current_rgb(env_u):
    """Read the camera live — env_u.front_rgb is stale on the first step of an episode."""
    return env_u.front_camera.data.output["rgb"][0]


def refresh_camera(env_u, passes: int = 2):
    """Make the camera reflect the post-reset scene (see eval_smolvla.refresh_camera)."""
    for _ in range(passes):
        env_u.sim.render()
        env_u.scene.update(dt=env_u.physics_dt)


def policy_state(env_u):
    """14D state vector, in the order the LeRobot dataset stores it."""
    return torch.cat([
        env_u.fingertip_midpoint_pos[0], env_u.fingertip_midpoint_quat[0], env_u.gripper[0],
        env_u.ee_linvel_fd[0], env_u.ee_angvel_fd[0],
    ], dim=-1)


def build_frame(env_u, image_key):
    return {
        image_key: current_rgb(env_u),
        "observation.state": policy_state(env_u),
        "task": env_u.instructions[0],
    }


def blend(a_r, a_h, alpha, human_grip_in_charge):
    """a_exec = alpha*a_R + (1-alpha)*a_H, with quaternions nlerp'd and gripper switched.

    Position and orientation average meaningfully; a half-open gripper does not, so
    open/close follows whichever controller is in charge this step.
    """
    out = alpha * a_r + (1.0 - alpha) * a_h
    q = out[3:7]
    # Antipodal guard: q and -q are the same rotation, so blend against the nearer sign.
    if torch.dot(a_r[3:7], a_h[3:7]) < 0:
        q = alpha * a_r[3:7] - (1.0 - alpha) * a_h[3:7]
    out[3:7] = q / q.norm().clamp_min(1e-8)
    out[7] = a_h[7] if human_grip_in_charge else a_r[7]
    return out


def episode_meta(env_u):
    """Instruction / target / block layout, captured at episode START.

    DirectRLEnv resets inside the step that reports `done`, so reading these after the
    loop would describe the NEXT episode.
    """
    meta = {"instruction": env_u.instructions[0],
            "target_idx": int(env_u.target_block_idx[0].item())}
    if getattr(env_u.cfg_task, "randomize_positions", False):
        meta["block_xy"] = env_u.rb_block_xy[0].detach().cpu().tolist()
    return meta


def make_rollout_dir():
    path = os.path.abspath(args_cli.rollout_dir or
                           os.path.join("logs", "rollouts", f"shared_autonomy_{args_cli.task}"))
    if os.path.exists(path):
        print(f"[WARNING] Rollout directory already exists:\n  {path}")
        if args_cli.yes:
            print("Overwriting (--yes).")
        else:
            if input("Overwrite? [y/N] ").strip().lower() != "y":
                raise SystemExit("Aborting.")
        import shutil
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def timestep_record(obs, a_h, a_r, a_exec, qpos, d_t, intervening):
    """One logged timestep. obs.* / base_action.* / action.* match play.py's layout.

    base_action.* is the HUMAN command (so convert_demos' default builds D_human) and
    exec_action.* is what ran; policy_action.* is the VLA's own proposal. action.* is
    kept for format compatibility and is the same as exec_action.*.
    """
    return {
        "obs.fingertip_pos":           obs[0:3].tolist(),
        "obs.fingertip_quat":          obs[3:7].tolist(),
        "obs.gripper":                 obs[7:8].tolist(),
        "obs.fingertip_pos_rel_fixed": obs[8:11].tolist(),
        "obs.fingertip_pos_rel_held":  obs[11:14].tolist(),
        "obs.ee_linvel_fd":            obs[14:17].tolist(),
        "obs.ee_angvel_fd":            obs[17:20].tolist(),
        "obs.qpos":                    qpos.tolist(),
        "base_action.fingertip_pos":   a_h[0:3].tolist(),
        "base_action.fingertip_quat":  a_h[3:7].tolist(),
        "base_action.gripper":         a_h[7:8].tolist(),
        "policy_action.fingertip_pos":  a_r[0:3].tolist(),
        "policy_action.fingertip_quat": a_r[3:7].tolist(),
        "policy_action.gripper":        a_r[7:8].tolist(),
        "exec_action.fingertip_pos":   a_exec[0:3].tolist(),
        "exec_action.fingertip_quat":  a_exec[3:7].tolist(),
        "exec_action.gripper":         a_exec[7:8].tolist(),
        "action.fingertip_pos":        a_exec[0:3].tolist(),
        "action.fingertip_quat":       a_exec[3:7].tolist(),
        "action.gripper":              a_exec[7:8].tolist(),
        "action.qpos":                 qpos.tolist(),
        "disagreement":                float(d_t),
        "intervening":                 bool(intervening),
    }


@hydra_task_config(TASK_ID, "rl_games_cfg_entry_point")
def run(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.vis.store_rgb = True
    # Markers render into the camera frame and the held marker sits on the target
    # block — an answer key the training data never had.
    env_cfg.vis.vis_obs = False
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.pilot_model = "knn"     # env-internal pilot is unused; a_H comes from our own instance
    env_cfg.pilot_type = "none"

    env = gym.make(TASK_ID, cfg=env_cfg)
    env_u = env.unwrapped

    robot = SmolVLA_Pilot(args_cli.checkpoint, device=args_cli.policy_device,
                          n_action_steps=args_cli.n_action_steps)
    image_key = resolve_image_key(robot)
    human = SpaceMousePilot(num_envs=1, device=str(env_u.device))

    rollout_path = None if args_cli.no_record else make_rollout_dir()
    alpha = args_cli.alpha
    print(f"[INFO] alpha={alpha:.2f}  (a_exec = {alpha:.2f}*robot + {1 - alpha:.2f}*human)")
    print(f"[INFO] idle gating: {'OFF (always blending)' if args_cli.blend_always else 'ON'}")
    print(f"[INFO] image key: {image_key} | replanning every "
          f"{robot.policy.config.n_action_steps} steps")

    obs_dict, _ = env.reset()
    robot.reset()
    refresh_camera(env_u)

    ep_stats = {}
    for ep in range(args_cli.num_episodes):
        meta = episode_meta(env_u)
        ep_name = f"episode_{ep:04d}"
        print(f"\n=== {ep_name}  |  {meta['instruction']} ===")

        buffers, frames = [], []
        grip_owner_until = -1
        prev_grip_cmd = None
        steps = 0
        success = False
        ended = False           # env reported terminated/truncated (so it already auto-reset)

        while steps < args_cli.max_steps:
            obs_t = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            obs_np = obs_t[0].detach().cpu().numpy()
            rgb = current_rgb(env_u)

            with torch.inference_mode():
                a_r = robot.act(build_frame(env_u, image_key)).to(env_u.device).view(-1).float()
                a_h = human.get_actions(
                    None, env_u.fingertip_midpoint_pos, env_u.fingertip_midpoint_quat,
                    env_u.gripper,
                )[0].to(env_u.device).float()

            # SpaceMouse reports gripper as -1/+1; the env's action space is [0, 1].
            a_h = a_h.clone()
            a_h[7] = 1.0 if float(a_h[7]) > 0 else 0.0
            if prev_grip_cmd is not None and float(a_h[7]) != prev_grip_cmd:
                grip_owner_until = steps + args_cli.grip_hold_steps
            prev_grip_cmd = float(a_h[7])

            intervening = args_cli.blend_always or not bool(getattr(human, "is_idle", True))
            d_t = float(torch.linalg.vector_norm(a_h[0:3] - a_r[0:3]))
            a_exec = blend(a_r, a_h, alpha, steps < grip_owner_until) if intervening else a_r.clone()

            if rollout_path is not None:
                buffers.append(timestep_record(
                    obs_np, a_h.cpu().numpy(), a_r.cpu().numpy(), a_exec.cpu().numpy(),
                    env_u.qpos_targets[0].detach().cpu().numpy(), d_t, intervening))
                frames.append(rgb.cpu().numpy().astype(np.uint8))

            with torch.inference_mode():
                obs_dict, _rew, terminated, truncated, _info = env.step(a_exec.view(1, -1))
            steps += 1
            print(f"\r[{ep_name}] step {steps:4d}  d_t {d_t*100:5.1f} cm  "
                  f"{'HUMAN' if intervening else 'robot'}   ", end="", flush=True)

            if bool((terminated | truncated)[0].item()):
                success = bool(env_u.ep_succeeded[0].item())
                ended = True
                break

        n_corr = sum(1 for b in buffers if b["intervening"])
        ds = [b["disagreement"] for b in buffers if b["intervening"]]
        print(f"\n[{ep_name}] {'SUCCESS' if success else 'fail'} in {steps} steps | "
              f"human intervened on {n_corr}/{steps} steps"
              + (f" | disagreement median {np.median(ds)*100:.1f} cm, "
                 f"p90 {np.percentile(ds, 90)*100:.1f} cm" if ds else ""))

        if rollout_path is not None:
            ep_dir = os.path.join(rollout_path, ep_name)
            os.makedirs(os.path.join(ep_dir, "robot"), exist_ok=True)
            os.makedirs(os.path.join(ep_dir, "camera_0", "rgb"), exist_ok=True)
            for t, entry in enumerate(buffers):
                with open(os.path.join(ep_dir, "robot", f"{t:06d}.json"), "w") as f:
                    json.dump(entry, f, indent=2)
            for t, img in enumerate(frames):
                cv2.imwrite(os.path.join(ep_dir, "camera_0", "rgb", f"{t:06d}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            ep_stats[ep_name] = {**meta, "success": success, "steps": steps,
                                 "alpha": alpha, "corrective_steps": n_corr}

        if not ended:
            # Hit the step cap without terminating: reset explicitly. Must be inside
            # inference_mode — the env's buffers were created under it and reset()
            # updates them in place.
            with torch.inference_mode():
                obs_dict, _ = env.reset()
        robot.reset()
        human.reset()
        refresh_camera(env_u)

    if rollout_path is not None:
        meta_dir = os.path.join(rollout_path, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        with open(os.path.join(meta_dir, "infos.json"), "w") as f:
            json.dump({"task": args_cli.task, "checkpoint": args_cli.checkpoint,
                       "alpha": alpha, "blend_always": args_cli.blend_always,
                       "n_action_steps": robot.policy.config.n_action_steps,
                       "seed": args_cli.seed}, f, indent=2)
        with open(os.path.join(meta_dir, "stats.json"), "w") as f:
            json.dump(ep_stats, f, indent=2)
        n_succ = sum(1 for v in ep_stats.values() if v["success"])
        print(f"\n[INFO] recorded {len(ep_stats)} episodes to {rollout_path} "
              f"({n_succ} successful)")

    env.close()


if __name__ == "__main__":
    run()
    simulation_app.close()
