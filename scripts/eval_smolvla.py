"""Evaluate a (finetuned) SmolVLA policy on XArm-RandomBlock-GuidedDiffusion.

Runs N fresh episodes with no human input: each step the env's front-camera RGB,
14D robot state, and the per-episode instruction are fed to SmolVLA, which
returns an 8D absolute end-effector action. Reports the overall success rate and
a per-colour breakdown.

Usage:
    python scripts/eval_smolvla.py --checkpoint outputs/train/rb_smolvla/checkpoints/last/pretrained_model
    python scripts/eval_smolvla.py --checkpoint <hf_user>/randomblock_smolvla --num_episodes 50
"""
import argparse
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate SmolVLA on RandomBlock.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="SmolVLA checkpoint dir (…/pretrained_model) or HF repo id.")
parser.add_argument("--task", type=str, default="RandomBlock",
                    help="Task name -> gym id XArm-<task>-GuidedDiffusion.")
parser.add_argument("--num_episodes", type=int, default=30)
parser.add_argument("--image_key", type=str, default="observation.images.front",
                    help="Image feature key the checkpoint was trained with.")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=None,
                    help="Env seed. Omit for a fresh random seed each run.")
parser.add_argument("--no_rand", action="store_true", default=False,
                    help="Disable admittance/param domain randomization.")
parser.add_argument("--max_steps", type=int, default=600,
                    help="Safety cap on steps per episode (env also has its own time-out).")
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

TASK_ID = f"XArm-{args_cli.task}-GuidedDiffusion"


def build_frame(env_u):
    state = torch.cat([
        env_u.fingertip_midpoint_pos[0],
        env_u.fingertip_midpoint_quat[0],
        env_u.gripper[0],
        env_u.ee_linvel_fd[0],
        env_u.ee_angvel_fd[0],
    ], dim=-1)  # (14,)
    return {
        args_cli.image_key: env_u.front_rgb[0],          # (H, W, 3) uint8
        "observation.state": state,
        "task": env_u.instructions[0],
    }


@hydra_task_config(TASK_ID, "rl_games_cfg_entry_point")
def run(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.vis.store_rgb = True
    env_cfg.pilot_model = "knn"      # base action is unused; kNN just needs its local demo file
    env_cfg.pilot_type = "none"
    if args_cli.no_rand:
        env_cfg.dmr.rand_ctrl = False
        env_cfg.dmr.aug_data = False

    env = gym.make(TASK_ID, cfg=env_cfg)
    env_u = env.unwrapped
    if not getattr(env_u.cfg_task, "single_target", False):
        print("[WARN] task is not single_target — instructions/target may be undefined.")

    pilot = SmolVLA_Pilot(args_cli.checkpoint, device=args_cli.device)

    obs, _ = env.reset()
    pilot.reset()
    if not hasattr(env_u, "front_rgb"):
        raise RuntimeError("env has no front_rgb — camera did not initialise (store_rgb / --enable_cameras).")

    results = []
    steps_in_ep = 0
    while simulation_app.is_running() and len(results) < args_cli.num_episodes:
        tgt_pre = int(env_u.target_block_idx[0].item())
        instr = env_u.instructions[0]

        with torch.inference_mode():
            action = pilot.act(build_frame(env_u)).to(env_u.device).view(1, -1)
            obs, _rew, terminated, truncated, _info = env.step(action)

        steps_in_ep += 1
        done = bool((terminated | truncated)[0].item()) or steps_in_ep >= args_cli.max_steps

        if done:
            succ = bool(env_u.ep_succeeded[0].item())
            color = env_u.cfg_task.target_colors[tgt_pre]
            results.append({"color": color, "success": succ, "steps": steps_in_ep})
            print(f"[{len(results):3d}/{args_cli.num_episodes}] {color:<5s} "
                  f"{'SUCCESS' if succ else 'fail   '}  ({steps_in_ep} steps)  «{instr}»")
            if steps_in_ep >= args_cli.max_steps and not (terminated | truncated)[0].item():
                # force the env to reset a hung episode
                env.reset()
            pilot.reset()
            steps_in_ep = 0

    n = len(results)
    n_succ = sum(r["success"] for r in results)
    print("\n" + "=" * 48)
    print(f"  SmolVLA @ {args_cli.checkpoint}")
    print(f"  overall success: {n_succ}/{n} = {100.0 * n_succ / max(n, 1):.1f}%")
    by_color = defaultdict(lambda: [0, 0])
    for r in results:
        by_color[r["color"]][0] += int(r["success"])
        by_color[r["color"]][1] += 1
    for color, (s, tot) in sorted(by_color.items()):
        print(f"    {color:<6s} {s}/{tot} = {100.0 * s / tot:.1f}%")
    print("=" * 48)

    env.close()


if __name__ == "__main__":
    run()
    simulation_app.close()
