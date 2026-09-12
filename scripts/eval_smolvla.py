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
parser.add_argument("--checkpoint", type=str, default=None,
                    help="SmolVLA checkpoint dir (…/pretrained_model) or HF repo id.")
parser.add_argument("--task", type=str, default="RandomBlock",
                    help="Task name -> gym id XArm-<task>-GuidedDiffusion.")
parser.add_argument("--num_episodes", type=int, default=30)
parser.add_argument("--image_key", type=str, default=None,
                    help="Image feature key the checkpoint was trained with. Defaults to reading it "
                         "off the checkpoint's own config — training with --rename_map (e.g. mapping "
                         "'front' to 'camera1' for smolvla_base) changes this, and a mismatch would "
                         "otherwise fail at the first inference step.")
# NB: not --device. AppLauncher.add_app_launcher_args() injects its own --device
# for the sim, and errors out on a collision.
parser.add_argument("--policy_device", type=str, default="cuda",
                    help="Device to run the SmolVLA policy on.")
parser.add_argument("--seed", type=int, default=None,
                    help="Env seed. Omit for a fresh random seed each run.")
parser.add_argument("--no_rand", action="store_true", default=False,
                    help="Disable admittance/param domain randomization.")
parser.add_argument("--max_steps", type=int, default=600,
                    help="Safety cap on steps per episode (env also has its own time-out).")
parser.add_argument("--n_action_steps", type=int, default=None,
                    help="How many steps of each predicted action chunk to execute before "
                         "re-planning. The checkpoint's default is 50 (3.3 s at 15 Hz) — longer than "
                         "half a typical episode, so it cannot react to a missed grasp. Lowering to "
                         "10-15 re-plans ~1x/second and needs NO retraining. Try 50 vs 10 to separate "
                         "'bad perception' from 'too much open-loop drift'.")
parser.add_argument("--debug_grounding", action="store_true", default=False,
                    help="At each episode start, report where the first predicted action chunk aims "
                         "versus the true block positions. Directly comparable to probe_smolvla.py's "
                         "grounding number, but computed on LIVE sim observations — so it separates "
                         "'wrong from the very first observation' (visual domain gap) from "
                         "'drifts later' (compounding closed-loop error).")
parser.add_argument("--dump_only", action="store_true", default=False,
                    help="Render the scene and save camera frames, then exit — no policy is loaded. "
                         "Use this to diff a live sim frame against a training frame BEFORE spending "
                         "hours on training. Writes <dump_frame> (settled) and a _t0 variant taken "
                         "immediately after reset, so an unsettled first render is also visible.")
parser.add_argument("--settle_steps", type=int, default=5,
                    help="Hold-still steps before the settled dump frame is captured.")
parser.add_argument("--dump_frame", type=str, default=None,
                    help="Save the first sim camera frame to this path (PNG) for visual comparison "
                         "against a training frame.")
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


def resolve_image_key(pilot):
    """Which image key this checkpoint expects.

    smolvla_base declares camera1/2/3, so finetuning our single-camera dataset
    needs --rename_map and the trained policy then expects `camera1`, not
    `front`. Read it off the config rather than making the caller remember.
    """
    if args_cli.image_key:
        return args_cli.image_key
    keys = sorted(pilot.policy.config.image_features)
    if not keys:
        raise RuntimeError("checkpoint declares no image features")
    if len(keys) > 1:
        print(f"[WARN] checkpoint declares {len(keys)} image features {keys}; using {keys[0]}. "
              f"Pass --image_key to override.")
    return keys[0]


def current_rgb(env_u):
    """Read the camera live rather than via env.front_rgb.

    `front_rgb` is only assigned inside _compute_intermediate_values(), which during
    a reset runs from step_sim_no_action() *inside* _set_assets_state — i.e. before
    the assets have been placed. _get_observations() never refreshes it, so the
    cached attribute is stale for the whole first step of every episode and shows
    the blocks at their USD spawn defaults. IsaacLab sensors update lazily on
    attribute access, so going through .data here gives the current frame.
    """
    return env_u.front_camera.data.output["rgb"][0]


def refresh_camera(env_u, passes: int = 2):
    """Make the camera reflect the post-reset scene.

    Isaac Lab updates sensor buffers in scene.update(), which only runs inside
    env.step()'s decimation loop — so the first observation after any reset shows
    the PREVIOUS scene (blocks at their USD spawn defaults). sim.render() alone is
    not enough: measured, 8 render passes changed the frame by exactly 0.00, while
    render()+scene.update() cut divergence from 22.9 to 1.8. Costs no episode steps.
    """
    for _ in range(passes):
        env_u.sim.render()
        env_u.scene.update(dt=env_u.physics_dt)


def build_frame(env_u, image_key):
    state = torch.cat([
        env_u.fingertip_midpoint_pos[0],
        env_u.fingertip_midpoint_quat[0],
        env_u.gripper[0],
        env_u.ee_linvel_fd[0],
        env_u.ee_angvel_fd[0],
    ], dim=-1)  # (14,)
    return {
        image_key: current_rgb(env_u),                   # (H, W, 3) uint8, read live
        "observation.state": state,
        "task": env_u.instructions[0],
    }


@hydra_task_config(TASK_ID, "rl_games_cfg_entry_point")
def run(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.vis.store_rgb = True
    # The cfg default is True, and the markers render INTO the camera frame — the
    # held marker sits on the target block. Training data is recorded without them,
    # so leaving them on here would feed the policy a scene it has never seen.
    env_cfg.vis.vis_obs = False
    # Isaac Lab renders sensors BEFORE applying a reset unless told otherwise, so with the
    # default of 0 the first camera frame of every episode is stale — it shows the blocks at
    # their USD spawn defaults rather than the randomized positions. The policy would then
    # commit a 50-step (3.3 s) open-loop chunk based on a scene that no longer exists, which
    # looks exactly like "reaches the same place regardless of the blocks or the prompt".
    # Recording is unaffected (its first frame is captured after a full env.step()).
    env_cfg.num_rerenders_on_reset = 2
    env_cfg.pilot_model = "knn"      # base action is unused; kNN just needs its local demo file
    env_cfg.pilot_type = "none"
    if args_cli.no_rand:
        env_cfg.dmr.rand_ctrl = False
        env_cfg.dmr.aug_data = False

    env = gym.make(TASK_ID, cfg=env_cfg)
    env_u = env.unwrapped
    if not getattr(env_u.cfg_task, "single_target", False):
        print("[WARN] task is not single_target — instructions/target may be undefined.")
    if not args_cli.dump_only and not args_cli.checkpoint:
        raise SystemExit("--checkpoint is required unless --dump_only is given")

    if args_cli.dump_only:
        from PIL import Image as _Image
        out = args_cli.dump_frame or "/tmp/eval_frame.png"
        env.reset()
        refresh_camera(env_u)
        _Image.fromarray(current_rgb(env_u).cpu().numpy().astype("uint8")).save(out)
        print(f"[INFO] wrote {out}")
        print(f"[INFO] instruction was: {env_u.instructions[0]}")
        env.close()
        return

    pilot = SmolVLA_Pilot(args_cli.checkpoint, device=args_cli.policy_device,
                          n_action_steps=args_cli.n_action_steps)
    print(f"[INFO] executing {pilot.policy.config.n_action_steps} of each "
          f"{pilot.policy.config.chunk_size}-step chunk before re-planning "
          f"({pilot.policy.config.n_action_steps / 15.0:.2f}s open-loop)")
    image_key = resolve_image_key(pilot)
    print(f"[INFO] feeding observations under image key: {image_key}")

    obs, _ = env.reset()
    pilot.reset()
    if not hasattr(env_u, "front_camera"):
        raise RuntimeError("env has no front_camera — not initialised (store_rgb / --enable_cameras).")

    results = []
    steps_in_ep = 0
    grounding = []
    aimed_at = []           # (target_colour, aimed_colour) per episode
    dumped = False
    lifted = False          # target block ever raised clear of the table this episode
    binned_any = None       # blocks_binned snapshot at episode end
    min_xy = 1e9            # closest the fingertip got to the target block (xy, m)
    min_xyz = 1e9
    approach = []           # per-episode closest approach, for the summary
    while simulation_app.is_running() and len(results) < args_cli.num_episodes:
        if steps_in_ep == 0:
            # DirectRLEnv auto-resets inside the step that reports `done`, so the
            # camera is stale at the first observation of every episode.
            refresh_camera(env_u)
        tgt_pre = int(env_u.target_block_idx[0].item())
        instr = env_u.instructions[0]

        # First observation of an episode: is the policy already aiming at the wrong
        # block, before any closed-loop drift could have accumulated?
        if args_cli.debug_grounding and steps_in_ep == 0:
            with torch.inference_mode():
                frame = build_frame(env_u, image_key)
                if args_cli.dump_frame and not dumped:
                    from PIL import Image as _Image
                    _Image.fromarray(current_rgb(env_u).cpu().numpy().astype("uint8")).save(args_cli.dump_frame)
                    print(f"[INFO] wrote sim camera frame to {args_cli.dump_frame}")
                    dumped = True
                chunk = pilot.predict_chunk(frame)
            reach = chunk[int(chunk[:, 2].argmin()), :2].numpy()   # xy where the chunk dips lowest
            blocks = env_u.rb_block_xy[0].cpu().numpy()            # (3, 2) true positions
            d = ((blocks - reach[None, :]) ** 2).sum(1) ** 0.5
            closest = int(d.argmin())
            grounding.append(closest == tgt_pre)
            aimed_at.append((env_u.cfg_task.target_colors[tgt_pre],
                             env_u.cfg_task.target_colors[closest]))
            print(f"    [grounding] aims at {env_u.cfg_task.target_colors[closest]:<5s} "
                  f"(target {env_u.cfg_task.target_colors[tgt_pre]:<5s})  "
                  f"d_target={d[tgt_pre]:.3f}  d_min={d.min():.3f}")

        with torch.inference_mode():
            action = pilot.act(build_frame(env_u, image_key)).to(env_u.device).view(1, -1)
            obs, _rew, terminated, truncated, _info = env.step(action)

        steps_in_ep += 1
        # Blocks rest at z ~= 0.011; treat a clear rise as "was actually grasped".
        blocks_z = torch.stack([
            env_u._block_a.data.root_pos_w[0, 2], env_u._block_b.data.root_pos_w[0, 2],
            env_u._block_c.data.root_pos_w[0, 2]]) - env_u.scene.env_origins[0, 2]
        if float(blocks_z[tgt_pre]) > 0.045:
            lifted = True

        # How close did the gripper actually get to the target block? This is the
        # decisive number for "right direction but misses": a 3 cm cube needs the
        # fingertip within roughly 1-1.5 cm to close on it.
        tgt_xy = env_u.rb_block_xy[0, tgt_pre]
        ft = env_u.fingertip_midpoint_pos[0]
        min_xy = min(min_xy, float(torch.linalg.vector_norm(ft[:2] - tgt_xy)))
        binned_any = env_u.blocks_binned[0].clone()

        done = bool((terminated | truncated)[0].item()) or steps_in_ep >= args_cli.max_steps

        if done:
            succ = bool(env_u.ep_succeeded[0].item())
            color = env_u.cfg_task.target_colors[tgt_pre]
            results.append({"color": color, "success": succ, "steps": steps_in_ep})
            cols = env_u.cfg_task.target_colors
            binned = [cols[i] for i in range(3) if bool(binned_any[i])] if binned_any is not None else []
            distractor_binned = [c for c in binned if c != color]
            if succ:
                why = "SUCCESS"
            elif distractor_binned:
                why = f"fail: binned DISTRACTOR {'+'.join(distractor_binned)}"
            elif color in binned:
                why = "fail: target binned but a distractor was too"
            elif lifted:
                why = "fail: grasped target, never binned it (dropped / didn't reach bin)"
            else:
                why = "fail: never lifted the target"
            results[-1]["reason"] = why
            approach.append(min_xy)
            results[-1]["min_xy"] = min_xy
            print(f"[{len(results):3d}/{args_cli.num_episodes}] {color:<5s} "
                  f"{why:<52s} ({steps_in_ep} steps)  closest to target: {min_xy*100:.1f} cm")
            if steps_in_ep >= args_cli.max_steps and not (terminated | truncated)[0].item():
                # Force-reset an episode that hit the step cap without terminating.
                # Must be inside inference_mode: the env's buffers were created under
                # it above, and reset() updates them in place — doing that outside
                # raises "Inplace update to inference tensor outside InferenceMode".
                with torch.inference_mode():
                    env.reset()
            pilot.reset()
            steps_in_ep = 0
            lifted = False
            binned_any = None
            min_xy = 1e9

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
    from collections import Counter as _C
    reasons = _C(r.get("reason", "?").split(":")[0] if r.get("reason") else "?" for r in results)
    print("  failure modes:")
    for k, v in _C(r.get("reason", "?") for r in results).most_common():
        print(f"    {v:>3}x  {k}")
    if approach:
        import numpy as _np
        a = _np.array(approach)
        print(f"  closest fingertip-to-target approach: median {_np.median(a)*100:.1f} cm  "
              f"min {a.min()*100:.1f} cm  max {a.max()*100:.1f} cm")
        print(f"    within 1.5 cm (graspable) in {int((a < 0.015).sum())}/{len(a)} episodes")
        print("    -> if it rarely gets under ~1.5 cm, the failure is PRECISION, not block choice")
    if aimed_at:
        cols = list(env_u.cfg_task.target_colors)
        print("  which block it AIMED AT, by instructed colour:")
        print(f"    {'target':<8s} " + " ".join(f"{c:>6s}" for c in cols) + "   n")
        for tc in cols:
            row = [a for (t, a) in aimed_at if t == tc]
            counts = [sum(1 for a in row if a == c) for c in cols]
            print(f"    {tc:<8s} " + " ".join(f"{n:>6d}" for n in counts) + f"   {len(row)}")
        allaims = [a for (_t, a) in aimed_at]
        from collections import Counter as _C2
        top, topn = _C2(allaims).most_common(1)[0]
        print(f"    -> aimed at '{top}' in {topn}/{len(allaims)} episodes "
              f"({100.0*topn/len(allaims):.0f}%) regardless of instruction")
        print("       (a near-100% single column = the prompt is being ignored;")
        print("        a strong diagonal = language grounding is working)")
    if grounding:
        g = sum(grounding)
        print(f"  first-chunk grounding: {g}/{len(grounding)} = "
              f"{100.0 * g / len(grounding):.0f}%  (chance 33%)")
        print("    high  -> it sees correctly; failure is closed-loop drift / grasping")
        print("    ~33%  -> wrong from the first frame; suspect a sim-vs-dataset visual gap")
    print("=" * 48)

    env.close()


if __name__ == "__main__":
    run()
    simulation_app.close()
