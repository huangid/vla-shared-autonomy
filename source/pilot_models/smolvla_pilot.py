"""Inference wrapper around a (finetuned) SmolVLA policy.

Mirrors ``bc_pilot.BC_Pilot`` but for ``SmolVLAPolicy``. It loads the policy and
its saved pre/post processors from ``model_id`` (a local checkpoint dir or an HF
repo), then exposes ``act(frame)`` which takes a single unbatched observation in
LeRobot format:

    {
        "observation.images.front": (H, W, 3) uint8  OR  (3, H, W) float in [0,1],
        "observation.state":        (14,) float array/tensor,
        "task":                     "pick up the green block and put it in the bin",
    }

and returns a (1, 8) absolute end-effector action (pos3 + quat4 + gripper1),
ready to feed straight into ``XArm-RandomBlock-GuidedDiffusion``.

SmolVLA predicts an action chunk and dequeues one action per ``select_action``
call, so ``reset()`` must be called at every episode boundary.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def _to_lerobot_image(img: Any) -> torch.Tensor:
    """Return a (C, H, W) float32 tensor in [0, 1] from an (H, W, C) uint8 frame
    (or pass through something already channel-first / normalized)."""
    t = torch.as_tensor(img)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
        if t.max() > 1.5:  # 0-255 stored as float
            t = t / 255.0
    if t.ndim == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
        t = t.permute(2, 0, 1).contiguous()
    return t


class SmolVLA_Pilot:
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.device = "cuda" if device is None else device

        self.policy: SmolVLAPolicy = SmolVLAPolicy.from_pretrained(model_id)
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    @torch.inference_mode()
    def act(self, frame: dict[str, Any]) -> torch.Tensor:
        """preprocess -> select_action -> postprocess. Returns an (1, 8) CPU tensor."""
        frame = dict(frame)
        for k, v in frame.items():
            if "image" in k:
                frame[k] = _to_lerobot_image(v)
            elif k != "task":
                frame[k] = torch.as_tensor(v).float()
        processed = self.preprocessor(frame)
        action = self.policy.select_action(processed)
        return self.postprocessor(action)

    def reset(self):
        """Clear the action-chunk queue. Call at every episode boundary."""
        self.policy.reset()


__all__ = ["SmolVLA_Pilot"]
