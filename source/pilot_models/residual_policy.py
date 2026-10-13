"""Residual correction policy on top of a frozen SmolVLA base (CR-DAgger-style).

The base policy is never updated. A small network learns, from shared-autonomy logs,
the position adjustment the human's corrections produced:

    target = a_exec[:3] - a_R[:3]        (zero on steps the human did not correct)

and at run time that adjustment is added to the base action. Orientation and gripper
stay with the base: on the two-camera correction set the executed orientation differs
from the policy's by a median of 0.8 deg, so there is nothing there to learn.

This is position-only residual learning from SpaceMouse blending, not CR-DAgger itself
(no force sensing, no compliant kinesthetic interface, no force output).

Deliberately free of Isaac imports so the offline scripts (feature extraction,
training) and the live evaluation share exactly one encoder and one network.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

STATE_DIM = 14      # fingertip pos3 + quat4 + gripper1 + linvel3 + angvel3
ACTION_DIM = 8      # pos3 + quat4 + gripper1
FEAT_DIM = 512      # ResNet-18 layer4 channels
GRID = 2            # layer4 map is pooled to GRID x GRID cells per view
N_CELLS = GRID * GRID

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class FrozenViewEncoder(nn.Module):
    """ImageNet ResNet-18 trunk -> (N, N_CELLS, 512) per image, never trained.

    Frames go in at native resolution (wrist 240x320, side 360x480). The layer4 map is
    pooled to a 2x2 grid rather than one global vector, because the residual has to
    judge where the block sits relative to the gripper and a global average throws most
    of that away. Models that want a single vector average the cells themselves.
    """

    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.trunk = nn.Sequential(*list(net.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(GRID)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):          # stays in eval mode whatever the caller does
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: (N, H, W, 3) or (H, W, 3), uint8 or float in [0, 255]."""
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        x = rgb.float().permute(0, 3, 1, 2) / 255.0
        x = (x - self.mean) / self.std
        f = self.pool(self.trunk(x))                      # (N, 512, GRID, GRID)
        return f.flatten(2).transpose(1, 2).contiguous()  # (N, N_CELLS, 512)


def low_dim_inputs(state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
    """(N, 14) state + (N, 8) base action -> (N, 25).

    The last three entries are where the base wants to go relative to where the
    fingertip is now. The network could derive that itself, but it is the quantity a
    correction most depends on, so it is handed over directly.
    """
    rel = base_action[:, 0:3] - state[:, 0:3]
    return torch.cat([state, base_action, rel], dim=-1)


LOW_DIM = STATE_DIM + ACTION_DIM + 3


class ResidualNet(nn.Module):
    """MLP: [wrist feats | side feats | low-dim] -> position adjustment (metres)."""

    VIEWS = {"both": ("wrist", "side"), "wrist": ("wrist",), "side": ("side",), "none": ()}

    def __init__(self, views: str = "both", pool: int = 1, hidden: int = 256, layers: int = 3,
                 dropout: float = 0.3):
        """
        Args:
            views: which camera features to use. "none" gives a network that sees only
                the robot state and the base action — on the two-camera correction set
                that variant generalised best to held-out episodes, the image features
                adding overfitting rather than information at 100 episodes.
            pool: 2 = keep the 2x2 feature cells per view, 1 = average them to one vector.
        """
        super().__init__()
        if pool not in (1, 2):
            raise ValueError("pool must be 1 (one vector per view) or 2 (2x2 cells per view)")
        if views not in self.VIEWS:
            raise ValueError(f"views must be one of {sorted(self.VIEWS)}")
        self.pool, self.views = pool, views
        feat_in = len(self.VIEWS[views]) * FEAT_DIM * (N_CELLS if pool == 2 else 1)
        dims = [feat_in + LOW_DIM] + [hidden] * layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 3)
        # Normalisation statistics, filled in from the training set and saved with the weights.
        self.register_buffer("feat_mean", torch.zeros(feat_in))
        self.register_buffer("feat_std", torch.ones(feat_in))
        self.register_buffer("low_mean", torch.zeros(LOW_DIM))
        self.register_buffer("low_std", torch.ones(LOW_DIM))
        self.register_buffer("out_scale", torch.ones(1))

    def pack_feats(self, wrist, side, n: int | None = None) -> torch.Tensor:
        """(N, N_CELLS, 512) per view -> (N, feat_in), keeping only the views in use."""
        parts = []
        for name, f in (("wrist", wrist), ("side", side)):
            if name in self.VIEWS[self.views]:
                f = f.float()
                parts.append((f.mean(1) if self.pool == 1 else f).flatten(1))
        if not parts:
            n = n if n is not None else len(wrist if wrist is not None else side)
            return self.low_mean.new_zeros((n, 0))
        return torch.cat(parts, dim=-1)

    def forward(self, wrist, side, low: torch.Tensor) -> torch.Tensor:
        """Returns the adjustment in units of out_scale; multiply by out_scale for metres."""
        f = (self.pack_feats(wrist, side, n=len(low)) - self.feat_mean) / self.feat_std
        lo = (low - self.low_mean) / self.low_std
        return self.head(self.body(torch.cat([f, lo], dim=-1)))


class ResidualPolicy:
    """Run-time wrapper: frozen encoder + trained ResidualNet + a safety cap."""

    def __init__(self, path: str, device: str = "cuda", cap: float = 0.05, zero: bool = False):
        """
        Args:
            cap: largest adjustment applied, metres. The training labels have a 99th
                percentile of 5.1 cm; anything beyond that is extrapolation.
            zero: run the whole path but apply no adjustment. With this set the result
                must match the base policy, which is the check that adding the residual
                machinery changed nothing else.
        """
        ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
        self.config = ckpt["config"]
        self.net = ResidualNet(views=self.config.get("views", "both"), pool=self.config["pool"],
                               hidden=self.config["hidden"], layers=self.config["layers"],
                               dropout=0.0).to(device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.views = ResidualNet.VIEWS[self.net.views]
        # A state-only residual needs no encoder and no camera frames at all.
        self.encoder = FrozenViewEncoder().to(device) if self.views else None
        self.device, self.cap, self.zero = device, float(cap), bool(zero)

    @torch.no_grad()
    def delta(self, wrist_rgb: torch.Tensor, side_rgb: torch.Tensor,
              state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        """Position adjustment (3,) in metres for one step, norm-capped."""
        dev = self.device
        fw = self.encoder(wrist_rgb.to(dev)) if "wrist" in self.views else None
        fs = self.encoder(side_rgb.to(dev)) if "side" in self.views else None
        low = low_dim_inputs(state.to(dev).float().view(1, -1),
                             base_action.to(dev).float().view(1, -1))
        d = (self.net(fw, fs, low) * self.net.out_scale).view(3)
        n = torch.linalg.vector_norm(d)
        if n > self.cap:
            d = d * (self.cap / n)
        return torch.zeros_like(d) if self.zero else d


__all__ = ["FrozenViewEncoder", "ResidualNet", "ResidualPolicy", "low_dim_inputs",
           "STATE_DIM", "ACTION_DIM", "LOW_DIM", "FEAT_DIM", "N_CELLS"]
