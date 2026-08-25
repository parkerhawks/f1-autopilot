"""The hybrid driving policy: frames + proprioceptive state -> continuous control.

Two encoders feeding a shared head:

  frames  (B, C, H, W) --CNN--> 256-d  \
                                        +--> MLP --> steer, throttle, brake
  state   (B, D)       --MLP--> 64-d   /

The CNN follows Bojarski et al.'s PilotNet: five convolutions, no pooling,
striding for downsample. It is small on purpose. F1 25 will take 6-8 GB of a
12 GB card and most of its throughput, so the compute budget here is a few
milliseconds, not a few hundred. Capacity is not the binding constraint in
imitation learning from a few hours of demonstrations -- data coverage is.

The state branch matters more than its size suggests. Speed and lateral offset
are directly observable there, so the CNN never has to learn to regress them
from pixels, and the policy degrades to something still driveable when the
vision branch is confused by an unfamiliar corner.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """PilotNet-style convolutional trunk."""

    def __init__(self, in_channels: int, in_hw: tuple[int, int], out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 24, 5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),
        )
        # Infer the flattened width rather than hardcoding it, so input
        # resolution stays a free parameter during the Phase 2 ablations.
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, *in_hw)).numel()
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, out_dim), nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim
        self.n_flat = n_flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class StateEncoder(nn.Module):
    """Small MLP over the telemetry-derived feature vector."""

    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DrivingPolicy(nn.Module):
    """steer in [-1, 1]; throttle and brake in [0, 1].

    The output ranges match both F1's CarTelemetryData convention and
    VirtualPad's input, so predictions can be fed straight to the pad and
    compared directly against recorded human actions with no rescaling.

    Set `use_vision` / `use_state` to False to build the single-modality
    variants for the ablation table -- keeping them in one class means the
    three arms differ only in their inputs, not their training code.
    """

    def __init__(
        self,
        state_dim: int,
        frame_channels: int = 3,
        frame_hw: tuple[int, int] = (96, 192),
        use_vision: bool = True,
        use_state: bool = True,
    ):
        super().__init__()
        if not (use_vision or use_state):
            raise ValueError("policy needs at least one input modality")

        self.use_vision = use_vision
        self.use_state = use_state

        fused = 0
        if use_vision:
            self.vision = VisionEncoder(frame_channels, frame_hw)
            fused += self.vision.out_dim
        if use_state:
            self.state = StateEncoder(state_dim)
            fused += self.state.out_dim

        self.head = nn.Sequential(
            nn.Linear(fused, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        frames: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = []
        if self.use_vision:
            if frames is None:
                raise ValueError("policy was built with vision but got no frames")
            parts.append(self.vision(frames))
        if self.use_state:
            if state is None:
                raise ValueError("policy was built with state but got none")
            parts.append(self.state(state))

        raw = self.head(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])

        # Squash into the physical ranges. Brake and throttle are independent
        # outputs rather than one signed pedal axis, because human demonstrators
        # genuinely overlap them (trail braking), and a single axis cannot
        # represent that.
        steer = torch.tanh(raw[:, 0:1])
        throttle = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
