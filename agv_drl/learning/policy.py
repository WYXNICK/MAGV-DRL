from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .features import FEATURE_DIM


@dataclass
class PPOConfig:
    learning_rate: float = 3.0e-4
    gamma: float = 0.985
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 2048
    hidden_size: int = 256


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Categorical
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for LTF-PPO training/inference. Install the "
            "conda environment from environment.yml."
        ) from exc
    return torch, nn, optim, Categorical


def make_actor_critic(hidden_size: int = 256):
    _, nn, _, _ = require_torch()

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(FEATURE_DIM, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden_size, 5)
            self.critic = nn.Linear(hidden_size, 1)

        def forward(self, obs):
            x = self.backbone(obs)
            return self.actor(x), self.critic(x).squeeze(-1)

    return ActorCritic()


class LocalFollowerPolicy:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        torch, _, _, Categorical = require_torch()
        self.torch = torch
        self.Categorical = Categorical
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location=device)
        ckpt_feature_dim = payload.get("feature_dim")
        if ckpt_feature_dim is not None and int(ckpt_feature_dim) != FEATURE_DIM:
            raise RuntimeError(
                f"Checkpoint feature_dim={ckpt_feature_dim}, but current code expects "
                f"{FEATURE_DIM}. Retrain after map/LTF feature changes."
            )
        action_space = payload.get("action_space")
        if action_space != "ltf_grid_move_intents":
            raise RuntimeError(
                "Checkpoint action_space is not 'ltf_grid_move_intents'. "
                "Retrain because the current implementation uses Learn-to-Follow "
                "grid-move intents rather than raw RWARE primitive actions."
            )
        hidden_size = int(payload.get("hidden_size", 256))
        self.model = make_actor_critic(hidden_size)
        self.model.load_state_dict(payload["model"])
        self.model.to(device)
        self.model.eval()
        self.device = device

    def act(
        self,
        features: np.ndarray,
    ):
        torch = self.torch
        with torch.no_grad():
            obs = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            logits, _ = self.model(obs)
            dist = self.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)
        return actions.cpu().numpy(), logp.cpu().numpy()
