"""PyTorch MLP reward model for RLHF preference learning."""

from pathlib import Path

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    MLP that maps trajectory feature vectors to a scalar reward score.

    Architecture: feat_dim → 128 → 64 → 1 (ReLU activations)

    Used in Bradley-Terry preference learning:
        P(A > B) = σ(r(A) - r(B))
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature tensor of shape (batch, feat_dim)
        Returns:
            scalar scores of shape (batch, 1)
        """
        return self.net(x)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Return scalar scores with shape (batch,)."""
        with torch.no_grad():
            return self.forward(features).squeeze(-1)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"feat_dim": self.feat_dim, "state_dict": self.state_dict()}, path
        )
        print(f"Reward model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "RewardModel":
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(feat_dim=checkpoint["feat_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Reward model loaded from {path}")
        return model
