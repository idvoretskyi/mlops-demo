from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


class SimpleNet(nn.Module):
    """A small MLP used for demo training and inference.

    Parameters
    ----------
    in_features: int
        Number of input features.
    hidden: int
        Hidden layer width.
    out_features: int
        Number of output classes (logits produced).
    """

    def __init__(self, in_features: int = 20, hidden: int = 32, out_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class CheckpointMeta:
    in_features: int
    classes: int
    data_source: str

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointMeta":
        return cls(
            in_features=int(d.get("in_features", 20)),
            classes=int(d.get("classes", 2)),
            data_source=str(d.get("data_source", "synthetic")),
        )

    def to_dict(self) -> Dict:
        return {
            "in_features": self.in_features,
            "classes": self.classes,
            "data_source": self.data_source,
        }


def save_checkpoint(model: nn.Module, path: str, meta: CheckpointMeta) -> None:
    """Save model state dict and metadata to a single .pt file."""
    torch.save({
        "model_state_dict": model.state_dict(),
        **meta.to_dict(),
    }, path)


def load_checkpoint(path: str, device: torch.device) -> Tuple[SimpleNet, CheckpointMeta]:
    """Load a checkpoint and return an initialized model and its metadata."""
    checkpoint = torch.load(path, map_location=device)
    meta = CheckpointMeta.from_dict(checkpoint)
    model = SimpleNet(in_features=meta.in_features, out_features=meta.classes)
    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[arg-type]
    model.to(device)
    model.eval()
    return model, meta
