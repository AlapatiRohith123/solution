"""Editable general-purpose byte trainer."""

from __future__ import annotations

import torch
from torch import nn


class ByteNet(nn.Module):
    """Tiny starter from coarse character-type ratios."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, num_labels),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Return label scores from coarse byte counts."""
        values = samples.to(torch.int64)
        lowercase = ((values >= 97) & (values <= 122)).to(torch.float32)
        uppercase = ((values >= 65) & (values <= 90)).to(torch.float32)
        digits = ((values >= 48) & (values <= 57)).to(torch.float32)
        spaces = (values == 32).to(torch.float32)
        punctuation = (
            (values == 33)
            | ((values >= 39) & (values <= 47))
            | ((values >= 58) & (values <= 63))
        ).to(torch.float32)
        features = torch.stack(
            [
                lowercase.mean(dim=1),
                uppercase.mean(dim=1),
                digits.mean(dim=1),
                spaces.mean(dim=1),
                punctuation.mean(dim=1),
            ],
            dim=1,
        )
        return self.head(features)


class Trainer:
    """Trainer stepped only through raw labeled batches."""

    def __init__(self, num_labels: int, train_steps: int) -> None:
        del train_steps
        self.model = ByteNet(num_labels)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def step(
        self,
        samples: torch.Tensor,
        label_indices: torch.Tensor,
    ) -> float:
        """Perform one update on a fed batch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            self.model(samples),
            label_indices,
        )
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        """Return scores without changing model state."""
        self.model.eval()
        return self.model(samples)
