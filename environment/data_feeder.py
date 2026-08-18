"""Shuffled batches from labeled fixed-width dialogue lines."""

from __future__ import annotations

import torch


class LineFeeder:
    """Shuffle cached dialogue lines on every pass."""

    def __init__(
        self,
        samples: torch.Tensor,
        targets: torch.Tensor,
        feed_size: int,
        device: torch.device,
    ) -> None:
        self.samples = samples.to(device)
        self.targets = targets.to(device)
        self.feed_size = feed_size
        self.device = device
        self.generator = torch.Generator()

    def feeds(self):
        """Yield one shuffled pass over every line."""
        order = torch.randperm(
            len(self.samples),
            generator=self.generator,
        )
        if self.device.type != "cpu":
            order = order.to(self.device)
        for begin in range(0, len(order), self.feed_size):
            rows = order[begin : begin + self.feed_size]
            yield (
                self.samples.index_select(0, rows),
                self.targets.index_select(0, rows),
            )
