"""GRU-based trainer for fixed-width byte sequences."""

import os
import torch
from torch import nn

class SequenceRNN(nn.Module):
    """GRU network for sequence classification."""

    def __init__(
        self,
        num_labels: int,
        embedding_dim: int = 24,
        hidden_dim: int = 40,
        dropout: float = 0.1
    ):
        super().__init__()
        # 256 possible byte values
        self.embedding = nn.Embedding(256, embedding_dim)
        
        # Batch first means inputs are (batch, seq, feature)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_labels)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        # samples: (batch, 72), uint8
        tokens = samples.to(torch.int64)
        embedded = self.embedding(tokens)
        
        # RNN outputs: out is (batch, seq, hidden_dim)
        out, _ = self.rnn(embedded)
        
        # Max pooling across the sequence dimension (dim=1)
        pooled, _ = torch.max(out, dim=1)
        
        features = self.dropout(pooled)
        return self.head(features)


class Trainer:
    """Trainer for the GRU model."""

    def __init__(self, num_labels: int, train_steps: int) -> None:
        self.train_steps = max(int(train_steps), 1)
        self.completed_steps = 0
        
        # Configurable from env or defaults
        embedding_dim = int(os.environ.get("GRU_EMBEDDING_DIM", "24"))
        hidden_dim = int(os.environ.get("GRU_HIDDEN_DIM", "40"))
        
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        
        self.batch_size = int(os.environ.get("GRU_BATCH_SIZE", "256" if device_name == "cuda" else "32"))
        dropout = float(os.environ.get("GRU_DROPOUT", "0.05"))
        
        self.model = SequenceRNN(
            num_labels=num_labels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)
        
        # Learning rate scaled slightly with batch size
        reference_lr = float(os.environ.get("GRU_LR", "0.015"))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=reference_lr,
            weight_decay=0.03
        )
        
        self.base_lr = reference_lr
        self.warmup_fraction = 0.1

    def _update_lr(self) -> None:
        progress = min(self.completed_steps / self.train_steps, 1.0)
        if progress < self.warmup_fraction:
            scale = progress / self.warmup_fraction
        else:
            scale = (1.0 - progress) / (1.0 - self.warmup_fraction)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr * max(scale, 0.0)

    def step(self, samples: torch.Tensor, label_indices: torch.Tensor) -> float:
        self.model.train()
        self._update_lr()
        
        samples = samples.to(self.device)
        targets = label_indices.to(self.device)
        
        losses = []
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i : i + self.batch_size]
            batch_targets = targets[i : i + self.batch_size]
            
            self.optimizer.zero_grad(set_to_none=True)
            
            logits = self.model(batch_samples)
            loss = torch.nn.functional.cross_entropy(logits, batch_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.optimizer.step()
            
            losses.append(loss.detach())
            
        self.completed_steps += 1
        return float(torch.stack(losses).mean().cpu().item())

    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(samples.to(self.device))
