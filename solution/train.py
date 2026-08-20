import torch
import torch.nn.functional as F
from torch import nn, optim


class SequenceCNN(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.embedding = nn.Embedding(256, 24)
        
        self.conv3 = nn.Conv1d(24, 30, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(24, 30, kernel_size=5, padding=2)
        
        self.fc = nn.Linear(120, num_labels)
        
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.transpose(1, 2) 
        
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        
        c3_max = F.max_pool1d(c3, c3.size(2)).squeeze(-1)
        c3_mean = F.avg_pool1d(c3, c3.size(2)).squeeze(-1)
        c5_max = F.max_pool1d(c5, c5.size(2)).squeeze(-1)
        c5_mean = F.avg_pool1d(c5, c5.size(2)).squeeze(-1)
        
        combined = torch.cat([c3_max, c3_mean, c5_max, c5_mean], dim=1)
        
        return self.fc(combined)

class Trainer:
    def __init__(self, num_labels: int, train_steps: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SequenceCNN(num_labels).to(self.device)
        self.train_steps = max(int(train_steps), 1)
        self.completed_steps = 0
        
        self.base_lr = 0.01
        self.warmup_fraction = 0.1
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.base_lr,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def _update_lr(self) -> None:
        progress = min(self.completed_steps / self.train_steps, 1.0)
        if progress < self.warmup_fraction:
            scale = progress / self.warmup_fraction
        else:
            scale = (1.0 - progress) / (1.0 - self.warmup_fraction)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr * max(scale, 0.0)

    def step(self, samples: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        self._update_lr()
        self.optimizer.zero_grad()
        
        samples = samples.to(self.device).to(torch.int64)
        labels = labels.to(self.device)
        logits = self.model(samples)
        
        loss = self.criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
        self.optimizer.step()
        
        self.completed_steps += 1
        return loss.item()

    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            samples = samples.to(self.device).to(torch.int64)
            logits = self.model(samples)
            return logits
