import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SequenceCNN(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        # 256 byte values
        self.embed = nn.Embedding(256, 16)
        # Deep narrow CNN
        self.conv1 = nn.Conv1d(16, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(24, 24, kernel_size=3, padding=1)
        self.head = nn.Linear(24, num_labels)

    def forward(self, x):
        # x is (batch, seq) -> embed to (batch, seq, 16) -> transpose to (batch, 16, seq)
        x = self.embed(x.long()).transpose(1, 2)
        
        # Apply convolutions with GELU
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        
        # Mean pooling across the sequence dimension
        x = torch.mean(x, dim=2)
        
        return self.head(x)

class Trainer:
    def __init__(self, num_labels: int, train_steps: int):
        self.model = SequenceCNN(num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.train_steps = train_steps

    def step(self, samples: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        samples = samples.to(torch.int64)
        logits = self.model(samples)
        
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            samples = samples.to(torch.int64)
            logits = self.model(samples)
            return logits
