import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SequenceCNN(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.embedding = nn.Embedding(256, 16)
        
        # Parallel convolutions
        self.conv3 = nn.Conv1d(16, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(16, 64, kernel_size=5, padding=2)
        
        self.fc = nn.Linear(128, num_labels)
        
    def forward(self, x):
        # x is (batch_size, 72)
        x = self.embedding(x)
        x = x.transpose(1, 2) # (batch_size, 16, 72)
        
        c3 = F.relu(self.conv3(x)) # (batch_size, 64, 72)
        c5 = F.relu(self.conv5(x)) # (batch_size, 64, 72)
        
        c3_max = F.adaptive_max_pool1d(c3, 1).squeeze(-1)
        c3_mean = F.adaptive_avg_pool1d(c3, 1).squeeze(-1)
        c5_max = F.adaptive_max_pool1d(c5, 1).squeeze(-1)
        c5_mean = F.adaptive_avg_pool1d(c5, 1).squeeze(-1)
        
        c3_pool = c3_max + c3_mean
        c5_pool = c5_max + c5_mean
        
        combined = torch.cat([c3_pool, c5_pool], dim=1) # (batch_size, 128)
        
        return self.fc(combined)

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
