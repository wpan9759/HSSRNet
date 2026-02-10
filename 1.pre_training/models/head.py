import torch.nn as nn

class BYOLHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256)
        )

    def forward(self, x):
        return self.projection(x)