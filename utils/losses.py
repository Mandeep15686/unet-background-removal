import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + 0.1 * self.mse(pred, target)
