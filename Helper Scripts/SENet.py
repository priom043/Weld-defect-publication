"""
This is the PyTorch Implementation of the renowned SENet attention module 
"""
import torch
import torch.nn as nn

# Define SENet (Squeeze-and-Excitation Network) module
class SEModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #There is no GlobalAveragePooling in PyTorch
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)
