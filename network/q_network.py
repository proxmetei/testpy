import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.conv(x)