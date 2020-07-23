import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 2)
        self.fc1 = nn.Linear(64*9*9, 256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.reshape(-1, 64*9*9)))
        x = self.fc2(x)
        return x
