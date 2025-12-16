import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNet(nn.Module):
    def __init__(self, board_size=10):
        super(SnakeNet, self).__init__()
        self.board_size = board_size
        
        # Convolutional Body
        # Input: 3 channels (Body, Head, Food)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Heads
        # Adjusted for flattening: 64 channels * H * W
        self.flat_size = 64 * board_size * board_size
        
        # Policy Head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1) # Reduce channels before flattening
        # 2 channels * H * W
        self.policy_flat_size = 2 * board_size * board_size
        self.policy_fc = nn.Linear(self.policy_flat_size, 3)

        # Value Head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_flat_size = 1 * board_size * board_size
        self.value_fc1 = nn.Linear(self.value_flat_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Policy Head
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, self.policy_flat_size)
        p = self.policy_fc(p)
        # Logsoftmax for training stability, or just logits. 
        # Usually MCTS uses Softmax probabilities.
        p = F.log_softmax(p, dim=1) 

        # Value Head
        v = F.relu(self.value_conv(x))
        v = v.view(-1, self.value_flat_size)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        # Removed Tanh to allow unbounded value estimation (e.g. sum of rewards)
        # v = torch.tanh(v)

        return p, v
