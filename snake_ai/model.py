import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNELS = 5  # Body occupancy, Head, Food, Hunger, Action ratios

def _best_group_count(num_channels: int, max_groups: int = 8) -> int:
    """
    Pick the largest group count <= max_groups that divides num_channels.
    Keeps GroupNorm valid even if channel counts change later.
    """
    g = min(int(max_groups), int(num_channels))
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return max(1, g)

class ResidualBlock(nn.Module):
    """
    AlphaZero-style residual block:
    Conv -> Norm -> ReLU -> Conv -> Norm + skip -> ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=_best_group_count(channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=_best_group_count(channels), num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = F.relu(out + x)
        return out

class SnakeNet(nn.Module):
    """
    Multi-size capable Snake neural network (AlphaZero-style).
    
    ARCHITECTURE FOR MULTI-SIZE SUPPORT:
    ====================================
    The model can process boards of different sizes (6x6, 8x8, 10x10, etc.) without
    architectural changes. This is achieved through:
    
    1. CONVOLUTIONAL LAYERS (size-agnostic):
       - All conv layers use padding=1, making them work with any input size
       - Input: (Batch, 5, H, W) where H=W can vary (6, 8, 10, etc.)
       - Output after trunk: (Batch, channels, H, W) - same spatial size
    
    2. ADAPTIVE POOLING (key to multi-size):
       - Before FC layers, we use nn.AdaptiveAvgPool2d(adaptive_pool_size)
       - This reduces ANY spatial size to a fixed size (e.g., 2x2)
       - Example: 6x6 -> 2x2, 8x8 -> 2x2, 10x10 -> 2x2
       - This allows the same FC layers to work with any board size
    
    3. FULLY CONNECTED LAYERS (fixed input size):
       - FC layers receive fixed-size input from adaptive pooling
       - Policy: 2 channels * 2*2 = 8 features -> 3 actions
       - Value: 1 channel * 2*2 = 4 features -> 64 -> 1 value
    
    TRAINING WITH MULTI-SIZE:
    - During training, batches can contain games from different board sizes
    - The model processes each independently (batch dimension handles this)
    - This acts as data augmentation and improves generalization
    """
    def __init__(self, board_size: int = 10, channels: int = 64, num_blocks: int = 4, adaptive_pool_size: int = 2):
        """
        Args:
            board_size: Default board size (for backward compatibility, but model now supports any size)
            channels: Number of channels in the network
            num_blocks: Number of residual blocks
            adaptive_pool_size: Size to pool to before FC layers (enables multi-size support)
                                Default 2 means 2x2 output regardless of input size
        """
        super(SnakeNet, self).__init__()
        self.board_size = board_size  # Kept for compatibility, but not strictly required
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.adaptive_pool_size = int(adaptive_pool_size)
        
        # ResNet trunk (AlphaZero-style)
        # Input: 5 channels (Body occupancy, Head, Food, Hunger, Flood)
        # All conv layers use padding=1, making them size-agnostic
        self.input_conv = nn.Conv2d(INPUT_CHANNELS, self.channels, kernel_size=3, padding=1, bias=False)
        # GroupNorm is typically more stable than BatchNorm for self-play training
        # where batches are non-i.i.d and may have varying composition.
        self.input_gn = nn.GroupNorm(num_groups=_best_group_count(self.channels), num_channels=self.channels)
        self.blocks = nn.Sequential(*[ResidualBlock(self.channels) for _ in range(self.num_blocks)])

        # Heads - Now size-agnostic using Adaptive Pooling
        # Policy Head: Use adaptive pooling to fixed size before FC
        # This allows processing any board size (6x6, 8x8, 10x10, etc.)
        self.policy_conv = nn.Conv2d(self.channels, 2, kernel_size=1)
        self.policy_pool = nn.AdaptiveAvgPool2d(self.adaptive_pool_size)  # Always outputs (adaptive_pool_size, adaptive_pool_size)
        self.policy_flat_size = 2 * self.adaptive_pool_size * self.adaptive_pool_size
        self.policy_fc = nn.Linear(self.policy_flat_size, 3)

        # Value Head: Use adaptive pooling to fixed size before FC
        self.value_conv = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.value_pool = nn.AdaptiveAvgPool2d(self.adaptive_pool_size)  # Always outputs (adaptive_pool_size, adaptive_pool_size)
        self.value_flat_size = 1 * self.adaptive_pool_size * self.adaptive_pool_size
        self.value_fc1 = nn.Linear(self.value_flat_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (Batch, C, H, W) - H and W can now vary!
        
        x = F.relu(self.input_gn(self.input_conv(x)))
        x = self.blocks(x)

        # Policy Head
        p = F.relu(self.policy_conv(x))
        p = self.policy_pool(p)  # Reduces to (adaptive_pool_size, adaptive_pool_size)
        p = torch.flatten(p, start_dim=1)
        p = self.policy_fc(p)
        # Logsoftmax for training stability, or just logits. 
        # Usually MCTS uses Softmax probabilities.
        p = F.log_softmax(p, dim=1) 

        # Value Head
        v = F.relu(self.value_conv(x))
        v = self.value_pool(v)  # Reduces to (adaptive_pool_size, adaptive_pool_size)
        v = torch.flatten(v, start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        # Removed Tanh to allow unbounded value estimation (e.g. sum of rewards)
        # v = torch.tanh(v)

        return p, v
