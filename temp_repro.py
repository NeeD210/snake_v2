
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'snake_ai'))

from game import SnakeGame 
from model import SnakeNet

def test_model():
    model_path = "snake_ai/experiments/train_v13/snake_net.pth"
    board_size = 6
    
    print(f"Loading model from {model_path}...")
    try:
        model = SnakeNet(board_size=board_size)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\n--- Checking BatchNorm Stats ---")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            mean = module.running_mean
            var = module.running_var
            
            print(f"Layer: {name}")
            print(f"  Mean - Min: {mean.min():.4f}, Max: {mean.max():.4f}, NaNs: {torch.isnan(mean).any()}, Infs: {torch.isinf(mean).any()}")
            print(f"  Var  - Min: {var.min():.4f}, Max: {var.max():.4f}, NaNs: {torch.isnan(var).any()}, Infs: {torch.isinf(var).any()}")
            
            if torch.isnan(mean).any() or torch.isinf(mean).any() or torch.isnan(var).any() or torch.isinf(var).any():
                print(f"  [CRITICAL] {name} has corrupted stats!")

if __name__ == "__main__":
    test_model()
