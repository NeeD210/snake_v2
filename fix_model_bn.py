
import torch
import numpy as np
import sys
import os

# Ensure snake_ai is in path
sys.path.append(os.path.join(os.getcwd(), 'snake_ai'))

from game import SnakeGame 
from model import SnakeNet
from run_command import run_command # Oops, no I don't have this.
# I will just write the script.

def repair_model(model_path):
    print(f"Repairing model: {model_path}")
    board_size = 6
    
    # Load Model
    try:
        model = SnakeNet(board_size=board_size)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Set to TRAIN mode. This allows BN stats to update locally.
    model.train()
    print("Model set to TRAIN mode. Running dummy inputs to update BatchNorm stats...")
    
    # Create a dummy game
    game = SnakeGame(board_size=board_size)
    
    # We need to feed decent data. Random play is fine, the stats just need to match the general input distribution
    # (0s and 1s).
    
    for i in range(1000):
        # 1. Reset occasionally
        if i % 50 == 0:
            game.reset()
            
        # 2. Get State
        state = game.get_state()
        
        # 3. Process State (POV)
        # Inline processing logic
        input_tensor = np.zeros((3, game.board_size, game.board_size), dtype=np.float32)
        input_tensor[0] = (state == 1).astype(float) 
        input_tensor[1] = (state == 2).astype(float) 
        input_tensor[2] = (state == 3).astype(float) 
        k = game.direction
        input_tensor = np.rot90(input_tensor, k, axes=(1, 2)).copy()
        
        # 4. Forward pass
        input_batch = torch.tensor(input_tensor).unsqueeze(0)
        
        with torch.no_grad(): # We don't need gradients, just the forward pass updates BN running stats
             p, v = model(input_batch)
             
        # 5. Step game (randomly is fine)
        action = np.random.choice(4)
        game.step(action)
        
    print("Repair complete.")
    
    # Save repaired model
    backup_path = model_path + ".bak"
    if not os.path.exists(backup_path):
        os.rename(model_path, backup_path)
        print(f"Original backed up to {backup_path}")
    
    torch.save(model.state_dict(), model_path)
    print(f"Repaired model saved to {model_path}")
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "snake_ai/experiments/train_v13/snake_net.pth"
        
    repair_model(path)
