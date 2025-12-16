
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'snake_ai'))

import numpy as np
import torch
import torch.nn.functional as F
from game import SnakeGame
from model import SnakeNet
from main import process_game_memory

def test_tail_collision_logic():
    print("Testing Tail Collision Fix...")
    game = SnakeGame(board_size=5)
    
    # Manually setup a snake that is chasing its tail
    # Snake length 4: Head(2,2), Body(2,1), Body(1,1), Tail(1,2)
    # Head is at (2,2), facing Down (2)??
    # Let's say Head is (2,2). Tail is (2,1).
    # If Head moves Up to (2,1), it enters Tail's spot.
    # But Tail moves to (1,1). So (2,1) becomes empty.
    
    # Let's force a scenario
    # Current head: (2,2)
    # Body: (2,1), (1,1)
    # Tail: (1,2)
    # If we move LEFT (from down?), we go to (1,2) which is TAIL.
    
    game.snake = [(2,2), (2,1), (1,1), (1,2)]
    game.direction = 1 # Facing Right (so relative left is Up? No.)
    # Let's use absolute moves to be sure.
    # Head is (2,2). Tail is (1,2).
    # Move LEFT (dx=-1) -> New Head (1,2).
    # Checks collision.
    
    # 3: Left
    action = 3 
    state, reward, done = game.step(action)
    
    if done and game.death_reason == "body":
        print("[FAIL] Snake died by hitting tail!")
    elif not done or game.death_reason != "body":
        if game.snake[0] == (1,2):
             print("[PASS] Snake successfully moved into tail position.")
        else:
             print(f"[FAIL] Snake moved somewhere else? {game.snake[0]}")
    else:
        print(f"[FAIL] Unexpected outcome: Done={done}, Reason={game.death_reason}")

def test_model_tanh():
    print("\nTesting Model Tanh Activation...")
    model = SnakeNet(board_size=10)
    input_tensor = torch.randn(1, 3, 10, 10)
    p, v = model(input_tensor)
    
    val = v.item()
    print(f"Value Output: {val}")
    if -1.0 <= val <= 1.0:
        print("[PASS] Value is within [-1, 1]")
    else:
        print("[FAIL] Value is OUTSIDE [-1, 1]")

def test_return_clipping():
    print("\nTesting Return Clipping...")
    # Mock memory: [state, policy, reward]
    # Reward sequence: -0.01, -0.01, ..., -100 (hypothetically)
    # Actually checking process_game_memory
    
    # Case 1: Huge hunger penalty
    huge_neg_memory = []
    # 100 steps of -1.0 reward
    for _ in range(100):
        huge_neg_memory.append([None, None, -1.0])
        
    processed = process_game_memory(huge_neg_memory)
    
    # Check value of first step (should be huge negative, but clipped)
    first_sample_val = processed[0][2] # (state, policy, value)
    
    print(f"Calculated Return (from -100 cumulative): {first_sample_val}")
    if first_sample_val == -1.0:
        print("[PASS] Return clipped to -1.0")
    else:
        print("[FAIL] Return NOT clipped correctly")

    # Case 2: Huge positive reward
    huge_pos_memory = [[None, None, 100.0]]
    processed_pos = process_game_memory(huge_pos_memory)
    print(f"Calculated Return (from +100): {processed_pos[0][2]}")
    if processed_pos[0][2] == 1.0:
        print("[PASS] Return clipped to 1.0")


if __name__ == "__main__":
    test_tail_collision_logic()
    test_model_tanh()
    test_return_clipping()
