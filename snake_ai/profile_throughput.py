import time
import torch
import numpy as np
from game import SnakeGame
from mcts import MCTS
from encoder import encode_pov

def profile_encoder(num_samples=1000):
    """Benchmarks the speed of the POV encoder and flood-fill."""
    print(f"--- Profiling Encoder ({num_samples} iterations) ---")
    game = SnakeGame(board_size=10)
    
    # Run a few steps to generate a non-trivial board state
    for _ in range(5):
        game.step(1)
        
    start_time = time.time()
    for _ in range(num_samples):
        encode_pov(game)
    end_time = time.time()
    
    dt = end_time - start_time
    print(f"Total Time: {dt:.4f} seconds")
    print(f"Encodings per second: {num_samples / dt:.2f}\n")

def profile_mcts(num_simulations=500):
    """Benchmarks the speed of MCTS tree traversal and expansion overhead."""
    print(f"--- Profiling MCTS ({num_simulations} simulations) ---")
    game = SnakeGame(board_size=10)
    
    # Make the snake much longer to see the impact of O(N) operations
    game.snake = [(5, y) for y in range(8, 0, -1)] + [(4, 1), (4, 2), (4, 3)]
    game.occ = set(game.snake)
    game.score = len(game.snake)
    
    # Dummy network to measure pure Python/CPU overhead of MCTS algorithm
    # rather than being bottlenecked by the PyTorch model inference.
    def dummy_predict(input_tensor):
        # Uniform policy
        p = np.ones(3) / 3.0
        v = 0.0
        return p, v

    mcts = MCTS(dummy_predict, n_simulations=num_simulations)
    
    start_time = time.time()
    # Execute a single massive search to profile tree traversal/cloning
    mcts.search(game, n_simulations=num_simulations)
    end_time = time.time()
    
    dt = end_time - start_time
    print(f"Total Time: {dt:.4f} seconds")
    print(f"Simulations per second: {num_simulations / dt:.2f}\n")

if __name__ == "__main__":
    print("Testing baseline throughput before optimizations...\n")
    profile_encoder(num_samples=1000)
    profile_mcts(num_simulations=1000)
