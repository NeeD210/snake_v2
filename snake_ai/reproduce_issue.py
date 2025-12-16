import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from snake_ai.game import SnakeGame

def test_initialization():
    print("Testing Initialization...")
    game = SnakeGame(board_size=10)
    print(f"Initial Snake Length: {len(game.snake)}")
    if len(game.snake) == 1:
        print("FAIL: Snake starts with length 1.")
    else:
        print(f"PASS: Snake starts with length {len(game.snake)}.")

def test_180_turn():
    print("\nTesting 180 Degree Turn...")
    # Setup specific scenario: Head at (5,5), Direction UP (0)
    game = SnakeGame(board_size=10)
    # Force state for consistency
    game.snake = [(5, 5)] 
    game.direction = 0 # UP
    
    # Try to move DOWN (2) - Opposite of UP
    # In current buggy version (len=1), it might just move down instantly.
    # In fixed version, it should ignore and move UP.
    
    print(f"Initial Head: {game.snake[0]}, Direction: {game.direction} (0=UP)")
    print("Attempting to move DOWN (2)...")
    
    game.step(2) # Try 2 (Down)
    
    new_head = game.snake[0]
    new_direction = game.direction
    
    print(f"New Head: {new_head}, Direction: {new_direction}")
    
    # Expected behavior for FIXED:
    # If ignored: Direction remains 0 (UP), Head moves to (5, 4)
    # If simply moved Down: Head moves to (5, 6)
    
    if new_head == (5, 6):
        print("FAIL: Snake allowed 180 degree turn (moved DOWN).")
    elif new_head == (5, 4):
         print("PASS: Snake ignored 180 turn and moved UP.")
    else:
        print(f"UNKNOWN: Snake moved to {new_head}")

if __name__ == "__main__":
    test_initialization()
    test_180_turn()
