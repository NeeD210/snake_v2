
from game import SnakeGame

def verify_rewards():
    print("Verifying Dense Rewards...")
    game = SnakeGame(board_size=10)
    
    # Manually setup a scenario
    # Snake at (5, 5), Food at (8, 5)
    # Distance = 3
    game.snake = [(5, 5), (4, 5), (3, 5)]
    game.food = (8, 5)
    game.direction = 1 # Right
    
    print(f"Snake Head: {game.snake[0]}, Food: {game.food}")
    
    # Move Right (towards food)
    # New Head (6, 5), New Dist = 2.
    # Delta = 3 - 2 = 1.
    # Reward = -0.01 (step) + 0.1 (shape) = 0.09
    
    print("Moving Right (Towards Food)...")
    _, reward, _ = game.step(1)
    print(f"Reward: {reward}")
    
    if abs(reward - 0.09) < 1e-6:
        print("PASS: Positive reward received.")
    else:
        print(f"FAIL: Expected 0.09, got {reward}")

    # Now at (6, 5). Food at (8, 5). Dist = 2.
    # Move Up (away from food indirectly, but dist changes)
    # New Head (6, 4). New Dist = |6-8| + |4-5| = 2 + 1 = 3.
    # Delta = 2 - 3 = -1.
    # Reward = -0.01 + (-0.1) = -0.11
    
    print("Moving Up (Away from Food)...")
    _, reward, _ = game.step(0)
    print(f"Reward: {reward}")
    
    if abs(reward - (-0.11)) < 1e-6:
        print("PASS: Negative reward received.")
    else:
        print(f"FAIL: Expected -0.11, got {reward}")

if __name__ == "__main__":
    verify_rewards()
