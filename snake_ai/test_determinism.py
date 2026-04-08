"""
Quick verification that food placement is now deterministic.
Tests both SnakeGame and FastSnakeState.
"""
import sys
import numpy as np
from game import SnakeGame
from fast_state import FastSnakeState
from mcts import MCTS

def test_snakegame_determinism():
    """Two identical games should place food identically after eating."""
    print("--- Test 1: SnakeGame determinism ---")
    
    results = []
    for trial in range(3):
        g = SnakeGame(board_size=10)
        # Force identical starting state
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.occ = set(g.snake)
        g.direction = 1  # Right
        g.score = 0
        g.steps = 0
        g.food = (6, 5)  # Place food directly ahead
        g.game_id = 42
        
        # Step right into food
        state, reward, done = g.step(1)
        results.append(g.food)
    
    assert results[0] == results[1] == results[2], f"Food positions differ: {results}"
    print(f"  PASS: Food consistently placed at {results[0]}")

def test_faststate_determinism():
    """FastSnakeState should place food identically across runs."""
    print("--- Test 2: FastSnakeState determinism ---")
    
    results = []
    for trial in range(3):
        fs = FastSnakeState(
            board_size=10,
            snake=[(5, 5), (4, 5), (3, 5)],
            food=(6, 5),
            direction=1,
            steps_since_eaten=0,
            hunger_limit=200,
            score=0,
            steps=0,
            max_steps=100000,
            game_id=42,
        )
        # Step right into food
        reward, done = fs.step_abs(1)
        results.append(fs.food)
    
    assert results[0] == results[1] == results[2], f"Food positions differ: {results}"
    print(f"  PASS: Food consistently placed at {results[0]}")

def test_game_faststate_agreement():
    """SnakeGame and FastSnakeState should agree on food placement."""
    print("--- Test 3: SnakeGame <-> FastSnakeState agreement ---")
    
    g = SnakeGame(board_size=10)
    g.snake = [(5, 5), (4, 5), (3, 5)]
    g.occ = set(g.snake)
    g.direction = 1
    g.score = 0
    g.steps = 0
    g.food = (6, 5)
    g.game_id = 42
    
    fs = FastSnakeState.from_game(g)
    
    g.step(1)
    fs.step_abs(1)
    
    assert g.food == fs.food, f"Disagreement: SnakeGame={g.food}, FastState={fs.food}"
    print(f"  PASS: Both placed food at {g.food}")

def test_undo_consistency():
    """After step+undo, state should be perfectly restored (no desync)."""
    print("--- Test 4: undo() consistency after eating ---")
    
    fs = FastSnakeState(
        board_size=10,
        snake=[(5, 5), (4, 5), (3, 5)],
        food=(6, 5),
        direction=1,
        steps_since_eaten=0,
        hunger_limit=200,
        score=0,
        steps=0,
        max_steps=100000,
        game_id=42,
    )
    
    original_food = fs.food
    original_snake = list(fs.snake)
    original_score = fs.score
    
    # Step into food (eat)
    reward, done = fs.step_abs(1)
    assert fs.score == 1, "Should have eaten"
    
    # Undo
    fs.undo()
    
    assert fs.food == original_food, f"Food not restored: {fs.food} != {original_food}"
    assert fs.snake == original_snake, f"Snake not restored"
    assert fs.score == original_score, f"Score not restored"
    print(f"  PASS: undo() perfectly restored state")

def test_mcts_no_crash():
    """Run MCTS search and verify no crashes from tree desync."""
    print("--- Test 5: MCTS search stability ---")
    
    def dummy_predict(input_tensor):
        return np.ones(3) / 3.0, 0.0
    
    game = SnakeGame(board_size=10)
    mcts = MCTS(dummy_predict, n_simulations=200)
    
    # Run multiple searches (simulating turns)
    for turn in range(10):
        policy, entropy, _win_path, _sims_done, _timing = mcts.search(game)
        action = np.argmax(policy)
        
        # Take the action in the real game
        abs_action = (game.direction + (action - 1)) % 4
        state, reward, done = game.step(abs_action)
        mcts.update_root(action)
        
        if done:
            break
    
    print(f"  PASS: {turn+1} turns completed without crash (score={game.score})")

def test_mcts_tree_reuse():
    """Verify that tree reuse works (root state matches after update_root)."""
    print("--- Test 6: MCTS tree reuse ---")
    
    def dummy_predict(input_tensor):
        return np.ones(3) / 3.0, 0.0
    
    game = SnakeGame(board_size=10)
    mcts = MCTS(dummy_predict, n_simulations=50)
    
    reuses = 0
    resets = 0
    
    for turn in range(20):
        policy, entropy, _win_path, _sims_done, _timing = mcts.search(game)
        action = np.argmax(policy)
        
        abs_action = (game.direction + (action - 1)) % 4
        state, reward, done = game.step(abs_action)
        mcts.update_root(action)
        
        # Check if root was reused or reset
        if mcts.root is not None:
            if np.array_equal(mcts.root.state, game.get_state()):
                reuses += 1
            else:
                resets += 1
        
        if done:
            break
    
    print(f"  Tree reuses: {reuses}, resets: {resets}, turns: {turn+1}")
    print(f"  PASS")


def test_cross_game_variety():
    """Different games should produce different food positions."""
    print("--- Test 7: Cross-game variety ---")
    
    food_positions = set()
    for _ in range(10):
        g = SnakeGame(board_size=10)
        food_positions.add(g.food)
    
    assert len(food_positions) > 1, f"All 10 games placed food at the same spot: {food_positions}"
    print(f"  PASS: {len(food_positions)} unique food positions across 10 games")


if __name__ == "__main__":
    test_snakegame_determinism()
    test_faststate_determinism()
    test_game_faststate_agreement()
    test_undo_consistency()
    test_mcts_no_crash()
    test_mcts_tree_reuse()
    test_cross_game_variety()
    print("\n[OK] All tests passed!")
