import argparse
import random
import numpy as np
import torch

from game import SnakeGame
from model import SnakeNet
from mcts import MCTS


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_pov(game: SnakeGame) -> np.ndarray:
    state = game.get_state()
    x = np.zeros((4, game.board_size, game.board_size), dtype=np.float32)

    # Channel 0: Lifetime/flow of the body
    snake = getattr(game, "snake", [])
    L = len(snake)
    if L > 1:
        for i in range(1, L):
            cx, cy = snake[i]
            x[0, cy, cx] = (L - i) / L

    # Channel 1: Head
    x[1] = (state == 2).astype(np.float32)
    # Channel 2: Food
    x[2] = (state == 3).astype(np.float32)
    # Channel 3: Hunger
    hunger_limit = max(1, getattr(game, "hunger_limit", 100))
    hunger = float(getattr(game, "steps_since_eaten", 0)) / hunger_limit
    x[3].fill(hunger)

    # POV rotation: head faces "up"
    k = game.direction
    return np.rot90(x, k, axes=(1, 2)).copy()


@torch.no_grad()
def evaluate(model_path: str, board_size: int, episodes: int, seed: int, sims: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SnakeNet(board_size=board_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores = []
    reasons = []
    steps_list = []

    for ep in range(episodes):
        set_all_seeds(seed + ep)
        game = SnakeGame(board_size=board_size)
        game.reset()

        mcts = MCTS(model, n_simulations=sims, dirichlet_epsilon=0.0)
        mcts.reset()

        steps = 0
        while not game.done:
            p_probs, _ = mcts.search(game)
            rel_action = int(np.argmax(p_probs))
            abs_action = (game.direction + (rel_action - 1)) % 4
            _s, _r, _d = game.step(abs_action)
            mcts.update_root(rel_action)
            steps += 1

        scores.append(game.score)
        reasons.append(game.death_reason)
        steps_list.append(steps)

    wins = reasons.count("won")
    print(f"Episodes: {episodes} | Board: {board_size} | Sims: {sims}")
    print(f"Win%: {wins/episodes*100:.1f}% | AvgScore: {sum(scores)/episodes:.2f} | MaxScore: {max(scores)}")
    print("Death reasons:", {r: reasons.count(r) for r in sorted(set(reasons))})
    print(f"AvgSteps: {sum(steps_list)/episodes:.1f} | MaxSteps: {max(steps_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Path to model weights (.pth)")
    parser.add_argument("--board_size", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        board_size=args.board_size,
        episodes=args.episodes,
        seed=args.seed,
        sims=args.sims,
    )

