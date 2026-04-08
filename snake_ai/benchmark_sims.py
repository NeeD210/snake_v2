"""
MCTS Simulation Budget Sweep

Evaluates a trained model at different MCTS simulation counts to find
the optimal tradeoff between decision quality and throughput.

Usage:
    python snake_ai/benchmark_sims.py <experiment_dir> [--episodes 30] [--board 6]

Example:
    python snake_ai/benchmark_sims.py snake_ai/experiments/train_v41
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from game import SnakeGame
from model import SnakeNet
from mcts import MCTS


SIM_BUDGETS = [10, 25, 50, 75, 100, 128, 200]


def load_model(experiment_dir: str, device: torch.device) -> torch.nn.Module:
    """Load the best model from an experiment directory."""
    best_path = os.path.join(experiment_dir, "best_snake_net.pth")
    latest_path = os.path.join(experiment_dir, "snake_net.pth")

    path = best_path if os.path.exists(best_path) else latest_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found in {experiment_dir}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # Infer model config from checkpoint keys
    state_dict = checkpoint if isinstance(checkpoint, dict) and "input_conv.weight" in checkpoint else checkpoint
    model = SnakeNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model from: {path}")
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, board_size: int, episodes: int, sims: int, seed: int = 42):
    """
    Play `episodes` games with greedy MCTS (no noise, no temperature).
    Returns dict with score/throughput stats.
    """
    device = next(model.parameters()).device
    scores = []
    steps_list = []
    reasons = []

    t0 = time.time()

    for ep in range(episodes):
        # Per-episode seed for reproducibility
        np.random.seed(seed + ep)
        import random
        random.seed(seed + ep)

        game = SnakeGame(board_size=board_size)
        game.reset()

        mcts = MCTS(model, n_simulations=sims, dirichlet_epsilon=0.0)
        mcts.reset()

        steps = 0
        while not game.done:
            p_probs, _entropy, _win_path, _sims_done, _timing = mcts.search(game)
            rel_action = int(np.argmax(p_probs))
            abs_action = (game.direction + (rel_action - 1)) % 4
            game.step(abs_action)
            mcts.update_root(rel_action)
            steps += 1

        scores.append(game.score)
        steps_list.append(steps)
        reasons.append(game.death_reason)

    dt = time.time() - t0
    wins = reasons.count("won")
    games_per_min = (episodes / dt) * 60 if dt > 0 else 0

    return {
        "sims": sims,
        "avg_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "win_pct": (wins / episodes) * 100,
        "avg_steps": float(np.mean(steps_list)),
        "games_per_min": games_per_min,
        "total_time": dt,
        "episodes": episodes,
        "deaths": {
            "wall": reasons.count("wall"),
            "body": reasons.count("body"),
            "starvation": reasons.count("starvation"),
            "won": wins,
        },
    }


def print_table(results: list):
    """Print a formatted comparison table."""
    # Header
    print()
    print(f"{'Sims':>6} | {'AvgScore':>9} | {'MaxScore':>9} | {'WinPct':>7} | {'Games/Min':>10} | {'AvgSteps':>9} | {'Score×Speed':>12} | Deaths")
    print("-" * 105)

    for r in results:
        score_x_speed = r["avg_score"] * r["games_per_min"]
        deaths = r["deaths"]
        death_str = f"W:{deaths['wall']} B:{deaths['body']} S:{deaths['starvation']} Won:{deaths['won']}"
        print(
            f"{r['sims']:>6} | "
            f"{r['avg_score']:>9.2f} | "
            f"{r['max_score']:>9} | "
            f"{r['win_pct']:>6.1f}% | "
            f"{r['games_per_min']:>10.1f} | "
            f"{r['avg_steps']:>9.1f} | "
            f"{score_x_speed:>12.1f} | "
            f"{death_str}"
        )

    print()

    # Recommendation
    best_quality = max(results, key=lambda r: r["avg_score"])
    best_efficiency = max(results, key=lambda r: r["avg_score"] * r["games_per_min"])

    print(f"Best quality:    {best_quality['sims']} sims (AvgScore={best_quality['avg_score']:.2f})")
    print(f"Best efficiency: {best_efficiency['sims']} sims (Score×Speed={best_efficiency['avg_score'] * best_efficiency['games_per_min']:.1f})")
    print()


def save_csv(results: list, output_path: str):
    """Save results to CSV."""
    import csv
    fieldnames = ["Sims", "AvgScore", "MaxScore", "MinScore", "WinPct", "AvgSteps", "GamesPerMin", "ScoreXSpeed", "TotalTime", "Episodes"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Sims": r["sims"],
                "AvgScore": f"{r['avg_score']:.2f}",
                "MaxScore": r["max_score"],
                "MinScore": r["min_score"],
                "WinPct": f"{r['win_pct']:.1f}",
                "AvgSteps": f"{r['avg_steps']:.1f}",
                "GamesPerMin": f"{r['games_per_min']:.1f}",
                "ScoreXSpeed": f"{r['avg_score'] * r['games_per_min']:.1f}",
                "TotalTime": f"{r['total_time']:.2f}",
                "Episodes": r["episodes"],
            })
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep MCTS simulation budgets")
    parser.add_argument("experiment_dir", help="Path to experiment directory with model weights")
    parser.add_argument("--episodes", type=int, default=30, help="Games per sim budget (default: 30)")
    parser.add_argument("--board", type=int, default=6, help="Board size (default: 6)")
    parser.add_argument("--sims", type=str, default=None,
                        help="Comma-separated sim budgets (default: 10,25,50,75,100,128,200)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.experiment_dir, device)

    sim_budgets = SIM_BUDGETS
    if args.sims:
        sim_budgets = [int(s.strip()) for s in args.sims.split(",")]

    print(f"\nSweeping sim budgets: {sim_budgets}")
    print(f"Board: {args.board}x{args.board} | Episodes per budget: {args.episodes} | Seed: {args.seed}")
    print(f"Device: {device}\n")

    results = []
    for sims in sim_budgets:
        print(f"Evaluating {sims} sims...", end=" ", flush=True)
        r = evaluate(model, args.board, args.episodes, sims, seed=args.seed)
        print(f"done ({r['total_time']:.1f}s) — AvgScore={r['avg_score']:.2f}, Games/Min={r['games_per_min']:.1f}")
        results.append(r)

    print_table(results)

    # Save CSV next to the model
    csv_path = os.path.join(args.experiment_dir, "sim_sweep.csv")
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
