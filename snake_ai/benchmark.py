from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from game import SnakeGame
from mcts import MCTS


@dataclass(frozen=True)
class BenchmarkConfig:
    board_size: int = 6
    episodes: int = 50
    sims: int = 5
    seed: int = 0


def _set_all_seeds(seed: int) -> None:
    # Keep eval deterministic across code changes.
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def run_benchmark(
    model: torch.nn.Module,
    cfg: BenchmarkConfig,
    generation: Optional[int] = None,
) -> Dict[str, object]:
    """Deterministic, comparable benchmark.

    - Greedy action selection: argmax over MCTS visit distribution
    - No Dirichlet noise
    - Fixed per-episode RNG seeds
    """

    device = next(model.parameters()).device
    model.eval()

    scores: list[int] = []
    steps_list: list[int] = []
    reasons: list[str] = []

    t0 = time.time()

    for ep in range(int(cfg.episodes)):
        _set_all_seeds(int(cfg.seed) + ep)
        game = SnakeGame(board_size=int(cfg.board_size))
        game.reset()

        mcts = MCTS(model, n_simulations=int(cfg.sims), dirichlet_epsilon=0.0)
        mcts.reset()

        guaranteed_win_path = None
        steps = 0
        while not game.done:
            if guaranteed_win_path:
                rel_action = guaranteed_win_path.pop(0)
            else:
                p_probs, _entropy, win_path, _sims_used, _timing = mcts.search(game)
                if win_path is not None:
                    guaranteed_win_path = win_path
                    rel_action = guaranteed_win_path.pop(0)
                else:
                    rel_action = int(np.argmax(p_probs))
            abs_action = (game.direction + (rel_action - 1)) % 4
            _s, _r, _d = game.step(abs_action)
            if not guaranteed_win_path:
                mcts.update_root(rel_action)
            steps += 1

        scores.append(int(game.score))
        steps_list.append(int(steps))
        reasons.append(str(game.death_reason))

    dt = time.time() - t0

    wins = reasons.count("won")
    avg_score = float(np.mean(scores)) if scores else 0.0
    max_score = int(np.max(scores)) if scores else 0
    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
    max_steps = int(np.max(steps_list)) if steps_list else 0

    # Stable columns (avoid dynamic dict columns per new death reason)
    death_wall = reasons.count("wall")
    death_body = reasons.count("body")
    death_timeout = reasons.count("timeout")
    death_starvation = reasons.count("starvation")

    row: Dict[str, object] = {
        # Folder name like "train_v25" (filled by append_benchmark_csv)
        "Run": "",
        "Gen": int(generation) if generation is not None else "",
        "Board": int(cfg.board_size),
        "Episodes": int(cfg.episodes),
        "Sims": int(cfg.sims),
        "Seed": int(cfg.seed),
        "Device": str(device),
        "Time": float(dt),
        "WinPct": float((wins / len(reasons) * 100.0) if reasons else 0.0),
        "AvgScore": float(avg_score),
        "MaxScore": int(max_score),
        "AvgSteps": float(avg_steps),
        "MaxSteps": int(max_steps),
        "DeathWall": int(death_wall),
        "DeathBody": int(death_body),
        "DeathTimeout": int(death_timeout),
        "DeathStarvation": int(death_starvation),
        "DeathWon": int(wins),
    }
    return row


def append_benchmark_csv(run_dir: str, row: Dict[str, object], filename: str = "benchmark.csv") -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, filename)

    fieldnames = [
        "Run",
        "Gen",
        "Board",
        "Episodes",
        "Sims",
        "Seed",
        "Device",
        "Time",
        "WinPct",
        "AvgScore",
        "MaxScore",
        "AvgSteps",
        "MaxSteps",
        "DeathWall",
        "DeathBody",
        "DeathTimeout",
        "DeathStarvation",
        "DeathWon",
    ]

    file_exists = os.path.isfile(path)
    run_name = os.path.basename(os.path.normpath(run_dir))
    row = dict(row)
    row["Run"] = row.get("Run") or run_name

    # If we changed columns, upgrade the CSV header while preserving rows.
    if file_exists:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and header != fieldnames:
                # Preserve existing rows by reading with the old header then rewriting.
                f.seek(0)
                dict_reader = csv.DictReader(f)
                data = list(dict_reader)
                with open(path, "w", newline="") as f_out:
                    w_out = csv.DictWriter(f_out, fieldnames=fieldnames)
                    w_out.writeheader()
                    for r in data:
                        # Old rows won't have Run; keep empty.
                        w_out.writerow(r)

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    return path
