## snake_v2 — NN + MCTS Snake (AlphaZero-style)

This repo trains a Snake agent via **self-play** using a small CNN (`SnakeNet`) guided by **MCTS** (`mcts.py`). Goal: learn to **fill the board** (first milestone: do it at least once during training).

### Requirements
- **Python**: 3.9+
- **Packages**:
  - `numpy`
  - `torch` (install from the official PyTorch instructions for your CPU/GPU)
  - `pygame` (for visualization / manual play)
  - `psutil` (optional, for training monitor stats)

Example install (CPU-only PyTorch may differ on your machine):

```bash
pip install numpy pygame psutil
pip install torch
```

### Train
Training is in `snake_ai/main.py`. It creates runs under `snake_ai/experiments/train_v*/`.

From repo root:

```bash
python snake_ai/main.py --board-size 6
```

Helpful flags:
- **Fast preset (recommended on laptops)**:

```bash
python snake_ai/main.py --board-size 6 --fast
```

- **Control parallelism (Windows spawn overhead makes “max cores” slower sometimes)**:

```bash
python snake_ai/main.py --board-size 6 --workers 4
```

- **Pin compute (disables schedules)**
  - Use this if you want predictable runtime:

```bash
python snake_ai/main.py --board-size 6 --sims 64 --games 12 --epochs 1
```

- **Resume the latest run**:

```bash
python snake_ai/main.py --resume
```

### Evaluate (deterministic MCTS eval)
Evaluates a saved model with MCTS and no Dirichlet noise:

```bash
python snake_ai/eval.py --model snake_ai/experiments/train_v15/snake_net.pth --board_size 6 --episodes 200 --sims 200
```

### Visualize
Watch a trained model play (NN-only or MCTS-assisted):

```bash
python snake_ai/visualize.py --model snake_ai/experiments/train_v15/snake_net.pth --board_size 6
python snake_ai/visualize.py --model snake_ai/experiments/train_v15/snake_net.pth --board_size 6 --mcts --sims 200
```

### Manual play (sanity check the environment)

```bash
python snake_ai/manual_play.py
```

### Notes (performance)
- Training uses **compute-efficient schedules** by default:
  - fewer MCTS sims early-game and early generations
  - more sims only in **endgame** (when filling the board is hard)
  - games/gen ramps up over time
- If you set `--sims` or `--games`, schedules are disabled for predictable compute.


