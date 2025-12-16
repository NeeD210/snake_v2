# Analysis Report: NN+MCTS Stagnation (Train v7)

**Date**: 2025-12-14
**Subject**: Experiment v7 Stagnation Analysis
**Author**: Antigravity (Data Scientist Agent)

## 1. Executive Summary
The training stagnation at Avg Score ~1.15 is primarily caused by a **critical architectural mismatch** between the Neural Network's Value Head (bounded by Tanh $[-1, 1]$) and the Reinforcement Learning Target Values (which exceed $1.0$ due to reward accumulation). The model effectively minimizes loss by predicting "Max Value" (1.0) for any survival state, failing to distinguish between "Surviving" and "Eating".

## 2. Methodology
- **Data Source**: `snake_ai/experiments/train_v7/training_report.csv`
- **Codebase Review**: `snake_ai/model.py`, `snake_ai/game.py`, `snake_ai/mcts.py`
- **Telemetry Analysis**: Correlated `Loss`, `AvgScore`, and `DeathReason` trends over 44 generations.

## 3. Findings

### 3.1 The "Happy Loser" Paradox
- **Observation**: `Loss` decreases steadily (1.46 -> 0.47) while `AvgScore` stagnates (~1.15).
- **Diagnosis**: The model is successfully learning to map inputs to the training targets, but the targets are flawed for the current architecture.
- **Root Cause**: 
    - **Value Head**: Uses `torch.tanh(self.value_fc2(v))`, enforcing output in range $[-1, 1]$.
    - **Rewards**: Eat = +2, Win = +10. Discounted returns ($G_t$) for eating even a single apple often exceed $2.0$.
    - **Result**: The "Eating" return (>2.0) is clamped to 1.0 by the Tanh saturation. The "Surviving" return (avoiding death) approaches 0 or slightly negative. 
    - **Consequence**: The model learns that "Eating" and "Surviving" both have the maximum possible value of 1.0. It has no incentive to take risks to eat; it simply tries to exist in the "1.0 zone" (alive), leading to random walks and eventual timeouts or self-collisions.

### 3.2 High "Death by Body"
- **Observation**: `DeathBody` is consistently high (~14-16 per 20 games) even in late generations.
- **Diagnosis**: On a small 6x6 board, a snake that only cares about "not hitting a wall" (Value > -1) will eventually spiral into itself or trap itself if it lacks a long-term plan to keep space open.
- **Contributing Factor**: `SIMULATIONS = 30` is extremely low.
    - MCTS Depth is roughly $\log_{3}(30) \approx 3$. The agent can only "see" 3 moves ahead.
    - It cannot foresee trapping itself in a cul-de-sac 5 moves away.

### 3.3 Entropy Decay
- **Observation**: Entropy drops from 0.79 to 0.55.
- **Interpretation**: The Policy Head is becoming confident. Since it's not learning to eat, it is likely becoming confident in "safe" moves (center of board) or repeating patterns that avoid immediate wall death, reinforcing the stagnation.

## 4. Recommendations / Upgrades

### Upgrade 1: Unbound Value Head (Critical)
**Proposal**: Remove the `Tanh` activation from the Value Head in `model.py`.
- **Why**: Allows the network to predict the true magnitude of returns (e.g., 2.0, 5.0).
- **Effect**: The model will learn that `Value(Eating State) = 2.0` > `Value(Survival State) = 0.0`. This provides the necessary gradient to pursue food.

### Upgrade 2: Boost Simulation Count
**Proposal**: Increase `SIMULATIONS` from 30 to **100**.
- **Why**: 30 simulations is insufficient for a 6x6 board to avoid simple traps. 
- **Effect**: Deeper search horizon allows MCTS to identify "Body Death" traps and find food beyond the immediate vicinity.

### Upgrade 3: Reward Normalization (Optional but Good)
**Proposal**: If Tanh is preferred for stability, normalize rewards:
- Apple: +1.0
- Death: -1.0
- Step: -0.01
*Note*: Even with this, multiple apples would exceed 1.0. **Upgrade 1 (Unbound)** is preferred for regression-style score maximization.

## 5. Proposed Next Steps
1.  Apply **Upgrade 1** and **Upgrade 2**.
2.  Start a new training run (`train_v8`).
3.  Monitor if `AvgScore` breaks the 1.5 barrier within 20 generations.
