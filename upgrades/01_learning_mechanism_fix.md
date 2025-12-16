# Upgrade Proposal: Fix Learned Helplessness

> [!IMPORTANT]
> **Priority**: Critical
> **Status**: Approved

> **Target Component**: `snake_ai/main.py`

## 1. The Problem: "Learned Helplessness"
Analysis of the current `main.py` reveals a critical flaw in the learning mechanism:

1.  **Ignored Rewards**: The immediate reward from `game.step()` (e.g., +1 for food, -0.01 for time) is **discarded** during self-play.
2.  **Impossible Thresholds**: The `final_value` backfilled into memory is binary/ternary based on the *total game score*:
    *   Score > 10: `+1` (Good)
    *   Score > 2: `0` (Neutral)
    *   Score <= 2: `-1` (Bad)
3.  **The Collapse**: An untrained random agent almost *never* reaches Score > 2. Thus, **100% of its training data** is labeled `-1` (Bad).
4.  **Result**: The network learns that *every* state and *every* action leads to pain. To minimize loss, it collapses to a deterministic, low-entropy policy (e.g., "Always go Left"), resulting in steady 0 scores.

## 2. The Solution: Discounted Returns (Monte Carlo)
We must shift from "Game Outcome" feedback to "Action Consequence" feedback.

### Key Changes
1.  **Track Rewards**: Store the specific reward `r` received at each step `t`.
2.  **Discounted Return ($G_t$)**: Instead of a flat `final_value`, calculate the return for each step:
    $$ G_t = r_t + \gamma \cdot G_{t+1} $$
    *   Where $\gamma$ (Gamma) is a discount factor (e.g., 0.9).
3.  **Backpropagation of Value**:
    *   If the snake eats food at Step 10 ($r_{10} = +1$):
    *   Step 9 gets: $0 + 0.9 \times 1 = 0.9$
    *   Step 8 gets: $0 + 0.9 \times 0.9 = 0.81$
    *   ...
    *   Step 0 gets a small positive signal indicating "This path leads to food."

## 3. Implementation Plan
### Modify `main.py`
- [ ] Update `Trainer.self_play` to store `(state, policy, reward)` tuples instead of waiting for `final_value`.
- [ ] Implement a `calculate_returns(rewards, gamma=0.9)` helper function.
- [ ] Replace the "Backfill memory" loop with the new discounted returns logic.

### Expected Outcome
- **Immediate Improvement**: The agent will start seeking food within the first few epochs because "Eating" now provides an accessible gradient of success.
- **Elimination of Bias**: The "Always Left" behavior will vanish as the Value Head learns to predict positive integers for food-seeking states.
