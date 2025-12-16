# Professional Analysis: Snake AI Implementation & Training Report (v12)

**Date**: December 16, 2025
**Analyst**: Antigravity (Data Scientist)
**Subject**: Neural Network & MCTS Implementation Review and Training Analysis

## 1. Executive Summary
The current implementation of the Snake AI follows a robust **AlphaZero-style architecture** with correctly implemented POV (Point of View) invariance. However, the training process is currently **failing to learn**, emphasized by a **catastrophic loss spike at Generation 11** that likely reset the model's learning. The agent displays a pathological tendency to collide with its own body ("Ouroboros behavior"), suggesting a failure in value estimation for self-termination.

**Verdict**: The Architecture is sound, but the Hyperparameters or Training Stability are broken.

---

## 2. Implementation Review (`nn_mcts_implementation.md`)

### 2.1 Strengths
*   **POV Invariance**: The rotation logic (`np.rot90` based on direction) is mathematically correct. By aligning "Forward" to "Up" in the tensor, the model generalizes left/right turns regardless of global direction.
*   **Tensor Encoding**: Separating Head, Body, and Food into distinct channels (`3x10x10`) provides clear spatial features to the CNN.
*   **MCTS Integration**: The separation of "Intuition" (NN) and "Reasoning" (MCTS) is effectively designed to bootstrap learning.

### 2.2 Potential Weaknesses
*   **Action Space**: The game logic uses relative moves (Left, Straight, Right) mapped to cardinals. While efficient, this requires the model to strictly adhere to the POV perspective. Any mismatch in rotation would be fatal. (Verified as correct in code).
*   **Reward Signal**: The "Time Penalty" (-0.01) is intended to prevent loops, but it might be too weak to counter a confused Value head that predicts avoiding walls (by curling into itself) is better than traversing open space.

---

## 3. Data Analysis (`training_report.csv`)

### 3.1 The "Event Horizon" (Generation 11)
The most critical finding is the **Loss Spike**:
*   **Gen 1-10**: Loss stable at `~1.47`.
*   **Gen 11**: Loss jumps to `5.63` (+280%).
*   **Impact**: After this spike, the **Entropy** drops sharply from `0.56` to `0.48`.
*   **Interpretation**: The model suffered "Catastrophic Forgetting" or a gradient explosion. The sharp drop in entropy suggests the model collapsed into a deterministic (but wrong) policy, effectively "giving up" on exploration.

### 3.2 The "Suicidal Snake" Anomaly
*   **Metric**: `DeathBody` is persistently the highest cause of death (17-19 out of 20 games per gen).
*   **Analysis**: In early training, random agents typically die by hitting walls (`DeathWall`). A high `DeathBody` count implies the agent is **actively turning into itself**.
*   **Root Cause**: The Neural Network likely perceives "Body" pixels as "Empty" or has learned that hitting a wall is "state -1" but hitting a body is "state -0.99" (incorrect value estimation). The MCTS simulations (N=50) are insufficient to look ahead far enough to see that curling up leads to unavoidable death.

### 3.3 Stagnation
*   **Average Score**: Fluctuates between 1.05 and 1.6. Random play on a 10x10 board usually achieves ~2-3. The AI is performing *worse* than random chance.
*   **Top Score**: Stalled at 3-5 apples.

---

## 4. Recommendations

### 4.1 Immediate Actions (Stability)
1.  **Gradient Clipping**: Implement gradient clipping (e.g., `clip_grad_norm_`) to prevent the Gen 11 explosion.
2.  **Lower Learning Rate**: reduce the LR (likely from 1e-3 to 1e-4) to ensure stable convergence.

### 4.2 Architectural Tuning (Performance)
3.  **Increase MCTS Simulations**: Increase `num_simulations` from 50 to 100 or 200. The current "Reasoning" depth is too shallow to identify self-trapping loops.
4.  **Value Head Weight**: Increase the weight of the Value Loss relative to the Policy Loss. The agent needs to prioritize "Survival" (Value) over "Copying MCTS moves" (Policy) if the MCTS itself is flawed.

### 4.3 Debugging
5.  **Sanity Check Inputs**: Visualize the *actual* tensor passed to the NN during training. Ensure the "Body" channel correctly displays the tail segments relative to the rotated head.

---

**Signed,**
*Antigravity, Lead Data Scientist*
