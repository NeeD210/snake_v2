# Neural Network & MCTS Implementation: A University Case Study

**Subject**: Advanced AI Architectures (AlphaZero Style)  
**Case Study**: Snake AI (`snake_v2`)  
**Date**: December 2025

---

## 1. Introduction & Use Case

This document explains the implementation of a Self-Play Reinforcement Learning system designed to master the game of Snake. The architecture is inspired by **AlphaZero**, utilizing two distinct but cooperating components:

1.  **Neural Network (NN)**: The "Intuition". It looks at the board and instantly predicts the best move and the game's outcome.
2.  **Monte Carlo Tree Search (MCTS)**: The "Reasoning". It uses the NN's intuition to look ahead, simulate future possibilities, and refine the decision.

**Inter-process Goal**:  
The detailed, slow reasoning of MCTS produces high-quality data. The Neural Network trains on this data to approximate the MCTS's output. Over time, the "Intuition" becomes as good as the "Reasoning" was, allowing the "Reasoning" to reach even deeper levels of play.

---

## 2. The Mental Model: "Brain" vs. "Planner"

Imagine a Chess master playing a game:
*   **Glance (NN)**: They look at the board and think, *"Moving the knight looks good, and I stick feel I'm winning."*
*   **Calculation (MCTS)**: They then pause to calculate. *"If I move the knight, they take my pawn... but then I check..."*.
*   **Action**: After calculating, they might realize the knight move was actually bad, and a pawn move is better.
*   **Learning**: The master remembers, *"Next time I see this pattern, don't trust the knight move; the pawn is better."*

In our code:
*   **`SnakeNet` (model.py)** is the **Glance**.
*   **`MCTS` (mcts.py)** is the **Calculation**.
*   **`Trainer` (main.py)** ensures the **Learning** happens.

---

## 3. The Neural Network (`SnakeNet`)

The Neural Network acts as a function approximator $f_\theta(s) = (\mathbf{p}, v)$.

### 3.1 Input: The "Eyes"
*   **Source**: `game.get_state()` transformed by `process_state`.
*   **Shape**: `(3, BoardSize, BoardSize)`.
*   **Channels**:
    1.  **Body**: Binary map of snake body segments.
    2.  **Head**: Binary map of the snake's head.
    3.  **Food**: Binary map of the food location.
*   **POV Transformation**: Crucially, the board is **rotated** so the snake's head is always facing "UP". This creates **invariance**—a left turn looks the same whether the snake is moving North or East on the global grid.

### 3.2 Outputs
1.  **Policy Head ($\mathbf{p}$)**: A probability distribution over 3 actions: `[Left, Straight, Right]`.
    *   *Meaning*: "How promising does each move look immediately?"
2.  **Value Head ($v$)**: A scalar between -1 and 1.
    *   **Activation**: `Tanh` (Hyperbolic Tangent) to strictly bound outputs.
    *   *Meaning*: "How likely am I to win from this state?" (1 = Win, -1 = Lose/Die).

### 3.3 Architecture
*   **Type**: Convolutional Neural Network (CNN).
*   **Layer Channels**: `[3 -> 32 -> 64 -> 64]`.
    *   Input: 3 Channels (Head, Body, Food)
    *   Hidden Layers: 32, 64, and 64 filters respectively.
*   **Body**: 3 Convolutional blocks with Batch Normalization and ReLU. This extracts spatial features (e.g., "Food is 3 blocks ahead", "Corner on the left").
*   **Dual Heads**: The network splits at the end into the Policy and Value heads (a standard "Two-Headed Monster" architecture).

---

## 4. Monte Carlo Tree Search (`MCTS`)

MCTS is the engine that generates the ground-truth data for training.

### 4.1 The Loop (Algorithm)
For every move in the game, MCTS performs `N` simulations (e.g., 50). Each simulation follows 4 steps:

1.  **Select (How we find a leaf)**:
    *   Start at the root.
    *   At every step, ask: "Which child has the highest PUCT score?"
    *   Move to that child. Repeat until we hit a node that has **no children** (a leaf).
    *   *Note on Loops*: This is a **Tree Search**, not a Graph Search. If the snake returns to the exact same board state, MCTS treats it as a *new* node deeper in the tree. We rely on the **Dynamic Hunger Penalty** to teach the AI that finding a loop is "bad" compared to just eating the apple quickly.

2.  **Expand**: When a leaf node is reached, ask the **Neural Network** to evaluate it.
    *   NN returns $\mathbf{p}$ (Policy) and $v$ (Value).
    *   Create child nodes for valid moves, initializing specific priors with $\mathbf{p}$.
    *   *Note*: This happens **immediately** on the first visit. The node is no longer a leaf after this step.

3.  **Backup (Backpropagation)**: Propagate the value $v$ *up* the tree.
    *   Update the visit count ($N$) and mean value ($Q$) for every node in the path.
    *   *Note*: Unlike traditional MCTS, we do **not** play random moves to the end of the game ("rollouts"). We trust the NN's value estimate $v$.

### 4.2 The Decision
After `N` simulations, we stop. The "real" policy $\pi$ is derived from the visit counts, not the scores:
$$ \pi(a) = \frac{N(a)^{\frac{1}{T}}}{\sum N(b)^{\frac{1}{T}}} $$
*   The system picks a move based on $\pi$.
*   By visiting nodes more often, MCTS has "smoothed out" the raw noise from the NN and found a safer path.

---

## 5. The Training Cycle (`main.py`)

The system improves iteratively through **Self-Play**.

### Step 1: Data Collection (Parallel)
*   Multiple worker processes play games using the *current* network.
*   In each step:
    1.  MCTS thinks (Simulates).
    2.  MCTS acts (chooses move based on visit counts).
    3.  We store `(State, MCTS_Policy, Reward)` in memory.

### Step 2: Goal Calculation
*   **Game Score**: The number of apples eaten (e.g., 5). This is just for humans to watch.
*   **Reward Signal**: The "Outcome" for the AI. It comes from the **Reward Mechanisms** in `game.py`:
    *   **+2** for eating an apple.
    *   **-1** for dying (hitting a wall or itself).
    *   **Dynamic Hunger**: Penalty starts at -0.01 and increases over time (forcing the snake to eat).
    *   **+0.1** for getting closer to food (Reward Shaping).
*   The game ends (Win/Die/timeout).
*   The final reward (e.g., -1 for death) is backpropagated to all previous steps in that game (Discounted Return).
*   **Normalization**: The returns are clipped to `[-1, 1]` to ensure they fit the Value Head's range.
*   Now we have tuples: `(State, Target_Policy, Target_Value)`.
    *   **State**: What the snake saw.
    *   **Target_Policy**: What MCTS decided was best (Robust).
    *   **Target_Value**: The actual outcome (Discounted Return).
        *   *Note*: In 2-player games (Chess), this is "Who Won". In Snake, this is "Did we eat or die?".

### Step 3: Backpropagation (Learning)
*   The NN is trained to minimize the error:
    $$ Loss = (v - Target\_Value)^2 - \pi \cdot \log(\mathbf{p}) $$
*   **Policy Loss** (Cross-Entropy): Force NN probability $\mathbf{p}$ to match MCTS distribution $\pi$.
*   **Value Loss** (MSE): Force NN value prediction $v$ to match actual game outcome.
*   **Gradient Clipping**: We clip gradients to norm 1.0 to prevent training instability.

---

## 6. Summary of Interaction

| Component | Responsibility | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| **Neural Net** | Rapid Intuition | Board State (Tensor) | `P` (Suggestions), `V` (Win Estimate) |
| **MCTS** | Deep Reasoning | Game Rules, `P`, `V` | `Pi` (Refined Move Probabilities) |
| **Trainer** | Experience Management | Game History | Updated Neural Net Weights |

**Conclusion**: The Neural Network bootstraps the MCTS, making the search more efficient. The MCTS provides better training targets than the Neural Network could generate on its own, allowing the Network to improve. This "loop" allows the AI to discover strategies (like surrounding food or trapping opponents) without human instruction.
