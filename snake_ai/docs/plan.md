# MCTS Lite Snake AI - Implementation Guide

This project implements a reinforcement learning agent for the game Snake using a simplified AlphaZero approach. It uses a lightweight Neural Network to guide a Monte Carlo Tree Search, prioritizing compute efficiency over massive scale.

## 1. Technology Stack

*   **Language**: Python 3.9+
*   **Deep Learning**: PyTorch (CPU inference is sufficient; lightweight operations).
*   **Data Handling**: NumPy (Board representation and vector math).
*   **Visualization**: PyGame (For rendering the game loop).

## 2. Architecture Overview

The system is composed of three distinct modules that interact in a loop:

1.  **The Environment (Game)**: The logic of Snake.
2.  **The Brain (Net)**: A neural network that predicts the best move (Policy) and the winning chance (Value) of a given state.
3.  **The Planner (MCTS)**: A search algorithm that simulates future moves to correct the Brain's mistakes.

### Directory Structure

```plaintext
snake_ai/
├── main.py           # Entry point (Train or Play modes)
├── game.py           # Snake game logic (headless & efficient)
├── model.py          # PyTorch Neural Network definition
├── mcts.py           # Monte Carlo Tree Search implementation
└── utils.py          # Helper functions (plotting, logging)
```

## 3. Step-by-Step Implementation

### Step 1: The Environment (`game.py`)
We need a fast, "headless" simulation. The Neural Network sees a matrix, not pixels.

*   **Grid**: $N \times N$ matrix.
*   **Encoding**:
    *   `0`: Empty space
    *   `1`: Snake Body
    *   `2`: Snake Head
    *   `3`: Food
*   **Action Space**: `[0: Up, 1: Right, 2: Down, 3: Left]`
*   **Reward Function**:
    *   `+1`: Ate food.
    *   `0`: Survived a step.
    *   `-1`: Died (Hit wall or self).
    *   *(Optimization)*: Add a small negative reward (`-0.01`) per step to discourage looping forever.

### Step 2: The Neural Network (`model.py`)
This network acts as the "intuition." It provides a fast approximation to guide the MCTS.

*   **Inputs**: A tensor of shape `(3, Height, Width)`.
    *   Channel 0: Binary mask of the snake body (1 where body is, 0 elsewhere).
    *   Channel 1: Binary mask of the snake head.
    *   Channel 2: Binary mask of the food.

*   **Architecture**:
    *   **Convolutional Body**:
        *   `Conv2d(3, 32, kernel=3, padding=1)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU`
        *   `Conv2d(32, 64, kernel=3, padding=1)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU`
        *   `Conv2d(64, 64, kernel=3, padding=1)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU`
    *   **Policy Head (Actor)**:
        *   `Conv2d(64, 2, kernel=1)` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(size, 4)`
        *   **Output**: Logits for 4 moves.
    *   **Value Head (Critic)**:
        *   `Conv2d(64, 1, kernel=1)` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(size, 64)` $\rightarrow$ `Linear(64, 1)` $\rightarrow$ `Tanh`
        *   **Output**: Scalar between -1 (Lose) and 1 (Win).

### Step 3: Monte Carlo Tree Search (`mcts.py`)
This is the core of the "Lite" logic. It runs ~50 simulations before deciding the actual move.

*   **Node Class**: Stores `N` (visit count), `Q` (mean value), `P` (prior probability from NN), and `children` (map of moves to Nodes).
*   **The Search Loop** (Repeat 50 times):
    1.  **Select**: Start at root. Traverse down using PUCT Formula:
        $$U(s, a) = Q(s, a) + C_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum N}}{1 + N(s, a)}$$
        Choose the child with the highest Score until a leaf node is reached.
    2.  **Expand & Evaluate**:
        *   If the node is a terminal state (Game Over), return actual result (-1 or 1).
        *   Otherwise, pass the board state into `model.py`.
        *   Get Policy vector and Value scalar.
        *   Create child nodes for valid moves using Policy.
    3.  **Backup**: Propagate the Value up the tree. Update `Q` (running average) and `N` (count) for every node in the path.

### Step 4: The Training Loop (`main.py`)
This replaces standard reinforcement learning (DQN). The agent learns from its own search results.

1.  **Self-Play Phase**:
    *   Start a new game.
    *   While game not over:
        *   Run MCTS 50 times.
        *   **Action**: Pick move based on MCTS visit counts (e.g., if "Up" was visited 40 times and "Down" 10, pick "Up" 80% of the time).
        *   **Store**: Save `(State, Search_Probabilities, Winner)` to a history buffer.
        *   Execute move in `game.py`.
    *   When game ends, assign the final result (Winner) to all stored steps in that game.

2.  **Training Phase**:
    *   Sample a batch from the history buffer.
    *   **Loss Function**:
        *   Minimize error between NN Policy and MCTS Search Probabilities (Cross Entropy).
        *   Minimize error between NN Value and Actual Winner (MSE).
        *   $$Loss = (z - v)^2 - \pi^T \log(p) + c||\theta||^2$$
    *   Update network weights.

## 4. Specific "Lite" Optimizations for Snake
To ensure this runs perfectly on low compute, we apply these specific constraints:

1.  **State Canonicalization**: Snake is rotationally symmetric. Before feeding a state to the NN, rotate the grid so the Snake's head always faces "North" relative to the input tensor. This reduces the state space by 4x.
2.  **Pruning**: In the Select phase of MCTS, immediately discard moves that lead to instant death (hitting a wall or immediate body segment). Do not waste simulations on suicide moves.
3.  **Temperature Parameter ($\tau$)**:
    *   **Early Game**: $\tau = 1$. Pick moves probabilistically based on visit counts to encourage exploration.
    *   **Late Game (Length > 20)**: $\tau \to 0$. Pick the move with the absolute highest visit count (Deterministic) to ensure precision.