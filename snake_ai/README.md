# MCTS Lite Snake AI

This directory contains a Python implementation of a Snake AI using **Monte Carlo Tree Search (MCTS)** guided by a **Deep Neural Network**, inspired by the AlphaZero algorithm.

## 🧠 Architecture

The system consists of three main components:

1.  **The Environment (`game.py`)**
    *   A headless implementation of Snake using a grid-based state representation.
    *   Optimized for training efficiency (no rendering overhead).
    *   **State**: $N \times N$ matrix (0: Empty, 1: Body, 2: Head, 3: Food).
    *   **Rewards**: `+5` (Eat), `-1` (Die), `-0.01` (Base Step penalty + Hunger), `+10` (Victory).

2.  **The Neural Network (`model.py`)**
    *   **Type**: Convolutional Neural Network (PyTorch).
    *   **Input**: $(3, H, W)$ tensor representing Body, Head, and Food. **Rotated** so the Head always faces Up (POV).
    *   **Outputs**:
        *   **Policy Head**: Probabilities for the next best **Relative** move (Left, Straight, Right).
        *   **Value Head**: Scalar estimating the **Expected Discounted Return** ($G_t$) from the current state.

3.  **The Planner (`mcts.py`)**
    *   Performs lookahead simulations to improve upon the Network's raw intuition.
    *   Uses **PUCT** (Predictor + Upper Confidence Bound applied to Trees) for node selection.

## 🔄 Training Process

Refactored to approximate **AlphaZero** more closely:

1.  **Self-Play (Generation)**: The agent plays `N` games using MCTS.
2.  **Data Augmentation**: States are **horizontally flipped** to double the training data (Rotation is disabled due to POV).
3.  **Training**: The network trains on the replay buffer for `M` epochs.
4.  **Repeat**: The improved network generates better data for the next generation.

## 📂 File Structure

*   `main.py`: The entry point. Manages the **Generation-based Training Loop** and **Data Augmentation**.
*   `game.py`: The core game logic.
*   `model.py`: PyTorch model definition.
*   `mcts.py`: MCTS algorithm logic.
*   `visualize.py`: Script to watch the trained agent play.

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install torch numpy pygame
    ```

2.  **Start Training**:
    ```bash
    python snake_ai/main.py
    ```
    This will start the training loop. Progress is saved to `snake_net.pth` and `training_report.csv` after every generation.

    To **resume** training from the latest checkpoint:
    ```bash
    python snake_ai/main.py --resume
    ```
    You can also resume from a specific version (e.g., `train_v1`) by passing the version number:
    ```bash
    python snake_ai/main.py --resume 1
    ```

## ⚙️ Configuration

Hyperparameters can be adjusted in `main.py`:

*   `BOARD_SIZE`: Grid dimensions (default: 6x6 for speed).
*   `SIMULATIONS`: MCTS searches per move (default: 30).
*   `GAMES_PER_GEN`: Games played before a training step (default: 20).
*   `MEMORY_SIZE`: Size of the replay buffer (default: 20000).
