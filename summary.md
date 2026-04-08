# Snake_V2: AlphaZero-Style Snake AI

## General Description
The project is a Reinforcement Learning system that trains an AI agent to play the game of Snake using an architecture inspired by DeepMind's AlphaZero. It learns purely through **self-play**, generating its own training data by playing thousands of games against itself. 

The core of the system relies on combining a **Convolutional Neural Network (CNN)** (`SnakeNet`) with **Monte Carlo Tree Search (MCTS)**. The neural network predicts the value of a board state and the probabilities of the best moves (policy), while MCTS uses these predictions to search ahead and create a stronger, more refined policy. Over time, the network learns to emulate the deep search of MCTS. The primary goal of the agent is to learn to survive and ultimately **fill the entire board**.

## System Workflow
The training pipeline consists of a continuous loop of data generation and model optimization:

1. **Self-Play (Simulation):** Multiple worker processes play games of Snake simultaneously. At each turn, MCTS uses the current Neural Network to explore possible future states.
2. **Action Selection:** Based on the MCTS search, an improved action policy is calculated. The agent selects a move, and the game state, the MCTS policy, and the ultimate game reward are stored in a replay memory buffer.
3. **Training Phase:** Once enough games are generated, the Neural Network is trained using the collected data. The model minimizes the difference between its own predictions and the improved targets:
   - **Policy Loss:** The network learns to predict the MCTS-improved policy.
   - **Value Loss:** The network learns to predict the actual outcome/score of the game.
4. **Iteration:** The newly improved model replaces the old one, and the self-play workers begin generating a new, higher-quality batch of games.

## Role of Files in `/snake_ai`

- **`main.py`**: The main entry point and orchestrator. It handles the training loop, manages multiprocessing for self-play workers, batches inference requests to the GPU/CPU, and runs the neural network optimization phase.
- **`game.py`**: Implements the base logic and rules of the standard Snake game environment (grid movement, eating food, collisions).
- **`fast_state.py`**: A highly optimized, lightweight representation of the game board. It is specifically designed to be cloned and advanced rapidly, which is essential for the heavy simulation workload of MCTS.
- **`mcts.py`**: Implements the Monte Carlo Tree Search algorithm. It expands a search tree using the network's predictions and Dirichlet noise (for exploration) to output an improved policy for state transitions.
- **`model.py`**: Contains `SnakeNet`, the PyTorch-based Convolutional Neural Network architecture that evaluates board states to provide value and policy logits.
- **`encoder.py`**: Responsible for transforming the discrete game state (snake head, body, tail, food) into multi-channel tensor matrices that the CNN can process.
- **`schedules.py`**: Implements compute-efficient training schedules, such as dynamically adjusting MCTS simulations (e.g., increasing them only in the endgame) or scaling up parameters as generations progress to save compute.
- **`visualize.py`**: A script that uses PyGame to render the environment, letting you watch a trained model play the game in real-time (using either raw NN inference or MCTS-assisted inference).
- **`eval.py`**: Evaluates the performance of a saved model objectively in read-only mode by running deterministic setups (MCTS with no exploration noise) to calculate win rates or average score.
- **`benchmark.py`**: Contains benchmarking tools to profile system throughput, taking measurements like MCTS node expansions per second or batched inference speeds.
- **`manual_play.py`**: A debugging utility that lets a human play the environment manually to verify that game rules and mechanics are functioning properly.
- **`utils.py`**: Contains shared utility functions, standard definitions, or logging helpers used across various parts of the codebase.
