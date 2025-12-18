import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import csv
import os
import time
import argparse
import re
from collections import deque
import torch.multiprocessing as mp
import math
import psutil
import subprocess
import queue
import traceback
import sys

from game import SnakeGame
from model import SnakeNet
from mcts import MCTS
from benchmark import BenchmarkConfig, append_benchmark_csv, run_benchmark
from encoder import encode_pov

# Hyperparameters (defaults; override via CLI)
LR = 5e-4
BATCH_SIZE = 128
MEMORY_SIZE = 50000
EPOCHS = 2
# Only used when schedules are disabled (see USE_SCHEDULES below)
GAMES_PER_GEN = 20
SIMULATIONS = 128
VALUE_LOSS_WEIGHT = 0.5

# Training stability (optional)
USE_VALUE_TARGET_NORM = False  # Phase 2.2 (optional)
VALUE_NORM_EPS = 1e-5

# Inference batching (defaults; override via CLI)
INFER_MAX_BATCH = 64
INFER_TIMEOUT_EMPTY = 0.01
INFER_TIMEOUT_NONEMPTY = 0.005
INFER_MODE = "central"  # "central" (batched inference in main) or "worker" (local inference in each worker)

# Compute-efficient schedules (enabled by default; can be disabled via CLI)
USE_SCHEDULES = True
SIMS_START = 64
SIMS_MID = 128
SIMS_END = 256  # Increased for better endgame precision
SIMS_ENDGAME_MULT = 2  # boost sims only when the board is mostly filled (>=75%)
GAMES_START = 20
GAMES_END = 50

# Dev/Quick mode for fast iteration (reduces compute without hurting learning signal)
DEV_MODE = False  # Enable via --dev flag

# Multi-size board training (curriculum learning)
USE_MULTI_SIZE = False  # Enable to train on multiple board sizes
BOARD_SIZES = [6, 8, 10]  # Sizes to use when multi-size is enabled
CURRICULUM_MODE = "progressive"  # "progressive" (start small, grow) or "mixed" (random mix)
CURRICULUM_START_SIZE = 6
CURRICULUM_END_SIZE = 10
CURRICULUM_RAMP_GENS = 50  # Generations to ramp from start to end size

def get_c_puct(generation):
    """
    Decays C_PUCT from 3.0 to 1.0 over 50 generations to favor exploration early on.
    """
    if generation < 50:
        return 3.0 - (2.0 * generation / 50)
    return 1.0

def get_temperature_threshold(generation):
    """
    Decays exploration steps (temperature=1) from 40 to 10.
    """
    if generation < 20:
        return 40
    elif generation < 50:
        return 20
    else:
        return 10


def get_simulations(generation: int, snake_len: int, board_size: int) -> int:
    """
    Compute-efficient simulation schedule:
    - ramp sims up across generations
    - spend extra compute only in endgame where precision matters
    
    Endgame boost: When snake covers >=75% of the board, simulations are multiplied
    by SIMS_ENDGAME_MULT. This is critical because:
    - Endgame requires perfect play to avoid self-collision
    - More simulations = better pathfinding through tight spaces
    - Example: 6x6 board (36 cells), threshold = 27 cells (75%)
              When snake_len >= 27, sims are doubled
    """
    if DEV_MODE:
        # Dev mode: much faster, lower sims
        if generation < 10:
            base = max(32, SIMS_START // 2)  # Half the sims
        elif generation < 30:
            base = max(64, SIMS_MID // 2)
        else:
            base = max(96, SIMS_END // 2)
        # No endgame boost in dev mode
    else:
        if generation < 10:
            base = SIMS_START
        elif generation < 30:
            base = SIMS_MID
        else:
            base = SIMS_END
        
        # Endgame boost: only when >=75% of the board is filled
        # This is when precision matters most to avoid self-collision
        endgame_threshold = int(0.75 * (board_size * board_size))
        if snake_len >= endgame_threshold:
            base *= SIMS_ENDGAME_MULT

    return int(base)


def get_games_per_gen(generation: int) -> int:
    # Linearly ramp games to stabilize training without exploding early compute.
    if DEV_MODE:
        # Dev mode: fewer games, faster iteration
        if generation <= 0:
            return max(8, GAMES_START // 2)  # Half the games
        if generation >= 20:  # Ramp faster in dev mode
            return max(16, GAMES_END // 2)
        t = generation / 20.0
        start = max(8, GAMES_START // 2)
        end = max(16, GAMES_END // 2)
        return int(round(start + t * (end - start)))
    else:
        if generation <= 0:
            return GAMES_START
        if generation >= 30:
            return GAMES_END
        t = generation / 30.0
        return int(round(GAMES_START + t * (GAMES_END - GAMES_START)))

def get_board_size_for_game(generation: int, game_index: int = None) -> int:
    """
    Returns the board size for a specific game based on curriculum learning strategy.
    
    Args:
        generation: Current generation number
        game_index: Optional game index (for mixed mode randomization)
    
    Returns:
        Board size to use for this game
    """
    if not USE_MULTI_SIZE:
        # Single size mode: return the default (will be overridden by Trainer.board_size)
        return CURRICULUM_START_SIZE
    
    if CURRICULUM_MODE == "progressive":
        # Progressive: start with small boards, gradually increase
        if generation <= 0:
            return CURRICULUM_START_SIZE
        if generation >= CURRICULUM_RAMP_GENS:
            return CURRICULUM_END_SIZE
        
        # Linear interpolation
        t = generation / CURRICULUM_RAMP_GENS
        size = CURRICULUM_START_SIZE + t * (CURRICULUM_END_SIZE - CURRICULUM_START_SIZE)
        # Round to nearest valid size
        return int(round(size))
    
    elif CURRICULUM_MODE == "mixed":
        # Mixed: randomly sample from BOARD_SIZES
        if game_index is not None:
            # Use game_index for deterministic but varied selection
            np.random.seed(generation * 1000 + game_index)
        return int(np.random.choice(BOARD_SIZES))
    
    else:
        # Default: use start size
        return CURRICULUM_START_SIZE

def play_games_worker(
    worker_id,
    board_size,  # Default/fallback size (used if game_queue doesn't specify)
    simulations,
    generation,
    c_puct,
    temp_threshold,
    game_queue,
    request_queue,
    response_queue,
    result_queue,
    infer_mode="central",
    model_state_dict=None,
):
    """
    Worker function to play games in a dedicated process.
    Uses batched inference by sending states to the main process via queues.

    IMPORTANT: Each process must have a unique (worker_id, response_queue) pair.
    This avoids cross-talk where one process consumes another's inference response.
    """
    # Re-seed random number generators for this process
    seed = os.getpid() + int(torch.randint(0, 10000, (1,)).item())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prediction client
    if infer_mode == "worker":
        # Local inference in each worker removes IPC bottlenecks but requires care
        # to avoid thread oversubscription.
        # Note: Model is now size-agnostic, so we can use any board_size for initialization
        torch.set_num_threads(1)
        device = torch.device("cpu")
        local_model = SnakeNet(board_size=board_size).to(device)  # Size doesn't matter anymore
        if model_state_dict is None:
            raise RuntimeError("infer_mode='worker' requires model_state_dict")
        local_model.load_state_dict(model_state_dict)
        local_model.eval()

        def predict_client(input_tensor):
            x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
            with torch.no_grad():
                p, v = local_model(x)
            p = torch.exp(p).squeeze(0).cpu().numpy()
            return p, float(v.item())
    else:
        # Central mode: send requests to the parent, wait for batched inference response.
        def predict_client(input_tensor):
            request_queue.put((worker_id, input_tensor))
            p, v = response_queue.get()
            return p, v

    try:
        # Dynamic scheduling: pull games from a shared queue.
        # Faster workers will naturally execute more games.
        while True:
            # IMPORTANT (Windows spawn): using get_nowait() can observe an "empty"
            # queue briefly during startup due to feeder-thread timing, causing
            # workers to exit prematurely. We use a blocking get() + sentinels.
            _game_token = game_queue.get()
            if _game_token is None:
                break
            
            # Support dynamic board sizes: game_token can be (game_id, board_size) or just game_id
            if isinstance(_game_token, tuple):
                game_id, game_board_size = _game_token
            else:
                game_id = _game_token
                game_board_size = board_size  # Fallback to default
            
            game = SnakeGame(board_size=game_board_size)

            # Fresh tree per game (n_simulations may be adjusted per-move if schedules enabled)
            mcts = MCTS(predict_client, n_simulations=simulations, c_puct=c_puct)
            mcts.reset()

            game_memory = []
            steps = 0
            total_entropy = 0
            move_count = 0
            state_tensor = game.reset()

            while not game.done:
                # Temperature: High exploration early, greedy later
                temp = 1.0 if steps < temp_threshold else 0.1

                # Dynamic sims: save compute early-game, spend it late-game
                if USE_SCHEDULES:
                    mcts.n_simulations = get_simulations(generation, len(game.snake), game.board_size)

                action_probs, entropy = mcts.search(game)
                total_entropy += entropy
                move_count += 1

                # Apply temperature
                if temp == 0:
                    rel_action = np.argmax(action_probs)
                else:
                    action_probs = action_probs ** (1 / temp)
                    action_probs = action_probs / np.sum(action_probs)
                    rel_action = np.random.choice(len(action_probs), p=action_probs)

                # Convert relative action to absolute action
                abs_action = (game.direction + (rel_action - 1)) % 4

                # Store input state for training
                input_state = process_state(game, state_tensor)

                state_tensor, reward, done = game.step(abs_action)
                mcts.update_root(rel_action)
                steps += 1

                game_memory.append([input_state, action_probs, reward])

            avg_entropy = total_entropy / move_count if move_count > 0 else 0
            result_queue.put((process_game_memory(game_memory), game.score, game.death_reason, avg_entropy, steps))
    except KeyboardInterrupt:
        # On Windows spawn, Ctrl+C is often delivered to child processes too.
        # If we don't catch it, multiprocessing prints a noisy traceback for each worker.
        # Exiting cleanly here is expected behavior when the user cancels training.
        return
    except Exception:
        # Propagate a traceback to the parent so we don't "hang" silently.
        result_queue.put(("__error__", worker_id, traceback.format_exc()))
    finally:
        result_queue.put(("__done__", worker_id))

def process_state(game, state):
    return encode_pov(game, state)

def augment_data(state, policy, value):
    """
    Generates symmetries. For POV, rotation is NOT valid as Up is fixed.
    Only Horizontal Flip is valid.
    """
    augmented_data = []
    
    state_np = state
    policy_np = np.array(policy)
    
    # Original
    augmented_data.append((state_np, policy_np, value))
    
    # Flip (Horizontal flip of the state)
    # Axis 2 is Width (Columns)
    flip_state = np.flip(state_np, axis=2).copy()
    
    # For relative policy: [Left, Straight, Right]
    # Flip swaps Left (0) and Right (2)
    p_flip = policy_np.copy()
    p_flip[0], p_flip[2] = p_flip[2], p_flip[0]
    
    augmented_data.append((flip_state, p_flip, value))
        
    return augmented_data

def process_game_memory(game_memory):
    processed_samples = []
    # Keep consistent with MCTS backup discount (see mcts.py Node.update)
    gamma = 0.95
    running_return = 0
    
    # Backpropagate actual returns
    for i in reversed(range(len(game_memory))):
        reward = game_memory[i][2]
        running_return = reward + gamma * running_return
        # Removed Clipping to allow Value Head to learn true returns (e.g. > 1.0)
        # running_return = max(-1.0, min(1.0, running_return))
        
        state = game_memory[i][0]
        policy = game_memory[i][1]
        value = running_return
        
        # Augment data (1 sample -> 8 samples)
        augmented = augment_data(state, policy, value)
        processed_samples.extend(augmented)
        
    return processed_samples

class RunningMeanStd:
    """
    Numerically-stable running mean/std (Welford-style) for scalar targets.
    Used to normalize value targets to reduce gradient scale spikes.
    """

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, x) -> None:
        """
        Update running mean/var from a batch of scalars.
        Accepts either a torch.Tensor or a numpy-like array.
        """
        if torch.is_tensor(x):
            if x.numel() == 0:
                return
            # Avoid materializing the whole tensor on CPU (keep stats as floats only).
            # Note: use unbiased=False for population variance.
            batch_mean = float(x.detach().mean().item())
            batch_var = float(x.detach().var(unbiased=False).item())
            batch_count = float(x.numel())
        else:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if x.size == 0:
                return
            batch_mean = float(x.mean())
            batch_var = float(x.var())
            batch_count = float(x.size)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta * delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = max(new_var, 0.0)
        self.count = tot_count

    @property
    def std(self) -> float:
        return float(math.sqrt(self.var) + VALUE_NORM_EPS)

class Trainer:
    def __init__(self, run_dir, start_gen=0, num_workers=None, board_size=6):
        self.board_size = board_size  # Default/fallback size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model is now size-agnostic, so board_size parameter is just for compatibility
        self.model = SnakeNet(board_size=self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.history = []
        self.generation = start_gen
        self.run_dir = run_dir
        self.value_rms = RunningMeanStd() if USE_VALUE_TARGET_NORM else None
        self.best_avg_score = -float('inf')
        self.best_benchmark_score = -float('inf')  # Best benchmark AvgScore
        self.best_benchmark_winpct = -float('inf')  # Best benchmark WinPct
        # Moving average for training avg_score (reduces noise)
        self.avg_score_window = deque(maxlen=5)  # Last 5 generations
        self.best_moving_avg_score = -float('inf')
        
        # Determine number of workers
        if num_workers is not None:
            self.num_workers = num_workers
        else:
            total_cores = mp.cpu_count()
            self.num_workers = max(1, total_cores - 1)
        
        print(f"Initializing Trainer with {self.num_workers} workers on device {self.device}")
        print(f"Training will be saved to: {self.run_dir}")


    def train_generation(self):
        print(f"--- Generation {self.generation + 1} ---")
        start_time = time.time()
        selfplay_start_time = start_time
        
        # 1. Collection Phase (Parallel)
        new_samples = []
        scores = []
        entropies = []
        death_reasons = []
        total_steps = 0
        min_score = float('inf')
        
        # Calculate dynamic parameters for this generation
        current_c_puct = get_c_puct(self.generation)
        current_temp_threshold = get_temperature_threshold(self.generation)
        
        # Allow schedules to control total self-play work per generation
        games_this_gen = get_games_per_gen(self.generation) if USE_SCHEDULES else GAMES_PER_GEN
        print(f"Gen {self.generation+1} Params: C_PUCT={current_c_puct:.2f}, TempThreshold={current_temp_threshold}, Games={games_this_gen}")
        
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        request_queue = None
        response_queues = None
        model_state_dict = None

        if INFER_MODE == "central":
            request_queue = ctx.Queue()
            response_queues = [ctx.Queue() for _ in range(self.num_workers)]
        elif INFER_MODE == "worker":
            # Snapshot model weights once per generation and ship to workers.
            model_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        else:
            raise ValueError(f"Unknown INFER_MODE: {INFER_MODE}")

        # Dynamic scheduling: shared game queue so workers load-balance automatically.
        game_queue = ctx.Queue()
        for i in range(games_this_gen):
            if USE_MULTI_SIZE:
                # Pass (game_id, board_size) tuple for multi-size training
                game_board_size = get_board_size_for_game(self.generation, game_index=i)
                game_queue.put((i, game_board_size))
            else:
                # Single size: just pass game_id
                game_queue.put(i)
        # One sentinel per worker so everyone can shut down cleanly.
        for _ in range(self.num_workers):
            game_queue.put(None)

        procs = []
        proc_by_worker = {}
        active_workers = 0
        for w_id in range(self.num_workers):
            p = ctx.Process(
                target=play_games_worker,
                args=(
                    w_id,
                    self.board_size,
                    SIMULATIONS,
                    self.generation,
                    current_c_puct,
                    current_temp_threshold,
                    game_queue,
                    request_queue,
                    (response_queues[w_id] if response_queues is not None else None),
                    result_queue,
                    INFER_MODE,
                    model_state_dict,
                ),
            )
            p.daemon = True
            p.start()
            procs.append(p)
            proc_by_worker[w_id] = p
            active_workers += 1
        
        # Inference Loop (Main Thread) - only in central mode
        if INFER_MODE == "central":
            self.model.eval()
        
        # Variables for monitoring
        last_monitor_time = time.time()
        inference_batches = 0

        # Telemetry (per-generation)
        total_inference_batches = 0
        total_inference_requests = 0
        
        done_workers = 0
        done_worker_ids = set()
        results_received = 0
        expected_results = games_this_gen
        # We consider the system "making progress" if we are either:
        # - receiving game results, OR
        # - processing inference requests from workers.
        #
        # A generation can legitimately go a while without finishing a game
        # (e.g. if the remaining worker is playing a long MCTS-heavy episode),
        # so "no results for X seconds" is not a reliable deadlock signal.
        last_activity_time = time.time()
        last_results_received = 0

        while results_received < expected_results:
            if INFER_MODE == "central":
                # Check for requests
                batch_reqs = []

                # Non-blocking collect up to INFER_MAX_BATCH OR until empty
                while len(batch_reqs) < INFER_MAX_BATCH:
                    try:
                        timeout = INFER_TIMEOUT_NONEMPTY if len(batch_reqs) > 0 else INFER_TIMEOUT_EMPTY
                        req = request_queue.get(timeout=timeout)
                        batch_reqs.append(req)
                    except queue.Empty:
                        break

                if batch_reqs:
                    inference_batches += 1
                    total_inference_batches += 1
                    total_inference_requests += len(batch_reqs)

                    worker_ids = [r[0] for r in batch_reqs]
                    input_np = np.array([r[1] for r in batch_reqs])
                    input_tensor = torch.tensor(input_np).to(self.device)

                    with torch.no_grad():
                        policies, values = self.model(input_tensor)

                    policies = torch.exp(policies).cpu().numpy()
                    values = values.cpu().numpy()

                    for i, w_id in enumerate(worker_ids):
                        response_queues[w_id].put((policies[i], values[i].item()))
                    last_activity_time = time.time()
                else:
                    time.sleep(0.001)
            else:
                # Worker mode: keep loop responsive while workers compute locally.
                time.sleep(0.001)

            # Drain finished game results (non-blocking)
            while True:
                try:
                    msg = result_queue.get_nowait()
                except queue.Empty:
                    break

                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__done__":
                    done_workers += 1
                    done_worker_ids.add(msg[1])
                    last_activity_time = time.time()
                    continue
                
                if isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "__error__":
                    _tag, w_id, tb = msg
                    print(f"\n[Worker {w_id}] ERROR:\n{tb}", flush=True)
                    for p in procs:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    for p in procs:
                        try:
                            p.join(timeout=2)
                        except Exception:
                            pass
                    raise RuntimeError(f"Worker {w_id} crashed. See traceback above.")

                samples, score, death_reason, entropy, steps_played = msg
                new_samples.extend(samples)
                scores.append(score)
                death_reasons.append(death_reason)
                entropies.append(entropy)
                total_steps += steps_played
                if score < min_score:
                    min_score = score
                results_received += 1
                last_activity_time = time.time()

            # Monitor periodically
            if time.time() - last_monitor_time > 5:
                alive = sum(1 for p in procs if p.is_alive())
                print(f"   [Progress] FinishedGames: {results_received}/{expected_results} | WorkersAlive: {alive}/{active_workers}", flush=True)
                # Detect unexpected worker exits early (e.g. hard crash / killed process)
                for w_id, p in proc_by_worker.items():
                    if w_id in done_worker_ids:
                        continue
                    if p.exitcode is not None and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker process {w_id} exited unexpectedly with exit code {p.exitcode} "
                            f"(FinishedGames {results_received}/{expected_results})."
                        )

                # CPU Usage
                cpu_pct = psutil.cpu_percent()
                
                # GPU Usage
                gpu_util = "N/A"
                gpu_mem = "N/A"
                try:
                    result = subprocess.run(
                        ['/usr/bin/nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if result.returncode == 0:
                        output = result.stdout.strip().split(',')
                        if len(output) >= 2:
                            gpu_util = f"{output[0].strip()}%"
                            gpu_mem = f"{output[1].strip()}MiB"
                except Exception:
                    pass
                    
                if INFER_MODE == "central":
                    batch_rate = f"{inference_batches/5:.1f}/s"
                else:
                    batch_rate = "N/A (worker)"

                print(
                    f"   [Monitor] CPU: {cpu_pct:.1f}% | GPU Util: {gpu_util} | GPU Mem: {gpu_mem} | "
                    f"Workers: {self.num_workers} | Batch Rate: {batch_rate}",
                    flush=True,
                )
                inference_batches = 0
                last_monitor_time = time.time()

            # Deadlock protection: if all workers are done but we didn't receive all games, fail fast.
            if done_workers >= active_workers and results_received < expected_results:
                raise RuntimeError(
                    f"All workers finished ({done_workers}/{active_workers}) but only received {results_received}/{expected_results} game results."
                )

            # Deadlock protection: no activity for too long (no inference requests and no game results).
            # This is more reliable than "no finished games" because the last remaining
            # episode can be long-running while still actively requesting inference.
            if INFER_MODE == "central" and time.time() - last_activity_time > 60 and results_received == last_results_received:
                alive = sum(1 for p in procs if p.is_alive())
                raise RuntimeError(
                    f"No activity for 60s (FinishedGames {results_received}/{expected_results}, WorkersAlive {alive}/{active_workers})."
                )
            last_results_received = results_received

        # Ensure workers are cleaned up
        for p in procs:
            p.join(timeout=5)

        selfplay_end_time = time.time()
        selfplay_time_s = selfplay_end_time - selfplay_start_time
            
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        avg_steps = total_steps / len(scores)
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0
        
        # Count death reasons
        death_counts = {
            "wall": death_reasons.count("wall"),
            "body": death_reasons.count("body"),
            "timeout": death_reasons.count("timeout"),
            "starvation": death_reasons.count("starvation"), # Should be rare if timeout covers it
            "won": death_reasons.count("won")
        }
        
        # Add to replay buffer
        self.memory.extend(new_samples)
        
        # 2. Training Phase
        train_start_time = time.time()
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        total_pred_acc = 0
        steps = 0
        
        # Train for epochs
        self.model.train()
        num_batches = len(self.memory) // BATCH_SIZE
        if num_batches > 0:
            for _ in range(EPOCHS):
                # Shuffle full memory
                full_batch = list(self.memory)
                random.shuffle(full_batch)
                
                for i in range(0, len(full_batch), BATCH_SIZE):
                    batch = full_batch[i:i+BATCH_SIZE]
                    if len(batch) < BATCH_SIZE: continue
                    
                    loss, p_loss, v_loss, pred_acc = self.train_step(batch)
                    total_loss += loss
                    total_p_loss += p_loss
                    total_v_loss += v_loss
                    total_pred_acc += pred_acc
                    steps += 1
        
        if steps > 0:
            avg_loss = total_loss / steps
            avg_p_loss = total_p_loss / steps
            avg_v_loss = total_v_loss / steps
            avg_pred_acc = total_pred_acc / steps
        else:
            avg_loss = 0
            avg_p_loss = 0
            avg_v_loss = 0
            avg_pred_acc = 0

        train_end_time = time.time()
        train_time_s = train_end_time - train_start_time
        avg_infer_batch_size = (total_inference_requests / total_inference_batches) if total_inference_batches > 0 else 0.0
            
        duration = time.time() - start_time
        # Step LR schedule once per generation (after at least one training step)
        if steps > 0:
            self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        print(f"Gen {self.generation+1} Finished. Avg Score: {avg_score:.2f}, Max Score: {max_score}, Loss: {avg_loss:.4f}, Time: {duration:.2f}s")
        print(f"Stats: LR={current_lr:.2e}, Entropy={avg_entropy:.4f}, Steps={avg_steps:.1f}, PredAcc={avg_pred_acc:.1%}, Deaths={death_counts}")
        print(
            f"Telemetry: InferMode={INFER_MODE} | SelfPlay={selfplay_time_s:.2f}s | Train={train_time_s:.2f}s | "
            f"InferReq={total_inference_requests} | InferBatches={total_inference_batches} | AvgInferBatch={avg_infer_batch_size:.1f}",
            flush=True,
        )
        if USE_VALUE_TARGET_NORM and self.value_rms is not None:
            print(
                f"ValueNorm: mean={self.value_rms.mean:.3f} std={self.value_rms.std:.3f} count={int(self.value_rms.count)}",
                flush=True,
            )
        
        self.history.append({
            'Gen': self.generation + 1, 
            'AvgScore': avg_score, 
            'MaxScore': max_score, 
            'MinScore': min_score,
            'Loss': avg_loss,
            'PolicyLoss': avg_p_loss,
            'ValueLoss': avg_v_loss,
            'PredAcc': avg_pred_acc,
            'Games': games_this_gen,
            'Time': duration,
            'AvgSteps': avg_steps,
            'AvgEntropy': avg_entropy,
            'ValueNormMean': (self.value_rms.mean if (USE_VALUE_TARGET_NORM and self.value_rms is not None) else ""),
            'ValueNormStd': (self.value_rms.std if (USE_VALUE_TARGET_NORM and self.value_rms is not None) else ""),
            'DeathWall': death_counts['wall'],
            'DeathBody': death_counts['body'],
            'DeathTimeout': death_counts['timeout'],
            'DeathStarvation': death_counts['starvation'],
            'DeathWon': death_counts['won']
        })
        
        # Update moving average of avg_score (reduces noise)
        self.avg_score_window.append(avg_score)
        moving_avg_score = sum(self.avg_score_window) / len(self.avg_score_window) if self.avg_score_window else avg_score
        
        # Save best model based on moving average (more stable than single value)
        # This helps avoid saving models during temporary spikes
        if moving_avg_score > self.best_moving_avg_score:
            previous_best = self.best_moving_avg_score
            self.best_moving_avg_score = moving_avg_score
            self.best_avg_score = avg_score  # Keep track of raw best too
            self.save_best_model()
            if previous_best > -float('inf'):
                print(f"New best model saved! MovingAvgScore: {moving_avg_score:.2f} (previous: {previous_best:.2f}, raw: {avg_score:.2f})")
            else:
                print(f"New best model saved! MovingAvgScore: {moving_avg_score:.2f} (raw: {avg_score:.2f})")
        
        self.generation += 1
        return avg_score

    def train_step(self, batch):
        inputs = torch.tensor(np.array([x[0] for x in batch])).to(self.device)
        target_pis = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(self.device)
        target_vs = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.optimizer.zero_grad()
        p, v = self.model(inputs)
        
        # Huber loss is more stable than pure MSE when returns can be large/outlier-y.
        if USE_VALUE_TARGET_NORM and self.value_rms is not None:
            # Update stats on raw targets, then normalize both sides. This keeps the model's value
            # output in the original (raw-return) scale while improving gradient conditioning.
            self.value_rms.update(target_vs.detach())
            mean = float(self.value_rms.mean)
            std = float(self.value_rms.std)
            v_n = (v - mean) / std
            t_n = (target_vs - mean) / std
            loss_v = F.smooth_l1_loss(v_n, t_n)
        else:
            loss_v = F.smooth_l1_loss(v, target_vs)
        loss_p = -torch.sum(target_pis * p) / target_pis.size(0)
        
        total_loss = VALUE_LOSS_WEIGHT * loss_v + loss_p
        
        # Calculate Prediction Accuracy
        with torch.no_grad():
             # p is log_softmax
             pred_actions = torch.argmax(p, dim=1)
             true_actions = torch.argmax(target_pis, dim=1)
             correct = (pred_actions == true_actions).float()
             pred_acc = torch.mean(correct).item()

        total_loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item(), loss_p.item(), loss_v.item(), pred_acc

    def save_report(self):
        filename = os.path.join(self.run_dir, "training_report.csv")
        file_exists = os.path.isfile(filename)
        
        fieldnames = [
            'Gen', 'AvgScore', 'MaxScore', 'MinScore',
            'Loss', 'PolicyLoss', 'ValueLoss', 'PredAcc',
            'Games', 'Time', 'AvgSteps', 'AvgEntropy',
            'ValueNormMean', 'ValueNormStd',
            'DeathWall', 'DeathBody', 'DeathTimeout', 'DeathStarvation', 'DeathWon',
        ]
        
        # Check if we need to update headers for existing file
        if file_exists:
            with open(filename, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and header != fieldnames:
                    print(f"Updating CSV header in {filename} to match new metrics...")
                    # Read all data
                    f.seek(0)
                    dict_reader = csv.DictReader(f)
                    data = list(dict_reader)
                    
                    # Rewrite with new header
                    with open(filename, 'w', newline='') as f_out:
                         writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                         writer.writeheader()
                         for row in data:
                             # Write existing data, new fields will be empty/null
                             writer.writerow(row)
                    # We have handled the file, so effectively we append now
        
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            # Only write the last entry in history
            if self.history:
                writer.writerow(self.history[-1])
        print(f"Report saved to {filename}")

    def save_model(self):
        path = os.path.join(self.run_dir, "snake_net.pth")
        torch.save(self.model.state_dict(), path)
    
    def save_best_model(self):
        path = os.path.join(self.run_dir, "best_snake_net.pth")
        torch.save(self.model.state_dict(), path)
    
    def update_best_from_benchmark(self, benchmark_row):
        """
        Update best model based on benchmark results (more reliable than self-play avg_score).
        Prioritizes WinPct, then AvgScore from benchmark.
        """
        bench_winpct = float(benchmark_row.get('WinPct', 0.0))
        bench_avgscore = float(benchmark_row.get('AvgScore', -float('inf')))
        
        # Prioritize WinPct (most important metric), then AvgScore
        should_save = False
        reason = ""
        
        if bench_winpct > self.best_benchmark_winpct:
            should_save = True
            reason = f"WinPct: {bench_winpct:.1f}% (previous best: {self.best_benchmark_winpct:.1f}%)"
            self.best_benchmark_winpct = bench_winpct
            self.best_benchmark_score = bench_avgscore
        elif bench_winpct == self.best_benchmark_winpct and bench_avgscore > self.best_benchmark_score:
            # Same WinPct but better AvgScore
            should_save = True
            reason = f"WinPct: {bench_winpct:.1f}%, AvgScore: {bench_avgscore:.2f} (previous: {self.best_benchmark_score:.2f})"
            self.best_benchmark_score = bench_avgscore
        
        if should_save:
            self.save_best_model()
            print(f"New best model saved from benchmark! {reason}")
            return True
        return False

def get_run_dir(base_dir="experiments", resume_version=None):
    os.makedirs(base_dir, exist_ok=True)
    
    if resume_version is not None:
        # Resume specific or latest
        if resume_version == -1:
            # Find latest
            versions = []
            for d in os.listdir(base_dir):
                match = re.match(r'train_v(\d+)', d)
                if match:
                    versions.append(int(match.group(1)))
            if not versions:
                print("No experiments found to resume.")
                exit(1)
            version = max(versions)
        else:
            version = resume_version
            
        run_dir = os.path.join(base_dir, f"train_v{version}")
        if not os.path.exists(run_dir):
             print(f"Experiment {run_dir} does not exist.")
             exit(1)
        return run_dir, True
    
    else:
        # New run
        versions = []
        for d in os.listdir(base_dir):
            match = re.match(r'train_v(\d+)', d)
            if match:
                versions.append(int(match.group(1)))
        
        new_version = max(versions) + 1 if versions else 1
        run_dir = os.path.join(base_dir, f"train_v{new_version}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir, False


def _list_experiment_runs(base_dir: str):
    """
    Return experiment directories as a list sorted by newest first.
    Each item: dict(name, path, version, mtime).
    """
    runs = []
    if not os.path.isdir(base_dir):
        return runs
    for d in os.listdir(base_dir):
        match = re.match(r"train_v(\d+)$", d)
        if not match:
            continue
        p = os.path.join(base_dir, d)
        if not os.path.isdir(p):
            continue
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0.0
        runs.append(
            {
                "name": d,
                "path": p,
                "version": int(match.group(1)),
                "mtime": float(mtime),
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def choose_run_dir(base_dir: str, *, prefer_new: bool) -> tuple[str, bool]:
    """
    Interactive selection of an experiment directory.
    - Ordered by newest (mtime).
    - Lets the user choose an existing run or create a new one.
    Returns (run_dir, is_resume).

    If stdin isn't interactive, defaults to NEW run.
    """
    runs = _list_experiment_runs(base_dir)
    versions = [r["version"] for r in runs]
    new_version = (max(versions) + 1) if versions else 1
    new_name = f"train_v{new_version}"
    new_path = os.path.join(base_dir, new_name)

    if not sys.stdin.isatty():
        os.makedirs(new_path, exist_ok=True)
        return new_path, False

    if runs:
        print("\nSelect experiment run (newest first):", flush=True)
        print(f"  [0] NEW: {new_name}", flush=True)
        for idx, r in enumerate(runs, start=1):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["mtime"])) if r["mtime"] else "unknown"
            print(f"  [{idx}] {r['name']}  (modified {ts})", flush=True)
    else:
        print("\nNo existing experiments found.", flush=True)
        print(f"  [0] NEW: {new_name}", flush=True)

    default_idx = 0 if prefer_new else (1 if runs else 0)
    default_label = "NEW" if default_idx == 0 else runs[default_idx - 1]["name"]
    prompt = f"Choice [default {default_idx}={default_label}]: "

    while True:
        choice = input(prompt).strip()
        if choice == "":
            choice = str(default_idx)

        # Allow: index number (0..N)
        if choice.isdigit():
            n = int(choice)
            if n == 0:
                os.makedirs(new_path, exist_ok=True)
                return new_path, False
            if 1 <= n <= len(runs):
                return runs[n - 1]["path"], True

            # Allow: direct version number (e.g. "25" -> train_v25)
            if n > 0:
                direct = os.path.join(base_dir, f"train_v{n}")
                if os.path.isdir(direct):
                    return direct, True
                print(f"Invalid choice: train_v{n} not found.", flush=True)
                continue

        # Allow: "train_v25"
        if re.match(r"^train_v\d+$", choice):
            direct = os.path.join(base_dir, choice)
            if os.path.isdir(direct):
                return direct, True
            print(f"Invalid choice: {choice} not found.", flush=True)
            continue

        print("Invalid choice. Enter an index number (e.g. 0, 1, 2...) or a version (e.g. 25).", flush=True)


if __name__ == "__main__":
    print("Starting main execution...", flush=True)
    # Required for Windows multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to spawn.", flush=True)
    except RuntimeError as e:
        print(f"Error setting start method: {e}", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs='?', const=-1, type=int, help="Resume training. Optional: provide version number (e.g. 1 for train_v1). Default is latest.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--board-size", type=int, default=6, help="Board size to train on (e.g. 6 for easier perfect-play learning)")
    parser.add_argument("--fast", action="store_true", help="Compute-efficient preset (recommended for laptops)")
    parser.add_argument("--dev", action="store_true", help="Dev mode: very fast iterations for testing changes (~3-5x faster, minimal impact on learning)")
    parser.add_argument("--sims", type=int, default=None, help="Base simulations per move (disables schedule if set)")
    parser.add_argument("--games", type=int, default=None, help="Games per generation (disables schedule if set)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per generation")
    parser.add_argument("--memory", type=int, default=None, help="Replay buffer size")
    parser.add_argument("--batch", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--infer-batch", type=int, default=None, help="Max inference batch size")
    parser.add_argument("--infer-wait-empty", type=float, default=None, help="Queue wait (seconds) when empty")
    parser.add_argument("--infer-wait-nonempty", type=float, default=None, help="Queue wait (seconds) when already batching")
    parser.add_argument("--infer-mode", type=str, default=None, choices=["central", "worker"], help="Inference mode: central (batched in main) or worker (local per worker)")
    parser.add_argument("--value-norm", action="store_true", help="Normalize value targets with running mean/std (Phase 2.2)")
    parser.add_argument("--gens", type=int, default=None, help="Run N generations then exit (useful for smoke tests)")
    parser.add_argument("--benchmark", action="store_true", help="Run deterministic benchmark and append to benchmark.csv")
    parser.add_argument("--bench-only", action="store_true", help="Run benchmark once and exit (no training)")
    parser.add_argument("--bench-every", type=int, default=1, help="Run benchmark every N generations (when --benchmark enabled)")
    parser.add_argument("--bench-episodes", type=int, default=50, help="Benchmark episodes (fixed for comparability)")
    parser.add_argument("--bench-sims", type=int, default=128, help="Benchmark MCTS sims per move (fixed for comparability)")
    parser.add_argument("--bench-seed", type=int, default=0, help="Benchmark base seed (fixed for comparability)")
    parser.add_argument("--multi-size", action="store_true", help="Enable multi-size board training (curriculum learning)")
    parser.add_argument("--board-sizes", type=int, nargs="+", default=[6, 8, 10], help="Board sizes to use when --multi-size is enabled")
    parser.add_argument("--curriculum-mode", type=str, default="progressive", choices=["progressive", "mixed"], help="Curriculum mode: progressive (grow over time) or mixed (random mix)")
    parser.add_argument("--curriculum-start", type=int, default=6, help="Starting board size for progressive curriculum")
    parser.add_argument("--curriculum-end", type=int, default=10, help="Ending board size for progressive curriculum")
    parser.add_argument("--curriculum-ramp", type=int, default=50, help="Generations to ramp from start to end size")
    args = parser.parse_args()

    # Apply CLI overrides / presets (module-level globals)
    if args.dev:
        # Dev mode: optimized for fast iteration when testing changes
        # Reduces compute by ~3-5x while maintaining learning signal quality
        DEV_MODE = True
        LR = 5e-4
        BATCH_SIZE = 128
        MEMORY_SIZE = 20000  # Smaller buffer for faster training
        EPOCHS = 1  # Single epoch is usually enough
        SIMS_START = 32
        SIMS_MID = 64
        SIMS_END = 96
        SIMS_ENDGAME_MULT = 1  # Disable endgame boost in dev mode
        GAMES_START = 8
        GAMES_END = 16
        INFER_MAX_BATCH = 64
        print("Dev mode enabled: Fast iteration mode (~3-5x faster)")
        print("  Games: 8-16, Sims: 32-96, Epochs: 1, Memory: 20k")
    
    if args.fast:
        # Aim: reach "fill board once" quickly without melting your CPU.
        LR = 5e-4
        BATCH_SIZE = 128
        MEMORY_SIZE = 30000
        EPOCHS = 1
        SIMS_START = 48
        SIMS_MID = 96
        SIMS_END = 140
        SIMS_ENDGAME_MULT = 2
        GAMES_START = 12
        GAMES_END = 24
        INFER_MAX_BATCH = 48

    if args.lr is not None:
        LR = args.lr
    if args.batch is not None:
        BATCH_SIZE = args.batch
    if args.memory is not None:
        MEMORY_SIZE = args.memory
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.infer_batch is not None:
        INFER_MAX_BATCH = args.infer_batch
    if args.infer_wait_empty is not None:
        INFER_TIMEOUT_EMPTY = float(args.infer_wait_empty)
    if args.infer_wait_nonempty is not None:
        INFER_TIMEOUT_NONEMPTY = float(args.infer_wait_nonempty)
    if args.infer_mode is not None:
        INFER_MODE = args.infer_mode
    if args.value_norm:
        USE_VALUE_TARGET_NORM = True
    
    # Multi-size training configuration
    if args.multi_size:
        USE_MULTI_SIZE = True
        BOARD_SIZES = args.board_sizes
        CURRICULUM_MODE = args.curriculum_mode
        CURRICULUM_START_SIZE = args.curriculum_start
        CURRICULUM_END_SIZE = args.curriculum_end
        CURRICULUM_RAMP_GENS = args.curriculum_ramp
        print(f"Multi-size training enabled: sizes={BOARD_SIZES}, mode={CURRICULUM_MODE}")
        if CURRICULUM_MODE == "progressive":
            print(f"  Progressive curriculum: {CURRICULUM_START_SIZE} -> {CURRICULUM_END_SIZE} over {CURRICULUM_RAMP_GENS} generations")

    # If user pins sims/games, treat it as disabling schedules.
    if args.sims is not None:
        SIMULATIONS = int(args.sims)
        USE_SCHEDULES = False
    if args.games is not None:
        GAMES_PER_GEN = int(args.games)
        USE_SCHEDULES = False
    
    # Setup directories
    # Note: Assuming this script is run from project root, experiments will be at snake_ai/experiments
    # If run from snake_ai/ folder, it will be at ./experiments
    # Let's anchor it relative to this file
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

    if args.resume is not None:
        run_dir, is_resume = get_run_dir(base_dir, args.resume)
    elif args.bench_only:
        # Bench-only should be scriptable/non-interactive: default to the newest existing run.
        runs = _list_experiment_runs(base_dir)
        if runs:
            # Prefer the newest run that actually has weights saved.
            picked = None
            for r in runs:
                if os.path.isfile(os.path.join(r["path"], "snake_net.pth")):
                    picked = r
                    break

            if picked is None:
                run_dir, is_resume = runs[0]["path"], True
                print(
                    "WARNING: No saved weights (snake_net.pth) found in any run; "
                    f"benchmarking untrained model in {os.path.basename(run_dir)}",
                    flush=True,
                )
            else:
                run_dir, is_resume = picked["path"], True
                print(f"Benchmark-only: using latest weights from {os.path.basename(run_dir)}", flush=True)
        else:
            # Fall back to a new run (benchmarking an untrained model) but warn loudly.
            run_dir, is_resume = get_run_dir(base_dir, None)
            print(
                "WARNING: No existing experiments found; benchmarking an untrained model "
                f"in {os.path.basename(run_dir)}",
                flush=True,
            )
    else:
        # Default behavior: start a new run unless explicitly resuming via --resume.
        run_dir, is_resume = get_run_dir(base_dir, None)
    
    start_gen = 0
    
    # Pre-loading logic to find start_gen and best metrics
    best_avg_score_from_history = -float('inf')
    best_benchmark_from_history = {'WinPct': -float('inf'), 'AvgScore': -float('inf')}
    if is_resume:
        report_path = os.path.join(run_dir, "training_report.csv")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    start_gen = int(last_row['Gen'])
                    print(f"Resuming from Generation {start_gen}")
                    # Find best avg_score from history (for moving average)
                    for row in rows:
                        try:
                            avg_score = float(row.get('AvgScore', -float('inf')))
                            if avg_score > best_avg_score_from_history:
                                best_avg_score_from_history = avg_score
                        except (ValueError, TypeError):
                            pass
        
        # Load best benchmark metrics if available
        benchmark_path = os.path.join(run_dir, "benchmark.csv")
        if os.path.exists(benchmark_path):
            with open(benchmark_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        winpct = float(row.get('WinPct', -float('inf')))
                        avgscore = float(row.get('AvgScore', -float('inf')))
                        if winpct > best_benchmark_from_history['WinPct']:
                            best_benchmark_from_history['WinPct'] = winpct
                            best_benchmark_from_history['AvgScore'] = avgscore
                        elif winpct == best_benchmark_from_history['WinPct'] and avgscore > best_benchmark_from_history['AvgScore']:
                            best_benchmark_from_history['AvgScore'] = avgscore
                    except (ValueError, TypeError):
                        pass
    
    trainer = Trainer(run_dir, start_gen=start_gen, num_workers=args.workers, board_size=args.board_size)
    if is_resume:
        # Initialize moving average with recent history
        if best_avg_score_from_history > -float('inf'):
            # Fill window with best score (conservative initialization)
            for _ in range(min(5, start_gen)):
                trainer.avg_score_window.append(best_avg_score_from_history)
            trainer.best_moving_avg_score = best_avg_score_from_history
            trainer.best_avg_score = best_avg_score_from_history
            print(f"Best moving avg_score from history: {best_avg_score_from_history:.2f}")
        
        # Initialize benchmark metrics
        if best_benchmark_from_history['WinPct'] > -float('inf'):
            trainer.best_benchmark_winpct = best_benchmark_from_history['WinPct']
            trainer.best_benchmark_score = best_benchmark_from_history['AvgScore']
            print(f"Best benchmark from history: WinPct={best_benchmark_from_history['WinPct']:.1f}%, AvgScore={best_benchmark_from_history['AvgScore']:.2f}")
    
    # Load previous model if exists
    model_path = os.path.join(run_dir, "snake_net.pth")
    if is_resume and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            trainer.model.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            print("\nERROR: Failed to load saved model weights.")
            print("This usually means the model architecture changed (e.g., input channels/features).")
            print("Start a new run (no --resume) or keep the old code for that run.")
            print(f"Details: {e}")
            # For benchmark-only runs, prefer producing a result over crashing.
            # (Benchmarking an untrained model is still useful as a sanity check.)
            if args.bench_only:
                print("Continuing with randomly-initialized weights for benchmark-only.", flush=True)
            else:
                raise

    # Benchmark-only mode: load weights (if present), run deterministic benchmark once, then exit.
    if args.bench_only:
        cfg = BenchmarkConfig(
            board_size=int(args.board_size),
            episodes=int(args.bench_episodes),
            sims=int(args.bench_sims),
            seed=int(args.bench_seed),
        )
        # We store Gen as whatever generation index the trainer is currently at (i.e., the next gen to run).
        # This matches the rest of the script where benchmark is run after a generation completes.
        gen_idx = int(trainer.generation)
        bench_row = run_benchmark(trainer.model, cfg, generation=gen_idx)
        bench_path = append_benchmark_csv(trainer.run_dir, bench_row)
        print(
            f"Benchmark saved: {bench_path} | "
            f"Win%={bench_row['WinPct']:.1f} AvgScore={bench_row['AvgScore']:.2f} "
            f"(Eps={cfg.episodes} Sims={cfg.sims} Seed={cfg.seed})",
            flush=True,
        )
        raise SystemExit(0)
    
    print("Starting Training Loop (Ctrl+C to stop)...")
    try:
        gens_target = args.gens
        gens_done = 0
        while True:
            trainer.train_generation()
            trainer.save_model()
            trainer.save_report()

            if args.benchmark and (int(args.bench_every) > 0):
                gen_idx = int(trainer.generation)  # generation increments at end of train_generation
                if (gen_idx % int(args.bench_every)) == 0:
                    cfg = BenchmarkConfig(
                        board_size=int(args.board_size),
                        episodes=int(args.bench_episodes),
                        sims=int(args.bench_sims),
                        seed=int(args.bench_seed),
                    )
                    bench_row = run_benchmark(trainer.model, cfg, generation=gen_idx)
                    bench_path = append_benchmark_csv(trainer.run_dir, bench_row)
                    print(
                        f"Benchmark saved: {bench_path} | "
                        f"Win%={bench_row['WinPct']:.1f} AvgScore={bench_row['AvgScore']:.2f} "
                        f"(Eps={cfg.episodes} Sims={cfg.sims} Seed={cfg.seed})",
                        flush=True,
                    )
                    # Update best model based on benchmark (more reliable than self-play)
                    trainer.update_best_from_benchmark(bench_row)

            gens_done += 1
            if gens_target is not None and gens_done >= int(gens_target):
                break
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        trainer.save_model()
        trainer.save_report()
        print("Saved.")
