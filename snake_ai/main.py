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

from game import SnakeGame
from model import SnakeNet
from mcts import MCTS

# Hyperparameters (defaults; override via CLI)
LR = 5e-4
BATCH_SIZE = 128
MEMORY_SIZE = 50000
EPOCHS = 2
# Only used when schedules are disabled (see USE_SCHEDULES below)
GAMES_PER_GEN = 20
SIMULATIONS = 128
VALUE_LOSS_WEIGHT = 0.5

# Inference batching (defaults; override via CLI)
INFER_MAX_BATCH = 64
INFER_TIMEOUT_EMPTY = 0.01
INFER_TIMEOUT_NONEMPTY = 0.005

# Compute-efficient schedules (enabled by default; can be disabled via CLI)
USE_SCHEDULES = True
SIMS_START = 64
SIMS_MID = 128
SIMS_END = 200
SIMS_ENDGAME_MULT = 2  # boost sims only when the board is mostly filled
GAMES_START = 20
GAMES_END = 50

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
    """
    if generation < 10:
        base = SIMS_START
    elif generation < 30:
        base = SIMS_MID
    else:
        base = SIMS_END

    # Endgame boost: only when >75% of the board is filled
    endgame_threshold = int(0.75 * (board_size * board_size))
    if snake_len >= endgame_threshold:
        base *= SIMS_ENDGAME_MULT

    return int(base)


def get_games_per_gen(generation: int) -> int:
    # Linearly ramp games to stabilize training without exploding early compute.
    if generation <= 0:
        return GAMES_START
    if generation >= 30:
        return GAMES_END
    t = generation / 30.0
    return int(round(GAMES_START + t * (GAMES_END - GAMES_START)))

def play_games_worker(worker_id, board_size, simulations, generation, c_puct, temp_threshold, n_games, request_queue, response_queue, result_queue):
    """
    Worker function to play N games in a dedicated process.
    Uses batched inference by sending states to the main process via queues.

    IMPORTANT: Each process must have a unique (worker_id, response_queue) pair.
    This avoids cross-talk where one process consumes another's inference response.
    """
    # Re-seed random number generators for this process
    seed = os.getpid() + int(torch.randint(0, 10000, (1,)).item())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prediction Client for MCTS
    def predict_client(input_tensor):
        # input_tensor is a numpy array (C, board, board)
        request_queue.put((worker_id, input_tensor))
        p, v = response_queue.get()
        return p, v

    try:
        for _ in range(n_games):
            game = SnakeGame(board_size=board_size)

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
    except Exception:
        # Propagate a traceback to the parent so we don't "hang" silently.
        result_queue.put(("__error__", worker_id, traceback.format_exc()))
    finally:
        result_queue.put(("__done__", worker_id))

def process_state(game, state):
    input_tensor = np.zeros((4, game.board_size, game.board_size), dtype=np.float32)

    # Channel 0: Lifetime/flow of the body (temporal information).
    snake = getattr(game, "snake", [])
    L = len(snake)
    if L > 1:
        for i in range(1, L):
            x, y = snake[i]
            input_tensor[0, y, x] = (L - i) / L

    input_tensor[1] = (state == 2).astype(float)
    input_tensor[2] = (state == 3).astype(float)
    hunger_limit = max(1, getattr(game, "hunger_limit", 100))
    hunger = float(getattr(game, "steps_since_eaten", 0)) / hunger_limit
    input_tensor[3].fill(hunger)
    
    # Rotate based on direction to enforce POV (Head Up)
    k = game.direction
    input_tensor = np.rot90(input_tensor, k, axes=(1, 2)).copy()
    
    return input_tensor

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

class Trainer:
    def __init__(self, run_dir, start_gen=0, num_workers=None, board_size=6):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SnakeNet(board_size=self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.history = []
        self.generation = start_gen
        self.run_dir = run_dir
        
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
        request_queue = ctx.Queue()
        response_queues = [ctx.Queue() for _ in range(self.num_workers)]
        result_queue = ctx.Queue()

        # Split games across dedicated processes (each has a unique response queue)
        base = games_this_gen // self.num_workers
        rem = games_this_gen % self.num_workers
        games_per_worker = [base + (1 if i < rem else 0) for i in range(self.num_workers)]

        procs = []
        active_workers = 0
        for w_id, n_games in enumerate(games_per_worker):
            if n_games <= 0:
                continue
            p = ctx.Process(
                target=play_games_worker,
                args=(
                    w_id,
                    self.board_size,
                    SIMULATIONS,
                    self.generation,
                    current_c_puct,
                    current_temp_threshold,
                    n_games,
                    request_queue,
                    response_queues[w_id],
                    result_queue,
                ),
            )
            p.daemon = True
            p.start()
            procs.append(p)
            active_workers += 1
        
        # Inference Loop (Main Thread)
        self.model.eval()
        
        # Variables for monitoring
        last_monitor_time = time.time()
        inference_batches = 0
        
        done_workers = 0
        results_received = 0
        expected_results = games_this_gen
        last_progress_time = time.time()
        last_results_received = 0

        while results_received < expected_results:
            # Check for requests
            batch_reqs = []
            
            # Non-blocking collect up to BATCH_SIZE (e.g. 64) OR until empty
            # We want to wait a tiny bit if empty to batch up, but not too long
            start_wait = time.time()
            while len(batch_reqs) < INFER_MAX_BATCH:
                try:
                    # If we have nothing, wait a bit (latency/throughput trade-off)
                    # If we have something, don't wait much for more
                    # TWEAK: Increased latency to force larger batches
                    timeout = INFER_TIMEOUT_NONEMPTY if len(batch_reqs) > 0 else INFER_TIMEOUT_EMPTY
                    req = request_queue.get(timeout=timeout)
                    batch_reqs.append(req)
                except queue.Empty:
                    break
            
            if batch_reqs:
                inference_batches += 1
                # Process Batch
                # unzip
                worker_ids = [r[0] for r in batch_reqs]
                input_np = np.array([r[1] for r in batch_reqs])
                
                input_tensor = torch.tensor(input_np).to(self.device)
                
                with torch.no_grad():
                    policies, values = self.model(input_tensor)
                    
                policies = torch.exp(policies).cpu().numpy()
                values = values.cpu().numpy()
                
                # Send back results
                for i, w_id in enumerate(worker_ids):
                    response_queues[w_id].put((policies[i], values[i].item()))
            else:
                 # No requests, sleep briefly to avoid CPU spin
                 time.sleep(0.001)

            # Drain finished game results (non-blocking)
            while True:
                try:
                    msg = result_queue.get_nowait()
                except queue.Empty:
                    break

                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__done__":
                    done_workers += 1
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
                last_progress_time = time.time()

            # Monitor periodically
            if time.time() - last_monitor_time > 5:
                alive = sum(1 for p in procs if p.is_alive())
                print(f"   [Progress] FinishedGames: {results_received}/{expected_results} | WorkersAlive: {alive}/{active_workers}", flush=True)

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
                    
                print(f"   [Monitor] CPU: {cpu_pct:.1f}% | GPU Util: {gpu_util} | GPU Mem: {gpu_mem} | Workers: {self.num_workers} | Batch Rate: {inference_batches/5:.1f}/s", flush=True)
                inference_batches = 0
                last_monitor_time = time.time()

            # Deadlock protection: if all workers are done but we didn't receive all games, fail fast.
            if done_workers >= active_workers and results_received < expected_results:
                raise RuntimeError(
                    f"All workers finished ({done_workers}/{active_workers}) but only received {results_received}/{expected_results} game results."
                )

            # Deadlock protection: no progress for too long
            if time.time() - last_progress_time > 60 and results_received == last_results_received:
                alive = sum(1 for p in procs if p.is_alive())
                raise RuntimeError(
                    f"No progress for 60s (FinishedGames {results_received}/{expected_results}, WorkersAlive {alive}/{active_workers})."
                )
            last_results_received = results_received

        # Ensure workers are cleaned up
        for p in procs:
            p.join(timeout=5)
            
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
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        total_pred_acc = 0
        steps = 0
        
        # Train for epochs
        self.model.train() # <--- CRITICAL FIX: Switch back to train mode so BatchNorm stats update
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
            
        duration = time.time() - start_time
        # Step LR schedule once per generation (after at least one training step)
        if steps > 0:
            self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        print(f"Gen {self.generation+1} Finished. Avg Score: {avg_score:.2f}, Max Score: {max_score}, Loss: {avg_loss:.4f}, Time: {duration:.2f}s")
        print(f"Stats: LR={current_lr:.2e}, Entropy={avg_entropy:.4f}, Steps={avg_steps:.1f}, PredAcc={avg_pred_acc:.1%}, Deaths={death_counts}")
        
        self.history.append({
            'Gen': self.generation + 1, 
            'AvgScore': avg_score, 
            'MaxScore': max_score, 
            'MinScore': min_score,
            'Loss': avg_loss,
            'PolicyLoss': avg_p_loss,
            'ValueLoss': avg_v_loss,
            'PredAcc': avg_pred_acc,
            'Games': GAMES_PER_GEN,
            'Time': duration,
            'AvgSteps': avg_steps,
            'AvgEntropy': avg_entropy,
            'DeathWall': death_counts['wall'],
            'DeathBody': death_counts['body'],
            'DeathTimeout': death_counts['timeout'],
            'DeathStarvation': death_counts['starvation'],
            'DeathWon': death_counts['won']
        })
        self.generation += 1
        return avg_score

    def train_step(self, batch):
        inputs = torch.tensor(np.array([x[0] for x in batch])).to(self.device)
        target_pis = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(self.device)
        target_vs = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        self.optimizer.zero_grad()
        p, v = self.model(inputs)
        
        # Huber loss is more stable than pure MSE when returns can be large/outlier-y.
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
        
        fieldnames = ['Gen', 'AvgScore', 'MaxScore', 'MinScore', 'Loss', 'PolicyLoss', 'ValueLoss', 'PredAcc', 'Games', 'Time', 'AvgSteps', 'AvgEntropy', 'DeathWall', 'DeathBody', 'DeathTimeout', 'DeathStarvation', 'DeathWon']
        
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
    parser.add_argument("--sims", type=int, default=None, help="Base simulations per move (disables schedule if set)")
    parser.add_argument("--games", type=int, default=None, help="Games per generation (disables schedule if set)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per generation")
    parser.add_argument("--memory", type=int, default=None, help="Replay buffer size")
    parser.add_argument("--batch", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--infer-batch", type=int, default=None, help="Max inference batch size")
    parser.add_argument("--infer-wait-empty", type=float, default=None, help="Queue wait (seconds) when empty")
    parser.add_argument("--infer-wait-nonempty", type=float, default=None, help="Queue wait (seconds) when already batching")
    args = parser.parse_args()

    # Apply CLI overrides / presets (module-level globals)
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
    
    run_dir, is_resume = get_run_dir(base_dir, args.resume)
    
    start_gen = 0
    
    # Pre-loading logic to find start_gen
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
    
    trainer = Trainer(run_dir, start_gen=start_gen, num_workers=args.workers, board_size=args.board_size)
    
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
            raise
    
    print("Starting Training Loop (Ctrl+C to stop)...")
    try:
        while True:
            trainer.train_generation()
            trainer.save_model()
            trainer.save_report()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        trainer.save_model()
        trainer.save_report()
        print("Saved.")
