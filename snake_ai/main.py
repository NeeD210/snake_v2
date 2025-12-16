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

from game import SnakeGame
from model import SnakeNet
from mcts import MCTS

# Hyperparameters
LR = 0.001
BATCH_SIZE = 128
MEMORY_SIZE = 20000 
EPOCHS = 5         
GAMES_PER_GEN = 20 
SIMULATIONS = 100   
VALUE_LOSS_WEIGHT = 0.5

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

    for _ in range(n_games):
        game = SnakeGame(board_size=board_size)

        # Fresh tree per game
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

    result_queue.put(("__done__", worker_id))

def process_state(game, state):
    input_tensor = np.zeros((4, game.board_size, game.board_size), dtype=np.float32)
    input_tensor[0] = (state == 1).astype(float)
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
    def __init__(self, run_dir, start_gen=0, num_workers=None):
        # We'll use 6x6 board as per original
        self.board_size = 6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SnakeNet(board_size=self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
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
        
        print(f"Gen {self.generation+1} Params: C_PUCT={current_c_puct:.2f}, TempThreshold={current_temp_threshold}")
        
        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue()
        response_queues = [ctx.Queue() for _ in range(self.num_workers)]
        result_queue = ctx.Queue()

        # Split games across dedicated processes (each has a unique response queue)
        base = GAMES_PER_GEN // self.num_workers
        rem = GAMES_PER_GEN % self.num_workers
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
        expected_results = GAMES_PER_GEN

        while results_received < expected_results:
            # Check for requests
            batch_reqs = []
            
            # Non-blocking collect up to BATCH_SIZE (e.g. 64) OR until empty
            # We want to wait a tiny bit if empty to batch up, but not too long
            start_wait = time.time()
            while len(batch_reqs) < 64:
                try:
                    # If we have nothing, wait a bit (latency/throughput trade-off)
                    # If we have something, don't wait much for more
                    # TWEAK: Increased latency to force larger batches
                    timeout = 0.005 if len(batch_reqs) > 0 else 0.01 
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

                samples, score, death_reason, entropy, steps_played = msg
                new_samples.extend(samples)
                scores.append(score)
                death_reasons.append(death_reason)
                entropies.append(entropy)
                total_steps += steps_played
                if score < min_score:
                    min_score = score
                results_received += 1

            # Monitor periodically
            if time.time() - last_monitor_time > 5:
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
        
        print(f"Gen {self.generation+1} Finished. Avg Score: {avg_score:.2f}, Max Score: {max_score}, Loss: {avg_loss:.4f}, Time: {duration:.2f}s")
        print(f"Stats: Entropy={avg_entropy:.4f}, Steps={avg_steps:.1f}, PredAcc={avg_pred_acc:.1%}, Deaths={death_counts}")
        
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
    args = parser.parse_args()
    
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
    
    trainer = Trainer(run_dir, start_gen=start_gen, num_workers=args.workers)
    
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
