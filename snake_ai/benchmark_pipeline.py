import torch
import torch.multiprocessing as mp
import numpy as np
import time
import queue
import argparse
import os
import sys

# Add current directory to path so we can import from local files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game import SnakeGame
from model import SnakeNet
from mcts import MCTS
from encoder import encode_pov
from shm_ipc import SHMManager, SHMWorkerClient

# Import necessary parts from main.py if possible, otherwise redefine locally
# Note: play_games_worker is in main.py but depends on main-level globals.
# For the benchmark, we'll use a simplified version.

def benchmark_worker(
    worker_id,
    board_size,
    obs_shm_name,
    simulations,
    game_queue,
    request_queue,
    response_queue,
    result_queue,
):
    """
    Simplified worker for benchmarking. 
    Focuses on MCTS and Inference loop without training data collection.
    """
    # Prediction client using Shared Memory with Async support
    class AsyncSHMClient:
        def __init__(self, client):
            self.client = client
            self.is_async = True
            self.pending_reqs = [] # List of (req_id, timestamp)
            
        def __call__(self, input_tensor):
            """Fallback for sync calls (e.g. root expansion)"""
            self.client.send_request(input_tensor, request_queue)
            return self.client.wait_for_response(response_queue)

        def send_async(self, input_tensor):
            req_id = len(self.pending_reqs)
            self.client.send_request(input_tensor, request_queue)
            self.pending_reqs.append(req_id)
            return req_id

        def poll_results(self, wait=True):
            if not self.pending_reqs: return []
            p, v = self.client.wait_for_response(response_queue)
            req_id = self.pending_reqs.pop(0)
            return [(req_id, p, v)]

    shm_client = SHMWorkerClient(worker_id, board_size, obs_shm_name)
    async_client = AsyncSHMClient(shm_client)
    
    def predict_client(input_tensor):
        return async_client(input_tensor)
    # Patch async methods
    predict_client.is_async = True
    predict_client.send_async = async_client.send_async
    predict_client.poll_results = async_client.poll_results

    while True:
        _game_token = game_queue.get()
        if _game_token is None:
            break
        
        game = SnakeGame(board_size=board_size)
        mcts = MCTS(predict_client, n_simulations=simulations)
        mcts.reset()

        state_tensor = game.reset()
        move_count = 0
        total_sims = 0
        total_mcts_logic_time = 0.0
        total_inf_wait_time = 0.0

        while not game.done:
            # Parallel search with Virtual Loss
            action_probs, entropy, win_path, sims_performed, timing = mcts.search(game, num_parallel=8)
            mcts_logic, inf_wait = timing
            total_mcts_logic_time += mcts_logic
            total_inf_wait_time += inf_wait
            
            total_sims += sims_performed
            move_count += 1

            rel_action = int(np.argmax(action_probs))
            abs_action = (game.direction + (rel_action - 1)) % 4
            state_tensor, reward, done = game.step(abs_action)
            mcts.update_root(rel_action)

        # Report timing and throughput data back
        result_queue.put({
            'worker_id': worker_id,
            'moves': move_count,
            'sims': total_sims,
            'mcts_logic': total_mcts_logic_time,
            'inf_wait': total_inf_wait_time
        })

def run_benchmark_config(num_workers, simulations, board_size, num_games):
    """Runs a single (workers, sims) configuration benchmark."""
    ctx = mp.get_context("spawn")
    
    game_queue = ctx.Queue()
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(num_workers)]
    result_queue = ctx.Queue()
    
    shm_manager = SHMManager(num_workers, board_size, ctx=ctx)

    # Fill game queue
    for i in range(num_games):
        game_queue.put(i)
    for _ in range(num_workers):
        game_queue.put(None)

    # Init model on CPU/GPU as available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SnakeNet(board_size=board_size).to(device)
    model.eval()

    procs = []
    for w_id in range(num_workers):
        p = ctx.Process(
            target=benchmark_worker,
            args=(w_id, board_size, shm_manager.obs_shm_names[w_id], simulations, game_queue, request_queue, response_queues[w_id], result_queue)
        )
        p.start()
        procs.append(p)

    # Inference Loop
    start_time = time.time()
    inf_busy_time = 0.0
    results_received = 0
    total_moves = 0
    total_sims = 0
    mcts_logic_sum = 0.0
    inf_wait_sum = 0.0
    
    # Constants
    INFER_MAX_BATCH = 64
    INFER_TIMEOUT_EMPTY = 0.01
    INFER_TIMEOUT_NONEMPTY = 0.005

    while results_received < num_games:
        batch_reqs = []
        while len(batch_reqs) < INFER_MAX_BATCH:
            try:
                timeout = INFER_TIMEOUT_NONEMPTY if len(batch_reqs) > 0 else INFER_TIMEOUT_EMPTY
                w_id = request_queue.get(timeout=timeout)
                batch_reqs.append(w_id)
            except queue.Empty:
                break
        
        # Resolve observation data from SHM
        if batch_reqs:
            t_compute_start = time.perf_counter()
            
            # batch_reqs now just contains worker_ids
            obs_list = [shm_manager.get_observation(w_id) for w_id in batch_reqs]
            group_inputs = torch.tensor(np.array(obs_list)).to(device)
            
            with torch.no_grad():
                p_group, v_group = model(group_inputs)
                
            p_group = torch.exp(p_group).cpu().numpy()
            v_group = v_group.cpu().numpy()
            
            for i, w_id in enumerate(batch_reqs):
                # obs was already read directly from shm_manager.get_observation(w_id)
                shm_manager.set_response(response_queues[w_id], p_group[i], v_group[i].item())
            
            inf_busy_time += (time.perf_counter() - t_compute_start)
        else:
            time.sleep(0.001)

        # 2. Results
        while True:
            try:
                msg = result_queue.get_nowait()
                total_moves += msg['moves']
                total_sims += msg['sims']
                mcts_logic_sum += msg['mcts_logic']
                inf_wait_sum += msg['inf_wait']
                results_received += 1
            except queue.Empty:
                break

    duration = time.time() - start_time
    for p in procs:
        p.join(timeout=1)
        if p.is_alive(): p.terminate()
    
    shm_manager.cleanup()

    # Calculate metrics
    games_per_min = (num_games / duration) * 60
    sims_per_sec = total_sims / duration
    worker_sat = (mcts_logic_sum / (mcts_logic_sum + inf_wait_sum)) * 100 if (mcts_logic_sum + inf_wait_sum) > 0 else 0
    inf_duty = (inf_busy_time / duration) * 100 if duration > 0 else 0
    
    return {
        'workers': num_workers,
        'sims': simulations,
        'g_min': games_per_min,
        's_sec': sims_per_sec,
        'worker_sat': worker_sat,
        'inf_duty': inf_duty,
        'time': duration
    }

def main():
    parser = argparse.ArgumentParser(description="System Pipeline Benchmark for Snake AI")
    parser.add_argument("--board-size", type=int, default=10, help="Board size for benchmark")
    parser.add_argument("--games", type=int, default=12, help="Games to run per configuration")
    parser.add_argument("--quick", action="store_true", help="Run a fast minimal sweep")
    args = parser.parse_args()

    if args.quick:
        WORKER_COUNTS = [1, 4, 8]
        SIM_COUNTS = [5, 15]
    else:
        # Deterministic max workers based on cores
        max_workers = os.cpu_count() or 4
        WORKER_COUNTS = [5, 10, 15, 20]
        WORKER_COUNTS = [w for w in WORKER_COUNTS if w <= max_workers + 4] # Allow slight oversubscription
        SIM_COUNTS = [5, 10, 15, 30, 64]

    print(f"=== Snake AI System Pipeline Benchmark ===")
    print(f"Board Size: {args.board_size}x{args.board_size} | Games per config: {args.games}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Inference Device: {device_name}")
    print("-" * 80)
    print(f"{'Workers':<8} {'Sims':<8} {'G/min':<10} {'S/sec':<10} {'W_Sat%':<10} {'Inf_Duty%':<12}")
    print("-" * 80)

    all_results = []
    print(f"DEBUG: SIM_COUNTS={SIM_COUNTS}, WORKER_COUNTS={WORKER_COUNTS}")
    for sims in SIM_COUNTS:
        for workers in WORKER_COUNTS:
            # Run benchmark configuration
            res = run_benchmark_config(workers, sims, args.board_size, args.games)
            all_results.append(res)
            print(f"{res['workers']:<8} {res['sims']:<8} {res['g_min']:<10.2f} {res['s_sec']:<10.2f} {res['worker_sat']:<10.1f} {res['inf_duty']:<12.1f}")

    print("-" * 80)
    print("\n💡 Bottleneck Analysis:")
    
    # Identify the best configuration for throughput
    best_config = max(all_results, key=lambda x: x['s_sec'])
    print(f"Max Throughput observed: {best_config['s_sec']:.2f} S/sec with {best_config['workers']} workers and {best_config['sims']} sims.")
    
    # Simple analysis based on best throughput config
    if best_config['worker_sat'] < 40:
        print("🔴 ANALYSIS: Pipeline is heavily bound by Inference Latency or IPC Overhead.")
        print("   Recommend: Faster IPC (Shared Memory) or increasing simulations per move to hide latency.")
    elif best_config['inf_duty'] > 80:
        print("🟡 ANALYSIS: Inference server/GPU is fully saturated.")
        print("   Recommend: Optimizing model depth or using TensorRT for faster forward pass.")
    else:
        print("🟢 ANALYSIS: Pipeline is well-balanced or limited by raw CPU speed (MCTS traversal).")
        print("   Recommend: Numba/Cython optimizations to speed up pure MCTS logic.")

if __name__ == "__main__":
    main()
