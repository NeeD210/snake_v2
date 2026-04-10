import numpy as np
from multiprocessing import shared_memory
import traceback

from encoder import UNIVERSAL_SIZE, NUM_CHANNELS

NUM_SLOTS = 16

class SHMWorkerClient:
    def __init__(self, worker_id, obs_shm_names):
        """
        obs_shm_names: list of SHM names for this worker's slots
        """
        self.worker_id = worker_id
        self.obs_shm_names = obs_shm_names
        
        self.obs_shms = []
        self.obs_bufs = []

        for name in obs_shm_names:
            try:
                shm = shared_memory.SharedMemory(name=name)
                self.obs_shms.append(shm)
                buf = np.ndarray((NUM_CHANNELS, UNIVERSAL_SIZE, UNIVERSAL_SIZE), dtype=np.float32, buffer=shm.buf)
                self.obs_bufs.append(buf)
            except FileNotFoundError:
                raise RuntimeError(f"Shared memory {name} not found for worker {worker_id}.")

    def send_request(self, slot_id, seq_id, observation, request_queue):
        """Writes observation to specific slot and puts (worker_id, slot_id, seq_id) in the queue."""
        self.obs_bufs[slot_id][:] = observation[:]
        request_queue.put((self.worker_id, slot_id, seq_id))
        
    def wait_for_response(self, response_queue):
        """Waits for (seq_id, p, v) in the response queue."""
        return response_queue.get()

    def close(self):
        for shm in self.obs_shms:
            shm.close()

class SHMManager:
    def __init__(self, num_workers, num_channels=NUM_CHANNELS):
        import os
        self.num_workers = num_workers
        self.num_channels = num_channels
        self.master_pid = os.getpid()
        
        # Grid of buffers: [worker_id, slot_id]
        self.obs_shms = [[None for _ in range(NUM_SLOTS)] for _ in range(num_workers)]
        self.obs_bufs = [[None for _ in range(NUM_SLOTS)] for _ in range(num_workers)]
        self.obs_shm_names = [[None for _ in range(NUM_SLOTS)] for _ in range(num_workers)]
        
        for w_id in range(num_workers):
            for s_id in range(NUM_SLOTS):
                name = f"snake_obs_p{self.master_pid}_w{w_id}_s{s_id}"
                self.obs_shm_names[w_id][s_id] = name
                size = num_channels * UNIVERSAL_SIZE * UNIVERSAL_SIZE * 4
                
                # Clean up legacy
                try:
                    temp = shared_memory.SharedMemory(name=name)
                    temp.close()
                    temp.unlink()
                except: pass

                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                self.obs_shms[w_id][s_id] = shm
                self.obs_bufs[w_id][s_id] = np.ndarray((num_channels, UNIVERSAL_SIZE, UNIVERSAL_SIZE), dtype=np.float32, buffer=shm.buf)

    def get_observation(self, worker_id, slot_id):
        return self.obs_bufs[worker_id][slot_id]

    def cleanup(self):
        for w_id in range(self.num_workers):
            for s_id in range(NUM_SLOTS):
                shm = self.obs_shms[w_id][s_id]
                if shm:
                    shm.close()
                    shm.unlink()
