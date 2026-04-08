import numpy as np
from multiprocessing import shared_memory
import traceback

class SHMWorkerClient:
    def __init__(self, worker_id, board_size, obs_shm_name, num_channels=5):
        self.worker_id = worker_id
        self.board_size = board_size
        self.num_channels = num_channels
        self.obs_shm_name = obs_shm_name
        
        # Obs buffer: (C, H, W)
        obs_size = num_channels * board_size * board_size * 4 # float32
        # Res buffer: (4,) -> [p0, p1, p2, value]
        res_size = 4 * 4 # float32
        
        try:
            self.obs_shm = shared_memory.SharedMemory(name=self.obs_shm_name)
        except FileNotFoundError:
            # If not created yet, this client is being initialized before the manager (shouldn't happen in worker)
            raise RuntimeError(f"Shared memory not found for worker {worker_id}. Ensure SHMManager is initialized in the master process.")

        self.obs_buf = np.ndarray((num_channels, board_size, board_size), dtype=np.float32, buffer=self.obs_shm.buf)

    def send_request(self, observation, request_queue):
        """Writes observation to SHM and puts worker_id in the queue."""
        self.obs_buf[:] = observation[:]
        request_queue.put(self.worker_id)
        
    def wait_for_response(self, response_queue):
        """Waits for the master to signal that the response is ready in the queue."""
        p, v = response_queue.get()
        # Return policy and value
        return p, v

    def close(self):
        self.obs_shm.close()

class SHMManager:
    def __init__(self, num_workers, board_size, num_channels=5, ctx=None):
        import os
        self.num_workers = num_workers
        self.board_size = board_size
        self.num_channels = num_channels
        self.master_pid = os.getpid()
        
        self.obs_shms = []
        self.obs_shm_names = []
        
        self.obs_bufs = []
        
        for i in range(num_workers):
            obs_shm_name = f"snake_obs_p{self.master_pid}_w{i}"
            self.obs_shm_names.append(obs_shm_name)
            obs_size = num_channels * board_size * board_size * 4
            
            # Clean up existing SHM if any
            try:
                temp = shared_memory.SharedMemory(name=obs_shm_name)
                temp.close()
                temp.unlink()
            except: pass

            o_shm = shared_memory.SharedMemory(name=obs_shm_name, create=True, size=obs_size)
            self.obs_shms.append(o_shm)
            self.obs_bufs.append(np.ndarray((num_channels, board_size, board_size), dtype=np.float32, buffer=o_shm.buf))

    def get_observation(self, worker_id):
        return self.obs_bufs[worker_id]

    def set_response(self, response_queue, policy, value):
        response_queue.put((policy, value))

    def cleanup(self):
        for shm in self.obs_shms:
            shm.close()
            shm.unlink()
