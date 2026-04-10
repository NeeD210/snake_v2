import math
import numpy as np
import torch

from fast_state import FastSnakeState
from encoder import encode_pov

class Node:
    def __init__(self, state, parent=None, action_taken=None, prior=0, reward=0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.reward = reward  # Immediate reward received getting to this state
        
        self.children = {} # Map action -> Node
        self.N = 0 # Visit count
        self.Q = 0 # Mean value
        self.vloss = 0 # Virtual Loss count
        
        self.is_dead = False # Dead-End Pruning flag
        
    def is_expanded(self):
        return len(self.children) > 0

    def select(self, c_puct=1.0):
        """
        Selects the child with the highest UCB score.
        """
        best_score = -float('inf')
        # Fallback if all children are dead (prevents None unpacking crash)
        best_action = list(self.children.keys())[0] if self.children else None
        best_child = self.children[best_action] if best_action is not None else None

        # Min-Max Scaling for Q-values (exclude dead nodes so safe nodes scale properly)
        q_values = []
        for child in self.children.values():
            if not getattr(child, 'is_dead', False):
                q_values.append(child.Q)
            
        if q_values:
            min_q = min(q_values)
            max_q = max(q_values)
        else:
            min_q = 0
            max_q = 0 # Should not happen unless all children are dead
            
        epsilon = 1e-4

        for action, child in self.children.items():
            if getattr(child, 'is_dead', False):
                score = -float('inf')
            else:
                # Normalize Q
                if max_q > min_q:
                    normalized_q = (child.Q - min_q) / (max_q - min_q)
                else:
                    normalized_q = 0.5 # Default if all equal
                    
                u = c_puct * child.prior * (math.sqrt(self.N + self.vloss) / (1 + child.N + child.vloss))
                
                # Virtual loss: penalize Q to discourage other in-flight sims
                # (Assuming normalized_q is [0,1], subtracting 1.0 is enough)
                score = (normalized_q - (1.0 if child.vloss > 0 else 0.0)) + u
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def expand(self, policy, valid_moves, game_snapshot):
        """
        Expands the node by creating children.
        policy: list of probabilities for all 4 moves
        valid_moves: list of valid action indices
        game_snapshot: the game object at this state
        """
        for action in valid_moves:
             if action not in self.children:
                # Simulate the next state
                if isinstance(game_snapshot, FastSnakeState):
                    reward, _done = game_snapshot.step_relative(action)
                    child_state = game_snapshot.get_state()
                    game_snapshot.undo()
                else:
                    next_game = game_snapshot.clone()
                    # action is relative (0, 1, 2)
                    abs_action = (next_game.direction + (action - 1)) % 4
                    _, reward, _done = next_game.step(abs_action) # Capture specific reward for this transition
                    child_state = next_game.get_state()
                self.children[action] = Node(
                    child_state, 
                    parent=self, 
                    action_taken=action, 
                    prior=policy[action],
                    reward=reward
                )

    def update(self, value, gamma=0.95):
        """
        Backpropagates the value up the tree using discounted returns.
        value: The estimated value of the future (from the child's perspective)
        """
        self.N += 1
        # Q tracks the average expected return from this state
        self.Q += (value - self.Q) / self.N
        
        # Propagation of dead ends
        if self.is_expanded():
            all_dead = True
            for child in self.children.values():
                if not getattr(child, 'is_dead', False):
                    all_dead = False
                    break
            self.is_dead = all_dead
        
        if self.parent:
            # The value of the parent is This Reward + Discounted Future Value
            parent_return = self.reward + gamma * value
            self.parent.update(parent_return, gamma)

class MCTS:
    def __init__(self, model_or_predict_fn, c_puct=1.0, n_simulations=50, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        if callable(model_or_predict_fn) and not isinstance(model_or_predict_fn, torch.nn.Module):
            self.predict_fn = model_or_predict_fn
            self.device = None # handled by predict_fn
        else:
            self.model = model_or_predict_fn
            self.device = next(self.model.parameters()).device
            self.predict_fn = self._default_predict

        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root = None

    def reset(self):
        self.root = None

    def update_root(self, action):
        if self.root and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None

    def search(self, game, n_simulations=None, num_parallel=8):
        """
        Runs MCTS simulations with Virtual Loss to support parallel in-flight inferences.
        """
        import time
        start_time = time.perf_counter()
        self.inf_wait_time = 0.0

        sims = self.n_simulations if n_simulations is None else int(n_simulations)
        if sims <= 0: sims = 1
        
        # Determine if we can use async (predict_fn must support it)
        is_async = hasattr(self.predict_fn, 'is_async') and self.predict_fn.is_async

        # Safety: non-blocking drain of any stale responses left from a prior search
        if is_async and hasattr(self.predict_fn, '_drain_stale'):
            self.predict_fn._drain_stale()

        root_sim = FastSnakeState.from_game(game)

        # Root initialization
        if self.root is None or not np.array_equal(self.root.state, game.get_state()):
            self.root = Node(game.get_state(), prior=0, reward=0)
            policy, _ = self.predict(root_sim)
            valid_moves = root_sim.get_valid_relative_moves()
            self.root.expand(policy, valid_moves, root_sim)
            self._add_dirichlet_noise(self.root, valid_moves)

        sims_done = 0
        sims_started = 0
        in_flight = [] # List of (seq_id, actions_taken, node, depth)
        winning_path = None

        while sims_done < sims:
            # 1. Fill in-flight pipeline
            while len(in_flight) < num_parallel and sims_started < sims:
                node = self.root
                depth = 0
                actions_taken = []
                
                # SELECT
                while node.is_expanded():
                    action, node = node.select(self.c_puct)
                    root_sim.step_relative(action)
                    actions_taken.append(action)
                    depth += 1
                
                if root_sim.done:
                    # Terminal node: sync backup
                    if root_sim.death_reason == "won":
                        winning_path = actions_taken.copy()
                    else:
                        node.is_dead = True
                    node.update(0) 
                    sims_done += 1
                    sims_started += 1
                    # Revert
                    for _ in range(depth): root_sim.undo()
                    if winning_path: break
                else:
                    # Expansion node: Apply Virtual Loss and push to inference
                    # Apply VLoss up the path
                    curr = node
                    while curr:
                        curr.vloss += 1
                        curr = curr.parent
                    
                    # Snapshot for expansion
                    input_tensor = encode_pov(root_sim)
                    # For a real implementation, we'd queue these. 
                    # If is_async is false, we just do it sync here but still track flow.
                    if not is_async:
                        res = self.predict_fn(input_tensor)
                        # Remove VLoss immediately
                        curr = node
                        while curr:
                            curr.vloss -= 1
                            curr = curr.parent
                        
                        valid_moves = root_sim.get_valid_relative_moves()
                        node.expand(res[0], valid_moves, root_sim)
                        node.update(res[1])
                        sims_done += 1
                    else:
                        # Queue request
                        seq_id = self.predict_fn.send_async(input_tensor)
                        in_flight.append((seq_id, actions_taken, node, depth))
                    
                    sims_started += 1
                    # Revert state for next sim
                    for _ in range(depth): root_sim.undo()

            if winning_path: break
            if not in_flight: 
                if sims_started >= sims: break
                continue

            # 2. Wait for ANY in-flight result (tagged)
            if is_async:
                results = self.predict_fn.poll_results(wait=True)
                for resp_seq, p, v in results:
                    # Find matching entry in in_flight
                    match_idx = -1
                    for i, (sid, _, _, _) in enumerate(in_flight):
                        if sid == resp_seq:
                            match_idx = i
                            break
                    
                    if match_idx == -1:
                        # Stale response from previous search, ignore
                        continue
                    
                    _, actions, node, depth = in_flight.pop(match_idx)
                    
                    # Remove VLoss
                    curr = node
                    while curr:
                        curr.vloss -= 1
                        curr = curr.parent
                    
                    # Fast-forward sim to this node to get valid moves
                    for a in actions: root_sim.step_relative(a)
                    valid_moves = root_sim.get_valid_relative_moves()
                    node.expand(p, valid_moves, root_sim)
                    # Backup result
                    node.update(v)
                    sims_done += 1
                    # Revert
                    for _ in range(depth): root_sim.undo()

            # Early stopping (Entropy-Based)
            if sims_done >= 5:
                curr_counts = np.zeros(3)
                for action, child in self.root.children.items():
                    curr_counts[action] = child.N
                sum_counts = np.sum(curr_counts)
                if sum_counts > 0:
                    probs = curr_counts / sum_counts
                    current_entropy = -np.sum(probs * np.log(probs + 1e-8))
                    if current_entropy < 0.15: break

        # Drain any in-flight async requests that were abandoned due to early
        # stopping or winning-path break.
        if is_async and in_flight:
            drain_start = time.perf_counter()
            while in_flight:
                # Fail-safe timeout: 5s
                if time.perf_counter() - drain_start > 5.0:
                    print(f"   [MCTS] WARNING: Drain timeout. Abandoning {len(in_flight)} requests.", flush=True)
                    # Force return slots for abandoned requests
                    for sid, _, _, _ in in_flight:
                        if hasattr(self.predict_fn, 'seq_to_slot') and sid in self.predict_fn.seq_to_slot:
                             s_id = self.predict_fn.seq_to_slot.pop(sid)
                             self.predict_fn.free_slots.append(s_id)
                    break
                    
                results = self.predict_fn.poll_results(wait=True)
                for resp_seq, p, v in results:
                    match_idx = -1
                    for i, (sid, _, _, _) in enumerate(in_flight):
                        if sid == resp_seq:
                            match_idx = i
                            break
                    
                    if match_idx != -1:
                        _, actions, node, depth = in_flight.pop(match_idx)
                        # Remove Virtual Loss
                        curr = node
                        while curr:
                            curr.vloss -= 1
                            curr = curr.parent
                        # Expand & backup to keep tree consistent
                        for a in actions:
                            root_sim.step_relative(a)
                        valid_moves = root_sim.get_valid_relative_moves()
                        node.expand(p, valid_moves, root_sim)
                        node.update(v)
                        sims_done += 1
                        for _ in range(depth):
                            root_sim.undo()
                    # else: stale response, ignore and loop until in_flight is empty

        # Calculate final policy
        counts = np.zeros(3)
        for action, child in self.root.children.items(): counts[action] = child.N
        if np.sum(counts) > 0: counts = counts / np.sum(counts)
        else: counts = np.array([0.33, 0.33, 0.33])
        entropy = -np.sum(counts * np.log(counts + 1e-8))
        
        total_time = time.perf_counter() - start_time
        search_logic_time = total_time - self.inf_wait_time
        return counts, entropy, winning_path, sims_done, (search_logic_time, self.inf_wait_time)

    def predict(self, game):
        import time
        start_inf = time.perf_counter()
        
        input_tensor = encode_pov(game)
        result = self.predict_fn(input_tensor)
        
        # Attribute wait time to the current search session
        if hasattr(self, 'inf_wait_time'):
            self.inf_wait_time += (time.perf_counter() - start_inf)
            
        return result

    def _default_predict(self, input_tensor):
        input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            p, v = self.model(input_tensor)
            
        p = torch.exp(p).squeeze().cpu().numpy()
        v = v.item()
        return p, v

    def _add_dirichlet_noise(self, node, valid_moves):
        """
        Adds Dirichlet noise to the prior probabilities of the node's children.
        This encourages exploration of different actions at the root.
        """
        if not self.dirichlet_epsilon > 0:
            return

        actions = list(node.children.keys())
        if not actions:
            return

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]
