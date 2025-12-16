import math
import numpy as np
import torch

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
        
    def is_expanded(self):
        return len(self.children) > 0

    def select(self, c_puct=1.0):
        """
        Selects the child with the highest UCB score.
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        # Min-Max Scaling for Q-values
        # To make Q and U comparable, we normalize Q to [0, 1]
        
        q_values = []
        for child in self.children.values():
            q_values.append(child.Q)
            
        if q_values:
            min_q = min(q_values)
            max_q = max(q_values)
        else:
            min_q = 0
            max_q = 0 # Should not happen if children exist
            
        epsilon = 1e-4

        for action, child in self.children.items():
            # Normalize Q
            if max_q > min_q:
                normalized_q = (child.Q - min_q) / (max_q - min_q)
            else:
                normalized_q = 0.5 # Default if all equal
                
            u = c_puct * child.prior * (math.sqrt(self.N) / (1 + child.N))
            
            # Score is Normalized Q + U
            score = normalized_q + u
            
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
                next_game = game_snapshot.clone()
                # action is relative (0, 1, 2)
                abs_action = (next_game.direction + (action - 1)) % 4
                _, reward, done = next_game.step(abs_action) # Capture specific reward for this transition
                
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

    def search(self, game):
        """
        Runs MCTS simulations from the current game state.
        Returns the refined policy (probabilities) and the entropy of the visit counts.
        """
        if self.root is None:
             self.root = Node(game.get_state(), prior=0, reward=0)
             policy, _ = self.predict(game)
             valid_moves = game.get_valid_relative_moves()
             self.root.expand(policy, valid_moves, game)
             self._add_dirichlet_noise(self.root, valid_moves)
        else:
            # Check if root state matches current game state (it might not if we updated root partially)
            # Actually, update_root should handle it, but if we reset or something...
            if not np.array_equal(self.root.state, game.get_state()):
                 self.root = Node(game.get_state(), prior=0, reward=0)
                 policy, _ = self.predict(game)
                 valid_moves = game.get_valid_relative_moves()
                 self.root.expand(policy, valid_moves, game)
                 self._add_dirichlet_noise(self.root, valid_moves)

        for _ in range(self.n_simulations):
            node = self.root
            simulation_game = game.clone()
            
            # 1. SELECT
            while node.is_expanded():
                action, node = node.select(self.c_puct)
                # action is relative (0, 1, 2)
                # Convert to absolute for simulation step
                abs_action = (simulation_game.direction + (action - 1)) % 4
                simulation_game.step(abs_action)

            # 2. EVALUATE & BACKUP
            if simulation_game.done:
                # Terminal state. Future value is 0.
                # The node.reward contains the death penalty or food reward.
                # Update starts recursively from here.
                node.update(0) 
                continue

            # Inference
            policy, value = self.predict(simulation_game)
            valid_moves = simulation_game.get_valid_relative_moves()
            
            # 3. EXPAND
            node.expand(policy, valid_moves, simulation_game)
            
            # 4. BACKUP
            # Value from network is the estimated return from that state onwards
            node.update(value)

        # Calculate final policy
        counts = np.zeros(3)
        for action, child in self.root.children.items():
            counts[action] = child.N
            
        if np.sum(counts) > 0:
            counts = counts / np.sum(counts)
        else:
            counts = np.array([0.33, 0.33, 0.33])
            
        entropy = -np.sum(counts * np.log(counts + 1e-8))
        return counts, entropy

    def predict(self, game):
        # Prepare input state
        state = game.get_state()
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
        
        return self.predict_fn(input_tensor)

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
