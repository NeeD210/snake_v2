import numpy as np
import random

class SnakeGame:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.reset()

    def clone(self):
        """Creates a lightweight copy of the game state."""
        new_game = SnakeGame(self.board_size)
        new_game.snake = self.snake[:] # Shallow copy of list of tuples (tuples are immutable)
        new_game.occ = set(self.occ)
        new_game.food = self.food
        new_game.done = self.done
        new_game.steps = self.steps
        new_game.score = self.score
        new_game.max_steps = self.max_steps
        new_game.steps_since_eaten = self.steps_since_eaten
        new_game.hunger_limit = self.hunger_limit
        new_game.direction = self.direction
        new_game.death_reason = self.death_reason
        new_game.game_id = self.game_id
        return new_game

    def reset(self):
        """Resets the game to the initial state."""
        # Initialize with 3 segments
        # Head at center, body to the left implies facing Right (1)
        start_x, start_y = self.board_size // 2, self.board_size // 2
        
        # Segments: Head, Body1, Body2
        self.snake = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ]
        # Occupancy for O(1) collision/valid-move checks.
        self.occ = set(self.snake)
        
        self.done = False
        self.steps = 0
        self.max_steps = 100000 # Large limit, relying on starvation instead
        self.score = 0
        self.steps_since_eaten = 0
        # Scale with board size so late-game routing on larger boards isn't artificially killed.
        self.hunger_limit = max(100, self.board_size * self.board_size * 2)
        self.direction = 1 # 1: Right ( Matches the body placement )
        self.death_reason = None # Track why the game ended
        self.game_id = random.randint(0, 2**31 - 1)  # Unique per game for food variety
        
        self.food = self._place_food()
        return self.get_state()

    def _place_food(self):
        """Places food deterministically based on current board state."""
        head = self.snake[0]
        seed = hash((self.game_id, head, len(self.snake), self.score, self.steps)) & 0x7FFFFFFF
        rng = np.random.RandomState(seed)

        free = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x, y) not in self.occ:
                    free.append((x, y))
        if not free:
            return self.snake[-1]  # Board full

        idx = rng.randint(0, len(free))
        return free[idx]

    def step(self, action):
        """
        Executes a step in the game.
        Action: 0: Up, 1: Right, 2: Down, 3: Left
        Returns: state, reward, done
        """
        if self.done:
            return self.get_state(), 0, True

        self.steps += 1
        # Small time penalty to discourage loops (still dominated by food/death).
        step_penalty = -0.005
        # Starvation check
        if self.steps_since_eaten >= self.hunger_limit:
            self.done = True
            self.death_reason = "starvation"
            return self.get_state(), -1.0, True
            
        if self.steps >= self.max_steps:
            self.done = True
            self.death_reason = "timeout"
            return self.get_state(), -1.0, True # Should not maximize this, same as starvation

        head_x, head_y = self.snake[0]
        
        # Calculate old distance to food
        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])

        # Prevent 180 degree turn
        # Directions: 0: Up, 1: Right, 2: Down, 3: Left
        # Opposites: 0<->2, 1<->3. Difference is always 2.
        if abs(action - self.direction) == 2:
             action = self.direction # Ignore input, continue straight
        
        self.direction = action

        # Movement delta
        dx, dy = 0, 0
        if action == 0:   dy = -1 # Up
        elif action == 1: dx = 1  # Right
        elif action == 2: dy = 1  # Down
        elif action == 3: dx = -1 # Left

        new_head = (head_x + dx, head_y + dy)

        # check collision with wall
        if (new_head[0] < 0 or new_head[0] >= self.board_size or
            new_head[1] < 0 or new_head[1] >= self.board_size):
            self.done = True
            self.death_reason = "wall"
            return self.get_state(), -1.0, True

        # Check collision with self
        # Note: Moving into the tail is safe only if we are NOT eating
        # (because then the tail moves away).
        tail = self.snake[-1]
        will_eat = (new_head == self.food)
        if new_head in self.occ and (will_eat or new_head != tail):
             self.done = True
             self.death_reason = "body"
             return self.get_state(), -1.0, True

        # Move snake
        self.snake.insert(0, new_head)
        self.occ.add(new_head)

        reward = 0
        # Check food
        if will_eat:
            self.score += 1
            reward = 0.2 # Eat
            if len(self.snake) == self.board_size * self.board_size:
                self.done = True # Victory
                self.death_reason = "won"
                return self.get_state(), 1.0, True # Win
            self.food = self._place_food()
            self.steps_since_eaten = 0 # Reset hunger
        else:
            tail_removed = self.snake.pop()
            # If we moved into the tail (legal when not eating), the tail position
            # remains occupied by the new head, so we must NOT remove it from occ.
            if tail_removed != new_head:
                self.occ.remove(tail_removed)
            reward = 0 # Not Dying
            self.steps_since_eaten += 1

        # Distance shaping (Reduced influence)
        # New distance
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        
        # If closer: old > new -> positive reward
        # If further: old < new -> negative reward
        # Scale 0.01: Enough to guide gently
        reward += (old_dist - new_dist) * 0.01
        reward += step_penalty

        return self.get_state(), reward, False

    def get_state(self):
        """
        Returns the board state as a numpy array.
        0: Empty, 1: Body, 2: Head, 3: Food
        """
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        for part in self.snake:
            board[part[1], part[0]] = 1
            
        head = self.snake[0]
        board[head[1], head[0]] = 2
        
        food = self.food
        board[food[1], food[0]] = 3
        
        return board

    def get_valid_moves(self):
        """
        Returns a mask of valid moves (those that don't immediately kill the snake).
        Note: This is 'greedy' validity, doesn't account for trapping self.
        """
        valid = []
        head_x, head_y = self.snake[0]
        tail = self.snake[-1]
        
        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for i, (dx, dy) in enumerate(moves):
            nx, ny = head_x + dx, head_y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                nxt = (nx, ny)
                # Moving into tail is only safe if not eating.
                if nxt not in self.occ or (nxt == tail and nxt != self.food):
                    valid.append(i)
                    
        return valid

    def get_valid_relative_moves(self):
        """
        Returns a list of valid relative moves (0: Left, 1: Straight, 2: Right).
        """
        valid_relative = []
        
        # Relative mappings: 0->Left, 1->Straight, 2->Right
        # Map to absolute changes: -1, 0, +1
        relative_changes = [-1, 0, 1]
        
        head_x, head_y = self.snake[0]
        tail = self.snake[-1]
        
        # Directions: 0: Up, 1: Right, 2: Down, 3: Left
        # Deltas for absolute directions
        # 0(Up): (0, -1), 1(Right): (1, 0), 2(Down): (0, 1), 3(Left): (-1, 0)
        abs_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for rel_action, change in enumerate(relative_changes):
            abs_dir = (self.direction + change) % 4
            dx, dy = abs_deltas[abs_dir]
            
            nx, ny = head_x + dx, head_y + dy
            
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                nxt = (nx, ny)
                if nxt not in self.occ or (nxt == tail and nxt != self.food):
                    valid_relative.append(rel_action)
                    
        return valid_relative
