from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from game import SnakeGame


Coord = Tuple[int, int]


@dataclass
class _Undo:
    prev_direction: int
    prev_food: Coord
    prev_steps_since_eaten: int
    prev_score: int
    prev_done: bool
    prev_death_reason: Optional[str]
    ate: bool
    tail_removed: Optional[Coord]
    new_head: Coord


class FastSnakeState:
    """
    Lightweight, simulation-oriented Snake state for MCTS.

    Goals:
    - Match SnakeGame reward/done semantics closely
    - Keep stepping fast (no object cloning, O(1) collision checks)
    - Support undo() so MCTS can apply/rollback actions cheaply when needed

    NOTE: This is not used for real gameplay outside MCTS; SnakeGame remains the source of truth.
    """

    def __init__(
        self,
        board_size: int,
        snake: List[Coord],
        food: Coord,
        direction: int,
        steps_since_eaten: int,
        hunger_limit: int,
        score: int,
        steps: int,
        max_steps: int,
        done: bool = False,
        death_reason: Optional[str] = None,
        game_id: int = 0,
    ):
        self.board_size = int(board_size)
        self.snake: List[Coord] = list(snake)  # [head, ..., tail]
        self.occ = set(self.snake)
        self.food = food
        self.direction = int(direction)
        self.steps_since_eaten = int(steps_since_eaten)
        self.hunger_limit = int(hunger_limit)
        self.score = int(score)
        self.steps = int(steps)
        self.max_steps = int(max_steps)
        self.done = bool(done)
        self.death_reason = death_reason
        self.game_id = int(game_id)
        self._undo_stack: List[_Undo] = []

    @classmethod
    def from_game(cls, game: SnakeGame) -> "FastSnakeState":
        return cls(
            board_size=game.board_size,
            snake=list(game.snake),
            food=game.food,
            direction=game.direction,
            steps_since_eaten=game.steps_since_eaten,
            hunger_limit=game.hunger_limit,
            score=game.score,
            steps=game.steps,
            max_steps=game.max_steps,
            done=game.done,
            death_reason=getattr(game, "death_reason", None),
            game_id=getattr(game, "game_id", 0),
        )

    def clone(self) -> "FastSnakeState":
        # For safety/debug paths. MCTS should prefer in-place stepping.
        c = FastSnakeState(
            board_size=self.board_size,
            snake=list(self.snake),
            food=self.food,
            direction=self.direction,
            steps_since_eaten=self.steps_since_eaten,
            hunger_limit=self.hunger_limit,
            score=self.score,
            steps=self.steps,
            max_steps=self.max_steps,
            done=self.done,
            death_reason=self.death_reason,
            game_id=self.game_id,
        )
        return c

    def get_state(self) -> np.ndarray:
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        for x, y in self.snake:
            board[y, x] = 1
        hx, hy = self.snake[0]
        board[hy, hx] = 2
        fx, fy = self.food
        board[fy, fx] = 3
        return board

    def get_valid_relative_moves(self) -> List[int]:
        valid: List[int] = []
        head_x, head_y = self.snake[0]
        tail = self.snake[-1]

        # Relative changes: -1 (Left), 0 (Straight), +1 (Right)
        relative_changes = (-1, 0, 1)
        abs_deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))

        for rel_action, change in enumerate(relative_changes):
            abs_dir = (self.direction + change) % 4
            dx, dy = abs_deltas[abs_dir]
            nx, ny = head_x + dx, head_y + dy
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                continue
            nxt = (nx, ny)
            # Tail exception only if not eating.
            if nxt not in self.occ or (nxt == tail and nxt != self.food):
                valid.append(rel_action)
        return valid

    def step_abs(self, action: int) -> Tuple[float, bool]:
        """
        Apply an absolute action (0:Up,1:Right,2:Down,3:Left).
        Returns (reward, done).
        """
        if self.done:
            # When the tree desyncs due to stochastic food, MCTS might step a dead snake.
            # We MUST push an undo so that the depth count explicitly matches the push count.
            self._push_undo(ate=False, tail_removed=None, new_head=(-1, -1))
            return 0.0, True

        # Mirror SnakeGame.step semantics
        self.steps += 1
        step_penalty = -0.005

        # Starvation / timeout checks
        if self.steps_since_eaten >= self.hunger_limit:
            self._push_undo(ate=False, tail_removed=None, new_head=(-1, -1))
            self.done = True
            self.death_reason = "starvation"
            return -1.0, True

        if self.steps >= self.max_steps:
            self._push_undo(ate=False, tail_removed=None, new_head=(-1, -1))
            self.done = True
            self.death_reason = "timeout"
            return -1.0, True

        head_x, head_y = self.snake[0]
        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])

        # Prevent 180 degree turn (safety)
        if abs(action - self.direction) == 2:
            action = self.direction
        prev_direction = self.direction
        self.direction = action

        dx, dy = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dx = 1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1

        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= self.board_size
            or new_head[1] < 0
            or new_head[1] >= self.board_size
        ):
            self._push_undo(ate=False, tail_removed=None, new_head=new_head, prev_direction=prev_direction)
            self.done = True
            self.death_reason = "wall"
            return -1.0, True

        tail = self.snake[-1]
        will_eat = new_head == self.food
        if new_head in self.occ and (will_eat or new_head != tail):
            self._push_undo(ate=False, tail_removed=None, new_head=new_head, prev_direction=prev_direction)
            self.done = True
            self.death_reason = "body"
            return -1.0, True

        # Apply move
        self.snake.insert(0, new_head)
        self.occ.add(new_head)

        reward = 0.0
        tail_removed: Optional[Coord] = None
        prev_food = self.food
        prev_steps_since_eaten = self.steps_since_eaten
        prev_score = self.score
        prev_done = self.done
        prev_death_reason = self.death_reason

        if will_eat:
            self.score += 1
            reward = 0.2
            if len(self.snake) == self.board_size * self.board_size:
                self.done = True
                self.death_reason = "won"
                self._undo_stack.append(
                    _Undo(
                        prev_direction=prev_direction,
                        prev_food=prev_food,
                        prev_steps_since_eaten=prev_steps_since_eaten,
                        prev_score=prev_score,
                        prev_done=prev_done,
                        prev_death_reason=prev_death_reason,
                        ate=True,
                        tail_removed=None,
                        new_head=new_head,
                    )
                )
                return 1.0, True

            # Place new food (slow path; rare in simulations relative to steps)
            self.food = self._place_food()
            self.steps_since_eaten = 0
        else:
            tail_removed = self.snake.pop()
            if tail_removed != new_head:
                self.occ.remove(tail_removed)
            self.steps_since_eaten += 1

        # Distance shaping + step penalty
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        reward += (old_dist - new_dist) * 0.01
        reward += step_penalty

        self._undo_stack.append(
            _Undo(
                prev_direction=prev_direction,
                prev_food=prev_food,
                prev_steps_since_eaten=prev_steps_since_eaten,
                prev_score=prev_score,
                prev_done=prev_done,
                prev_death_reason=prev_death_reason,
                ate=will_eat,
                tail_removed=tail_removed,
                new_head=new_head,
            )
        )
        return reward, self.done

    def step_relative(self, rel_action: int) -> Tuple[float, bool]:
        # rel_action: 0=Left,1=Straight,2=Right
        abs_action = (self.direction + (rel_action - 1)) % 4
        return self.step_abs(abs_action)

    def undo(self) -> None:
        u = self._undo_stack.pop()

        # Restore simple fields
        self.direction = u.prev_direction
        self.food = u.prev_food
        self.steps_since_eaten = u.prev_steps_since_eaten
        self.score = u.prev_score
        self.done = u.prev_done
        self.death_reason = u.prev_death_reason

        # Undo move if we actually inserted head (we always insert before pushing undo for non-terminal)
        # For terminal wall/body cases, we didn't mutate snake/occ except possibly direction; nothing to undo.
        if u.new_head == self.snake[0]:
            # We inserted head; remove it
            head = self.snake.pop(0)
            # Rebuild occ incrementally
            # Remove head from occ unless it is also present in body (shouldn't happen)
            if head in self.occ:
                self.occ.remove(head)

            # If we didn't eat, we removed a tail; restore it
            if not u.ate and u.tail_removed is not None:
                self.snake.append(u.tail_removed)
                self.occ.add(u.tail_removed)

        # steps counter is only used for timeout (large), safe to decrement
        self.steps = max(0, self.steps - 1)

    def _push_undo(
        self,
        ate: bool,
        tail_removed: Optional[Coord],
        new_head: Coord,
        prev_direction: Optional[int] = None,
    ) -> None:
        if prev_direction is None:
            prev_direction = self.direction
        self._undo_stack.append(
            _Undo(
                prev_direction=prev_direction,
                prev_food=self.food,
                prev_steps_since_eaten=self.steps_since_eaten,
                prev_score=self.score,
                prev_done=self.done,
                prev_death_reason=self.death_reason,
                ate=ate,
                tail_removed=tail_removed,
                new_head=new_head,
            )
        )

    def _place_food(self) -> Coord:
        """
        Deterministic food placement for MCTS consistency.

        The seed is derived from the current board state so that any
        simulation reaching the same game state always places food
        identically.  This eliminates tree desynchronization that
        previously corrupted undo() semantics.
        """
        head = self.snake[0]
        seed = hash((self.game_id, head, len(self.snake), self.score, self.steps)) & 0x7FFFFFFF
        rng = np.random.RandomState(seed)

        free = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if (x, y) not in self.occ:
                    free.append((x, y))
        if not free:
            return self.snake[-1]  # Board full (win check fires before this)

        idx = rng.randint(0, len(free))
        return free[idx]



