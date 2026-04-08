from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
from numba import njit

Coord = Tuple[int, int]

# Channels (POV, rotated so head faces "up")
# 0: Body occupancy (excluding head)        {0,1}
# 1: Head                                   {0,1}
# 2: Food                                   {0,1}
# 3: Hunger (scalar filled on full grid)    [0,1]
# 4: Action-space ratio at targets          [0,1] (head cell forced to 1)
NUM_CHANNELS = 5


@njit(cache=True)
def _njit_dir_to_delta(d: int) -> tuple[int, int]:
    # 0:Up, 1:Right, 2:Down, 3:Left
    if d == 0: return (0, -1)
    if d == 1: return (1, 0)
    if d == 2: return (0, 1)
    return (-1, 0)

@njit(cache=True)
def _njit_flood_fill(
    n: int,
    sx: int,
    sy: int,
    sd: int,
    is_blocked: np.ndarray,
) -> int:
    """
    Numba-accelerated BFS for flood fill.
    is_blocked: 2D boolean array (n, n)
    """
    if not (0 <= sx < n and 0 <= sy < n):
        return 0
    if is_blocked[sy, sx]:
        return 0

    # visited[y, x, d]
    visited = np.zeros((n, n, 4), dtype=np.uint8)
    # visited_cells[y, x]
    visited_cells = np.zeros((n, n), dtype=np.uint8)
    
    # Queue for BFS: max size is n*n*4
    q = np.empty((n * n * 4, 3), dtype=np.int32)
    head = 0
    tail = 0
    
    q[tail] = (sx, sy, sd)
    tail += 1
    visited[sy, sx, sd] = 1
    visited_cells[sy, sx] = 1
    area = 1

    while head < tail:
        x, y, d = q[head]
        head += 1
        
        for rel in range(3): # 0, 1, 2
            nd = (d + (rel - 1)) % 4
            delta = _njit_dir_to_delta(nd)
            nx, ny = x + delta[0], y + delta[1]
            
            if 0 <= nx < n and 0 <= ny < n:
                if not is_blocked[ny, nx] and not visited[ny, nx, nd]:
                    visited[ny, nx, nd] = 1
                    q[tail] = (nx, ny, nd)
                    tail += 1
                    if visited_cells[ny, nx] == 0:
                        visited_cells[ny, nx] = 1
                        area += 1
    return area

def flood_fill_area_3dir(
    board_size: int,
    start: Coord,
    start_dir: int,
    blocked: set[Coord] | np.ndarray,
) -> int:
    """
    Accelerated 3-action flood fill.
    """
    n = int(board_size)
    if isinstance(blocked, set):
        is_blocked = np.zeros((n, n), dtype=np.bool_)
        for bx, by in blocked:
            if 0 <= bx < n and 0 <= by < n:
                is_blocked[by, bx] = True
    else:
        is_blocked = blocked

    return _njit_flood_fill(n, int(start[0]), int(start[1]), int(start_dir) % 4, is_blocked)


def encode_pov(game, state: np.ndarray | None = None) -> np.ndarray:
    """
    Encode a SnakeGame-like object into a POV tensor: (C, H, W), float32.
    Works with both SnakeGame and FastSnakeState (used by MCTS).
    """
    if state is None:
        state = game.get_state()

    n = int(getattr(game, "board_size", state.shape[0]))
    x = np.zeros((NUM_CHANNELS, n, n), dtype=np.float32)

    snake = list(getattr(game, "snake", []))
    if snake:
        # Channel 0: body occupancy (all segments except head)
        for sx, sy in snake[1:]:
            x[0, sy, sx] = 1.0

    # Channel 1: head
    x[1] = (state == 2).astype(np.float32)

    # Channel 2: food
    x[2] = (state == 3).astype(np.float32)

    # Channel 3: hunger scalar
    hunger_limit = max(1, int(getattr(game, "hunger_limit", 100)))
    hunger = float(getattr(game, "steps_since_eaten", 0)) / float(hunger_limit)
    x[3].fill(np.float32(hunger))

    # Channel 4: immediate action-space ratio signal at the 3 neighbor target cells.
    # - head cell is always 1
    # - each candidate next-head cell gets flood_after_area / flood_current_area (0 if invalid)
    cur_dir = int(getattr(game, "direction", 0)) % 4
    if snake:
        head = snake[0]
        food = getattr(game, "food", None)

        is_blocked_cur = np.zeros((n, n), dtype=np.bool_)
        for sx, sy in snake[1:]:
            is_blocked_cur[sy, sx] = True
        
        area = flood_fill_area_3dir(n, head, cur_dir, is_blocked_cur)

        hx, hy = head
        x[4, hy, hx] = 1.0
        denom = float(area) if int(area) > 0 else 1.0

        for rel in (0, 1, 2):
            nd = (cur_dir + (rel - 1)) % 4
            delta = _njit_dir_to_delta(nd)
            nh = (head[0] + delta[0], head[1] + delta[1])
            # If move goes out of bounds, leave channel empty (all zeros).
            if not (0 <= nh[0] < n and 0 <= nh[1] < n):
                continue

            will_eat = (food is not None and nh == food)
            
            # Efficiently build next blocked mask
            is_blocked_next = is_blocked_cur.copy()
            if will_eat:
                # Head was already 0 in is_blocked_cur, now it's part of body[1:] for the next step
                is_blocked_next[hy, hx] = True
            else:
                # Tail (snake[-1]) is removed from blocked
                tx, ty = snake[-1]
                is_blocked_next[ty, tx] = False
                # Head becomes blocked for the next step
                is_blocked_next[hy, hx] = True
                
            # nh must not be in blocked
            if is_blocked_next[nh[1], nh[0]]:
                continue

            area_after = flood_fill_area_3dir(n, nh, nd, is_blocked_next)
            ratio = float(area_after) / denom
            x[4, nh[1], nh[0]] = np.float32(max(0.0, min(1.0, ratio)))
    else:
        x[4].fill(0.0)

    # Rotate based on direction to enforce POV (Head Up)
    # k=0 (Up) -> 0 rot
    # k=1 (Right) -> 1 rot (90 deg CCW) -> Right becomes Up
    k = int(getattr(game, "direction", 0))
    x = np.rot90(x, k, axes=(1, 2)).copy()

    return x

