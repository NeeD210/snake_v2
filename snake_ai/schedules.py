from __future__ import annotations


def get_mcts_simulations(
    base_sims: float,
    snake_len: int,
    board_size: int,
    *,
    dev_mode: bool = False,
    sims_endgame_mult: int = 2,
) -> int:
    """
    Progressive MCTS simulation schedule derived from Adaptive Curriculum.
    - Uses the base_sims dynamically calculated by the parent process.
    - Applies the Endgame boost when >75% of the board is filled.
    """
    s_len = int(snake_len)
    n = int(board_size)
    base = int(round(base_sims))

    if dev_mode:
        # Dev mode: minimal sims for fast iteration
        return max(3, base)

    # Endgame boost: only when >75% of the board is filled
    endgame_threshold = int(0.75 * (n * n))
    if s_len >= endgame_threshold:
        base *= int(sims_endgame_mult)

    return base
