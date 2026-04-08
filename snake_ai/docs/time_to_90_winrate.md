# Time-to-90% Winrate (6x6): fastest path in this codebase

Goal: reach **≥ 90% benchmark winrate on 6x6** in the least wall-clock time (not just highest eventual strength).

This note is grounded in the current implementation in:
- `snake_ai/main.py` (training loop + multiprocessing + schedules)
- `snake_ai/mcts.py` (MCTS + FastSnakeState rollouts)
- `snake_ai/encoder.py` (POV encoding + flood-fill feature)
- `snake_ai/game.py` / `snake_ai/fast_state.py` (rewards + termination semantics)
- `snake_ai/benchmark.py` (deterministic benchmark, WinPct)

---

## 0) Definitions: what “winrate” means here

In `snake_ai/game.py`, `death_reason="won"` happens only when the snake **fills the entire board**:
- Eat: `+1.0`
- Win (fill board): `+2.0` and terminate
- Death (wall/body/timeout/starvation): `-2.0` and terminate
- Step penalty: `-0.01` every step
- Distance shaping: `0.05 * (old_manhattan_dist - new_manhattan_dist)`
- Returns used for value targets are discounted in `process_game_memory()` with `gamma=0.95`.

This makes “90% winrate” a **late-game precision** target: small errors near full board dominate whether an episode ends in “won” vs “body/wall”.

---

## 1) Measure the right metric: optimize for benchmark WinPct, not AvgScore

Training currently saves `best_snake_net.pth` when `AvgScore` improves. That is not guaranteed to match WinPct (especially late-game).

### 1.1 Add train WinPct (cheap) + optional early-stop target
Each generation already records deaths (including `DeathWon`) and games played. Define:

- **TrainWinPct** = `100 * DeathWon / Games`

This is not as strict as the deterministic benchmark (because self-play has temperature + Dirichlet noise), but it is a very cheap “progress meter”.

**Plan improvement**
- Add `TrainWinPct` to console output and `training_report.csv`.
- Add an early stop condition for “time-to-X” measurements:
  - `--stop-winpct 90` (stop when TrainWinPct ≥ 90 for a generation)
  - `--stop-wins 1` (stop when you first win a game; useful for debugging / time-to-first-win)

Note: using `--stop-winpct 1` is ambiguous (1% vs “win one game”). A dedicated `--stop-wins` is clearer.

**Upgrade (high ROI):**
- Run `snake_ai/benchmark.py` regularly and treat **WinPct** as the primary success metric.
- Save “best model” by benchmark WinPct (optionally tie-break by AvgSteps).

### 1.2 Benchmark cadence: dev checks vs final pass (manual)
**Dev benchmark (automated, fast signal)**
- Run 50 episodes every generation (or every 2 gens) to guide decisions quickly.
- Optional refinement (matches your preference): once WinPct first reaches 90%, switch to benchmarking every 5 generations after that (to reduce overhead while still tracking regressions).

**Final pass benchmark (manual, strict gate)**
- Run a deterministic benchmark of **200 episodes** (fixed seed + fixed sims) on the candidate checkpoint.
- If **WinPct ≥ 90%**, call it solved.

Instead of writing a one-off markdown report, append the result to the experiment’s `benchmark.csv` and record the training wall-clock time separately (e.g., copy from console output or log it in the run folder).

CLI examples:
- Training + benchmark every gen:
  - `python snake_ai/main.py --board-size 6 --benchmark --bench-every 1 --bench-episodes 50 --bench-sims 128`
- Benchmark-only (use latest saved weights):
  - `python snake_ai/main.py --bench-only --board-size 6 --bench-episodes 200 --bench-sims 128`

---

## 2) Where time goes (this codebase’s hotspots)

### 2.1 Encoding cost: `encode_pov()` flood fill is expensive
`snake_ai/encoder.py` builds channel 4 (“action-space ratio”) via `flood_fill_area_3dir(...)`:
- BFS over `(x, y, dir)` and visited arrays `(H, W, 4)`
- computed up to 3 times (for left/straight/right targets) per encoded state
- plus `np.rot90(...).copy()`

This is called from:
- `MCTS.predict()` (for every inference during MCTS)
- `process_state()` during data collection

For 6x6, this can become a dominant cost in self-play throughput.

### 2.2 MCTS rollout overhead: avoid per-simulation cloning
`snake_ai/mcts.py` uses `FastSnakeState` but still does:
- `root_sim = FastSnakeState.from_game(game)`
- inside each simulation: `simulation_state = root_sim.clone()`

Even though `FastSnakeState` supports `undo()`, cloning each simulation is expensive and scales with sims.

### 2.3 IPC overhead (central inference mode)
In central mode, each worker sends an entire encoded float tensor over a `multiprocessing.Queue`.
If your monitor shows low GPU/CPU utilization but low inference batch rate, you’re likely IPC-bound.

### 2.4 Training loop overhead
Training currently:
- materializes `list(self.memory)` every epoch
- shuffles in Python
- builds tensors from NumPy each batch

This is usually smaller than self-play cost until you make self-play much faster; then it matters.

---

## 3) Tier 1 upgrades (biggest impact on time-to-90%)

### 3.1 Make model selection match the goal (WinPct-based best model)
**Why**: time-to-90% is measured by WinPct, so best checkpoints must be WinPct-best.

**Implementation idea**
- When `--benchmark` is enabled, compute `bench_row["WinPct"]`.
- Keep `best_bench_winpct` and save `best_snake_net.pth` on improvement.

**Expected effect**
- Faster “usable model” selection.
- Avoid chasing AvgScore improvements that don’t translate to wins.

---

### 3.2 Stop cloning `FastSnakeState` per simulation (use undo)
**Why**: reduces MCTS cost per simulation and increases self-play throughput.

**Implementation sketch**
- Keep one `FastSnakeState` instance.
- During selection, record actions taken (or push undo records).
- After evaluation/expansion/backup, undo back to root.

**Expected effect**
- Typically a large speedup (often 1.5–3× for Python MCTS-heavy loops).

**Risk**
- Undo correctness bugs can silently corrupt rollouts; add a debug cross-check mode (optional) comparing rollouts against `SnakeGame.clone()` for a few random sequences.

---

### 3.3 Gate/replace the flood-fill channel (encoder speed)
**Why**: channel 4 is one of the most expensive parts of encoding and is used extremely frequently.

**Options (choose based on risk tolerance)**
- **Option A (lowest risk, high speed): gate it**
  - Compute channel 4 only in late-game (e.g., when `len(snake)` exceeds a threshold) and fill zeros otherwise.
  - Or compute it only every K steps and reuse last values (cheap, approximate).
- **Option B (medium risk): replace with cheaper heuristic**
  - Replace BFS over `(pos,dir)` with a simpler connected free-space estimate ignoring direction.
  - Or encode only immediate-valid-move mask + “free neighbors count”.
- **Option C (high ROI on 6x6): memoize by occupancy bitmask**
  - On 6x6, represent occupancy as a 36-bit mask and cache flood-fill results.
  - MCTS revisits similar local states; caching can be very effective.

**Expected effect**
- Often one of the biggest wall-clock improvements, especially on CPU.

---

### 3.4 Reduce exploration *after* the policy is competent (convert “can win” → “wins consistently”)

In `play_games_worker()`:
- early steps: `temp=1.0`
- after threshold: `temp=0.1` (still stochastic)

In `MCTS`:
- Dirichlet noise is applied at root by default (`dirichlet_epsilon=0.25`).

**Upgrades**
- Schedule temperature so late-game becomes deterministic earlier.
- Schedule Dirichlet noise down over time.

**What to drive these schedules with**
- Prefer **WinPct/TrainWinPct** or **policy entropy** over raw loss.
  - Loss is noisy (and scale-dependent if you change value normalization / gamma / batch sizes).
  - Entropy and winrate are closer to the behavior we actually care about.

**Example rule (conceptual)**
- If benchmark WinPct (or TrainWinPct) > 70%: reduce late-game temperature (more greedy).
- If benchmark WinPct (or TrainWinPct) > 85%: set late-game temperature to 0 and reduce Dirichlet epsilon close to 0.
- Schedule Dirichlet noise down:
  - high early generations; near-zero late generations.

**Why it helps time-to-90%**
- Once the agent is near-solved, remaining failures are usually rare late-game blunders.
- Reducing stochasticity generates cleaner targets and reduces “self-inflicted” random losses in training data.

---

## 4) Tier 2 upgrades (learning stability / fewer samples to 90%)

### 4.1 Improve LR scheduling (tie to optimizer steps, not generations)
Current scheduler: `ExponentialLR(gamma=0.99)` stepped once per generation.

**Why it can be suboptimal**
- Training steps per generation change with buffer size, games/gen, epochs.
- LR decay per generation can become “too fast” or “too slow” depending on throughput and settings.

**Better schedules**
- Warmup + cosine decay on *optimizer steps*.
- Plateau-based decay keyed off benchmark WinPct stagnation.
- Switch optimizer to AdamW with small weight decay (often improves stability/generalization).

---

### 4.2 Replay sampling: prioritize recency and/or “hard” states
Current: full replay shuffle each epoch.

**Why**
- As the policy improves, most stored samples become easy/stale.
- Late-game mistakes are rare; you want to upweight them.

**Upgrades**
- Recency bias: sample a large fraction from the most recent N games.
- Prioritized replay: use value loss / policy entropy / TD error as a proxy for “hardness”.

---

### 4.3 Tune discount factor for 6x6 endgame
Current: `gamma=0.95`.

For small boards where victory requires long sequences of precise moves, consider `gamma` in `0.97–0.99`:
- propagates endgame/win signal further back
- can improve late-game consistency

If increasing gamma causes value scale spikes, enable `--value-norm` (running mean/std normalization).

---

## 5) Tier 3 upgrades (MCTS/targets quality per unit compute)

### 5.1 Spend sims where it matters: make endgame boosting earlier for 6x6
You already have:
- `SIMS_ENDGAME_MULT`
- endgame threshold at `0.75 * board_cells`

For 6x6, consider boosting sims earlier (e.g., 0.60–0.70 full) because the “danger zone” starts earlier.

**Snake length vs score**
- In this environment, initial snake length is 3, so `snake_len = score + 3`.
- That means thresholds based on **snake length** vs **score** are effectively equivalent (just shifted by 3).
- Prefer `snake_len` for “% of board filled” logic (it directly matches occupancy), but either is fine.

### 5.2 Separate “training sims” and “benchmark sims”
Keep benchmark sims fixed for comparability (`--bench-sims`), but allow training sims to be scheduled aggressively:
- low sims early generation / early game
- high sims late generation / late game

---

## 6) Suggested execution order (fastest path)

1) **Benchmark-first training loop**
- Save best by benchmark WinPct (not AvgScore)
- Benchmark every generation (50 eps) until ~85%

2) **Throughput upgrades (largest wall-clock wins)**
- Remove per-simulation cloning in MCTS using undo
- Gate/optimize flood-fill channel in encoder
- If still IPC-bound: shared memory ring buffer or worker inference mode

3) **Convert near-solved → solved**
- Temperature schedule to deterministic late-game
- Dirichlet epsilon schedule down late

4) **If you still stall at 85–89%**
- Improve LR schedule
- Recency/priority replay
- Gamma sweep + value norm

---

## 7) What to log per run (minimal, high signal)

From training output (already mostly printed):
- `Time` per generation
- `Telemetry: SelfPlay vs Train time`, `InferBatches`, `AvgInferBatch`
- Death breakdown (watch late failures: mostly “body” vs “wall”)

From benchmark:
- `WinPct`, `AvgScore`, `AvgSteps`, death breakdown

Stop condition:
- WinPct ≥ 90% on **confirmation** benchmark (200–500 episodes).

