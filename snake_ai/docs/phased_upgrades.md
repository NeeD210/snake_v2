# Phased Upgrades Plan (Snake v2)

Goal: apply upgrades in **small, testable steps** so you can measure (speed + learning) after each change and quickly isolate regressions.

## Baseline: how to measure impact (use this for every step)

### Metrics to record (copy/paste into a note per run)
- **Self-play throughput**
  - Gen time (seconds): printed as `Time: ...s`
  - Inference batch rate: printed as `Batch Rate: .../s`
  - Games per gen: `--games` (or schedule output)
- **Learning signal**
  - `AvgScore`, `MaxScore`, `Deaths` breakdown (wall/body/timeout/starvation/won)
  - `Loss`, `PredAcc`, `Entropy`
- **System**
  - Workers, board size, sims, CPU utilization

### Recommended “repeatable test configs”
Use fixed configs (disable schedules) when you want apples-to-apples runtime comparisons:

- **Quick throughput smoke test (fast, noisy)**
  - `--board-size 6 --workers 4 --games 8 --sims 64 --epochs 0`
  - Purpose: measure self-play speed only (no training time).
- **Throughput + minimal training (still quick)**
  - `--board-size 6 --workers 4 --games 12 --sims 64 --epochs 1 --memory 5000 --batch 128`
  - Purpose: ensure training code still runs and produces stable metrics.

On PowerShell:

```bash
python snake_ai/main.py --board-size 6 --workers 4 --games 8 --sims 64 --epochs 0
python snake_ai/main.py --board-size 6 --workers 4 --games 12 --sims 64 --epochs 1 --memory 5000 --batch 128
```

### Rollback rule
After each upgrade:
- If performance regresses or behavior breaks, **git revert** (or restore files) and move to the next upgrade later.
- Don’t stack multiple upgrades without measuring in between.

---

## Phase 0 — Observability & correctness (low risk, enables everything)

### Upgrade 0.1 — Fix CSV logging for games/gen (schedule-safe)
**Why**: current CSV history hardcodes `'Games': GAMES_PER_GEN` even when schedules/CLI override are used. That makes performance/learning analysis misleading.

**Files**
- `snake_ai/main.py`

**Steps**
1. In `Trainer.train_generation()`, keep `games_this_gen` accessible where the history dict is written.
2. Change history field from `GAMES_PER_GEN` to `games_this_gen`.
3. Optional: also log `SIMULATIONS` or “avg sims used” (see 0.2).
4. Run 1 generation and confirm the CSV has the right Games value.

**Validation**
- CSV `Games` equals the printed “Games=…” value for that generation.

**Expected impact**
- Speed: none
- Model: none
- Debuggability: high

---

### Upgrade 0.2 — Add per-generation profiling counters (cheap telemetry)
**Why**: to compute real impact of later upgrades, you need to know if you’re **IPC-bound**, **MCTS-bound**, or **training-bound**.

**Files**
- `snake_ai/main.py`

**Steps**
1. Track counters during the generation:
   - `inference_requests` (sum of requests pulled from `request_queue`)
   - `avg_batch_size` (total_requests / inference_batches)
   - `selfplay_time` vs `train_time` (split timers)
2. Print these at end of gen and optionally write to CSV.

**Validation**
- Counters are non-zero and look plausible.

**Expected impact**
- Speed: ~0
- Debuggability: very high

---

## Phase 1 — Throughput upgrades (high ROI, mostly performance refactors)

### Upgrade 1.1 — O(1) collision via occupancy set/grid in `SnakeGame`
**Why**: MCTS calls `step()` constantly; `new_head in self.snake[:-1]` is O(length). Replace with O(1) occupancy checks.

**Files**
- `snake_ai/game.py`

**Steps**
1. Add an occupancy structure:
   - easiest: `self.occ = set(self.snake)` on reset
2. Update it on each move:
   - On normal move: add new head; remove tail (since it moves)
   - On food: add new head; do **not** remove tail
3. Replace collision check `new_head in self.snake[:-1]` with occupancy logic that still allows moving into the tail:
   - safe rule: if `new_head` is occupied **and** `new_head != tail_position` then collision
4. Update `clone()` to copy occupancy too (or reconstruct from snake list).
5. Update `get_valid_moves()` and `get_valid_relative_moves()` to use occupancy logic consistently.

**Validation**
1. Quick manual sanity:
   - Run `manual_play.py` (if used) and ensure collisions behave correctly.
2. Run a short training generation and confirm no weird spike in “body deaths” due to a bug.

**Expected impact**
- Throughput: **+15% to +60%** (depends on average snake length + sims)
- Model: unchanged

**Common pitfalls**
- Forgetting to allow “move into tail” when the tail will move away.
- Not keeping occupancy consistent on food vs non-food steps.

---

### Upgrade 1.2 — Reduce MCTS simulation overhead (fast internal state + undo)
**Why**: biggest hotspot is Python sim + clone overhead in `mcts.py`.

**Files**
- New: `snake_ai/fast_state.py` (or similar)
- `snake_ai/mcts.py`

**Approach**
Implement a minimal state for simulations:
- snake body as `collections.deque`
- occupancy as `set` or `np.ndarray` grid
- head position, direction, food, hunger counters
- `step_relative(rel_action)` returns `(reward, done)` and records an undo record
- `undo()` restores last step

**Steps**
1. Create `FastSnakeState.from_game(game: SnakeGame)` and `to_game()` (optional).
2. Implement `step_relative()` and `get_valid_relative_moves()` in this fast class.
3. In `MCTS.search()`:
   - replace `simulation_game = game.clone()` with `state = FastSnakeState.from_game(game)`
   - selection loop applies `step_relative()`
   - expansion uses fast stepping for child creation (or generates next states cheaply)
4. Keep the **exact same reward logic** as `SnakeGame.step()` (including hunger + shaping), otherwise you change the learning task.
5. Add a debug mode that cross-checks rewards/done between `SnakeGame` and `FastSnakeState` for random short rollouts.

**Validation**
- Correctness:
  - Run 1k random steps comparing `SnakeGame` vs `FastSnakeState` transitions (reward + done + positions).
- Performance:
  - Compare “self-play only” gen time before/after at same `--games --sims --workers`.

**Expected impact**
- Throughput: **2×–6×** (often the biggest win)
- Model: unchanged (if rewards match)

**Common pitfalls**
- Tiny reward mismatches silently change what the agent learns.
- Undo bugs (off-by-one tail update) cause rare invalid states.

---

### Upgrade 1.3 — Reduce IPC overhead (shared memory indices or per-worker inference)
**Why**: right now every inference request sends a full `(4,H,W)` float tensor via pickled Queue.

**Two options (pick one)**

**Option A: shared memory ring buffer (best speed)**
1. Allocate a shared `torch.Tensor` or `multiprocessing.shared_memory` buffer sized `[SLOTS, 4, H, W]`.
2. Workers write their state into a free slot and send only `(worker_id, slot_id)` through the queue.
3. Main reads the batch by indexing the shared tensor.
4. Main writes results into per-worker shared result buffers (or queues with tiny messages).

**Option B: inference inside workers (simpler)**
1. Load the model in each worker process (CPU only).
2. Set `torch.set_num_threads(1)` in workers to avoid CPU oversubscription.
3. Remove request/response queues; `predict_client()` calls model directly.

**Validation**
- Compare self-play throughput on the “epochs 0” config.
- Confirm results are consistent (scores/death reasons in the same ballpark).

**Expected impact**
- If you’re IPC-bound: **+30% to +150%**
- If you’re MCTS-bound: smaller improvement

---

## Phase 2 — Training stability (moderate risk, usually improves learning)

### Upgrade 2.1 — Replace BatchNorm with GroupNorm
**Why**: BatchNorm can be unstable with non-i.i.d self-play + varying batch composition; GroupNorm is often more stable.

**Files**
- `snake_ai/model.py`

**Steps**
1. Replace `nn.BatchNorm2d(C)` with `nn.GroupNorm(num_groups=8, num_channels=C)` (tune groups if needed).
2. Keep everything else identical.
3. Start a new experiment directory (don’t overwrite old weights).

**Validation**
- Watch for fewer loss spikes and smoother entropy decay.

**Expected impact**
- Model/sample-efficiency: **+5–20%** (typical)
- Throughput: similar (sometimes slightly slower)

---

### Upgrade 2.2 — Value target normalization (optional)
**Why**: unbounded returns can vary in scale; normalizing can reduce gradient spikes.

**Files**
- `snake_ai/main.py`

**Steps**
1. Maintain running mean/std of target values in trainer.
2. Normalize `target_vs` before loss, and de-normalize only for reporting if needed.
3. Keep reward structure unchanged.

**Validation**
- Loss becomes more stable; fewer sudden collapses.

**Expected impact**
- Stability/sample-efficiency: **+5–25%**

---

## Phase 3 — Model strength (higher cost, do after throughput wins)

### Upgrade 3.1 — ResNet trunk (AlphaZero-style)
**Why**: improves representational power; usually helps value and policy accuracy.

**Files**
- `snake_ai/model.py`

**Steps**
1. Implement `ResidualBlock(channels=64)` with Conv->Norm->ReLU->Conv->Norm + skip.
2. Replace current 3-layer conv trunk with:
   - initial conv to 64
   - N residual blocks (start with 4)
3. Keep heads (policy/value) similar, adjusting input channels.
4. Start a fresh run (new experiment).

**Validation**
- Compare `PredAcc`, `AvgScore` trend vs baseline at same sims/games.

**Expected impact**
- Strength/sample-efficiency: **+10–40%**
- Throughput: inference slower (**1.5×–3×**), so do after Phase 1.

---

## Suggested execution order (recommended)
1. **0.1** (CSV fix) → **0.2** (telemetry)
2. **1.1** (occupancy) → measure
3. **1.2** (fast MCTS state) → measure
4. **1.3** (IPC reduction) → measure
5. **2.1** (GroupNorm) → measure learning stability
6. **3.1** (ResNet) → measure strength (only after speed is acceptable)


