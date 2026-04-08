## Analysis n1

1. Analysis of Current Status (Wholistic View)
The project has a solid AlphaZero foundation (NN + MCTS + POV Invariance), but it suffers from a critical "State Representation Amnesia".
The "Ouroboros" Defect: The primary failure mode ("DeathBody") occurs because the snake treats its own body as a static wall. The binary input channel (0 = Empty, 1 = Body) destroys the temporal information of the snake. The agent cannot distinguish between a tail segment that will move away in 1 turn (safe) and a neck segment that will stay for 50 turns (deadly).
Scale Mismatch: Training on a 6x6 board with low simulation counts (30-50) created a "toy" agent that cannot generalize to real Snake strategy (trapping/filling).
Value Confusion: The previous tanh activation saturated at 1.0, making "Eating (+1)" and "Winning (+2)" indistinguishable from "Surviving".
2. Implemented Improvements ("Perfect Play" Upgrade)
I have applied the following upgrades directly to the codebase:
A. "Lifetime" State Representation (Crucial Fix)
I replaced the binary Body channel with a Gradient Lifetime Channel.
Old: Body = 1.0 (Static Wall)
New: Body = 1.0 (Head) $\to$ 0.1 (Tail).
Effect: The CNN can now "see" the flow of the snake. It can calculate if a gap will open up in time to squeeze through, enabling advanced trapping and cycle behaviors.
B. Architectural & Training Tuning
Board Size: Upgraded from 6x6 to 10x10 (Standard Snake).
Simulations: Increased from 100 to 200 (Deeper search for complex traps).
Stability: Reduced Learning Rate (5e-4), Increased Games Per Gen (50), and reduced Epochs (2) to prevent the "Catastrophic Forgetting" observed in previous runs.
Memory: Increased buffer to 50,000 to ensure a diverse training distribution.
3. Next Steps
The system is now configured for a serious training run.
Run Training: Start the training loop.
    python snake_ai/main.py
Monitor: Watch for DeathBody rates. With the new input channel, the agent should learn to "chase its tail" rather than crash into it.


## Analysis n2
# Holistic Analysis: Snake AI Project Status & Improvement Roadmap (v13)

**Date**: December 16, 2025  
**Analyst**: AI Assistant  
**Purpose**: Comprehensive analysis identifying all weaknesses and proposing integrated improvements for achieving perfect play

---

## Executive Summary

The Snake AI project implements an AlphaZero-style architecture (NN + MCTS) but suffers from **multiple interconnected weaknesses** preventing convergence to perfect play. While individual components are correctly implemented, **systemic issues** in reward propagation, MCTS terminal handling, training stability, and search depth create a cascading failure mode. The agent performs worse than random (~1.15 avg score vs ~2-3 expected) due to these fundamental misalignments.

**Critical Path**: Fix MCTS terminal state handling → Stabilize training → Increase search depth → Optimize reward structure → Scale to perfect play.

---

## 1. Current Architecture Overview

### 1.1 Components
- **Neural Network** (`SnakeNet`): CNN with Policy & Value heads (unbounded value, correctly fixed)
- **MCTS**: 100 simulations, PUCT selection, Dirichlet noise
- **Training**: Self-play with replay buffer, gradient clipping, Adam optimizer
- **Game**: 6x6 board, relative actions (Left/Straight/Right), POV invariance

### 1.2 Strengths
✅ POV invariance correctly implemented  
✅ Value head unbounded (Tanh removed)  
✅ Gradient clipping implemented  
✅ Batched inference for efficiency  
✅ Proper data augmentation (horizontal flip)

---

## 2. Critical Weaknesses (Root Cause Analysis)

### 2.1 **MCTS Terminal State Handling Bug** (CRITICAL)

**Location**: `mcts.py:161-166`

**Problem**: When MCTS reaches a terminal state, it calls `node.update(0)` instead of using the terminal reward. This breaks value propagation:

```python
if simulation_game.done:
    node.update(0)  # ❌ WRONG: Ignores terminal reward
    continue
```

**Impact**: 
- Terminal states (death/eat) propagate value=0 instead of their actual reward (-1.0, +1.0, +2.0)
- MCTS Q-values become systematically wrong
- Network learns incorrect value estimates
- Policy head receives wrong training signals

**Why This Matters**: MCTS is supposed to provide "ground truth" for training. If terminal states are misvalued, the entire learning loop breaks.

**Fix Required**: Extract terminal reward from the game state and propagate it correctly.

---

### 2.2 **Gamma Mismatch Between MCTS and Training**

**Location**: 
- MCTS: `mcts.py:87` uses `gamma=0.9`
- Training: `main.py:160` uses `gamma=0.95`

**Problem**: Different discount factors create inconsistent value estimates:
- MCTS backpropagates with γ=0.9
- Training computes returns with γ=0.95
- Network learns values that don't match MCTS's internal logic

**Impact**: Value head predictions diverge from MCTS Q-values, causing policy confusion.

**Fix Required**: Use consistent gamma (0.95 recommended for longer horizons).

---

### 2.3 **Insufficient MCTS Search Depth**

**Current**: 100 simulations on 6x6 board  
**Problem**: 
- Effective depth ≈ log₃(100) ≈ 4 moves
- Snake needs to plan 5-10 moves ahead to avoid traps
- Cannot foresee self-trapping scenarios

**Evidence**: High "DeathBody" rate (17-19/20 games) indicates inability to avoid traps.

**Impact**: MCTS provides poor training targets because it can't see far enough ahead.

**Fix Required**: Increase to 200-400 simulations, or implement progressive widening.

---

### 2.4 **Reward Structure Suboptimal**

**Current Rewards**:
- Eat: +1.0
- Death: -1.0  
- Win: +2.0
- Distance shaping: ±0.05 per step
- Starvation: -2.0

**Problems**:
1. **Death penalty too weak**: -1.0 is same magnitude as eating (+1.0). Agent doesn't strongly avoid death.
2. **Distance shaping too weak**: ±0.05 is negligible compared to ±1.0 rewards. Doesn't guide exploration effectively.
3. **No time penalty**: Agent can loop indefinitely without negative signal (starvation only triggers at 100 steps).

**Impact**: Agent lacks clear survival incentive, leading to random walks and self-collisions.

**Fix Required**: Increase death penalty, strengthen distance shaping, add step penalty.

---

### 2.5 **Training Instability**

**Current**:
- Learning rate: 0.001 (1e-3)
- Gradient clipping: 1.0
- Batch size: 128
- Epochs: 5

**Problems**:
1. **LR too high**: Analysis v12 shows catastrophic loss spike at Gen 11, suggesting instability.
2. **No learning rate schedule**: Fixed LR doesn't adapt as training progresses.
3. **Memory size**: 20,000 samples may be insufficient for stable learning.

**Impact**: Training collapses periodically, resetting progress.

**Fix Required**: Lower LR (1e-4), add LR scheduler, increase memory size.

---

### 2.6 **Board Size Too Small**

**Current**: 6x6 board  
**Problem**: 
- Very small board (36 cells) makes perfect play trivial but training harder
- Easy to trap self
- Limited strategic depth
- Random play achieves ~2-3 apples, but AI stagnates at ~1.15

**Impact**: Training environment doesn't scale well. Perfect play on 6x6 is achievable but not impressive.

**Fix Required**: Consider 8x8 or 10x10 for more strategic depth, or keep 6x6 but optimize for it.

---

### 2.7 **Missing Value Loss Weighting**

**Current**: `loss = loss_v + loss_p` (equal weights)  
**Problem**: 
- Policy loss dominates early training (high entropy)
- Value loss needs higher weight to prioritize survival
- No adaptive weighting

**Impact**: Network focuses on matching MCTS policy before learning value, leading to poor survival.

**Fix Required**: Weight value loss higher (e.g., 2x) or use adaptive weighting.

---

### 2.8 **MCTS Node Expansion Timing Issue**

**Location**: `mcts.py:173`

**Problem**: Node is expanded AFTER getting NN prediction, but the expansion happens on a game state that may have changed during selection. The `simulation_game` state might not match the node's state.

**Impact**: Potential state mismatch causing incorrect value propagation.

**Fix Required**: Ensure state consistency or expand before prediction.

---

## 3. Secondary Issues

### 3.1 **No Early Stopping**
Training runs indefinitely without convergence detection.

### 3.2 **Limited Diagnostics**
No visualization of MCTS tree structure, value distributions, or policy entropy over time.

### 3.3 **No Curriculum Learning**
Training starts with full difficulty. No progressive difficulty increase.

### 3.4 **Temperature Schedule Suboptimal**
Temperature decays linearly but may be too aggressive, reducing exploration too quickly.

---

## 4. Integrated Improvement Proposal

### Phase 1: Critical Fixes (Immediate)

#### Fix 1.1: MCTS Terminal State Handling
**File**: `mcts.py`  
**Change**: Extract terminal reward from game state and propagate correctly:

```python
if simulation_game.done:
    # Get terminal reward from game state
    terminal_reward = simulation_game.score * 1.0  # Or extract from last step
    if simulation_game.death_reason == "won":
        terminal_reward = 2.0
    elif simulation_game.death_reason in ["wall", "body"]:
        terminal_reward = -1.0
    elif simulation_game.death_reason in ["starvation", "timeout"]:
        terminal_reward = -2.0
    node.update(terminal_reward)  # Use actual reward, not 0
    continue
```

**Priority**: CRITICAL - Blocks all learning

---

#### Fix 1.2: Unify Gamma Values
**Files**: `mcts.py`, `main.py`  
**Change**: Use `gamma=0.95` consistently in both MCTS and training.

**Priority**: CRITICAL - Value consistency

---

#### Fix 1.3: Increase MCTS Simulations
**File**: `main.py`  
**Change**: `SIMULATIONS = 200` (or 400 for deeper search)

**Priority**: HIGH - Search depth

---

### Phase 2: Training Stability

#### Fix 2.1: Lower Learning Rate + Scheduler
**File**: `main.py`  
**Changes**:
- `LR = 0.0001` (1e-4)
- Add `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)` or `ReduceLROnPlateau`

**Priority**: HIGH - Prevents collapse

---

#### Fix 2.2: Increase Memory Size
**File**: `main.py`  
**Change**: `MEMORY_SIZE = 50000` (or 100000)

**Priority**: MEDIUM - Stability

---

#### Fix 2.3: Weight Value Loss Higher
**File**: `main.py`  
**Change**: `total_loss = 2.0 * loss_v + loss_p` (or adaptive weighting)

**Priority**: MEDIUM - Survival focus

---

### Phase 3: Reward Optimization

#### Fix 3.1: Strengthen Death Penalty
**File**: `game.py`  
**Change**: 
- Wall/Body death: `-2.0` (was -1.0)
- Starvation: `-3.0` (was -2.0)

**Priority**: MEDIUM - Clear survival signal

---

#### Fix 3.2: Strengthen Distance Shaping
**File**: `game.py`  
**Change**: `reward += (old_dist - new_dist) * 0.1` (was 0.05)

**Priority**: LOW - Exploration guide

---

#### Fix 3.3: Add Step Penalty
**File**: `game.py`  
**Change**: `reward -= 0.01` per step (time penalty)

**Priority**: LOW - Prevents loops

---

### Phase 4: Architecture Enhancements

#### Fix 4.1: Progressive MCTS Simulations
**File**: `main.py`  
**Change**: Start with 50 simulations, increase to 400 over generations:
```python
def get_simulations(generation):
    if generation < 10:
        return 50
    elif generation < 30:
        return 100
    elif generation < 50:
        return 200
    else:
        return 400
```

**Priority**: LOW - Gradual complexity

---

#### Fix 4.2: Add Value Head Regularization
**File**: `model.py`  
**Change**: Add L2 regularization to value head to prevent overfitting.

**Priority**: LOW - Generalization

---

## 5. Implementation Priority

### Critical Path (Must Fix First):
1. ✅ MCTS Terminal State Handling (Fix 1.1)
2. ✅ Unify Gamma (Fix 1.2)
3. ✅ Increase Simulations (Fix 1.3)
4. ✅ Lower Learning Rate (Fix 2.1)

### High Priority (Next):
5. Strengthen Death Penalty (Fix 3.1)
6. Weight Value Loss (Fix 2.3)
7. Increase Memory (Fix 2.2)

### Medium Priority (Polish):
8. Distance Shaping (Fix 3.2)
9. Step Penalty (Fix 3.3)
10. Progressive Simulations (Fix 4.1)

---

## 6. Expected Outcomes

### After Critical Fixes:
- **MCTS provides correct value estimates** → Network learns accurate values
- **Consistent gamma** → Value predictions align with MCTS
- **Deeper search** → Better training targets, fewer self-traps
- **Stable training** → No catastrophic collapses

### Target Metrics:
- **Gen 1-10**: Avg Score 1.5-2.0 (baseline)
- **Gen 11-30**: Avg Score 2.5-4.0 (learning)
- **Gen 31-50**: Avg Score 4.0-6.0 (improving)
- **Gen 50+**: Avg Score 6.0+ (approaching perfect play on 6x6)

### Perfect Play Definition:
- **6x6 board**: Score 34 (fill entire board, 36 cells - 2 initial body = 34 apples)
- **10x10 board**: Score 98 (100 cells - 2 initial body = 98 apples)

---

## 7. Testing Strategy

1. **Unit Tests**: Verify MCTS terminal handling with known terminal states
2. **Integration Tests**: Check gamma consistency across MCTS and training
3. **Ablation Studies**: Test each fix independently to measure impact
4. **Convergence Monitoring**: Track loss, score, entropy over generations

---

## 8. Long-Term Vision

### Perfect Play Path:
1. **Fix critical bugs** → Stable learning
2. **Optimize rewards** → Clear survival signal
3. **Increase search depth** → Better planning
4. **Scale to larger boards** → Strategic depth
5. **Add curriculum learning** → Progressive difficulty
6. **Implement self-play improvements** → Stronger opponents

### Advanced Features (Future):
- **Graph-based MCTS**: Detect loops, avoid revisiting states
- **Value function pretraining**: Supervised learning on expert games
- **Multi-scale features**: CNN with attention for long-range dependencies
- **Ensemble methods**: Multiple networks voting

---

## Conclusion

The project has a **solid architectural foundation** but suffers from **fundamental bugs** in MCTS terminal handling and **training instability**. Fixing these critical issues will unlock learning. The proposed improvements are **integrated and prioritized** to create a clear path to perfect play.

**Next Steps**: Implement Phase 1 fixes immediately, then proceed with Phase 2-4 based on training results.


## Analysis n3
1. Analysis of Current Status (Wholistic View)
The project has a solid AlphaZero foundation (NN + MCTS + POV Invariance), but it suffers from a critical "State Representation Amnesia".
The "Ouroboros" Defect: The primary failure mode ("DeathBody") occurs because the snake treats its own body as a static wall. The binary input channel (0 = Empty, 1 = Body) destroys the temporal information of the snake. The agent cannot distinguish between a tail segment that will move away in 1 turn (safe) and a neck segment that will stay for 50 turns (deadly).
Scale Mismatch: Training on a 6x6 board with low simulation counts (30-50) created a "toy" agent that cannot generalize to real Snake strategy (trapping/filling).
Value Confusion: The previous tanh activation saturated at 1.0, making "Eating (+1)" and "Winning (+2)" indistinguishable from "Surviving".
2. Implemented Improvements ("Perfect Play" Upgrade)
I have applied the following upgrades directly to the codebase:
A. "Lifetime" State Representation (Crucial Fix)
I replaced the binary Body channel with a Gradient Lifetime Channel.
Old: Body = 1.0 (Static Wall)
New: Body = 1.0 (Head) $\to$ 0.1 (Tail).
Effect: The CNN can now "see" the flow of the snake. It can calculate if a gap will open up in time to squeeze through, enabling advanced trapping and cycle behaviors.
B. Architectural & Training Tuning
Board Size: Upgraded from 6x6 to 10x10 (Standard Snake).
Simulations: Increased from 100 to 200 (Deeper search for complex traps).
Stability: Reduced Learning Rate (5e-4), Increased Games Per Gen (50), and reduced Epochs (2) to prevent the "Catastrophic Forgetting" observed in previous runs.
Memory: Increased buffer to 50,000 to ensure a diverse training distribution