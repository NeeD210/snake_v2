# 🐍 Snake AI - Development Backlog

This document tracks the evolution of the Snake AI project, categorizing tasks by priority, impact, and difficulty.

## 📊 Epoch 0: Pipeline Observability
*Goal: Implement real-time telemetry to identify hardware and software bottlenecks.*

| Task | Category | Difficulty | Impact | Status |
|:---|:---:|:---:|:---:|:---|
| **Pipeline Saturation Monitor** | Telemetry | Medium| High | ✅ Completed |
| *Description:* Per-gen telemetry (SelfPlay/Train time, InferBatches, AvgSims) + real-time monitor (CPU%, GPU%, Batch Rate). Full sweep via `benchmark_pipeline.py`. | | | | |

---

## ⚡ Epoch 1: Throughput & Scaling (Immediate ROI)
*Goal: Remove bottlenecks to make 10x10 training computationally viable.*

| Task | Category | Difficulty | Impact | Status |
|:---|:---:|:---:|:---:|:---|
| **Cython/Numba Encoder** | Performance | Medium | High | ✅ Completed |
| *Description:* JIT compile `flood_fill_area_3dir`. Achieved: 1,268 enc/s on 10x10 via `@njit`. | | | | |
| **Shared Memory IPC** | Infrastructure | High | High | ✅ Completed |
| *Description:* Replace standard Queues with `shared_memory`. Integrated into `main.py` training pipeline. | | | | |
| **Virtual Loss (MCTS)** | Search | High | High | ✅ Completed |
| *Description:* Multi-In-Flight simulations via `AsyncSHMClient` with `num_parallel=8`. | | | | |

---

## 🌀 Epoch 2: Universal Architecture & Reverse Curriculum
*Goal: Master the 10x10 board and generalize downwards.*

| Task | Category | Difficulty | Impact | Status |
|:---|:---:|:---:|:---:|:---|
| **Universal 10x10 POV** | Architecture | Medium | High | ✅ Completed |
| *Description:* `encode_pov` always produces a `(5, 10, 10)` tensor, center-padding smaller boards with wall signals. | | | | |
| **Reverse Curriculum** | Strategy | Medium | High | ✅ Completed |
| *Description:* Default to 10x10 training. Removed multi-size curriculum in favor of mastering the full board first. | | | | |

## Epoch 3: Policy Distillation
| **Policy Distillation** | Training | Medium | High | 🚧 In Progress |
| *Description:* Implement a standalone "Student" fast-policy network for deployment. | | | | |

---

## 🚀 Secondary Enhancements

| Task | Category | Difficulty | Impact |
|:---|:---:|:---:|:---:|
| **Replay Buffer Prioritization (PER)** | Training | Medium | Medium |
| *Description:* Focus training on "surprising" states where the model's error is highest. | | | |
| **Attention/Transformer Hybrid** | Architecture | High | High |
| *Description:* Self-attention layers to handle long-range dependencies on 10x10+ boards. | | | |
| **Competitive Self-Play** | Training | High | Medium |
| *Description:* Train against previous iterations to prevent strategy stagnation. | | | |

---

## ✅ Completed Milestones

| Milestone | Date | Impact |
|:---|:---:|:---:|
| **CSV Report Rounding** | 2026-04-08 | Quality of Life |
| **Pipeline Saturation Monitor** | 2026-04-08 | Bottleneck Visibility |
| **Shared Memory IPC + Virtual Loss** | 2026-04-08 | **10x10 Viable** |
| **Universal 10x10 POV** | 2026-04-08 | **Fixed Architecture** |
| **Reverse Curriculum (10x10 Focus)** | 2026-04-08 | **Strategic Simplification** |
| **Numba JIT Encoder** | 2026-04-07 | +4x Encoder Speed |
| **Adaptive Curriculum Learning** | 2026-04-06 | **Game Changer** |
| **Inhomogeneous Batching** | 2026-04-05 | Critical Stability |
| **Deterministic Food Placement** | 2026-04-05 | 100% Tree Reuse |
| **MCTS `.undo()` Optimization** | 2026-04-04 | +70% Throughput |
| **Value Head Re-bounding (Tanh)** | 2026-04-04 | Training Stability |
| **Dynamic MCTS Budget (Entropy-Based)**| 2026-04-04 | Search Efficiency |
| **Endgame Specialization (Oversampling)**| 2026-04-04 | Late-Game Logic |

---

> [!IMPORTANT]
> **Universal Viewport**: The 10x10 fixed POV is now implemented. All encoder outputs are `(5, 10, 10)`. Existing model weights are **incompatible** — start fresh from Gen 0.

> [!TIP]
> Use `python snake_ai/profile_throughput.py` to benchmark any optimization before merging.
