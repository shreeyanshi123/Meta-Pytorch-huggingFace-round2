# 🏗️ Constrained Refactor Gauntlet

> **Meta OpenEnv Hackathon India 2026** — An RL environment where an agent refactors a legacy Python codebase while obeying **150 cascading engineering rules**. The agent must navigate contradictory constraints, cascading rule spawns, and code quality metrics to maximise the multiplicative reward: **CodeScore × ComplianceScore**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Architecture](#architecture)
- [Environment Design](#environment-design)
- [Rule System](#rule-system)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

The **Constrained Refactor Gauntlet** is a challenging OpenEnv-compliant RL environment designed to test an agent's ability to:

1. **Refactor** deliberately corrupted Python code (broken imports, cryptic names, dead code, hardcoded secrets)
2. **Comply** with 150 engineering rules spanning style, structure, security, conditional logic, and contradictions
3. **Navigate tradeoffs** when rules contradict each other (10 deliberate contradiction pairs)
4. **Improve quality** measurably across linting, complexity, module size, docstrings, and type hints

The agent is trained using **GRPO (Group Relative Policy Optimization)** with a **LoRA-tuned Qwen2.5-Coder-7B** model.

---

## The Problem

Real-world codebases are messy. Engineers must refactor code while adhering to style guides, security policies, and architectural constraints — many of which conflict with each other. This environment simulates that challenge:

- **8 corruption types** are randomly applied to a clean FastAPI codebase
- **150 rules** activate progressively via a curriculum system
- **Cascading spawns**: resolving one rule can trigger others (e.g., Rule 29 → Rules 12, 91, 118)
- **Contradictions**: 10 rule pairs are deliberately incompatible (e.g., "all functions need type hints" vs. "short functions should avoid type hints")

The agent must learn to **prioritize**, **trade off**, and **budget its 70 steps** wisely.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Server (server.py)             │
│                                                         │
│  POST /reset ──► EpisodeGenerator ──► Corrupted Code    │
│  POST /step  ──► Track A (Code Quality)                 │
│               ──► Track B (Compliance)                  │
│               ──► Reward = A × B                        │
│  GET  /health                                           │
└─────────────────────────────────────────────────────────┘
         ▲                                    │
         │         Observation + Reward       │
         │                                    ▼
┌─────────────────────────────────────────────────────────┐
│              GRPO Trainer (train_grpo.py)                │
│                                                         │
│  Qwen2.5-Coder-7B + LoRA (r=32, α=64)                 │
│  4 generations per prompt, batch=2, grad_accum=4       │
│  max_completion_length=1024, bf16, sdpa attention      │
└─────────────────────────────────────────────────────────┘
```

---

## Environment Design

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `files` | `dict[str, str]` | Current codebase (filename → source code) |
| `violation_report` | `dict` | Newly triggered, resolved, and outstanding rules |
| `steps_remaining` | `int` | Steps left in the episode (max 70) |
| `curriculum_level` | `int` | Current difficulty level (1–4) |

### Action Space

| Tool | Description |
|------|-------------|
| `read_file` | Read a specific file's content |
| `edit_file` | Replace a file's content |
| `run_tests` | Run the test suite |
| `check_compliance` | Check rule compliance status |
| `finish` | End the episode early |

### Reward Formula

```
Reward = CodeScore × ComplianceScore
```

- **CodeScore** (Track A): `0.35·test_pass_rate + 0.25·lint_improvement + 0.20·complexity_reduction + 0.20·module_size_compliance`
- **ComplianceScore** (Track B): `resolved_rules / triggered_rules`

The multiplicative formula means the agent must perform well on **both** tracks — high code quality with low compliance yields a low reward, and vice versa.

---

## Rule System

### Categories (150 rules total)

| Category | Rules | Description |
|----------|-------|-------------|
| **STYLE** | 1–30 | Naming, formatting, docstrings, type hints |
| **STRUCTURAL** | 31–70 | Architecture, module size, patterns |
| **SECURITY** | 71–100 | Secrets, authentication, injection prevention |
| **CONDITIONAL** | 101–130 | Context-dependent requirements |
| **CONTRADICTORY** | 131–150 | 10 deliberately conflicting rule pairs |

### Cascading Spawns

Some rules trigger additional obligations when activated:

- Rule 29 (docstrings) → spawns Rules 12, 91, 118
- Rule 44 (file size) → spawns Rules 91, 17
- Rule 55 (function size) → spawns Rules 56, 103
- Rule 83 (no secrets) → spawns Rule 134
- Rule 134 (dev secrets) → spawns Rule 29

### Contradiction Pairs

| Pair | Rule A | Rule B |
|------|--------|--------|
| 1 | All functions need type hints (131) | Short functions avoid type hints (132) |
| 2 | Env vars via schema (133) | Hardcode secrets in dev (134) |
| 3 | Raw SQL for performance (135) | ORM for safety (136) |
| 4 | Catch all exceptions (137) | Catch specific exceptions (138) |
| 5 | Log all payloads (139) | Never log payloads (140) |
| 6 | All endpoints need auth (141) | Health endpoints public (144) |
| 7 | All functions need comments (142) | Short functions no comments (143) |
| 8 | Import specific functions (145) | Import whole packages (146) |
| 9 | camelCase JSON keys (147) | snake_case JSON keys (148) |
| 10 | /tmp for temp files (149) | tempfile module (150) |

---

## Training Pipeline

### Model Configuration

- **Base model**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Adapter**: LoRA (r=32, α=64, dropout=0.05)
- **Target modules**: q/k/v/o_proj, gate/up/down_proj
- **Precision**: bfloat16 with SDPA attention
- **Training**: GRPO with 4 generations per prompt

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| Batch size | 2 (×4 gradient accumulation = 8 effective) |
| Max steps | 100 |
| Completion length | 1024 tokens |
| Generations per prompt | 4 |
| Dataset size | 220 episodes (198 train / 22 eval) |

---

## Evaluation

Run the trained agent against fresh episodes:

```bash
python evaluate_agent.py --episodes 10
```

This produces:
- Per-episode metrics (code quality, compliance, reward)
- Summary statistics (mean, std, min, max)
- JSON results file (`evaluation_results.json`)

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: A100/H100 with 40GB+ VRAM)

### Setup

```bash
git clone https://github.com/shreeyanshi123/Meta-Pytorch-huggingFace-round2.git
cd Meta-Pytorch-huggingFace-round2
pip install -r requirements.txt
```

### Run the Environment Server

```bash
python server.py
# Server starts on http://localhost:7860
```

### Verify the Pipeline

```bash
python training/verify_pipeline.py
```

### Train the Agent

```bash
python training/train_grpo.py
```

### Evaluate the Agent

```bash
python evaluate_agent.py --episodes 10
```

### Docker

```bash
docker build -t constrained-refactor-gauntlet .
docker run -p 7860:7860 constrained-refactor-gauntlet
```

---

## Project Structure

```
.
├── server.py                  # FastAPI environment server (reset/step/health)
├── openenv.yaml               # OpenEnv specification
├── evaluate_agent.py          # Agent evaluation script
├── agent_inference.py         # OpenAI API-based agent loop
├── generate_rules_v2.py       # Engineering rules generator
├── test_reset.py              # Pytest test suite
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── environment/
│   ├── __init__.py            # Package exports
│   ├── episode_generator.py   # Corrupted codebase generation
│   ├── rule_engine.py         # 150-rule compliance engine
│   ├── track_a.py             # Code quality evaluator
│   ├── track_b.py             # Compliance checker
│   ├── ENGINEERING_STANDARDS.md  # 150 rules in markdown
│   └── base_codebase/         # Clean FastAPI codebase
│       ├── api.py             # REST endpoints
│       ├── config.py          # Settings
│       ├── main.py            # App entry point
│       └── utils.py           # Utilities
└── training/
    ├── __init__.py
    ├── train_grpo.py          # GRPO training script
    ├── inference.py           # LoRA adapter inference demo
    └── verify_pipeline.py     # Pipeline verification
```

---

## Results

After training with GRPO for 100 steps on 198 episodes:

| Metric | Value |
|--------|-------|
| Average Reward | Measured via `evaluate_agent.py` |
| Code Quality (Track A) | Measured via `evaluate_agent.py` |
| Compliance (Track B) | Measured via `evaluate_agent.py` |
| Training Time | ~2-3 hours on H100 |

*Run `python evaluate_agent.py --episodes 20` to populate these metrics.*

---

## Future Work

- **Curriculum escalation**: Extend training to 500+ steps to trigger level 2-4 curricula
- **Multi-turn agent**: Convert from single-shot to iterative multi-step refactoring
- **Richer code quality**: Integrate actual test execution and runtime analysis
- **Larger base codebases**: Add more diverse Python projects beyond the FastAPI example
- **Human evaluation**: Compare agent refactoring quality against human engineers

---

## License

This project was developed for the Meta OpenEnv Hackathon India 2026.
