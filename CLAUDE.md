# SPA Reproduction Repo — Complete Guide

## Top-Level Structure

```
SPA_reproduction/
├── spa_original/          # The SPA paper's implementation (main focus)
│   ├── SPA_agent/         # Core agent code
│   ├── config/            # Hydra configs
│   ├── sft/               # SFT training code
│   ├── run_spa.sh         # Master pipeline script
│   ├── run_baseline.sh    # Baseline (no SFT, direct RL)
│   └── train_ppo_sfted.sh # PPO launcher (called by run_spa.sh)
├── ragen/                 # RAGEN framework (underlying training infra)
│   ├── train.py           # RL training entry point
│   ├── ragen/             # Core library (trainer, workers, envs, agents)
│   └── verl/              # Upstream verl library (FSDP, vLLM integration)
├── scripts/               # Utility scripts (download data, local data gen)
└── data/                  # Downloaded datasets
```

## The 3-Stage Pipeline (`run_spa.sh`)

```
Stage 1: Data Generation  →  Stage 2: SFT  →  Stage 3: PPO (RL)
```

## Stage 1: Self-Play Data Generation

**Purpose:** Generate (state, action, next_state) trajectory data using the base LLM interacting with the environment, then replace the LLM's predicted states with ground-truth states from the environment.

| File | Role |
|---|---|
| `SPA_agent/generate_sft_data.py` | **Entry point.** Loads base model, runs rollouts, converts to SFT format, saves parquet |
| `SPA_agent/agent_proxy.py` | `LLMAgentProxy.rollout()` — orchestrates the multi-turn loop: get LLM input → generate → parse → step env → repeat |
| `SPA_agent/agent_proxy.py` | `VllmWrapperWg` — wraps vLLM to generate text from `Qwen2.5-1.5B-Instruct` |
| `SPA_agent/ctx_manager.py` | `ContextManager.get_lm_inputs()` — builds the prompt with environment state, format instructions, and `<observation>`/`<prediction>` template |
| `SPA_agent/ctx_manager.py` | `ContextManager.get_env_inputs()` → `_parse_response_add_worldmodel()` — extracts actions from LLM output |
| `SPA_agent/es_manager.py` | `EnvStateManager` — manages parallel envs, executes actions, collects ground-truth states |
| `config/envs.yaml` | `env_instruction_add_worldmodel` — the prompt template with few-shot example |
| `config/base.yaml` | Model path (`Qwen/Qwen2.5-1.5B-Instruct`), `ctx_manager.mode: add_worldmodel` |

**Key function in `generate_sft_data.py`:**
- `convert_to_sft_format_add_worldmodel()` — replaces `<observation>` and `<prediction>` content with real environment states via regex substitution

**Output:** `wm_train.parquet` and `wm_val.parquet` in `sftdata/`

## Stage 2: SFT (Supervised Fine-Tuning)

**Purpose:** Train the base model to predict accurate world states — i.e., learn "given this state + action, the next state is X". This internalizes a world model into the LLM weights.

| File | Role |
|---|---|
| `sft/finetune_ft.sh` | **Entry point.** Launches `torchrun -m sft.spa_sft_trainer` with data paths, LR, epochs, model size |
| `sft/spa_sft_trainer.py` | `FSDPSFTTrainer` — full FSDP training with bfloat16, optional LoRA, special token handling for `<observation>`/`<prediction>` |
| `sft/spa_sft_dataset.py` | `SFTDataset` — loads parquet, formats multi-turn conversation via `apply_chat_template` |
| `sft/filter_sft_by_tag.py` | Optional post-filter: keeps only rows with valid `<think>...<observation>...<prediction>...</think>` structure |
| `sft/config/sft_trainer.yaml` | Config: `batch_size=256`, `lr=1e-5`, `epochs=4`, `max_length=1024` |

**Key detail:** The loss mask in `ctx_manager.py`'s `get_masks_and_scores_add_worldmodel()` is designed so the model only learns to predict the **state tokens** (between `<observation>`/`</observation>` and `<prediction>`/`</prediction>` tags), not the action tokens. This focuses the SFT on world modeling.

**Output:** SFT checkpoint in `sftckpt/checkpoints_2_sokoban-1.5B-.../`

## Stage 3: PPO (Reinforcement Learning)

**Purpose:** Starting from the SFT-initialized model (which now has an internalized world model), train the policy to select **optimal actions** using environment rewards.

| File | Role |
|---|---|
| `train_ppo_sfted.sh` | **Entry point.** Calls `python ../train.py` with SFT checkpoint as `model_path`, 4 GPUs, 1000 steps |
| `ragen/train.py` | Main RL entry point: sets up Ray, FSDP actor/critic/ref workers, instantiates `RayAgentTrainer`, calls `trainer.fit()` |
| `ragen/ragen/trainer/agent_trainer.py` | `RayAgentTrainer` — the full PPO loop: rollout → compute advantages → actor update → critic update → validate |
| `ragen/ragen/trainer/core_algos.py` | Advantage estimation (GAE or GRPO) |
| `ragen/ragen/trainer/rollout_filter.py` | Filters rollouts by variance (`rollout_filter_ratio=0.25`, `std` mode) — skips low-variance groups |
| `ragen/ragen/workers/fsdp_workers.py` | `ActorRolloutRefWorker`, `CriticWorker` — FSDP-sharded model workers |
| `SPA_agent/agent_proxy.py` | Same `LLMAgentProxy.rollout()` but now called during PPO rollout collection |
| `SPA_agent/ctx_manager.py` | Same context manager, but now in `mode=base` (no `<observation>`/`<prediction>` tags in the prompt during RL) |
| `config/base.yaml` | PPO config: `clip_ratio=0.2`, `entropy_coeff=0.001`, `gamma=1.0`, `lam=1.0`, `adv_estimator=gae` |

**Key difference from Stage 1:** During RL, `ctx_manager.mode` switches to `base` — the model no longer explicitly predicts states with `<observation>`/`<prediction>` tags. The world model knowledge is already baked into the weights from SFT, so the model uses it implicitly to reason about consequences in `<think>` and choose better actions.

## How the Configs Chain Together

```
_2_sokoban.yaml  →  inherits base.yaml  →  inherits ppo_trainer.yaml + envs.yaml
```

- `envs.yaml`: defines environment-specific instructions and parameters
- `base.yaml`: model path, rollout settings, agent_proxy settings, es_manager settings
- `ppo_trainer.yaml`: verl PPO hyperparameters (LR, clipping, FSDP, critic, etc.)
- `ctx_manager.mode` is set via env var `MODE` — `add_worldmodel` for Stage 1, `base` for Stage 3

## Baseline Comparison (`run_baseline.sh`)

Skips Stages 1 & 2 entirely — runs PPO directly from `Qwen/Qwen2.5-1.5B-Instruct` without SFT warm-starting. This measures how much the world model internalization (SPA's contribution) actually helps.

## Quick Reference: What Calls What

```
run_spa.sh
  ├── python -m SPA_agent.generate_sft_data    # Stage 1
  ├── bash sft/finetune_ft.sh                   # Stage 2
  │     └── torchrun -m sft.spa_sft_trainer
  └── bash train_ppo_sfted.sh                   # Stage 3
        └── python ../train.py (ragen/train.py)
              └── RayAgentTrainer.fit()
                    └── LLMAgentProxy.rollout() per step
```
