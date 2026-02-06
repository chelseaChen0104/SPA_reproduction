# Reproducing SPA: Step-by-Step Guide

## Overview

SPA (Self-Play Agent) training consists of **3 stages**:
1. **Self-Play Data Generation** - Generate trajectories with `<observation>` and `<prediction>`
2. **SFT (Supervised Fine-Tuning)** - Train world model on the generated data
3. **PPO Training** - Reinforce policies using the learned world model

---

## Prerequisites

### 1. Clone Repositories

```bash
# Clone RAGEN (base framework)
git clone https://github.com/RAGEN-AI/RAGEN.git
cd RAGEN

# Clone SPA inside RAGEN
git clone https://github.com/shiqichen17/SPA.git
```

### 2. Environment Setup

```bash
# Run RAGEN setup script
bash scripts/setup_ragen.sh

# Install specific versions (CRITICAL - use exact versions)
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip uninstall -y vllm flash-attn flash_attn
pip install vllm==0.8.5.post1

# Install flash-attention (for CUDA 12.4 + PyTorch 2.6)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Verify installation
python -c "import torch; import flash_attn; import vllm; print('✅ All modules loaded successfully.')"
```

### 3. Hardware Requirements

| Stage | GPU Memory | Recommended |
|-------|------------|-------------|
| Data Generation | ~24GB | 1x A100 40GB |
| SFT Training | ~40GB per GPU | 4x A100 40GB |
| PPO Training | ~80GB total | 8x A100 80GB |

---

## Option A: Use Pre-trained Models (Quick Start)

SPA provides pre-trained models and datasets on Hugging Face:

| Environment | Dataset | Model |
|-------------|---------|-------|
| Sokoban | [SPA-sokoban-data](https://huggingface.co/datasets/tyzhu/SPA-sokoban-data) | [SPA-sokoban-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct) |
| FrozenLake | [SPA-frozenlake-data](https://huggingface.co/datasets/tyzhu/SPA-frozenlake-data) | [SPA-frozenlake-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-frozenlake-qwen2.5-1.5b-instruct) |
| Sudoku | [SPA-sudoku-data](https://huggingface.co/datasets/tyzhu/SPA-sudoku-data) | [SPA-sudoku-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-sudoku-qwen2.5-1.5b-instruct) |

### Download and Use Pre-trained Model

```bash
# Download model
huggingface-cli download tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct --local-dir ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/

# Run PPO only (skip data generation and SFT)
cd SPA
bash run_spa.sh _2_sokoban last False
```

---

## Option B: Full Reproduction (From Scratch)

### Stage 1: Generate Self-Play Data

```bash
cd SPA

# Generate data for Sokoban
python -m SPA_agent.generate_sft_data --config-name _2_sokoban

# Or for other environments:
# python -m SPA_agent.generate_sft_data --config-name _3_frozen_lake
# python -m SPA_agent.generate_sft_data --config-name _10_sudoku
```

**Output:** `./sftdata/_2_sokoban-1.5B-text_with_coordinates/`
- `wm_train.parquet` - Training data
- `wm_val.parquet` - Validation data

### Stage 2: SFT Training

```bash
# Run SFT (4 GPUs recommended)
bash sft/finetune_ft.sh _2_sokoban 4 \
    ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/ \
    ./sftdata/_2_sokoban-1.5B-text_with_coordinates \
    1.5B
```

**Key SFT Parameters:**
- Learning rate: `1e-4`
- Batch size: `16` (total)
- Micro batch: `1` per GPU
- Epochs: `5`
- Max length: `2048`

**Output:** `./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/global_step_XXX/`

### Stage 3: PPO Training

```bash
# Run PPO with the SFT checkpoint
bash run_spa.sh _2_sokoban last False
```

Or run the full pipeline in one command:
```bash
bash run_spa.sh _2_sokoban last True
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SPA Training Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Stage 1    │    │   Stage 2    │    │   Stage 3    │       │
│  │              │    │              │    │              │       │
│  │  Self-Play   │───▶│     SFT      │───▶│     PPO      │       │
│  │   Data Gen   │    │  (World      │    │  (Policy     │       │
│  │              │    │   Model)     │    │   Learning)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  wm_train.parquet    SFT checkpoint     Trained Agent           │
│  wm_val.parquet                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Format

### Self-Play Trajectory Format

```xml
<think>
<observation>
######
#___O#
#__X_#
###P_#
###__#
######
Player (P) at (3,3); box (X) at (2,3); goal at (1,4).
</observation>
<prediction>
######
#___O#
#____#
###X_#
###P_#
######
</prediction>
</think>
<answer>Up</answer>
```

### Loss Function (From Paper Section 2.2)

```
L_W(θ) = -1/(Σᵢ Mᵢ) × Σᵢ Mᵢ log pθ(τᵢ | τ<ᵢ)

where M_i = 𝟙[τ_i ∈ (span(<think>,</think>) ∪ span(<answer>,</answer>))]
```

**Trained on:**
- ✅ `<observation>` content (state representation)
- ✅ `<prediction>` content (transition modeling)
- ✅ `<answer>` content (action)

**Masked (no loss):**
- ❌ Input prompt (User message with raw state)

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/base.yaml` | Base training settings |
| `config/_2_sokoban.yaml` | Sokoban environment config |
| `config/_3_frozen_lake.yaml` | FrozenLake environment config |
| `config/_10_sudoku.yaml` | Sudoku environment config |
| `sft/config/sft_trainer.yaml` | SFT trainer config |

### Key Config Parameters

```yaml
# Model
model_path: Qwen/Qwen2.5-1.5B-Instruct

# Training
trainer.total_training_steps: 1000  # PPO steps
agent_proxy.max_turn: 20            # Max turns per episode
es_manager.train.env_groups: 32     # Number of environment groups
```

---

## Expected Results

After 1000 PPO training steps:

| Environment | Vanilla RL | + SPA |
|-------------|-----------|-------|
| Sokoban | 25.6% | **59.8%** |
| FrozenLake | 22.1% | **70.9%** |
| Sudoku | 0.0% | **59.6%** |

---

## Troubleshooting

### 1. CUDA Out of Memory
- Reduce `micro_batch_size_per_gpu`
- Enable gradient checkpointing: `model.enable_gradient_checkpointing=True`
- Use fewer GPUs with larger accumulation steps

### 2. Flash Attention Errors
- Ensure exact version match: `flash_attn-2.7.3+cu12torch2.6`
- Check CUDA version compatibility

### 3. RAGEN Import Errors
- Ensure SPA is inside RAGEN directory
- Run from RAGEN root: `cd RAGEN && python -m SPA.SPA_agent.generate_sft_data`

### 4. Ray Worker Failures
- Set `RAY_object_spilling_threshold=0.99`
- Increase system memory or reduce batch size

---

## File Structure

```
RAGEN/
├── SPA/                              # SPA code
│   ├── SPA_agent/
│   │   ├── generate_sft_data.py     # Data generation
│   │   ├── agent_proxy.py           # Agent interface
│   │   └── ctx_manager.py           # Context manager
│   ├── sft/
│   │   ├── spa_sft_trainer.py       # SFT trainer
│   │   ├── spa_sft_dataset.py       # Dataset class
│   │   └── finetune_ft.sh           # SFT script
│   ├── config/                       # Configs
│   ├── run_spa.sh                    # Main script
│   └── train_ppo_sfted.sh           # PPO script
├── ragen/                            # RAGEN framework
│   ├── env/                          # Environments
│   ├── trainer/                      # PPO trainer
│   └── ...
└── scripts/
    └── setup_ragen.sh                # Setup script
```
