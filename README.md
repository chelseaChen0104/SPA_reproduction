# SPA Reproduction

Reproduction of the SPA (Self-Play Agent) paper: **"Internalizing World Models via Self-Play Finetuning for Agentic RL"**

Paper: [arXiv:2510.15047](https://arxiv.org/abs/2510.15047)

## Repository Structure

```
SPA_reproduction/
├── spa_original/     # Official SPA codebase (cloned from github.com/shiqichen17/SPA)
├── ragen/            # RAGEN framework (required by SPA)
├── data/             # Training data
└── scripts/          # Helper scripts
```

## Setup

### 1. Environment Setup

SPA requires the RAGEN framework. From the repo root:

```bash
cd ragen
bash scripts/setup_ragen.sh

# Install specific versions (important!)
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip uninstall -y vllm flash-attn flash_attn
pip install vllm==0.8.5.post1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Verify installation
python -c "import torch; import flash_attn; import vllm; print('All modules loaded successfully.')"
```

### 2. Download Pre-trained Data/Models

| Environment | Dataset | Model |
|-------------|---------|-------|
| Sokoban | [SPA-sokoban-data](https://huggingface.co/datasets/tyzhu/SPA-sokoban-data) | [SPA-sokoban-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct) |
| FrozenLake | [SPA-frozenlake-data](https://huggingface.co/datasets/tyzhu/SPA-frozenlake-data) | [SPA-frozenlake-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-frozenlake-qwen2.5-1.5b-instruct) |
| Sudoku | [SPA-sudoku-data](https://huggingface.co/datasets/tyzhu/SPA-sudoku-data) | [SPA-sudoku-qwen2.5-1.5b-instruct](https://huggingface.co/tyzhu/SPA-sudoku-qwen2.5-1.5b-instruct) |

### 3. Run SPA Training

```bash
cd spa_original

# Full pipeline: Generate Data -> SFT -> PPO
bash run_spa.sh _2_sokoban last True

# PPO only (using existing checkpoint)
bash run_spa.sh _2_sokoban last False
```

**Config options:**
- `_2_sokoban` - Sokoban environment
- `_3_frozen_lake` - FrozenLake environment
- `_10_sudoku` - Sudoku environment

## Expected Results

| Environment | Model | Pass@1 | Pass@8 |
|-------------|-------|--------|--------|
| Sokoban | Qwen2.5-1.5B | 59.8% | 69.5% |
| FrozenLake | Qwen2.5-1.5B | 70.9% | 75.0% |
| Sudoku | Qwen2.5-1.5B | 59.6% | 94.9% |

## SPA Data Format

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

## Citation

```bibtex
@misc{chen2025spa,
    title={Internalizing World Models via Self-Play Finetuning for Agentic RL},
    author={Shiqi Chen and Tongyao Zhu and others},
    year={2025},
    eprint={2510.15047},
    archivePrefix={arXiv}
}
```
