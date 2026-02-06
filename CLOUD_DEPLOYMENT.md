# SPA Cloud Deployment Guide

## Quick Start for Cloud GPU (H100/A100)

This guide provides step-by-step instructions for running SPA on cloud GPU instances.

---

## Option 1: Use Pre-trained Models (Fastest - 30 min setup)

If you just want to run PPO training with an already-trained world model:

```bash
# 1. Clone repos
git clone https://github.com/RAGEN-AI/RAGEN.git
cd RAGEN
git clone https://github.com/shiqichen17/SPA.git

# 2. Setup environment (creates conda env 'ragen')
bash scripts/setup_ragen.sh

# 3. Activate and install specific versions
conda activate ragen
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y vllm flash-attn flash_attn
pip install vllm==0.8.5.post1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 4. Verify installation
python -c "import torch; import flash_attn; import vllm; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# 5. Download pre-trained model
cd SPA
huggingface-cli download tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct \
    --local-dir ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/

# 6. Run PPO training only
bash run_spa.sh _2_sokoban last False
```

---

## Option 2: Full Pipeline (Data Gen + SFT + PPO)

### Step 1: Environment Setup

```bash
# Clone repos
git clone https://github.com/RAGEN-AI/RAGEN.git
cd RAGEN
git clone https://github.com/shiqichen17/SPA.git

# Setup RAGEN environment
bash scripts/setup_ragen.sh
conda activate ragen

# Install exact versions (CRITICAL)
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip uninstall -y vllm flash-attn flash_attn
pip install vllm==0.8.5.post1

# Flash attention (Linux + CUDA 12.4 + Python 3.12)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Verify
python -c "import torch; import flash_attn; import vllm; print('All modules loaded.')"
```

### Step 2: Generate Self-Play Data

```bash
cd SPA

# Set environment variables
export MODE="add_worldmodel"
export RENDER_MODE="text_with_coordinates"
export BT_NUM=5  # Batch size (reduce if OOM)

# Generate data for Sokoban
python -m SPA_agent.generate_sft_data --config-name _2_sokoban

# Output: ./sftdata/_2_sokoban-1.5B-text_with_coordinates/
#   - wm_train.parquet
#   - wm_val.parquet
```

**GPU Memory:** ~24GB (A100 40GB recommended)

### Step 3: SFT Training

```bash
# Run SFT with 4 GPUs
bash sft/finetune_ft.sh _2_sokoban 4 \
    ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/ \
    ./sftdata/_2_sokoban-1.5B-text_with_coordinates \
    1.5B

# Key parameters (editable in script):
# - Learning rate: 1e-4
# - Batch size: 16 total
# - Micro batch: 1 per GPU
# - Epochs: 5
# - Max length: 2048
```

**GPU Memory:** ~40GB per GPU (4x A100 40GB recommended)
**Duration:** ~2-4 hours

### Step 4: PPO Training

```bash
# Run PPO with trained checkpoint
bash run_spa.sh _2_sokoban last False

# Or specify checkpoint step
bash run_spa.sh _2_sokoban 1000 False
```

**GPU Memory:** ~80GB total (8x A100 80GB recommended)
**Duration:** ~6-12 hours for 1000 steps

---

## One-Command Full Pipeline

```bash
cd SPA
bash run_spa.sh _2_sokoban last True
```

This runs: Data Generation → SFT → PPO

---

## Environment Configurations

| Environment | Config Name | Data | Model |
|-------------|-------------|------|-------|
| Sokoban | `_2_sokoban` | [HF Dataset](https://huggingface.co/datasets/tyzhu/SPA-sokoban-data) | [HF Model](https://huggingface.co/tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct) |
| FrozenLake | `_3_frozen_lake` | [HF Dataset](https://huggingface.co/datasets/tyzhu/SPA-frozenlake-data) | [HF Model](https://huggingface.co/tyzhu/SPA-frozenlake-qwen2.5-1.5b-instruct) |
| Sudoku | `_10_sudoku` | [HF Dataset](https://huggingface.co/datasets/tyzhu/SPA-sudoku-data) | [HF Model](https://huggingface.co/tyzhu/SPA-sudoku-qwen2.5-1.5b-instruct) |

---

## Hardware Requirements

| Stage | GPU Memory | Recommended Config |
|-------|------------|-------------------|
| Data Generation | ~24GB | 1x A100 40GB |
| SFT Training | ~40GB/GPU | 4x A100 40GB |
| PPO Training | ~80GB total | 8x A100 80GB |

### Cost Estimates (AWS/GCP)

| Stage | Instance | Cost/Hour | Est. Duration | Total |
|-------|----------|-----------|---------------|-------|
| Data Gen | p4d.24xlarge | ~$32 | 1-2 hours | ~$50 |
| SFT | p4d.24xlarge | ~$32 | 2-4 hours | ~$100 |
| PPO | p4de.24xlarge | ~$40 | 6-12 hours | ~$400 |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size for data generation
export BT_NUM=2

# Enable gradient checkpointing for SFT
# Add to finetune_ft.sh:
model.enable_gradient_checkpointing=True

# Reduce micro batch for PPO
# Edit train_ppo_sfted.sh
```

### Flash Attention Errors

```bash
# Check CUDA version
nvcc --version  # Should be 12.4+

# Check torch CUDA compatibility
python -c "import torch; print(torch.version.cuda)"

# Reinstall exact version
pip uninstall flash-attn flash_attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

### JAVA_HOME Error

```bash
# Install Java 21
sudo apt update && sudo apt install -y openjdk-21-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Or edit run_spa.sh to use your Java path
```

### Ray Worker Failures

```bash
# Increase spilling threshold
export RAY_object_spilling_threshold=0.99

# Reduce batch size
export BT_NUM=2
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

## File Outputs

```
SPA/
├── sftdata/
│   └── _2_sokoban-1.5B-text_with_coordinates/
│       ├── wm_train.parquet      # Training data
│       └── wm_val.parquet        # Validation data
├── sftckpt/
│   └── checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/
│       └── global_step_XXX/      # SFT checkpoint
└── outputs/                       # PPO training outputs
    └── experiment_name/
        ├── checkpoints/
        └── logs/
```

---

## Local Testing (No GPU)

For testing data generation locally without GPU:

```bash
cd SPA_reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn pyarrow

# Generate test data (no LLM needed)
python scripts/generate_data_local.py --env sokoban --num-trajectories 100
```
