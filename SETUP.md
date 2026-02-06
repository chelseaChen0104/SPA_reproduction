# SPA Data Generation Setup Guide

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (tested on H100)
- **CUDA**: 12.4+
- **Python**: 3.10+

## Step 1: Setup RAGEN Framework

```bash
cd /Users/siyunchen/Documents/GitHub/SPA_reproduction

# Install RAGEN
cd ragen
bash scripts/setup_ragen.sh
```

## Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM and flash-attn
pip uninstall -y vllm flash-attn flash_attn
pip install vllm==0.8.5.post1

# Install flash-attention (Linux only)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Verify
python -c "import torch; import vllm; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Step 3: Generate SPA Data

```bash
cd /Users/siyunchen/Documents/GitHub/SPA_reproduction/spa_original

# Set environment variables
export MODE="add_worldmodel"
export RENDER_MODE="text_with_coordinates"
export OUTPUT_DIR="./sftdata/sokoban"
export BT_NUM=5  # Number of batches

# Generate data for Sokoban
python -m SPA_agent.generate_sft_data --config-name _2_sokoban
```

## Step 4: Check Output

```bash
ls -la ./sftdata/sokoban/
# Should contain:
# - raw_trajectories_*.json
# - wm_train.parquet
# - wm_val.parquet
```

## Alternative: Use Pre-generated Data

If you don't have GPU access, download official data:

```bash
cd /Users/siyunchen/Documents/GitHub/SPA_reproduction
python scripts/download_data.py --env sokoban --output-dir data
```

## Configuration Options

| Environment | Config Name | Description |
|-------------|-------------|-------------|
| Sokoban | `_2_sokoban` | 6x6 grid, 1 box |
| FrozenLake | `_3_frozen_lake` | 4x4 grid |
| Sudoku | `_10_sudoku` | 4x4 puzzle |

## Troubleshooting

### JAVA_HOME Error
The script expects Java for some dependencies. Either:
1. Install Java 21: `brew install openjdk@21` (macOS)
2. Or comment out the Java check in `run_spa.sh`

### CUDA Out of Memory
Reduce batch size:
```bash
export BT_NUM=2  # Smaller batches
```

### vLLM Issues on macOS
vLLM requires Linux with CUDA. For macOS, use the pre-generated data instead.
