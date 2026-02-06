# SFT Training with Original Sokoban Data

## Quick Reference

**Data Location (already downloaded):**
```
data/original/SPA-sokoban-data/
├── wm_train.parquet   (4,809 samples)
└── wm_val.parquet     (1,203 samples)
```

---

## Option A: Use Pre-trained Checkpoint (Skip SFT)

If you just want to skip SFT and go straight to PPO:

```bash
cd spa_original

# Download pre-trained SFT checkpoint
huggingface-cli download tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct \
    --local-dir ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/global_step_1505

# Run PPO directly
bash run_spa.sh _2_sokoban last False
```

---

## Option B: Train SFT from Scratch

### Step 1: Upload to Cloud

```bash
# On your local machine
cd /Users/siyunchen/Documents/GitHub/SPA_reproduction

# Upload the whole repo (or use git clone on cloud)
rsync -avz . user@cloud-server:/path/to/SPA_reproduction/
```

### Step 2: Setup Environment on Cloud

```bash
# SSH into cloud server
ssh user@cloud-server

# Clone RAGEN (if not using rsync)
git clone https://github.com/RAGEN-AI/RAGEN.git
cd RAGEN

# Setup environment
bash scripts/setup_ragen.sh
conda activate ragen

# Install exact versions
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip uninstall -y vllm flash-attn flash_attn
pip install vllm==0.8.5.post1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Verify
python -c "import torch; import flash_attn; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: Copy Original Data

```bash
# Copy downloaded data to expected location
cd spa_original
mkdir -p sftdata/_2_sokoban-1.5B-text_with_coordinates

cp ../data/original/SPA-sokoban-data/wm_train.parquet \
   sftdata/_2_sokoban-1.5B-text_with_coordinates/

cp ../data/original/SPA-sokoban-data/wm_val.parquet \
   sftdata/_2_sokoban-1.5B-text_with_coordinates/
```

### Step 4: Run SFT Training

```bash
cd spa_original

# Create checkpoint directory
mkdir -p sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen

# Run SFT with 4 GPUs
bash sft/finetune_ft.sh _2_sokoban 4 \
    ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen \
    ./sftdata/_2_sokoban-1.5B-text_with_coordinates \
    1.5B
```

**Training Parameters:**
| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-1.5B-Instruct |
| Learning Rate | 1e-4 |
| Batch Size | 16 (total) |
| Micro Batch | 1 per GPU |
| Max Length | 2048 tokens |
| Epochs | 5 |

**Expected Duration:** ~2-4 hours on 4x A100 40GB

### Step 5: Run PPO Training

After SFT completes:

```bash
# Run PPO with trained checkpoint
bash run_spa.sh _2_sokoban last False
```

---

## Single GPU Setup (for smaller instances)

If you only have 1 GPU, modify the command:

```bash
# 1 GPU with gradient accumulation
bash sft/finetune_ft.sh _2_sokoban 1 \
    ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen \
    ./sftdata/_2_sokoban-1.5B-text_with_coordinates \
    1.5B \
    data.train_batch_size=4 \
    model.enable_gradient_checkpointing=True
```

---

## Troubleshooting

### OOM Error
```bash
# Reduce batch size and enable gradient checkpointing
bash sft/finetune_ft.sh _2_sokoban 1 \
    ./sftckpt/... \
    ./sftdata/... \
    1.5B \
    data.train_batch_size=2 \
    data.micro_batch_size_per_gpu=1 \
    model.enable_gradient_checkpointing=True
```

### JAVA_HOME Error (for PPO)
```bash
# Install Java
sudo apt update && sudo apt install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

---

## Expected Output

After SFT training:
```
sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/
└── global_step_XXXX/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

After PPO training:
```
outputs/
└── _2_sokoban-1.5B-RENDER_MODEtext_with_coordinates-spa-.../
    ├── checkpoints/
    └── logs/
```
