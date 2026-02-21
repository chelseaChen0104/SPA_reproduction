# Complete SPA Reproduction Guide on AutoDL

This guide covers the full workflow from local setup to SFT and PPO training on AutoDL (Chinese cloud GPU platform).

---

## Overview

```
Local Mac                          AutoDL Cloud
─────────────────────────────────────────────────────────────
1. Prepare repo        ──scp──>    2. Setup environment
                                   3. Run SFT (~30 min)
                                   4. Run PPO (~8-16 hours)
5. Pull results        <──git──    Push to GitHub
```

---

## Part 1: Local Preparation (Mac)

### 1.1 Clone/Prepare Your Repo

```bash
cd ~/Documents/GitHub
git clone https://github.com/YOUR_USERNAME/SPA_reproduction.git
cd SPA_reproduction
```

### 1.2 Clone Required Submodules

```bash
# Clone ragen (training framework)
git clone https://github.com/RAGEN-AI/RAGEN.git ragen

# Clone verl inside ragen
cd ragen
git clone https://github.com/volcengine/verl.git verl
cd ..

# Clone SPA original code
git clone https://github.com/shiqichen17/SPA.git spa_original
```

### 1.3 Remove Nested .git Folders (Important!)

```bash
rm -rf ragen/.git spa_original/.git ragen/verl/.git
```

### 1.4 Fix verl Structure

The verl repo has nested structure. Fix it:

```bash
mv ragen/verl/verl/* ragen/verl/
# Or after cloning, the correct verl package should be at: ragen/verl/
```

---

## Part 2: Upload to AutoDL

### 2.1 Create AutoDL Instance

1. Go to https://www.autodl.com
2. Click "租用新实例" (Rent New Instance)
3. Select:
   - **GPU**: A100 40GB/80GB or A800 (recommended)
   - **镜像**: PyTorch 2.1+ / CUDA 12.1+
   - **Python**: 3.12

### 2.2 Upload via SCP

On your **Mac terminal** (not cloud):

```bash
cd ~/Documents/GitHub
scp -rP [PORT] SPA_reproduction root@[HOST]:/root/autodl-tmp/
```

Replace `[PORT]` and `[HOST]` with your AutoDL SSH details.

**Example:**
```bash
scp -rP 51479 SPA_reproduction root@connect.nma1.seetacloud.com:/root/autodl-tmp/
```

---

## Part 3: Environment Setup on AutoDL

SSH into your instance:
```bash
ssh -p [PORT] root@[HOST]
```

### 3.1 Set Chinese Mirrors (Critical for China servers)

```bash
# HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# Pip mirror
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 Check PyTorch/CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

If PyTorch works, skip to 3.4.

### 3.3 Install PyTorch (if needed)

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3.4 Install Flash Attention

```bash
pip install flash-attn==2.7.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**If build fails (out of memory):**
- The build process needs lots of RAM
- Try a pre-built wheel or use a larger instance

### 3.5 Install verl/ragen

```bash
cd /root/autodl-tmp/SPA_reproduction/ragen
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.6 Verify Installation

```bash
python -c "from verl.utils.torch_functional import get_cosine_schedule_with_warmup; print('OK')"
python -c "import flash_attn; print(flash_attn.__version__)"
```

### 3.7 Install Java (Required for PPO)

```bash
apt update && apt install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
```

---

## Part 4: Download Data and Model

### 4.1 Download Training Data

```bash
cd /root/autodl-tmp/SPA_reproduction/spa_original
mkdir -p sftdata/_2_sokoban-1.5B-text_with_coordinates

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download tyzhu/SPA-sokoban-data \
    --repo-type dataset \
    --local-dir ./sftdata/_2_sokoban-1.5B-text_with_coordinates/
```

### 4.2 Download Base Model

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
```

---

## Part 5: Run SFT Training

### 5.1 Create Checkpoint Directory

```bash
cd /root/autodl-tmp/SPA_reproduction/spa_original
mkdir -p sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen
```

### 5.2 Run SFT

```bash
bash sft/finetune_ft.sh _2_sokoban 1 \
    ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/ \
    ./sftdata/_2_sokoban-1.5B-text_with_coordinates \
    1.5B \
    model.partial_pretrain=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
```

**Parameters:**
- `_2_sokoban`: Environment config name
- `1`: Number of GPUs
- 3rd arg: Checkpoint save path
- 4th arg: Data path
- `1.5B`: Model size

**Expected Time:** ~30-60 minutes on A100/A800

**Expected Output:**
```
Epoch 5/5: 100%|██████████| 300/300
val/loss: ~2.9-3.0
```

### 5.3 Verify Checkpoint

```bash
ls sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/
# Should see: global_step_1500/
```

---

## Part 6: Run PPO Training

### 6.1 Fix JAVA_HOME in Script

```bash
sed -i 's|export JAVA_HOME=.*|export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64|' run_spa.sh
```

### 6.2 Run PPO

```bash
cd /root/autodl-tmp/SPA_reproduction/spa_original
bash run_spa.sh _2_sokoban last False
```

**Parameters:**
- `_2_sokoban`: Environment config
- `last`: Use latest SFT checkpoint
- `False`: Don't regenerate SFT data

**Expected Time:** ~8-16 hours on single A100/A800

### 6.3 Run in Background (Recommended)

```bash
nohup bash run_spa.sh _2_sokoban last False > ppo_train.log 2>&1 &

# Check progress
tail -f ppo_train.log
```

---

## Part 7: Push Results to GitHub

### 7.1 Configure Git

```bash
cd /root/autodl-tmp/SPA_reproduction
git config --global user.email "your-email@example.com"
git config --global user.name "your-username"
git config --global --add safe.directory /root/autodl-tmp/SPA_reproduction
```

### 7.2 Remove Nested Git Repos

```bash
rm -rf spa_original/.git ragen/.git ragen/verl/.git 2>/dev/null
```

### 7.3 Initialize and Push

```bash
git init
git remote add origin https://YOUR_USERNAME@github.com/YOUR_USERNAME/SPA_reproduction.git
git add .
git commit -m "SFT training complete"
git branch -M main
git push -u origin main
```

When prompted for password, use a **GitHub Personal Access Token**:
1. Go to https://github.com/settings/tokens
2. Generate new token (classic) with `repo` scope
3. Use token as password

---

## Part 8: Pull Results Locally

On your Mac:

```bash
cd ~/Documents/GitHub/SPA_reproduction
git pull origin main
```

---

## Troubleshooting

### HuggingFace Connection Failed
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Flash Attention Build Killed
- Out of memory during compilation
- Use pre-built wheel or downgrade to torch 2.6.0

### "No module named 'verl.utils'"
- verl not installed correctly
- Check: `ls ragen/verl/` should have `utils/`, `trainer/`, etc.
- Reinstall: `cd ragen && pip install -e .`

### Git "dubious ownership"
```bash
git config --global --add safe.directory /root/autodl-tmp/SPA_reproduction
```

### Git "does not have a commit checked out"
```bash
rm -rf spa_original/.git ragen/.git ragen/verl/.git
```

---

## Expected Results

After PPO training (~1000 steps):

| Metric | Value |
|--------|-------|
| SFT Val Loss | ~2.9-3.0 |
| Success Rate (before PPO) | ~25% |
| Success Rate (after PPO) | ~60% |

---

## File Structure

```
SPA_reproduction/
├── ragen/                    # Training framework
│   └── verl/                 # verl package (not nested!)
├── spa_original/             # SPA code
│   ├── sft/                  # SFT trainer
│   ├── sftdata/              # Training data
│   ├── sftckpt/              # Checkpoints
│   ├── run_spa.sh            # Main script
│   └── train_ppo_sfted.sh    # PPO script
├── scripts/                  # Helper scripts
├── data/                     # Downloaded data
└── AUTODL_COMPLETE_GUIDE.md  # This guide
```

---

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Check verl
python -c "from verl.utils.torch_functional import get_cosine_schedule_with_warmup; print('OK')"

# Monitor training
tail -f sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/train.log

# Monitor PPO
tail -f ppo_train.log
```
