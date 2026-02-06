# SPA on AutoDL (Option A: Pre-trained Checkpoint)

## Step 1: Create Instance on AutoDL

1. Login to https://www.autodl.com
2. Click "租用新实例" (Rent New Instance)
3. Select:
   - **GPU**: A100 40GB or A100 80GB (推荐)
   - **镜像**: PyTorch 2.1+ / CUDA 12.1+
   - **Python**: 3.10 or 3.12

---

## Step 2: SSH into Instance

```bash
# AutoDL provides SSH command like:
ssh -p [PORT] root@[IP]
# Or use JupyterLab web interface
```

---

## Step 3: Setup Environment

```bash
# Clone RAGEN repo
cd /root/autodl-tmp
git clone https://github.com/RAGEN-AI/RAGEN.git
cd RAGEN

# Clone SPA into RAGEN
git clone https://github.com/shiqichen17/SPA.git

# Run setup script
bash scripts/setup_ragen.sh

# Activate environment
source activate ragen
# or: conda activate ragen
```

---

## Step 4: Fix Dependencies (CRITICAL)

```bash
# Uninstall conflicting versions
pip uninstall -y torch torchvision torchaudio
pip uninstall -y vllm flash-attn flash_attn

# Install exact versions for CUDA 12.4
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
pip install vllm==0.8.5.post1

# Install Flash Attention (Linux + CUDA 12 + Python 3.12)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# If Python 3.10, use this instead:
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Verify installation
python -c "import torch; import flash_attn; import vllm; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Step 5: Install Java (Required for PPO)

```bash
# Install Java 21
apt update && apt install -y openjdk-21-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Verify
java -version

# Add to ~/.bashrc for persistence
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
```

---

## Step 6: Download Pre-trained Checkpoint

```bash
cd /root/autodl-tmp/RAGEN/SPA

# Create checkpoint directory
mkdir -p sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen

# Download from HuggingFace (may need mirror in China)
# Option 1: Direct download
huggingface-cli download tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct \
    --local-dir ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/global_step_1505

# Option 2: If HuggingFace is slow, use mirror
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct \
    --local-dir ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/global_step_1505
```

---

## Step 7: Update JAVA_HOME in Script

```bash
# Edit run_spa.sh to fix JAVA_HOME path
sed -i 's|export JAVA_HOME=.*|export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64|' run_spa.sh
```

---

## Step 8: Run PPO Training

```bash
cd /root/autodl-tmp/RAGEN/SPA

# Run SPA with pre-trained checkpoint
bash run_spa.sh _2_sokoban last False
```

---

## Quick Copy-Paste Script

Run this entire block on AutoDL:

```bash
#!/bin/bash
set -e

# === 1. Clone repos ===
cd /root/autodl-tmp
git clone https://github.com/RAGEN-AI/RAGEN.git
cd RAGEN
git clone https://github.com/shiqichen17/SPA.git

# === 2. Setup conda env ===
bash scripts/setup_ragen.sh
source activate ragen

# === 3. Fix dependencies ===
pip uninstall -y torch torchvision torchaudio vllm flash-attn flash_attn -y 2>/dev/null || true
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# === 4. Install Java ===
apt update && apt install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# === 5. Download checkpoint (with China mirror) ===
cd SPA
mkdir -p sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct \
    --local-dir ./sftckpt/checkpoints_2_sokoban-1.5B-text_with_coordinates-qwen/global_step_1505

# === 6. Fix JAVA_HOME in script ===
sed -i 's|export JAVA_HOME=.*|export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64|' run_spa.sh

# === 7. Verify ===
python -c "import torch; import flash_attn; import vllm; print('All OK!')"
java -version

echo "=== Setup Complete! ==="
echo "Run: bash run_spa.sh _2_sokoban last False"
```

---

## Troubleshooting

### HuggingFace Download Slow
```bash
# Use China mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Flash Attention Version Mismatch
```bash
# Check your Python version first
python --version

# For Python 3.10:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# For Python 3.11:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version

# If CUDA 11.x, use different torch:
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### OOM During PPO
```bash
# Edit train_ppo_sfted.sh to reduce batch size
# Find and modify: BT_NUM or batch_size
```

---

## Expected Runtime

| Stage | GPU | Time |
|-------|-----|------|
| Setup | - | ~10 min |
| Download checkpoint | - | ~5 min |
| PPO Training (1000 steps) | A100 80GB | ~6-12 hours |

---

## Cost Estimate

| GPU | AutoDL Price | Est. Total |
|-----|--------------|------------|
| A100 40GB | ~¥3-5/hour | ~¥30-60 |
| A100 80GB | ~¥5-8/hour | ~¥50-100 |
