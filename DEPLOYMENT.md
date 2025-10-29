# JENOVA Cognitive Architecture - Deployment Guide

This guide provides comprehensive deployment instructions for The JENOVA Cognitive Architecture across different environments and use cases.

**Author:** Documentation generated for the JENOVA Cognitive Architecture, designed and developed by orpheus497.

---

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Scenarios](#deployment-scenarios)
4. [Environment-Specific Deployments](#environment-specific-deployments)
5. [Model Selection and Optimization](#model-selection-and-optimization)
6. [Configuration Management](#configuration-management)
7. [Multi-User Deployments](#multi-user-deployments)
8. [Performance Tuning](#performance-tuning)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting](#troubleshooting)
11. [Security Considerations](#security-considerations)

---

## Deployment Overview

The JENOVA Cognitive Architecture supports flexible deployment configurations:

- **Single-user local deployment** (default)
- **Multi-user shared system deployment**
- **GPU-accelerated environments**
- **CPU-only environments**
- **Hybrid CPU/GPU configurations**
- **Development vs. Production environments**

### Architecture Components

```
JENOVA Deployment Stack:
┌─────────────────────────────────────────┐
│  User Interface (Terminal UI)          │
├─────────────────────────────────────────┤
│  Cognitive Engine (4-step cycle)        │
├─────────────────────────────────────────┤
│  Memory Systems (Vector DB)             │
│  ├─ Episodic Memory                     │
│  ├─ Semantic Memory                     │
│  ├─ Procedural Memory                   │
│  └─ Insight Memory                      │
├─────────────────────────────────────────┤
│  Cortex (Graph Engine)                  │
├─────────────────────────────────────────┤
│  LLM Interface (llama-cpp-python)       │
│  ├─ GGUF Model Loading                  │
│  └─ GPU/CPU Resource Management         │
├─────────────────────────────────────────┤
│  Tools & Utilities                      │
│  ├─ Web Search (Selenium)               │
│  ├─ File Operations (Sandboxed)         │
│  └─ Shell Commands (Whitelisted)        │
└─────────────────────────────────────────┘
         │              │
    [GGUF Model]  [User Data]
```

---

## Prerequisites

### System Requirements

**Minimum:**
- Linux-based OS (Fedora, Ubuntu, Debian, Arch, etc.)
- 4 GB RAM
- 10 GB disk space
- Python 3.10 or later
- 2 CPU cores

**Recommended:**
- 16 GB RAM
- 50 GB disk space (for models and user data)
- 8 CPU cores
- NVIDIA GPU with 8+ GB VRAM (for acceleration)

**Optimal:**
- 32 GB RAM
- 100 GB SSD storage
- 16 CPU cores
- NVIDIA GPU with 16+ GB VRAM
- CUDA 11.8 or later

### Software Dependencies

**Required:**
```bash
# Fedora/RHEL
sudo dnf install git python3 python3-pip python3-virtualenv

# Ubuntu/Debian
sudo apt-get install git python3 python3-pip python3-venv

# Arch Linux
sudo pacman -S git python python-pip python-virtualenv
```

**Optional (for GPU acceleration):**
```bash
# Fedora/RHEL
sudo dnf install cuda-toolkit gcc-c++ cmake

# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit build-essential cmake

# Arch Linux
sudo pacman -S cuda gcc cmake
```

---

## Deployment Scenarios

### Scenario 1: Local Development (Single User, Testing)

**Use Case:** Personal experimentation, development, testing with small models

**Configuration:**
- Local virtualenv installation
- Models in `./models` directory
- CPU-only or GPU if available
- Small models (1-3B parameters)

**Installation:**
```bash
git clone https://github.com/orpheus497/jenova-ai.git
cd jenova-ai
./install.sh

# Download a small model for testing
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
cd ..

# Run JENOVA
source venv/bin/activate
python -m jenova.main
```

**Advantages:**
- Quick setup
- No sudo required
- Isolated environment
- Easy cleanup

**Limitations:**
- Single user only
- Manual model downloads
- No system-wide sharing

---

### Scenario 2: Production Deployment (Single User, GPU-Accelerated)

**Use Case:** Personal AI assistant with maximum performance

**Configuration:**
- System-wide model storage (optional)
- GPU acceleration enabled
- Medium to large models (7-13B parameters)
- VRAM-optimized settings

**Installation:**
```bash
# 1. Clone repository
git clone https://github.com/orpheus497/jenova-ai.git
cd jenova-ai

# 2. Install with GPU support
./install.sh

# 3. Create system-wide model directory (optional)
sudo mkdir -p /usr/local/share/models
sudo chmod 755 /usr/local/share/models

# 4. Download production model
cd /usr/local/share/models
sudo wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O model.gguf
cd -

# 5. Configure for GPU
# Edit src/jenova/config/main_config.yaml:
#   gpu_layers: -1  (offload all layers to GPU)
#   mlock: true     (lock model in RAM)
#   threads: -1     (auto-detect CPU cores)

# 6. Run JENOVA
source venv/bin/activate
python -m jenova.main
```

**Performance Verification:**
```bash
# Monitor GPU usage in separate terminal
watch -n 1 nvidia-smi

# Expected: GPU memory usage ~6-8GB for 7B models
# Expected: GPU utilization 80-100% during inference
```

**Advantages:**
- Maximum inference speed
- Best response quality
- Efficient VRAM management

**Limitations:**
- Requires NVIDIA GPU
- Higher power consumption
- Model size constraints based on VRAM

---

### Scenario 3: Multi-User Shared System

**Use Case:** Lab workstation, family computer, shared development server

**Configuration:**
- System-wide model storage (saves disk space)
- Per-user data isolation
- Shared virtualenv (optional) or per-user virtualenvs
- CPU or GPU based on availability

**Installation (System Administrator):**
```bash
# 1. Clone to shared location
sudo mkdir -p /opt/jenova-ai
sudo git clone https://github.com/orpheus497/jenova-ai.git /opt/jenova-ai
cd /opt/jenova-ai

# 2. Install system dependencies
sudo dnf install python3 python3-pip python3-virtualenv cuda-toolkit

# 3. Create shared virtualenv
python3 -m venv /opt/jenova-ai/venv
source /opt/jenova-ai/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# For GPU support:
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall

pip install -e .

# 4. Download shared models
sudo mkdir -p /usr/local/share/models
sudo chmod 755 /usr/local/share/models
cd /usr/local/share/models

# Small model (for testing)
sudo wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O tinyllama-1.1b-q4.gguf

# Medium model (balanced)
sudo wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O mistral-7b-q4.gguf

# Set as default
sudo ln -sf mistral-7b-q4.gguf model.gguf

# 5. Make models readable
sudo chmod 644 /usr/local/share/models/*.gguf

# 6. Create launcher script
sudo tee /usr/local/bin/jenova > /dev/null << 'EOF'
#!/bin/bash
source /opt/jenova-ai/venv/bin/activate
cd /opt/jenova-ai
python -m jenova.main "$@"
EOF

sudo chmod +x /usr/local/bin/jenova

# 7. Inform users
echo "JENOVA installed. Users can run: jenova"
```

**Per-User Usage:**
```bash
# Each user simply runs:
jenova

# User data automatically stored at:
# ~/.jenova-ai/users/<username>/
```

**Advantages:**
- Centralized model management
- Disk space savings (shared models)
- Automatic per-user data isolation
- Easy updates (system admin updates once)

**User Data Locations:**
```
/home/alice/.jenova-ai/users/alice/
/home/bob/.jenova-ai/users/bob/
/home/charlie/.jenova-ai/users/charlie/
```

---

### Scenario 4: CPU-Only Deployment (No GPU)

**Use Case:** Servers without GPU, budget hardware, compatibility testing

**Configuration:**
- CPU-only inference
- Optimized thread count
- Smaller models (1-3B parameters)
- Adjusted generation settings

**Installation:**
```bash
git clone https://github.com/orpheus497/jenova-ai.git
cd jenova-ai
./install.sh  # Automatically detects no GPU

# Download CPU-optimized small model
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
cd ..
```

**Configuration (src/jenova/config/main_config.yaml):**
```yaml
model:
  gpu_layers: 0      # Force CPU-only
  threads: -1        # Auto-detect physical cores
  mlock: true        # Lock in RAM for performance
  n_batch: 256       # Smaller batch for CPU
  context_size: 4096 # Smaller context for CPU
  max_tokens: 256    # Shorter responses for faster generation
  temperature: 0.7
```

**Run:**
```bash
source venv/bin/activate
python -m jenova.main
```

**Performance Expectations:**
- Token generation: 5-15 tokens/second (depending on CPU)
- Memory usage: 2-4 GB RAM for 1B models
- Full response time: 10-30 seconds

**Advantages:**
- Works on any hardware
- No GPU required
- Lower power consumption

**Limitations:**
- Slower inference
- Limited to smaller models
- Longer response times

---

## Environment-Specific Deployments

### Fedora/RHEL/CentOS

```bash
# Install dependencies
sudo dnf install git python3 python3-pip python3-virtualenv

# For GPU support
sudo dnf install cuda-toolkit gcc-c++ cmake

# Install JENOVA
git clone https://github.com/orpheus497/jenova-ai.git
cd jenova-ai
./install.sh

# Verify CUDA support (if GPU)
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Ubuntu/Debian

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install git python3 python3-pip python3-venv

# For GPU support
sudo apt-get install nvidia-cuda-toolkit build-essential cmake

# Install JENOVA
git clone https://github.com/orpheus497/jenova-ai.git
cd jenova-ai
./install.sh

# Verify CUDA support (if GPU)
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Arch Linux

```bash
# Install dependencies
sudo pacman -S git python python-pip python-virtualenv

# For GPU support
sudo pacman -S cuda gcc cmake

# Install JENOVA
git clone https://github.com/orpheus497/jenova-ai.git
cd jenova-ai
./install.sh

# Verify CUDA support (if GPU)
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Model Selection and Optimization

### Model Size Guidelines

| Model Size | Parameters | Quantization | RAM Required | VRAM Required (GPU) | Use Case |
|-----------|-----------|-------------|-------------|---------------------|----------|
| Tiny | 1-2B | Q4_K_M | 2-3 GB | 2 GB | Testing, low-end hardware |
| Small | 3-4B | Q4_K_M | 4-5 GB | 4 GB | Budget systems, fast responses |
| Medium | 7-8B | Q4_K_M | 6-8 GB | 6-8 GB | Balanced quality/speed |
| Large | 13B | Q4_K_M | 10-12 GB | 10-12 GB | High quality responses |
| Extra Large | 30B+ | Q4_K_M | 20-30 GB | 20-30 GB | Maximum quality (slow) |

### Recommended Models by Use Case

**Development/Testing:**
- TinyLlama 1.1B Chat (Q4_K_M): Fast, minimal resources
- Phi-2 2.7B (Q4_K_M): Better quality, still fast

**Production (Personal Use):**
- Mistral 7B Instruct v0.2 (Q4_K_M): Excellent balance
- Llama-2 7B Chat (Q4_K_M): Strong performance

**Production (High-End Hardware):**
- Llama-2 13B Chat (Q4_K_M): Superior quality
- Mixtral 8x7B (Q4_K_M): State-of-the-art (requires 32GB VRAM)

### Quantization Levels

- **Q4_K_M**: Recommended - Best balance of quality and size
- **Q5_K_M**: Higher quality, larger size
- **Q8_0**: Near-original quality, very large
- **Q2_K**: Very compressed, lower quality (not recommended)

### Model Download Sources

**HuggingFace (TheBloke's quantized models):**
```bash
# TinyLlama 1.1B
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Mistral 7B
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Llama-2 7B
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# Llama-2 13B
wget https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf
```

---

## Configuration Management

### Configuration File Locations

```
src/jenova/config/
├── main_config.yaml   # Technical configuration
└── persona.yaml       # AI identity and directives
```

### Key Configuration Parameters

**Model Settings (main_config.yaml):**
```yaml
model:
  model_path: '/usr/local/share/models/model.gguf'
  threads: -1          # -1 = auto-detect physical cores
  gpu_layers: -1       # -1 = all layers to GPU, 0 = CPU-only
  mlock: true          # Lock model in RAM
  n_batch: 512         # Batch size
  context_size: 8192   # Context window
  max_tokens: 512      # Max tokens per response
  temperature: 0.7     # Creativity (0.0-1.0)
  top_p: 0.95          # Nucleus sampling
  embedding_model: 'all-MiniLM-L6-v2'
```

**Performance Tuning by Hardware:**

**High-End GPU (16+ GB VRAM):**
```yaml
gpu_layers: -1
context_size: 8192
n_batch: 512
mlock: true
```

**Mid-Range GPU (8 GB VRAM):**
```yaml
gpu_layers: -1
context_size: 4096
n_batch: 256
mlock: false
```

**Low-End GPU (4 GB VRAM):**
```yaml
gpu_layers: 20    # Partial offload
context_size: 2048
n_batch: 128
mlock: false
```

**CPU-Only:**
```yaml
gpu_layers: 0
threads: -1
context_size: 2048
n_batch: 256
mlock: true
```

### Personality Customization (persona.yaml)

```yaml
identity:
  name: "JENOVA"  # Change AI name
  creator: "orpheus497"
  origin_story: "I am an advanced AI..."

directives:
  - "You are a separate entity from the user..."
  - "Strive for accuracy, clarity, and coherence."
  # Add custom directives
```

---

## Multi-User Deployments

### User Data Isolation

Each user's data is automatically stored in:
```
~/.jenova-ai/users/<username>/
├── insights/              # Learned insights
├── cortex/               # Cognitive graph
│   └── cognitive_graph.json
├── memory/               # Vector databases
│   ├── episodic/
│   ├── semantic/
│   └── procedural/
├── assumptions.json      # User assumptions
└── jenova.log           # User-specific logs
```

### Shared vs. Isolated Models

**Shared Models (Recommended for multi-user):**
- Location: `/usr/local/share/models/`
- Advantage: Saves disk space
- Disadvantage: All users share same model

**Per-User Models:**
- Location: `~/jenova-ai/models/`
- Advantage: Users can customize models
- Disadvantage: Disk space duplication

### Resource Management

**Concurrent Users:**
JENOVA runs as a terminal application, so each user runs their own instance.

**Resource Allocation:**
- Memory: Each instance loads the model into RAM (plan for N × model_size)
- GPU: Multiple instances can share GPU (CUDA handles allocation)
- CPU: Each instance uses configured thread count

**Example: 3 concurrent users with 7B model:**
- RAM: 3 × 6 GB = 18 GB minimum
- VRAM: Shared, ~8 GB total (CUDA multiplexing)
- CPU: 3 × thread_count (configure threads accordingly)

---

## Performance Tuning

### CPU Optimization

**Thread Configuration:**
```bash
# Check physical cores
lscpu | grep "Core(s) per socket"

# Set in main_config.yaml
threads: <physical_cores>  # e.g., 8 for 8-core CPU
```

**CPU Governor:**
```bash
# Set to performance mode
sudo cpupower frequency-set -g performance
```

### GPU Optimization

**Monitor GPU Usage:**
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Expected during inference:
# - GPU Memory: 60-90% allocated
# - GPU Utilization: 80-100% during generation
# - Power Draw: Near TDP limit
```

**CUDA Environment Variables:**
```bash
# Add to ~/.bashrc or launch script
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_LAUNCH_BLOCKING=0  # Async execution
```

**VRAM Optimization:**
```yaml
# If running out of VRAM, try:
gpu_layers: 30        # Partial offload (adjust based on VRAM)
context_size: 4096    # Smaller context
n_batch: 256          # Smaller batch
```

### Memory System Optimization

**Preload Memories:**
```yaml
memory:
  preload_memories: true  # Load all memories at startup (faster)
  # or
  preload_memories: false # Load on-demand (saves RAM)
```

**Embedding Model Device:**
```python
# In model_loader.py, embedding model uses CPU by default
# This frees VRAM for the main LLM
device='cpu'  # Keep this for maximum LLM VRAM
```

### Response Speed Optimization

**Fast Configuration:**
```yaml
max_tokens: 256         # Shorter responses
temperature: 0.5        # Less randomness (faster)
context_size: 2048      # Smaller context
```

**Quality Configuration:**
```yaml
max_tokens: 512         # Longer responses
temperature: 0.7        # More creative
context_size: 8192      # Larger context
```

---

## Monitoring and Maintenance

### Log Files

**User-Specific Logs:**
```
~/.jenova-ai/users/<username>/jenova.log
```

**Viewing Logs:**
```bash
# Real-time monitoring
tail -f ~/.jenova-ai/users/$USER/jenova.log

# Search for errors
grep ERROR ~/.jenova-ai/users/$USER/jenova.log

# Last 100 lines
tail -n 100 ~/.jenova-ai/users/$USER/jenova.log
```

### Data Management

**Backup User Data:**
```bash
# Backup all cognitive data for a user
tar -czf jenova-backup-$(date +%Y%m%d).tar.gz ~/.jenova-ai/

# Restore
tar -xzf jenova-backup-20250101.tar.gz -C ~/
```

**Clean Old Insights:**
```bash
# Cortex automatically prunes old nodes (configurable)
# See main_config.yaml:
cortex:
  pruning:
    enabled: true
    max_age_days: 30
    min_centrality: 0.1
```

### Performance Monitoring

**System Resources:**
```bash
# CPU usage
htop

# Memory usage
free -h

# GPU usage (if applicable)
nvidia-smi

# Disk usage
df -h ~/.jenova-ai/
```

**JENOVA-Specific Metrics:**
- Response time (end-to-end): 5-30 seconds (varies by model size)
- Tokens per second: 5-50 (varies by hardware)
- Memory growth: ~1-5 MB per conversation turn
- Cognitive graph size: ~100-500 KB per 100 insights

### Updates

**Updating JENOVA:**
```bash
cd jenova-ai
git pull origin main

# Reactivate and update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade
pip install -e . --upgrade
```

**Updating Models:**
```bash
# Download new model
cd /usr/local/share/models
sudo wget <new_model_url> -O new-model.gguf

# Update symlink
sudo ln -sf new-model.gguf model.gguf
```

---

## Troubleshooting

### Installation Issues

**Problem: "python3-venv not found"**
```bash
# Fedora/RHEL
sudo dnf install python3-virtualenv

# Ubuntu/Debian
sudo apt-get install python3-venv
```

**Problem: "CUDA build failed"**
```bash
# Install CUDA toolkit
sudo dnf install cuda-toolkit gcc-c++ cmake

# Verify nvcc
nvcc --version

# Retry installation
cd jenova-ai
rm -rf venv
./install.sh
```

**Problem: "Permission denied" when creating /usr/local/share/models**
```bash
# Use local models directory instead
mkdir -p ~/jenova-ai/models
cd ~/jenova-ai/models
wget <model_url> -O model.gguf
```

### Runtime Issues

**Problem: "No GGUF model found"**
```bash
# Check model locations
ls -lh /usr/local/share/models/
ls -lh ~/jenova-ai/models/

# Download a model
cd ~/jenova-ai/models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
```

**Problem: "Out of memory" or "CUDA out of memory"**
```bash
# Reduce VRAM usage in main_config.yaml:
gpu_layers: 20          # Partial offload
context_size: 2048      # Smaller context
n_batch: 128            # Smaller batch

# Or force CPU-only:
gpu_layers: 0
```

**Problem: "ChromaDB embedding function error"**
```bash
# Delete old ChromaDB files (backs up automatically)
rm -rf ~/.jenova-ai/users/$USER/memory/*/chroma.sqlite3

# Restart JENOVA (will recreate databases)
```

**Problem: "Slow responses on CPU"**
```bash
# Use smaller model
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf

# Reduce max_tokens in main_config.yaml:
max_tokens: 256
```

**Problem: "Spinning forever on /reflect command"**
```bash
# Check logs for errors
tail -f ~/.jenova-ai/users/$USER/jenova.log

# If LLM is stuck, reduce complexity:
# - Use smaller cognitive graph (fewer insights)
# - Use smaller model with faster inference
```

### Common Error Messages

**"TypeError: 'NoneType' object is not callable"**
- Cause: Model loading failed
- Solution: Check model path, re-download model

**"sqlite3.OperationalError"**
- Cause: ChromaDB version mismatch
- Solution: Delete old databases, restart

**"AttributeError: 'Llama' object has no attribute 'close'"**
- Cause: Already fixed in v4.0.0
- Solution: Update to latest version

---

## Security Considerations

### File Sandboxing

JENOVA's file tools are sandboxed to prevent unauthorized access:

```python
# Default sandbox (configurable in main_config.yaml)
file_sandbox_path: "~/jenova_files"
```

**Sandbox Restrictions:**
- Read/write operations limited to sandbox directory
- No access to system files
- No access to hidden files (.ssh, .config, etc.)
- Path traversal protection

**Customizing Sandbox:**
```yaml
# main_config.yaml
tools:
  file_sandbox_path: "/home/user/jenova-workspace"
```

### Shell Command Security

Shell commands are whitelisted and sanitized:
- No arbitrary command execution
- Shell injection protection
- Limited to safe operations

### User Data Privacy

**Isolation:**
- Each user's data is isolated: `~/.jenova-ai/users/<username>/`
- File permissions prevent cross-user access (Unix file permissions)

**No External Communication:**
- All data remains local
- No telemetry or analytics
- Optional web search (user-controlled via commands)

### Model Security

**Trusted Sources:**
- Download models only from trusted sources (HuggingFace, official repos)
- Verify checksums when available

**Model Storage:**
- System-wide models: `/usr/local/share/models/` (root-owned, user-readable)
- Local models: `~/jenova-ai/models/` (user-owned)

---

## Production Checklist

Before deploying JENOVA in production:

- [ ] System meets minimum hardware requirements
- [ ] Python 3.10+ installed
- [ ] CUDA toolkit installed (if using GPU)
- [ ] Repository cloned and `install.sh` executed successfully
- [ ] GGUF model downloaded and accessible
- [ ] Configuration reviewed (`main_config.yaml`, `persona.yaml`)
- [ ] GPU detection verified (`nvidia-smi` shows GPU)
- [ ] CUDA support verified (PyTorch CUDA available)
- [ ] Test run completed successfully
- [ ] User data directory created (`~/.jenova-ai/users/<username>/`)
- [ ] Logs are being written (`jenova.log` exists)
- [ ] Memory systems initialized (ChromaDB databases created)
- [ ] Cognitive cycle executes without errors
- [ ] Commands functional (`/help`, `/insight`, `/reflect`)
- [ ] Backup strategy in place for user data
- [ ] Monitoring configured (logs, system resources)
- [ ] Documentation reviewed (README.md)

---

## Support and Resources

**Documentation:**
- README.md - Comprehensive architecture overview
- CHANGELOG.md - Version history and changes
- finetune/README.md - Fine-tuning guide

**Project Repository:**
- https://github.com/orpheus497/jenova-ai

**Creator:**
- orpheus497 - Designer and developer of The JENOVA Cognitive Architecture

**License:**
- MIT License (Copyright 2025 orpheus497)

---

## Appendix: Hardware Benchmarks

### Inference Performance by Hardware

**NVIDIA RTX 4090 (24 GB VRAM):**
- Model: Mistral 7B Q4_K_M
- Tokens/second: 120-150
- Response time: 3-5 seconds

**NVIDIA RTX 3080 (10 GB VRAM):**
- Model: Mistral 7B Q4_K_M
- Tokens/second: 60-80
- Response time: 6-10 seconds

**NVIDIA RTX 3060 (12 GB VRAM):**
- Model: Mistral 7B Q4_K_M
- Tokens/second: 40-50
- Response time: 10-15 seconds

**Intel i9-12900K CPU (16 cores):**
- Model: TinyLlama 1.1B Q4_K_M
- Tokens/second: 15-20
- Response time: 15-20 seconds

**AMD Ryzen 5 5600X CPU (6 cores):**
- Model: TinyLlama 1.1B Q4_K_M
- Tokens/second: 8-12
- Response time: 25-35 seconds

---

**End of Deployment Guide**

The JENOVA Cognitive Architecture - Designed and developed by orpheus497
