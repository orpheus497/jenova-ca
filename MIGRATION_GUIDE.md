# Migration Guide: v3.1.x → v3.2.0

## Overview

Version 3.2.0 introduces a major architectural change, migrating from HuggingFace transformers to llama-cpp-python with GGUF models. This guide helps you migrate smoothly.

## What's Changed

### Core Changes
- **Model System**: HuggingFace transformers → llama-cpp-python (GGUF)
- **Installation**: System-wide → Local virtualenv
- **Model Source**: Auto-download → User-provided GGUF files
- **Fine-tuning**: Active system → Deprecated (RAG-based personalization)
- **Document Processing**: Auto-insight generation → Canonical storage

### What's Preserved
- ✅ User data directory (`~/.jenova-ai/`)
- ✅ Memory databases (ChromaDB)
- ✅ Cognitive graph structure
- ✅ Embedding model (sentence-transformers)
- ✅ All commands (except /train deprecated)
- ✅ RAG and memory systems

## Migration Steps

### 1. Backup Your Data

```bash
# Backup your user data
cp -r ~/.jenova-ai ~/.jenova-ai.backup

# Verify backup
ls -la ~/.jenova-ai.backup
```

### 2. Uninstall Old Version

```bash
# Remove system-wide installation (if installed with sudo)
sudo pip uninstall jenova-ai

# Or remove user installation
pip uninstall jenova-ai
```

### 3. Update Repository

```bash
cd jenova-ca
git pull origin main
```

### 4. Install New Version

```bash
# Run installer (no sudo needed!)
./install.sh

# This will:
# - Create a Python virtualenv in ./venv/
# - Install all dependencies
# - Build llama-cpp-python (with CUDA if GPU detected)
# - Create models/ directory
```

### 5. Download a GGUF Model

Choose a model based on your hardware:

**For Testing (2-4GB RAM):**
```bash
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
cd ..
```

**For Production (8GB+ RAM):**
```bash
cd models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O model.gguf
cd ..
```

See `models/README.md` for more model options.

### 6. Update Configuration

Edit `src/jenova/config/main_config.yaml`:

```yaml
model:
  model_path: './models/model.gguf'  # Update if needed
  threads: 8                          # Adjust for your CPU
  gpu_layers: -1                      # -1 for all, 0 for CPU only
  mlock: true
  n_batch: 512
  context_size: 2048
  max_tokens: 512
  temperature: 0.7
  top_p: 0.95
  embedding_model: 'all-MiniLM-L6-v2'
```

**GPU Settings:**
- `gpu_layers: -1` → Offload all layers to GPU (fastest with NVIDIA GPU)
- `gpu_layers: 0` → CPU only (no GPU)
- `gpu_layers: 20` → Offload 20 layers to GPU (mixed mode)

### 7. Run JENOVA

```bash
# Activate virtualenv
source venv/bin/activate

# Run JENOVA
python -m jenova.main

# Or use the CLI entry point
jenova
```

## Troubleshooting

### Model Not Found

**Error:** `Model file not found: ./models/model.gguf`

**Solution:**
1. Download a GGUF model (see step 5 above)
2. Place it in `models/` directory
3. Update `model_path` in config

### GPU Not Detected

**Error:** No CUDA support / slow inference

**Solution:**
1. Install CUDA toolkit: `sudo dnf install cuda-toolkit`
2. Reinstall llama-cpp-python with CUDA:
   ```bash
   source venv/bin/activate
   CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```
3. Set `gpu_layers: -1` in config

### Out of Memory

**Error:** CUDA out of memory / System RAM exhausted

**Solution:**
1. Use a smaller model (e.g., TinyLlama instead of Mistral)
2. Reduce `gpu_layers` (try 20-30 instead of -1)
3. Set `mlock: false` to allow OS to swap
4. Reduce `context_size` in config

### Import Error: llama_cpp

**Error:** `ModuleNotFoundError: No module named 'llama_cpp'`

**Solution:**
```bash
source venv/bin/activate
pip install llama-cpp-python
```

### Virtualenv Issues

**Error:** Command not found / module not found

**Solution:**
Always activate the virtualenv first:
```bash
source venv/bin/activate
python -m jenova.main
```

## Feature Changes

### Fine-Tuning (Deprecated)

The `/train` command is deprecated. GGUF models are pre-quantized and not directly fine-tunable.

**Alternatives for Personalization:**
- ✅ Semantic Memory: Your interactions are stored and retrieved contextually
- ✅ RAG System: Documents enhance responses with relevant context
- ✅ Cognitive Graph: Insights and memories provide deep personalization

**For Advanced Users:**
If you need to fine-tune:
1. Fine-tune in native format (PyTorch, safetensors)
2. Convert to GGUF after training
3. Use external tools (llama.cpp training, Axolotl)

See `finetune/DEPRECATED.md` for details.

### Document Processing

Documents are now stored as canonical source knowledge without automatic insight generation.

**Before (v3.1.x):**
- Documents analyzed for insights automatically
- Insights extracted and stored separately

**After (v3.2.0):**
- Documents stored as-is in cognitive graph
- Available for RAG retrieval
- Manual analysis via commands if needed

## Validation

### Test Your Installation

```bash
# 1. Activate virtualenv
source venv/bin/activate

# 2. Check dependencies
pip list | grep -E "llama-cpp|sentence-transformers|chromadb"

# 3. Verify model exists
ls -lh models/*.gguf

# 4. Test JENOVA
python -m jenova.main

# 5. Try basic commands
# In JENOVA:
# - Have a conversation
# - Try /help
# - Try /reflect
# - Add a document to src/jenova/docs/ and use /develop_insight
```

### Verify GPU Usage (if applicable)

```bash
# In another terminal while JENOVA is running:
nvidia-smi

# You should see:
# - jenova process listed
# - GPU memory usage
# - GPU utilization %
```

## Rollback

If you need to rollback to v3.1.x:

```bash
# 1. Restore user data if needed
rm -rf ~/.jenova-ai
mv ~/.jenova-ai.backup ~/.jenova-ai

# 2. Checkout old version
git checkout v3.1.1

# 3. Reinstall
sudo ./install.sh
```

## Getting Help

- Check `models/README.md` for model recommendations
- Check `finetune/DEPRECATED.md` for fine-tuning alternatives
- Check `CHANGELOG.md` for detailed changes
- Open an issue on GitHub for problems

## Summary

✅ **What You Get:**
- Faster inference with optimized GGUF models
- Better GPU support
- Full control over models (no auto-downloads)
- Isolated virtualenv environment
- Enhanced privacy (no external API calls)

✅ **What You Keep:**
- All your user data and memories
- All existing features (RAG, memory, cognitive engine)
- All commands and UI
- Your conversation history and insights

---

**Migration Time:** ~10-15 minutes (plus model download time)

**Recommended:** Start with TinyLlama for testing, then upgrade to larger models based on your hardware.
