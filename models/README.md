# JENOVA AI Models Directory

Place your GGUF model files in this directory.

## Getting a GGUF Model

You can download GGUF models from HuggingFace. Some recommended models:

**Small models (1-3B parameters) - good for testing:**
- TinyLlama 1.1B: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- Phi-2 2.7B: https://huggingface.co/TheBloke/phi-2-GGUF

**Medium models (7-8B parameters) - good balance:**
- Llama-2 7B: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
- Mistral 7B: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

**Large models (13B+ parameters) - best quality, requires more RAM/VRAM:**
- Llama-2 13B: https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF

## Download Example

```bash
# Download a small model (TinyLlama)
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
cd ..
```

## Configuration

After placing a model file, update the `model_path` in `src/jenova/config/main_config.yaml`:

```yaml
model:
  model_path: './models/model.gguf'
```

Make sure the filename matches your downloaded model.

## Model Size Considerations

**RAM Requirements (approximate):**
- 1B model: ~2GB RAM
- 3B model: ~4GB RAM
- 7B model: ~8GB RAM
- 13B model: ~16GB RAM

**GPU Acceleration:**
If you have an NVIDIA GPU with CUDA:
1. Set `gpu_layers: -1` in config to offload all layers to GPU
2. Requires CUDA toolkit installed
3. Significantly improves inference speed

**Quantization:**
Most GGUF models come in different quantization levels:
- Q4_K_M: Good balance of quality and size (recommended)
- Q5_K_M: Better quality, larger size
- Q8_0: Highest quality, largest size
- Q3_K_S: Smallest size, lower quality
