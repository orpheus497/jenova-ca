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

## System-Wide Installation

For system-wide model storage, place models in /usr/local/share/models (requires sudo).
