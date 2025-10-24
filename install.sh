#!/bin/bash
# JENOVA AI Local Installation Script
# This script creates a Python virtualenv and installs JENOVA AI locally.
# Run without sudo/root privileges.

set -e

echo "======================================================================"
echo "        JENOVA AI - Local Installation (Virtualenv)"
echo "======================================================================"
echo

# 1. Check we're NOT running as root
if [ "$(id -u)" -eq 0 ]; then
    echo "[ERROR] This script should NOT be run with root privileges (sudo)."
    echo "Please run it as a regular user: ./install.sh"
    exit 1
fi

# 2. Verify Dependencies
echo "--> Checking for dependencies (python3, python3-venv, git)..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed."
    echo "Please install python3 (e.g., 'sudo dnf install python3' on Fedora)."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "[ERROR] git is not installed."
    echo "Please install git (e.g., 'sudo dnf install git' on Fedora)."
    exit 1
fi

# Check for venv module
if ! python3 -m venv --help &> /dev/null; then
    echo "[ERROR] python3-venv is not installed."
    echo "Please install it (e.g., 'sudo dnf install python3-virtualenv' on Fedora)."
    exit 1
fi

echo "✓ Dependencies verified."
echo

# 3. Create virtualenv
echo "--> Creating Python virtual environment in ./venv/..."
if [ -d "venv" ]; then
    echo "[INFO] Virtual environment already exists. Removing and recreating..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment."
    exit 1
fi
echo "✓ Virtual environment created."
echo

# 4. Activate virtualenv and upgrade pip
echo "--> Activating virtual environment and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded."
echo

# 5. Install dependencies
echo "--> Installing Python dependencies (this may take a few minutes)..."
echo "    Note: llama-cpp-python will be compiled from source for optimal performance."
echo "    For GPU support, ensure CUDA is installed and CMAKE_ARGS are set."
echo

# Check if user wants GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected. Building llama-cpp-python with CUDA support..."
    echo "       If build fails, install: sudo dnf install gcc-c++ cmake cuda-toolkit"
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir || {
        echo "[WARNING] GPU build failed. Installing CPU-only version..."
        pip install llama-cpp-python
    }
else
    echo "[INFO] No NVIDIA GPU detected. Installing CPU-only version of llama-cpp-python..."
    pip install llama-cpp-python
fi

# Install other dependencies
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi
echo "✓ Dependencies installed."
echo

# 6. Install the package in editable mode
echo "--> Installing JENOVA AI in development mode..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install JENOVA AI package."
    exit 1
fi
echo "✓ JENOVA AI package installed."
echo

# 7. Create models directory
echo "--> Creating models directory..."
mkdir -p models
if [ ! -f "models/.gitkeep" ]; then
    touch models/.gitkeep
fi

# Create README for models directory
cat > models/README.md << 'EOF'
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
EOF

echo "✓ Models directory created with README.md"
echo

# 8. Display completion message
echo
echo "======================================================================"
echo "✅ JENOVA AI installation complete!"
echo "======================================================================"
echo
echo "Next steps:"
echo
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo
echo "2. Download a GGUF model and place it in ./models/"
echo "   See ./models/README.md for download links and instructions."
echo "   Example:"
echo "     cd models"
echo "     wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf"
echo "     cd .."
echo
echo "3. Update the model path in src/jenova/config/main_config.yaml"
echo "   Set model_path to point to your downloaded .gguf file"
echo
echo "4. Run JENOVA:"
echo "   python -m jenova.main"
echo
echo "User data, memories, and insights will be stored at:"
echo "  ~/.jenova-ai/users/<username>/"
echo
echo "======================================================================"
echo
echo "For GPU support on NVIDIA cards:"
echo "  - Install CUDA toolkit: sudo dnf install cuda-toolkit"
echo "  - Set gpu_layers in config to -1 (offload all layers to GPU)"
echo
echo "======================================================================"
