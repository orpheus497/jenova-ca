#!/bin/bash
# JENOVA AI System-Wide Installation Script
# This script installs JENOVA AI for all users on the system.
# It must be run with root privileges (e.g., using 'sudo').

set -e

echo "--- Installing JENOVA AI for All Users ---"

# 1. Check for Root Privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] This script must be run with root privileges."
    echo "Please run it again using: sudo ./install.sh"
    exit 1
fi

# 2. Verify Dependencies
echo "--> Checking for dependencies (python3, pip, git)..."
if ! command -v python3 &> /dev/null || ! command -v pip &> /dev/null || ! command -v git &> /dev/null; then
    echo "[ERROR] Missing essential dependencies. Please ensure python3, python3-pip, and git are installed."
    exit 1
fi

# 3. Upgrade Pip
echo "--> Upgrading system's pip..."
pip install --upgrade pip > /dev/null

# 4. Create system-wide model directory
echo "--> Creating model directory at /usr/local/share/jenova-ai/models..."
mkdir -p /usr/local/share/jenova-ai/models
chmod 755 /usr/local/share/jenova-ai
chmod 755 /usr/local/share/jenova-ai/models

# 5. Download Gemma 3 4B (NoVision) model
echo "--> Downloading Gemma 3 4B (NoVision) model from HuggingFace..."
echo "    This may take a few minutes..."

# Clear Hugging Face cache to ensure clean download
echo "    Clearing Hugging Face cache..."
rm -rf ~/.cache/huggingface/

# Install specific version of transformers and torch to ensure compatibility
echo "    Installing transformers 4.42.3 and dependencies..."
pip install torch accelerate transformers==4.42.3 > /dev/null 2>&1

# Force-reinstall a compatible tokenizer version
echo "    Ensuring compatible tokenizers version..."
pip uninstall -y tokenizers > /dev/null 2>&1 || true
pip install --ignore-installed "tokenizers>=0.14,<0.19"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install compatible tokenizers version."
    exit 1
fi

# Download model using Python
python3 << 'PYTHON_SCRIPT'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/usr/local/share/jenova-ai/models"
model_name = "gghfez/gemma-3-4b-novision"

print(f"Downloading {model_name}...")
try:
    # Download tokenizer
    print("  - Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir, trust_remote_code=True)
    
    # Download model
    print("  - Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    print(f"✓ Model successfully downloaded to {model_dir}")
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    exit(1)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to download Gemma 3 4B (NoVision) model."
    exit 1
fi

# 6. Install the Package
echo "--> Installing JENOVA AI package globally..."
if ! pip install --ignore-installed .; then
    echo "[ERROR] Installation failed. Please check setup.py and ensure all dependencies can be installed."
    exit 1
fi

echo
echo "======================================================================"
echo "✅ JENOVA AI has been successfully installed for all users."
echo
echo "Model: Gemma 3 4B (NoVision)"
echo "Location: /usr/local/share/jenova-ai/models"
echo
echo "Any user can now run the application by typing:"
echo "  jenova"
echo
echo "User-specific data, memories, and insights will be stored at:"
echo "  ~/.jenova-ai/users/<username>/"
echo "======================================================================"
