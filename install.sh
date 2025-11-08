#!/bin/bash
# JENOVA AI Local Installation Script
# This script creates a Python virtualenv and installs JENOVA AI locally.
# Run without sudo/root privileges.

set -euo pipefail

echo "======================================================================"
echo "        JENOVA AI - Local Installation (Virtualenv)"
echo "======================================================================"
echo

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# 1. Check we're NOT running as root
if [ "$(id -u)" -eq 0 ]; then
    print_error "This script should NOT be run with root privileges (sudo)."
    echo "Please run it as a regular user: ./install.sh"
    exit 1
fi

# 2. Verify Dependencies
echo "--> Checking for dependencies (python3, python3-venv, git)..."
if ! command -v python3 &> /dev/null; then
    print_error "python3 is not installed."
    echo "Please install python3 (e.g., 'sudo dnf install python3' on Fedora)."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

print_info "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10 or higher is required. You have Python $PYTHON_VERSION"
    echo "Please upgrade your Python installation."
    exit 1
fi

if [ "$PYTHON_MINOR" -ge 13 ]; then
    print_warning "Python 3.13 detected. This requires torch>=2.5.1 (will be installed automatically)"
fi

if ! command -v git &> /dev/null; then
    print_error "git is not installed."
    echo "Please install git (e.g., 'sudo dnf install git' on Fedora)."
    exit 1
fi

# Check for venv module
if ! python3 -m venv --help &> /dev/null; then
    print_error "python3-venv is not installed."
    echo "Please install it (e.g., 'sudo dnf install python3-virtualenv' on Fedora)."
    exit 1
fi

print_success "Dependencies verified."
echo

# 3. Create virtualenv
echo "--> Creating Python virtual environment in ./venv/..."
if [ -d "venv" ]; then
    print_info "Virtual environment already exists. Removing and recreating..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment."
    exit 1
fi
print_success "Virtual environment created."
echo

# 4. Activate virtualenv and upgrade pip
echo "--> Activating virtual environment and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Pip upgraded."
echo

# 5. Detect CUDA and set build flags
echo "======================================================================"
echo "                    GPU/CUDA DETECTION"
echo "======================================================================"
echo

USE_CUDA=0
CUDA_VERSION=""
CUDA_CMAKE_ARGS=""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected."

    # Check for CUDA compiler (nvcc)
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        print_success "CUDA toolkit found: version $CUDA_VERSION"

        # Check for required CUDA libraries
        if ldconfig -p | grep -q libcuda.so && ldconfig -p | grep -q libcudart.so; then
            print_success "CUDA libraries detected"
            USE_CUDA=1

            # Modern llama.cpp uses GGML_CUDA (not LLAMA_CUDA)
            # Also enable CUBLAS for better performance
            CUDA_CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUBLAS=on"

            print_success "GPU acceleration will be enabled"
            print_info "Using CMAKE_ARGS: $CUDA_CMAKE_ARGS"
        else
            print_warning "CUDA toolkit found but libraries not in ldconfig cache"
            print_info "Falling back to CPU-only installation"
        fi
    else
        print_warning "CUDA compiler (nvcc) not found"
        print_info "For GPU support, install: sudo dnf install cuda-toolkit"
        print_info "Installing CPU-only version for now..."
    fi
else
    print_info "No NVIDIA GPU detected (nvidia-smi not found)"
    print_info "Installing CPU-only version"
fi

echo
echo "======================================================================"
echo "                  INSTALLING DEPENDENCIES"
echo "======================================================================"
echo

# 6. Install llama-cpp-python with appropriate flags
echo "--> Installing llama-cpp-python..."
print_info "This may take 3-10 minutes depending on your system..."
echo

if [ $USE_CUDA -eq 1 ]; then
    print_info "Building llama-cpp-python with CUDA support..."
    print_info "This requires compiling C++ code and may take several minutes..."
    echo

    # Set environment variables for CUDA build
    export CMAKE_ARGS="$CUDA_CMAKE_ARGS"
    export FORCE_CMAKE=1

    # Try to build with CUDA, with fallback to CPU
    if CMAKE_ARGS="$CUDA_CMAKE_ARGS" FORCE_CMAKE=1 pip install 'llama-cpp-python>=0.3.0,<0.4.0' --no-cache-dir --verbose 2>&1 | tee /tmp/llama_build.log; then
        print_success "llama-cpp-python built with CUDA support"

        # Verify CUDA was actually enabled
        if grep -q "GGML_CUDA" /tmp/llama_build.log || grep -q "CUBLAS" /tmp/llama_build.log; then
            print_success "CUDA acceleration confirmed in build"
        else
            print_warning "Build succeeded but CUDA may not be enabled"
            print_info "The installation will work but may run on CPU only"
        fi
    else
        print_warning "CUDA build failed. This may be due to:"
        echo "  - Missing CUDA development files (cuda-toolkit)"
        echo "  - Incompatible CUDA/GCC versions"
        echo "  - Insufficient permissions"
        echo
        print_info "Falling back to CPU-only build..."
        echo

        # Unset CUDA flags and try CPU build
        unset CMAKE_ARGS
        unset FORCE_CMAKE

        if pip install 'llama-cpp-python>=0.3.0,<0.4.0' --no-cache-dir; then
            print_success "llama-cpp-python installed (CPU-only mode)"
            print_warning "To enable GPU support later:"
            echo "  1. Install CUDA toolkit: sudo dnf install cuda-toolkit gcc-c++ cmake"
            echo "  2. Reinstall llama-cpp-python: pip uninstall llama-cpp-python"
            echo "  3. Rebuild: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-cache-dir"
        else
            print_error "Failed to install llama-cpp-python even in CPU mode"
            echo
            echo "Troubleshooting steps:"
            echo "  1. Ensure you have build essentials: sudo dnf install gcc-c++ cmake"
            echo "  2. Check /tmp/llama_build.log for detailed error messages"
            echo "  3. Try installing an older version: pip install 'llama-cpp-python==0.2.90'"
            exit 1
        fi
    fi
else
    print_info "Installing llama-cpp-python (CPU-only mode)..."

    if pip install 'llama-cpp-python>=0.3.0,<0.4.0' --no-cache-dir; then
        print_success "llama-cpp-python installed (CPU-only)"
    else
        print_warning "Failed to install latest version, trying 0.2.90..."
        if pip install 'llama-cpp-python==0.2.90' --no-cache-dir; then
            print_success "llama-cpp-python 0.2.90 installed (CPU-only)"
        else
            print_error "Failed to install llama-cpp-python"
            echo "Please ensure you have build tools: sudo dnf install gcc-c++ cmake"
            exit 1
        fi
    fi
fi

# Clean up build log
rm -f /tmp/llama_build.log

echo
echo "--> Installing remaining dependencies from requirements.txt..."
echo
print_info "Installing core dependencies with version constraints..."
print_info "This ensures compatibility and prevents breaking changes"
echo

# Install with explicit version constraints to avoid conflicts
if pip install -r requirements.txt; then
    print_success "Core dependencies installed successfully"
else
    print_error "Failed to install dependencies from requirements.txt"
    echo
    echo "This usually indicates one of the following issues:"
    echo "  1. Network connectivity problems"
    echo "  2. Missing system libraries (build-essential, cmake, etc.)"
    echo "  3. Incompatible dependency versions"
    echo
    echo "Troubleshooting steps:"
    echo "  1. Check internet: ping pypi.org"
    echo "  2. Install build tools: sudo dnf install gcc-c++ cmake python3-devel"
    echo "  3. Try with verbose output: pip install -r requirements.txt --verbose"
    echo "  4. Check specific package errors and install individually"
    echo "  5. Ensure protobuf<5.0.0 and numpy<2.0.0 constraints are met"
    exit 1
fi

print_success "Dependencies installed."
echo

# Verify critical dependencies
print_info "Verifying critical dependency versions..."
python3 -c "import sys; from google.protobuf import __version__ as pb_ver; sys.exit(0 if pb_ver.split('.')[0] < '5' else 1)" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "protobuf version is compatible (<5.0.0)"
else
    print_warning "Could not verify protobuf version or it may be >=5.0.0"
    print_info "If you encounter gRPC issues, reinstall: pip install 'protobuf>=4.25.2,<5.0.0' --force-reinstall"
fi

python3 -c "import sys; import numpy as np; sys.exit(0 if np.__version__.split('.')[0] < '2' else 1)" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "numpy version is compatible (<2.0.0)"
else
    print_warning "Could not verify numpy version or it may be >=2.0.0"
    print_info "If you encounter ML package issues, reinstall: pip install 'numpy>=1.26.4,<2.0.0' --force-reinstall"
fi
echo

# 7. Install the package in editable mode
echo "--> Installing JENOVA AI in development mode..."
print_info "This will install the jenova package and compile Protocol Buffers..."
echo

pip install -e .

if [ $? -ne 0 ]; then
    print_error "Failed to install JENOVA AI package."
    echo
    echo "Common issues:"
    echo "  - Proto compilation may have failed (this is usually non-fatal)"
    echo "  - Try running manually: python build_proto.py"
    echo "  - Then retry: pip install -e ."
    exit 1
fi

print_success "JENOVA AI package installed."
echo

# 8. Compile Protocol Buffers (if not already done)
echo "--> Ensuring Protocol Buffers are compiled..."
if python3 build_proto.py; then
    print_success "Protocol Buffers compiled successfully"
else
    print_warning "Proto compilation had issues but installation can continue"
    print_info "Distributed features may not work until protos are compiled"
fi
echo

# 9. Set up models directory
echo "======================================================================"
echo "                    MODELS DIRECTORY SETUP"
echo "======================================================================"
echo

# Try to create system-wide directory first
if sudo -n mkdir -p /usr/local/share/models 2>/dev/null; then
    print_success "Created system-wide models directory: /usr/local/share/models"
    MODELS_DIR="/usr/local/share/models"
    sudo chmod 755 /usr/local/share/models

    # Create README in system directory
    sudo tee /usr/local/share/models/README.md > /dev/null << 'EOF'
# JENOVA AI System Models Directory

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
cd /usr/local/share/models
sudo wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
```

## Alternative Local Directory

If you don't have sudo access, models can also be placed in the local ./models directory
in the jenova-ca project folder.
EOF
else
    print_info "Cannot create system-wide directory (no sudo access). Creating local directory..."
    mkdir -p models
    MODELS_DIR="./models"

    # Create README for local models directory
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

## System-Wide Installation

For system-wide model storage, place models in /usr/local/share/models (requires sudo).
EOF
fi

print_success "Models directory ready: $MODELS_DIR"
echo

# 10. Run compatibility test
echo "======================================================================"
echo "                DEPENDENCY COMPATIBILITY CHECK"
echo "======================================================================"
echo
print_info "Running compatibility tests to verify installation..."
echo

if python3 test_compatibility.py; then
    print_success "Compatibility test passed!"
else
    print_warning "Compatibility test found some issues"
    print_info "Review the output above for details"
    print_info "You can run 'python3 test_compatibility.py' anytime to recheck"
fi

# 11. Display completion message
echo
echo "======================================================================"
echo "✅ JENOVA AI installation complete!"
echo "======================================================================"
echo
print_success "Installation Summary:"
echo "  - Python version: $PYTHON_VERSION"
if [ $USE_CUDA -eq 1 ]; then
    echo "  - GPU support: ENABLED (CUDA $CUDA_VERSION)"
else
    echo "  - GPU support: CPU-only mode"
fi
echo "  - Models directory: $MODELS_DIR"
echo

echo "Next steps:"
echo
echo "1. Activate the virtual environment:"
echo "   ${GREEN}source venv/bin/activate${NC}"
echo
echo "2. Download a GGUF model and place it in the models directory:"
if [ "$MODELS_DIR" = "/usr/local/share/models" ]; then
echo "   ${BLUE}System-wide: /usr/local/share/models (recommended)${NC}"
echo "   Example:"
echo "     cd /usr/local/share/models"
echo "     sudo wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf"
echo "   ${BLUE}Alternative local: ./models${NC}"
else
echo "   ${BLUE}Local directory: ./models${NC}"
echo "   Example:"
echo "     cd models"
echo "     wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf"
echo "     cd .."
fi
echo "   See README.md in the models directory for more model options."
echo
echo "3. Run JENOVA (models will be auto-discovered):"
echo "   ${GREEN}python -m jenova.main${NC}"
echo
echo "   The system will automatically find models in:"
echo "   - /usr/local/share/models (system-wide)"
echo "   - ./models (local)"
echo

if [ $USE_CUDA -eq 0 ]; then
    echo "======================================================================"
    echo "                      GPU ACCELERATION"
    echo "======================================================================"
    echo
    print_info "Your installation is CPU-only."
    echo
    echo "To enable GPU support on NVIDIA cards:"
    echo "  1. Install CUDA toolkit:"
    echo "     ${GREEN}sudo dnf install cuda-toolkit gcc-c++ cmake${NC}"
    echo
    echo "  2. Reinstall llama-cpp-python with CUDA:"
    echo "     ${GREEN}source venv/bin/activate${NC}"
    echo "     ${GREEN}pip uninstall llama-cpp-python${NC}"
    echo "     ${GREEN}CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --no-cache-dir${NC}"
    echo
    echo "  3. Set gpu_layers in src/jenova/config/main_config.yaml:"
    echo "     ${GREEN}gpu_layers: -1  # Offload all layers to GPU${NC}"
    echo
fi

echo "User data, memories, and insights will be stored at:"
echo "  ${BLUE}~/.jenova-ai/users/<username>/${NC}"
echo
echo "To verify your installation anytime, run:"
echo "  ${GREEN}python3 test_compatibility.py${NC}"
echo
echo "======================================================================"
echo
print_success "Installation complete! Enjoy using JENOVA AI!"
echo
