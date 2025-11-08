#!/data/data/com.termux/files/usr/bin/bash

# The JENOVA Cognitive Architecture - Termux Installation Script
# Copyright (c) 2024, orpheus497. All rights reserved.
# Licensed under the MIT License
#
# Supports: Android (Termux), iOS (iSH/a-Shell)
# Platform: Unix-only (ARM architecture)

set -e  # Exit on error

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print section header
print_section() {
    echo
    print_message "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_message "$CYAN" "$1"
    print_message "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
}

# Print error and exit
error_exit() {
    print_message "$RED" "✗ Error: $1"
    exit 1
}

# Print success message
print_success() {
    print_message "$GREEN" "✓ $1"
}

# Print warning message
print_warning() {
    print_message "$YELLOW" "⚠ $1"
}

# Print info message
print_info() {
    print_message "$BLUE" "ℹ $1"
}

# Banner
print_section "JENOVA Cognitive Architecture - Termux Installation"
print_info "Platform: Android/iOS via Termux"
print_info "Architecture: ARM (Mobile devices)"
print_info "Creator: orpheus497"
print_info "License: MIT"
echo

# Step 1: Verify Termux environment
print_section "Step 1: Verify Termux Environment"

if [ -z "$TERMUX_VERSION" ] && [ ! -d "$PREFIX" ]; then
    error_exit "This script is for Termux only! For Linux/macOS, use: ./install.sh"
fi

print_success "Termux environment detected"
print_info "Termux version: ${TERMUX_VERSION:-Unknown}"
print_info "Prefix: ${PREFIX:-/data/data/com.termux/files/usr}"
print_info "Architecture: $(uname -m)"

# Step 2: Update package repositories
print_section "Step 2: Update Package Repositories"

print_info "Updating pkg repositories..."
pkg update -y || error_exit "Failed to update pkg repositories"
print_success "Package repositories updated"

print_info "Upgrading existing packages..."
pkg upgrade -y || print_warning "Some packages failed to upgrade (non-critical)"
print_success "Package upgrade complete"

# Step 3: Install system dependencies
print_section "Step 3: Install System Dependencies"

print_info "Installing Python and build tools..."
REQUIRED_PACKAGES=(
    "python"           # Python 3.10+
    "clang"           # C/C++ compiler
    "cmake"           # Build system
    "ninja"           # Fast build tool
    "git"             # Version control
    "wget"            # File download
    "openssl"         # SSL/TLS support
    "libxml2"         # XML parsing
    "libxslt"         # XSLT support
    "zlib"            # Compression
    "libjpeg-turbo"   # Image processing
    "libpng"          # PNG support
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    print_info "Installing $package..."
    pkg install -y "$package" || print_warning "Failed to install $package (may already be installed)"
done

print_success "System dependencies installed"

# Step 4: Verify Python installation
print_section "Step 4: Verify Python Installation"

if ! command -v python3 &> /dev/null; then
    error_exit "Python 3 not found after installation"
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION installed"

PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    error_exit "Python 3.10 or higher required (found $PYTHON_VERSION)"
fi

print_success "Python version check passed (>= 3.10)"

# Step 5: Upgrade pip and install base packages
print_section "Step 5: Upgrade pip and Install Base Packages"

print_info "Upgrading pip..."
python3 -m pip install --upgrade pip || error_exit "Failed to upgrade pip"
print_success "pip upgraded successfully"

print_info "Installing wheel and setuptools..."
pip install wheel setuptools || error_exit "Failed to install wheel and setuptools"
print_success "Base Python packages installed"

# Step 6: Install llama-cpp-python (CPU-only for ARM)
print_section "Step 6: Install llama-cpp-python (ARM/CPU)"

print_info "Building llama-cpp-python for ARM architecture..."
print_info "This may take 10-20 minutes on mobile devices..."
print_warning "CPU-only build (no GPU acceleration on mobile)"

# Set build flags for ARM CPU-only
export CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF"
export FORCE_CMAKE=1

print_info "Build configuration: CPU-only (ARM)"
pip install llama-cpp-python>=0.3.0 --no-cache-dir || {
    print_warning "First attempt failed, trying with version fallback..."
    pip install llama-cpp-python==0.2.90 --no-cache-dir || error_exit "Failed to install llama-cpp-python"
}

print_success "llama-cpp-python installed successfully"

# Step 7: Install other Python dependencies
print_section "Step 7: Install Python Dependencies"

if [ ! -f "requirements.txt" ]; then
    error_exit "requirements.txt not found. Are you in the jenova-ca directory?"
fi

print_info "Installing dependencies from requirements.txt..."
print_info "This may take 15-30 minutes on mobile devices..."

# Install in batches to avoid memory issues on mobile
print_info "Installing core dependencies..."
pip install torch>=2.5.1 --no-cache-dir || error_exit "Failed to install torch"
pip install chromadb>=0.5.20 --no-cache-dir || error_exit "Failed to install chromadb"
pip install sentence-transformers>=3.3.0 --no-cache-dir || error_exit "Failed to install sentence-transformers"

print_info "Installing remaining dependencies..."
pip install -r requirements.txt || print_warning "Some dependencies may have failed (check output above)"

print_success "Python dependencies installed"

# Step 8: Verify installation
print_section "Step 8: Verify Installation"

print_info "Testing Python imports..."

python3 << EOF || error_exit "Import verification failed"
import sys
print(f"Python: {sys.version}")

try:
    import llama_cpp
    print("✓ llama-cpp-python")
except ImportError as e:
    print(f"✗ llama-cpp-python: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except ImportError as e:
    print(f"✗ torch: {e}")
    sys.exit(1)

try:
    import chromadb
    print("✓ chromadb")
except ImportError as e:
    print(f"✗ chromadb: {e}")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers")
except ImportError as e:
    print(f"✗ sentence-transformers: {e}")
    sys.exit(1)

print("All core dependencies verified!")
EOF

print_success "Import verification passed"

# Step 9: Create model directory
print_section "Step 9: Setup Model Directory"

# Create model directory in Termux home and shared storage
MODEL_DIR="$HOME/jenova-models"
mkdir -p "$MODEL_DIR"
print_success "Model directory created: $MODEL_DIR"

# Try to setup shared storage access
if command -v termux-setup-storage &> /dev/null; then
    print_info "Setting up shared storage access..."
    print_info "Grant storage permission when prompted..."
    termux-setup-storage || print_warning "Could not setup shared storage (permissions denied)"

    if [ -d "$HOME/storage/shared" ]; then
        SHARED_MODEL_DIR="$HOME/storage/shared/jenova-models"
        mkdir -p "$SHARED_MODEL_DIR"
        print_success "Shared model directory created: $SHARED_MODEL_DIR"
        print_info "You can access this from Android file manager"
    fi
fi

# Step 10: Create JENOVA data directory
print_section "Step 10: Setup JENOVA Data Directory"

JENOVA_DATA_DIR="$HOME/.jenova-ai"
mkdir -p "$JENOVA_DATA_DIR"
print_success "JENOVA data directory created: $JENOVA_DATA_DIR"

# Step 11: Installation summary
print_section "Installation Complete!"

print_success "JENOVA Cognitive Architecture successfully installed on Termux"
echo
print_info "Installation Summary:"
print_info "  Python Version: $PYTHON_VERSION"
print_info "  Model Directory: $MODEL_DIR"
print_info "  Data Directory: $JENOVA_DATA_DIR"
print_info "  Architecture: $(uname -m)"
echo

print_section "Next Steps"
echo
print_message "$GREEN" "1. Download a GGUF model:"
print_message "$BLUE" "   Small models for mobile (< 2GB RAM):"
print_message "$BLUE" "     • TinyLlama 1.1B: ~600MB"
print_message "$BLUE" "       wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O $MODEL_DIR/model.gguf"
echo
print_message "$BLUE" "   Medium models (4GB+ RAM):"
print_message "$BLUE" "     • Qwen 1.8B: ~1.1GB"
print_message "$BLUE" "       wget https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf -O $MODEL_DIR/model.gguf"
echo
print_message "$GREEN" "2. Start JENOVA:"
print_message "$BLUE" "   python -m jenova.main"
echo
print_message "$GREEN" "3. For help and documentation:"
print_message "$BLUE" "   cat README.md"
echo

print_section "Termux-Specific Notes"
echo
print_message "$YELLOW" "⚠ Performance Considerations:"
print_message "$BLUE" "  • Mobile devices are CPU-only (no GPU acceleration)"
print_message "$BLUE" "  • Expect 2-5 tokens/second on modern smartphones"
print_message "$BLUE" "  • Use small models (< 2B parameters) for better performance"
print_message "$BLUE" "  • Close other apps to free up RAM"
echo
print_message "$YELLOW" "⚠ Storage:"
print_message "$BLUE" "  • Models are stored in: $MODEL_DIR"
print_message "$BLUE" "  • Memory database in: $JENOVA_DATA_DIR"
print_message "$BLUE" "  • Shared storage (if enabled): ~/storage/shared/jenova-models"
echo
print_message "$YELLOW" "⚠ Battery Usage:"
print_message "$BLUE" "  • LLM inference is CPU-intensive"
print_message "$BLUE" "  • Keep device plugged in for extended sessions"
print_message "$BLUE" "  • Consider using battery optimization exceptions"
echo

print_section "Troubleshooting"
echo
print_message "$YELLOW" "If you encounter issues:"
print_message "$BLUE" "  1. Out of memory: Use smaller models (TinyLlama 1.1B)"
print_message "$BLUE" "  2. Import errors: Try reinstalling dependencies"
print_message "$BLUE" "  3. Performance issues: Reduce context_size in config"
print_message "$BLUE" "  4. Storage errors: Check permissions with termux-setup-storage"
echo

print_message "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_message "$GREEN" "Installation completed successfully!"
print_message "$GREEN" "Created by orpheus497 | MIT License"
print_message "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
