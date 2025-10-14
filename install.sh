#!/bin/bash
# Jenova AI System-Wide Installation Script
# This script installs Jenova AI with Ground-Up Rebuilt CPA for all users on the system.
# It must be run with root privileges (e.g., using 'sudo').

set -e

echo "--- Installing Jenova AI (v3.1.0) with Ground-Up Rebuilt CPA ---"
echo "    Ground-Up Rebuild: Stable, performant, no complex hardware profiles"
echo "    Large Persistent Cache: 5GB default RAM/VRAM cache for model layers"
echo "    Safe JIT Compilation: Surgical optimization with robust error handling"
echo "    Hard-Coded Defaults: 16 threads, all GPU layers for reliable performance"
echo ""

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

# 3. Verify Python version (>=3.10 required)
echo "--> Checking Python version (3.10+ required)..."
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "[ERROR] Python 3.10 or higher is required. Current version: $python_version"
    exit 1
fi

# 4. Upgrade Pip
echo "--> Upgrading system's pip..."
pip install --upgrade pip > /dev/null

# 5. Install the Package with all dependencies
# This installs the package into the system's site-packages directory.
# The 'jenova' command will be placed in a system-wide bin location (e.g., /usr/local/bin).
echo "--> Installing Jenova AI package globally..."
echo "    Installing core dependencies: llama-cpp-python, chromadb, sentence-transformers, rich..."
echo "    Installing CPA dependencies: numba (JIT compilation), psutil (system monitoring)..."
if ! pip install --ignore-installed .; then
    echo "[ERROR] Installation failed. Please check setup.py and ensure all dependencies can be installed."
    exit 1
fi

echo
echo "======================================================================"
echo "✅ Jenova AI v3.1.0 with Ground-Up Rebuilt CPA"
echo "   has been successfully installed for all users."
echo ""
echo "🚀 KEY IMPROVEMENTS:"
echo "   • Ground-Up CPA Rebuild - Eliminates 'stuck on thinking' bugs"
echo "   • Large Persistent Cache - 5GB default for instant model access"
echo "   • Safe JIT Compilation - Maximum performance with stability"
echo "   • Hard-Coded Defaults - 16 threads, all GPU layers"
echo "   • Thread-Safe UI - No race conditions on multi-core systems"
echo "   • No Complex Profiles - Simple, stable, performant by default"
echo ""
echo "Any user can now run the application by simply typing the command:"
echo "  jenova"
echo ""
echo "User-specific data, memories, insights, and CPA state will be"
echo "automatically stored separately for each user in their home directory at:"
echo "  ~/.jenova-ai/users/<username>/"
echo ""
echo "The CPA maintains persistent learning state at:"
echo "  ~/.jenova-ai/users/<username>/.cpa_state/"
echo ""
echo "For help and available commands, run 'jenova' and type '/help'"
echo "======================================================================"
