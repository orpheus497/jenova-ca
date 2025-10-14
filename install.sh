#!/bin/bash
# Jenova AI System-Wide Installation Script
# This script installs Jenova AI with Cognitive Process Accelerator (CPA) for all users on the system.
# It must be run with root privileges (e.g., using 'sudo').

set -e

echo "--- Installing Jenova AI (v3.1.0) with CPA for All Users ---"
echo "    Features: Persistent State Management, Proactive Cognitive Engagement"
echo "    Performance Optimization with Profile-Guided JIT Compilation"
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
echo "✅ Jenova AI v3.1.0 with Cognitive Process Accelerator (CPA)"
echo "   has been successfully installed for all users."
echo ""
echo "🚀 NEW FEATURES:"
echo "   • Persistent State Management - AI remembers across sessions"
echo "   • Proactive Cognitive Engagement - Always thinking and learning"
echo "   • Profile-Guided JIT Compilation - Enterprise-grade performance"
echo "   • 2.5x Faster Response Time - Enhanced activity level"
echo "   • Thread-Safe UI - Resolved race condition on multi-core systems"
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
