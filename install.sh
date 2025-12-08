#!/bin/bash
# Jenova AI System-Wide Installation Script
# This script installs Jenova AI for all users on the system.
# It must be run with root privileges (e.g., using 'sudo').

set -e

echo "--- Installing Jenova AI for All Users ---"

# 1. Check for Root Privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] This script must be run with root privileges."
    echo "Please run it again using: sudo ./install.sh"
    exit 1
fi

# 2. Verify Dependencies
##Purpose: Check for essential runtime dependencies (python3, pip, git)
echo "--> Checking for dependencies (python3, pip, git)..."
if ! command -v python3 &> /dev/null || ! command -v pip &> /dev/null || ! command -v git &> /dev/null; then
    echo "[ERROR] Missing essential dependencies. Please ensure python3, python3-pip, and git are installed."
    exit 1
fi

# 3. Check for Build Dependencies
##Purpose: Verify that C++ compiler and Python development headers are available
##These are required for building packages like numpy from source when pre-built wheels are unavailable
echo "--> Checking for build dependencies (C++ compiler, Python dev headers)..."
MISSING_BUILD_DEPS=0

# Check for C++ compiler (g++ or clang++)
##Purpose: Verify a C++ compiler is available for building native extensions
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "[WARNING] No C++ compiler found (g++ or clang++)."
    echo "          Some packages may need to be built from source."
    MISSING_BUILD_DEPS=1
fi

# Check for Python development headers
##Purpose: Verify Python development headers are installed (required for building Python extensions)
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
if [ ! -f "/usr/include/python${PYTHON_VERSION}/Python.h" ] && \
   [ ! -f "/usr/local/include/python${PYTHON_VERSION}/Python.h" ] && \
   [ ! -f "$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')/Python.h" ] 2>/dev/null; then
    echo "[WARNING] Python development headers not found."
    echo "          Required for building Python extensions."
    MISSING_BUILD_DEPS=1
fi

# If build dependencies are missing, provide installation instructions
##Purpose: Guide user to install missing build dependencies before proceeding
if [ $MISSING_BUILD_DEPS -eq 1 ]; then
    echo ""
    echo "[ERROR] Missing build dependencies detected."
    echo ""
    echo "To install missing build dependencies, run:"
    echo ""
    if command -v dnf &> /dev/null; then
        echo "  sudo dnf install gcc-c++ python3-devel"
    elif command -v yum &> /dev/null; then
        echo "  sudo yum install gcc-c++ python3-devel"
    elif command -v apt-get &> /dev/null; then
        echo "  sudo apt-get install build-essential python3-dev"
    elif command -v pacman &> /dev/null; then
        echo "  sudo pacman -S gcc python"
    elif command -v zypper &> /dev/null; then
        echo "  sudo zypper install gcc-c++ python3-devel"
    else
        echo "  Install: gcc-c++ (or g++) and python3-devel (or python3-dev)"
    fi
    echo ""
    echo "After installing build dependencies, run this script again."
    exit 1
fi

# 4. Upgrade Pip
echo "--> Upgrading system's pip..."
pip install --upgrade pip > /dev/null

# 5. Install the Package
##Purpose: Install Jenova AI package into system-wide site-packages directory
##The 'jenova' command will be placed in a system-wide bin location (e.g., /usr/local/bin).
echo "--> Installing Jenova AI package globally..."
if ! pip install --ignore-installed .; then
    echo "[ERROR] Installation failed."
    echo ""
    echo "Common causes:"
    echo "  - Missing build dependencies (C++ compiler, Python dev headers)"
    echo "  - Network connectivity issues"
    echo "  - Incompatible Python version (recommended: Python 3.10-3.12)"
    echo ""
    echo "If you're using Python 3.14 or newer, some packages may need to be built from source."
    echo "Ensure all build dependencies are installed (see warnings above)."
    exit 1
fi

echo

echo "======================================================================"

echo "✅ Jenova AI has been successfully installed for all users."

echo

echo "Any user can now run the application by simply typing the command:"

echo "  jenova"

echo

echo "User-specific data, memories, and insights will be automatically"

echo "stored separately for each user in their home directory at:"

echo "  ~/.jenova-ai/users/<username>/"

echo "======================================================================"
