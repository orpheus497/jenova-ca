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
echo "--> Checking for dependencies (python3, pip, git)..."
if ! command -v python3 &> /dev/null || ! command -v pip &> /dev/null || ! command -v git &> /dev/null; then
    echo "[ERROR] Missing essential dependencies. Please ensure python3, python3-pip, and git are installed."
    exit 1
fi

# Check for python3-psutil
echo "--> Checking for python3-psutil system package..."
if ! python3 -c "import psutil" &> /dev/null; then
    echo "[WARNING] python3-psutil system package not found."
    echo "          This package is required for hardware detection."
    echo "          On Debian/Ubuntu, install it with: sudo apt-get install python3-psutil"
    echo "          On other systems, it will be installed via pip."
fi

# 3. Upgrade Pip
echo "--> Upgrading system's pip..."
pip install --upgrade pip > /dev/null

# 4. Install the Package
# This installs the package into the system's site-packages directory.
# The 'jenova' command will be placed in a system-wide bin location (e.g., /usr/local/bin).
echo "--> Installing Jenova AI package globally..."
if ! pip install --ignore-installed .; then
    echo "[ERROR] Installation failed. Please check setup.py and ensure all dependencies can be installed."
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
