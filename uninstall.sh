#!/bin/bash
# JENOVA Cognitive Architecture - Uninstallation Script
# This script removes JENOVA from the system.
# It can be run with or without root privileges, depending on installation method.

set -e

echo "=============================================="
echo "  JENOVA Cognitive Architecture Uninstaller"
echo "=============================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. Check Installation Method
##Purpose: Determine if installed system-wide (requires root) or user-wide
INSTALLED_SYSTEM_WIDE=0
if [ "$(id -u)" -eq 0 ]; then
    INSTALLED_SYSTEM_WIDE=1
elif pip show jenova-ai &> /dev/null; then
    # Check if installed in user site-packages or system site-packages
    INSTALL_LOCATION=$(pip show jenova-ai | grep "Location:" | awk '{print $2}')
    if [[ "$INSTALL_LOCATION" == /usr* ]] || [[ "$INSTALL_LOCATION" == /usr/local* ]]; then
        echo "[WARNING] JENOVA appears to be installed system-wide, but this script is not running as root."
        echo "          Some components may not be removed."
        echo "          To fully uninstall, run: sudo ./uninstall.sh"
        INSTALLED_SYSTEM_WIDE=1
    fi
fi

# 2. Uninstall the Package
##Purpose: Remove JENOVA package using pip
echo "--> Uninstalling JENOVA package..."
if pip show jenova-ai &> /dev/null; then
    if [ "$INSTALLED_SYSTEM_WIDE" -eq 1 ] && [ "$(id -u)" -ne 0 ]; then
        echo "[ERROR] Package is installed system-wide. Please run with sudo:"
        echo "        sudo ./uninstall.sh"
        exit 1
    fi
    
    if pip uninstall -y jenova-ai; then
        echo "✓ Package uninstalled successfully"
    else
        echo "[WARNING] Package uninstallation had issues, but continuing..."
    fi
else
    echo "[INFO] JENOVA package not found in pip. It may have been removed already."
fi

# 3. Remove TUI Binary
##Purpose: Remove the compiled Bubble Tea TUI binary
echo ""
echo "--> Removing Bubble Tea TUI binary..."
TUI_BINARY="$SCRIPT_DIR/tui/jenova-tui"
if [ -f "$TUI_BINARY" ]; then
    rm -f "$TUI_BINARY"
    echo "✓ TUI binary removed"
else
    echo "[INFO] TUI binary not found (may have been removed already)"
fi

# 4. Remove Virtual Environment (if exists)
##Purpose: Remove the venv directory if present
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo ""
    read -p "Remove virtual environment (venv/)? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$SCRIPT_DIR/venv"
        echo "✓ Virtual environment removed"
    else
        echo "[INFO] Virtual environment preserved"
    fi
fi

# 5. Remove User Data (Optional)
##Purpose: Ask user if they want to remove their personal data
echo ""
echo "======================================================================"
echo "User Data Removal"
echo "======================================================================"
echo ""
echo "JENOVA stores user-specific data in:"
echo "  ~/.jenova-ai/users/<username>/"
echo ""
echo "This includes:"
echo "  - Memory databases (episodic, semantic, procedural)"
echo "  - Insights and cognitive graph data"
echo "  - Configuration files"
echo ""
read -p "Do you want to remove your personal data? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    USER_DATA_DIR="$HOME/.jenova-ai"
    if [ -d "$USER_DATA_DIR" ]; then
        echo "--> Removing user data directory: $USER_DATA_DIR"
        rm -rf "$USER_DATA_DIR"
        echo "✓ User data removed"
    else
        echo "[INFO] User data directory not found: $USER_DATA_DIR"
    fi
else
    echo "[INFO] User data preserved at: ~/.jenova-ai/"
fi

# 6. Remove Models Directory (Optional)
##Purpose: Ask user if they want to remove downloaded models
if [ -d "$SCRIPT_DIR/models" ] && [ "$(ls -A "$SCRIPT_DIR/models" 2>/dev/null)" ]; then
    echo ""
    echo "======================================================================"
    echo "Models Removal"
    echo "======================================================================"
    echo ""
    echo "GGUF model files found in: $SCRIPT_DIR/models/"
    echo ""
    read -p "Do you want to remove downloaded models? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$SCRIPT_DIR/models"
        echo "✓ Models directory removed"
    else
        echo "[INFO] Models preserved at: $SCRIPT_DIR/models/"
    fi
fi

echo ""
echo "======================================================================"
echo "✅ JENOVA Cognitive Architecture has been uninstalled."
echo ""
if [ "$INSTALLED_SYSTEM_WIDE" -eq 1 ]; then
    echo "The 'jenova' command should no longer be available system-wide."
else
    echo "The 'jenova' command should no longer be available for your user."
fi
echo "======================================================================"
