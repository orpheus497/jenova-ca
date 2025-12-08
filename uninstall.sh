#!/bin/bash
# Jenova AI System-Wide Uninstallation Script
# This script removes Jenova AI from the system.
# It can be run with or without root privileges, depending on installation method.

set -e

echo "--- Uninstalling Jenova AI ---"

# 1. Check Installation Method
##Purpose: Determine if installed system-wide (requires root) or user-wide
INSTALLED_SYSTEM_WIDE=0
if [ "$(id -u)" -eq 0 ]; then
    INSTALLED_SYSTEM_WIDE=1
elif pip show jenova-ai &> /dev/null; then
    # Check if installed in user site-packages or system site-packages
    INSTALL_LOCATION=$(pip show jenova-ai | grep "Location:" | awk '{print $2}')
    if [[ "$INSTALL_LOCATION" == /usr* ]] || [[ "$INSTALL_LOCATION" == /usr/local* ]]; then
        echo "[WARNING] Jenova AI appears to be installed system-wide, but this script is not running as root."
        echo "          Some components may not be removed."
        echo "          To fully uninstall, run: sudo ./uninstall.sh"
        INSTALLED_SYSTEM_WIDE=1
    fi
fi

# 2. Uninstall the Package
##Purpose: Remove Jenova AI package using pip
echo "--> Uninstalling Jenova AI package..."
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
    echo "[INFO] Jenova AI package not found in pip. It may have been removed already."
fi

# 3. Remove User Data (Optional)
##Purpose: Ask user if they want to remove their personal data
echo ""
echo "======================================================================"
echo "User Data Removal"
echo "======================================================================"
echo ""
echo "Jenova AI stores user-specific data in:"
echo "  ~/.jenova-ai/users/<username>/"
echo ""
echo "This includes:"
echo "  - Memory databases (episodic, semantic, procedural)"
echo "  - Insights and saved conversations"
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

# 4. Remove System-Wide Models (if installed system-wide)
##Purpose: Check for and optionally remove system-wide model files
if [ "$INSTALLED_SYSTEM_WIDE" -eq 1 ] && [ "$(id -u)" -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "System-Wide Models Removal"
    echo "======================================================================"
    echo ""
    
    # Check common system-wide model locations
    MODEL_LOCATIONS=(
        "/usr/local/lib/python*/site-packages/jenova/models"
        "/usr/lib/python*/site-packages/jenova/models"
    )
    
    MODELS_FOUND=0
    for pattern in "${MODEL_LOCATIONS[@]}"; do
        for model_dir in $pattern; do
            if [ -d "$model_dir" ] && [ "$(ls -A $model_dir 2>/dev/null)" ]; then
                MODELS_FOUND=1
                echo "Found models directory: $model_dir"
                read -p "Do you want to remove system-wide models? (y/N): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "--> Removing models directory: $model_dir"
                    rm -rf "$model_dir"
                    echo "✓ Models removed"
                else
                    echo "[INFO] Models preserved at: $model_dir"
                fi
                break
            fi
        done
        [ $MODELS_FOUND -eq 1 ] && break
    done
    
    if [ $MODELS_FOUND -eq 0 ]; then
        echo "[INFO] No system-wide models directory found."
    fi
fi

echo ""
echo "======================================================================"
echo "✅ Jenova AI has been uninstalled."
echo ""
if [ "$INSTALLED_SYSTEM_WIDE" -eq 1 ]; then
    echo "The 'jenova' command should no longer be available system-wide."
else
    echo "The 'jenova' command should no longer be available for your user."
fi
echo "======================================================================"
