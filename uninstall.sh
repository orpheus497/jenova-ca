#!/bin/bash
# JENOVA AI Local Uninstallation Script
# This script removes JENOVA AI, all user data, and the downloaded models.

set -euo pipefail

echo "--- JENOVA AI Local Uninstaller ---"
echo "This script will offer to remove the virtual environment, local models,"
echo "and all user-generated data. Each step requires explicit confirmation."

# 1. Remove the virtual environment
echo
echo "--> Step 1: Removing the Python virtual environment..."
if [ -d "venv" ]; then
    read -p "Are you sure you want to DELETE the virtual environment at ./venv? [y/N] " confirm_venv
    if [[ "$confirm_venv" == "y" || "$confirm_venv" == "Y" ]]; then
        echo "Deleting ./venv/..."
        rm -rf venv
        echo "✓ Virtual environment removed."
    else
        echo "Skipped virtual environment removal."
    fi
else
    echo "i Virtual environment not found. Skipping."
fi

# 2. Remove the local models directory
echo
echo "--> Step 2: Removing local models directory..."
if [ -d "models" ]; then
    read -p "Are you sure you want to DELETE the local models at ./models? This is irreversible. [y/N] " confirm_model
    if [[ "$confirm_model" == "y" || "$confirm_model" == "Y" ]]; then
        echo "Deleting ./models/..."
        rm -rf models
        echo "✓ Local models removed."
    else
        echo "Skipped local model removal."
    fi
else
    echo "i Local models directory not found. Skipping."
fi

# 3. Remove user-specific data
echo
echo "--> Step 3: Removing user data..."
if [ -d "$HOME/.jenova-ai" ]; then
    echo "[WARNING] This will permanently delete all of JENOVA's memories, insights,"
    echo "          and cognitive data stored in ~/.jenova-ai."
    read -p "Are you sure you want to DELETE all user data? This is irreversible. [y/N] " confirm_user
    if [[ "$confirm_user" == "y" || "$confirm_user" == "Y" ]]; then
        echo "Deleting ~/.jenova-ai/..."
        rm -rf "$HOME/.jenova-ai"
        echo "✓ User data removed."
    else
        echo "Skipped user data removal."
    fi
else
    echo "i User data directory not found. Skipping."
fi

echo
echo "======================================================================"
echo "✅ JENOVA AI uninstallation process complete."
echo "The project directory 'jenova-ca' has not been removed."
echo "You can manually delete it if you wish."
echo "======================================================================"
