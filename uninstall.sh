#!/bin/bash
# JENOVA AI System-Wide Uninstallation Script
# This script removes JENOVA AI, all user data, and the downloaded model.
# It must be run with root privileges (e.g., using 'sudo').

set -e

echo "--- JENOVA AI System-Wide Uninstaller ---"

# 1. Check for Root Privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] This script must be run with root privileges."
    echo "Please run it again using: sudo ./uninstall.sh"
    exit 1
fi

# 2. Uninstall the pip package
echo
echo "--> Step 1: Uninstalling the 'jenova-ai' package..."
if pip show jenova-ai &> /dev/null; then
    read -p "Are you sure you want to uninstall the 'jenova-ai' Python package? [y/N] " confirm_pip
    if [[ "$confirm_pip" == "y" || "$confirm_pip" == "Y" ]]; then
        if ! pip uninstall -y jenova-ai; then
            echo "[WARNING] Failed to uninstall 'jenova-ai' package. It might have been installed in a different Python environment."
        else
            echo "✓ 'jenova-ai' package uninstalled."
        fi
    else
        echo "Skipped package uninstallation."
    fi
else
    echo "i 'jenova-ai' package not found. Skipping."
fi

# 3. Remove system-wide model and shared files
echo
echo "--> Step 2: Removing system-wide model directory..."
if [ -d "/usr/local/share/jenova-ai" ]; then
    read -p "Are you sure you want to DELETE the model and shared files at /usr/local/share/jenova-ai? This is irreversible. [y/N] " confirm_model
    if [[ "$confirm_model" == "y" || "$confirm_model" == "Y" ]]; then
        echo "Deleting /usr/local/share/jenova-ai..."
        rm -rf /usr/local/share/jenova-ai
        echo "✓ System-wide files removed."
    else
        echo "Skipped system-wide file removal."
    fi
else
    echo "i System-wide directory not found. Skipping."
fi

# 4. Remove user-specific data
echo
echo "--> Step 3: Searching for and removing user data..."
found_user_data=false
for home_dir in /home/*; do
    if [ -d "$home_dir/.jenova-ai" ]; then
        found_user_data=true
        username=$(basename "$home_dir")
        echo
        read -p "Found user data for '$username' at $home_dir/.jenova-ai. Are you sure you want to DELETE it? This is irreversible. [y/N] " confirm_user
        if [[ "$confirm_user" == "y" || "$confirm_user" == "Y" ]]; then
            echo "Deleting data for user '$username'..."
            rm -rf "$home_dir/.jenova-ai"
            echo "✓ User data for '$username' removed."
        else
            echo "Skipped data removal for user '$username'."
        fi
    fi
done

if [ "$found_user_data" = false ]; then
    echo "i No user data directories found in /home/*/.jenova-ai."
fi

echo
echo "======================================================================"
echo "✅ JENOVA AI uninstallation process complete."
echo "Some files may remain if you skipped any steps."
echo "======================================================================"
