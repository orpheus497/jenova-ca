#!/bin/bash
# Jenova AI Definitive Installation Script (Scorched Earth)
# Designed and Developed by orpheus497

echo "--- Installing Jenova AI (Perfected Architecture v1.0) ---"

VENV_DIR="venv"

# Create and Activate Python Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "--> Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
echo "--> Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# --- SCORCHED EARTH REBUILD ---
echo "--> Uninstalling any existing 'jenova-ai' package to ensure a clean state..."
pip uninstall jenova-ai -y
echo "--> Purging all Python cache files (__pycache__)..."
find . -type d -name "__pycache__" -exec rm -r {} +

# --- INSTALLATION ---
echo "--> Installing Jenova AI and all dependencies..."
if ! pip install -e .; then
    echo "!!!!!! ERROR: Installation failed. Please check setup.py and requirements.txt."
    exit 1
fi

echo ""
echo "=========================================================="
echo "Jenova AI v1.0 installation complete!"
echo "To run the application, ensure your virtual environment"
echo "is active and use the 'jenova' command:"
echo ""
echo "  source venv/bin/activate"
echo "  jenova"
echo "=========================================================="