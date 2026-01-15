#!/bin/bash
# JENOVA Cognitive Architecture - Virtual Environment Setup Script
# Recommended installation method for development and Python 3.13+ compatibility

set -e

echo "=============================================="
echo "  JENOVA Virtual Environment Setup"
echo "=============================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Go (required for Bubble Tea TUI)
echo "--> Checking for Go installation..."
if ! command -v go &> /dev/null; then
    echo "[ERROR] Go is not installed."
    echo "Go 1.21+ is required to build the Bubble Tea TUI."
    echo ""
    echo "Install Go from: https://go.dev/dl/"
    echo "Or use your package manager:"
    echo "  Ubuntu/Debian: sudo apt install golang-go"
    echo "  Fedora: sudo dnf install golang"
    echo "  Arch: sudo pacman -S go"
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
echo "✓ Go $GO_VERSION detected"

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo ""
    echo "--> Removing old virtual environment..."
    rm -rf venv
fi

# Create new venv
echo ""
echo "--> Creating new virtual environment..."
python3 -m venv venv

# Activate venv
echo "--> Activating virtual environment..."
source venv/bin/activate

# Upgrade pip, setuptools, wheel
echo "--> Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies from requirements.txt first
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install the package in editable mode
echo "Installing JENOVA in editable mode..."
pip install -e .

# Fix chromadb compatibility issues
echo "Applying chromadb compatibility fixes..."
python3 << 'PYTHON_FIX'
import os
import re
import site

def find_chromadb_config():
    """Find chromadb config.py in site-packages"""
    chromadb_paths = []
    for site_packages in site.getsitepackages():
        config_path = os.path.join(site_packages, 'chromadb', 'config.py')
        if os.path.exists(config_path):
            chromadb_paths.append(config_path)
    
    # Try user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        config_path = os.path.join(user_site, 'chromadb', 'config.py')
        if os.path.exists(config_path) and config_path not in chromadb_paths:
            chromadb_paths.append(config_path)
    
    return chromadb_paths

chromadb_paths = find_chromadb_config()

if chromadb_paths:
    for config_path in chromadb_paths:
        print(f"Fixing chromadb config at: {config_path}")
        
        # Read the file
        with open(config_path, 'r') as f:
            content = f.read()
        
        original_content = content
        fixes_applied = []
        
        # Fix 1: Add Optional import if missing
        if 'from typing import' in content and 'Optional' not in content.split('from typing import')[1].split('\n')[0]:
            # Add Optional to existing typing import
            content = re.sub(
                r'from typing import ([^\\n]+)',
                lambda m: f'from typing import {m.group(1)}, Optional' if 'Optional' not in m.group(1) else m.group(0),
                content,
                count=1
            )
            fixes_applied.append("Added Optional import")
        
        # Fix 2: Fix port fields to use Optional[str] instead of str = None
        port_fields = ['clickhouse_port', 'chroma_server_host', 'chroma_server_http_port', 'chroma_server_grpc_port']
        for field in port_fields:
            pattern = rf'(\s+){field}:\s*str\s*=\s*None'
            if re.search(pattern, content):
                content = re.sub(pattern, rf'\1{field}: Optional[str] = None', content)
                fixes_applied.append(f"Fixed {field} type annotation")
        
        # Fix 3: Add type annotation for chroma_coordinator_host if missing
        if 'chroma_coordinator_host = ' in content and 'chroma_coordinator_host:' not in content:
            content = re.sub(
                r'(\s+)chroma_coordinator_host\s*=\s*"localhost"',
                r'\1chroma_coordinator_host: str = "localhost"',
                content
            )
            fixes_applied.append("Fixed chroma_coordinator_host type annotation")
        
        # Fix 4: Add type annotation for chroma_logservice_port if missing
        if 'chroma_logservice_port = ' in content and 'chroma_logservice_port:' not in content:
            content = re.sub(
                r'(\s+)chroma_logservice_port\s*=\s*(\d+)',
                r'\1chroma_logservice_port: int = \2',
                content
            )
            fixes_applied.append("Fixed chroma_logservice_port type annotation")
        
        # Fix 5: Fix chroma_server_nofile access in System.__init__ (if it exists)
        if 'if settings["chroma_server_nofile"]' in content:
            old_pattern = r'if settings\["chroma_server_nofile"\] is not None:'
            new_pattern = '''# Note: chroma_server_nofile is commented out in Settings class
        # Use getattr with default None to handle missing attribute gracefully
        chroma_server_nofile = getattr(settings, "chroma_server_nofile", None)
        if chroma_server_nofile is not None:'''
            content = re.sub(old_pattern, new_pattern, content)
            
            if 'desired_soft = settings["chroma_server_nofile"]' in content:
                content = content.replace(
                    'desired_soft = settings["chroma_server_nofile"]',
                    'desired_soft = chroma_server_nofile'
                )
            fixes_applied.append("Fixed chroma_server_nofile access")
        
        # Write back if changes were made
        if content != original_content:
            with open(config_path, 'w') as f:
                f.write(content)
            print(f"  ✓ Applied {len(fixes_applied)} fixes:")
            for fix in fixes_applied:
                print(f"    - {fix}")
        else:
            print(f"  ✓ No fixes needed (already patched)")
else:
    print("⚠ Warning: Could not find chromadb config.py. Fixes will be applied at runtime.")
PYTHON_FIX

# Build the Bubble Tea TUI
echo ""
echo "--> Building Bubble Tea TUI..."
cd "$SCRIPT_DIR"
if [ -f "./build_tui.sh" ]; then
    chmod +x ./build_tui.sh
    if ./build_tui.sh; then
        echo "✓ Bubble Tea TUI built successfully"
    else
        echo "[ERROR] Failed to build Bubble Tea TUI."
        echo "You can try building it manually later with: ./build_tui.sh"
    fi
else
    echo "[WARNING] build_tui.sh not found. Build TUI manually with: cd tui && go build -o jenova-tui ."
fi

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "IMPORTANT: Before running JENOVA, download a GGUF model:"
echo "  mkdir -p models"
echo "  wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf"
echo ""
echo "To run JENOVA:"
echo "  source venv/bin/activate"
echo "  jenova"
echo ""

