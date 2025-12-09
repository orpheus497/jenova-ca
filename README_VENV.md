# JENOVA Virtual Environment Setup

This guide helps you set up JENOVA in a clean virtual environment with all compatibility fixes applied.

## Quick Start

1. **Run the setup script:**
   ```bash
   ./setup_venv.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Run JENOVA:**
   ```bash
   jenova
   ```

## What the Setup Script Does

1. Creates a fresh Python 3.14 virtual environment
2. Upgrades pip, setuptools, and wheel
3. Installs all dependencies from `requirements.txt`
4. Installs JENOVA in editable mode
5. Applies chromadb compatibility fixes for Python 3.14 and Pydantic 2.12

## ChromaDB Compatibility Fixes Applied

The setup script automatically fixes the following chromadb compatibility issues:

- **Type annotations**: Adds missing type annotations for `chroma_coordinator_host` and `chroma_logservice_port`
- **Optional types**: Fixes port fields to use `Optional[str]` instead of `str = None`
- **Missing imports**: Adds `Optional` import if missing
- **Attribute access**: Fixes `chroma_server_nofile` access to handle missing attributes gracefully

## Manual Setup (if needed)

If the setup script doesn't work, you can set up manually:

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Upgrade tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install JENOVA
pip install -e .

# Apply chromadb fixes
python fix_chromadb_compat.py
```

## Troubleshooting

If you encounter issues:

1. **Permission errors**: Make sure you own the project directory
   ```bash
   sudo chown -R $USER:$USER .
   ```

2. **ChromaDB import errors**: Run the fix script manually
   ```bash
   source venv/bin/activate
   python fix_chromadb_compat.py
   ```

3. **Clean reinstall**: Remove venv and start fresh
   ```bash
   rm -rf venv
   ./setup_venv.sh
   ```

## Notes

- The virtual environment uses Python 3.14
- ChromaDB compatibility fixes are applied automatically
- All fixes are applied to the chromadb installation in the venv
- The fixes are compatible with Pydantic 2.12

