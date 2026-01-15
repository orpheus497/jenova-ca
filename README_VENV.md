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

3. **Download a GGUF model:**
   ```bash
   mkdir -p models
   wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf
   ```

4. **Run JENOVA:**
   ```bash
   jenova
   ```

## Prerequisites

- **Python 3.10+** (tested with 3.10, 3.11, 3.12, 3.13)
- **Go 1.21+** (required for building the Bubble Tea TUI)
- **C++ compiler** (g++ or clang++, for building llama-cpp-python)

## What the Setup Script Does

1. Checks for Go installation (required for TUI)
2. Creates a fresh Python virtual environment
3. Upgrades pip, setuptools, and wheel
4. Installs all dependencies from `requirements.txt`
5. Installs JENOVA in editable mode
6. Applies ChromaDB compatibility fixes for Python 3.13+ and Pydantic 2.12
7. Builds the Bubble Tea TUI binary

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

- Tested with Python 3.10, 3.11, 3.12, and 3.13
- ChromaDB compatibility fixes are applied automatically
- All fixes are applied to the ChromaDB installation in the venv
- The fixes are compatible with Pydantic 2.12
- The Bubble Tea TUI is built automatically during setup

## Fixed Issues

The following ChromaDB compatibility issues have been resolved:

1. **Non-annotated attributes**: Added type annotations for `chroma_coordinator_host` and `chroma_logservice_port`
2. **Optional type annotations**: Fixed port fields (`chroma_server_http_port`, `chroma_server_grpc_port`, `clickhouse_port`) to use `Optional[str]` instead of `str = None`
3. **Missing imports**: Added `Optional` import to chromadb's config.py
4. **Attribute access errors**: Fixed `chroma_server_nofile` access to handle missing attributes gracefully using `getattr()`
5. **Environment variable validation**: Fixed pydantic v2 validation errors for empty string integer fields

## Additional Fixes

- Fixed `CacheManager` UnboundLocalError in `memory_search.py` by removing redundant local import

