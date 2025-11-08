# JENOVA AI - Installation Guide

This comprehensive guide covers the installation process, dependency management, and troubleshooting for JENOVA AI.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Dependency Constraints](#dependency-constraints)
4. [Installation Methods](#installation-methods)
5. [Troubleshooting](#troubleshooting)
6. [Verification](#verification)

---

## Quick Start

For most users, the automated installation script is the easiest method:

```bash
# Clone the repository
git clone https://github.com/orpheus497/jenova-ai.git
cd jenova-ai

# Run the installation script
./install.sh

# Verify installation
python3 test_compatibility.py
```

---

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Fedora 35+, or similar)
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk Space**: 10 GB free space

### Recommended for GPU Acceleration

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit**: 12.1 or higher
- **VRAM**: 4 GB minimum (8 GB+ recommended)
- **System Packages**:
  - `gcc-c++` or `g++` (C++ compiler)
  - `cmake` (build system)
  - `python3-devel` or `python3-dev` (Python headers)
  - `cuda-toolkit` (for GPU support)

### Installing System Dependencies

**Fedora/RHEL:**
```bash
sudo dnf install python3 python3-devel python3-pip git gcc-c++ cmake
# For GPU support:
sudo dnf install cuda-toolkit
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-dev python3-pip python3-venv git build-essential cmake
# For GPU support:
sudo apt install nvidia-cuda-toolkit
```

---

## Dependency Constraints

JENOVA AI uses carefully selected version constraints to ensure compatibility. Understanding these constraints helps prevent installation issues.

### Critical Version Constraints

#### 1. **protobuf < 5.0.0** (CRITICAL)

**Why?** Protobuf 5.x introduces breaking API changes that are incompatible with:
- `grpcio` (gRPC framework used for distributed computing)
- Many machine learning packages (TensorBoard, ONNX Runtime)
- ChromaDB dependencies

**Constraint:** `protobuf>=4.25.2,<5.0.0`

**Issue if violated:**
```
TypeError: Couldn't build proto file into descriptor pool
ImportError: cannot import name 'builder' from 'google.protobuf.internal'
```

**Fix:**
```bash
pip install 'protobuf>=4.25.2,<5.0.0' --force-reinstall
```

---

#### 2. **numpy < 2.0.0** (CRITICAL)

**Why?** NumPy 2.0 introduced breaking changes in:
- Array API changes
- Data type promotions
- C API modifications

This breaks compatibility with:
- `sentence-transformers`
- `chromadb`
- Many PyTorch operations
- Scientific computing packages

**Constraint:** `numpy>=1.26.4,<2.0.0`

**Issue if violated:**
```
AttributeError: module 'numpy' has no attribute 'float'
AttributeError: module 'numpy' has no attribute 'int'
```

**Fix:**
```bash
pip install 'numpy>=1.26.4,<2.0.0' --force-reinstall
```

---

#### 3. **torch < 2.6.0**

**Why?** PyTorch minor versions can introduce:
- CUDA API changes
- Model compatibility changes
- Breaking changes in nn.Module

**Constraint:** `torch>=2.5.1,<2.6.0`

**Notes:**
- torch 2.5.1+ is required for Python 3.13 support
- Upper bound prevents unexpected breaking changes

---

#### 4. **chromadb < 0.6.0** & **sentence-transformers < 3.4.0**

**Why?** These packages must be compatible with:
- Each other (shared dependencies)
- PyTorch version
- NumPy version
- ONNX Runtime (bundled with ChromaDB)

**Constraints:**
- `chromadb>=0.5.20,<0.6.0`
- `sentence-transformers>=3.3.0,<3.4.0`

**Issue if violated:**
```
RuntimeError: Inconsistent version of ONNX runtime
TypeError: 'NoneType' object is not subscriptable (in embedding functions)
```

---

### Why Use Upper Bounds?

Upper bounds (e.g., `<2.0.0`) prevent:
1. **Automatic breaking changes** when running `pip install --upgrade`
2. **Dependency conflicts** between packages with different requirements
3. **Unexpected behavior** from major version updates
4. **Production instability** from untested package versions

### Maintenance Strategy

- **Lower bounds** (`>=x.y.z`): Ensure minimum required features
- **Upper bounds** (`<x+1.0.0`): Prevent breaking changes
- **Regular updates**: Periodically test and update bounds for new versions

---

## Installation Methods

### Method 1: Automated Installation (Recommended)

The `install.sh` script handles all installation steps automatically:

```bash
./install.sh
```

**What it does:**
1. Validates Python version (3.10-3.13)
2. Creates Python virtual environment
3. Detects NVIDIA GPU and CUDA toolkit
4. Builds llama-cpp-python with CUDA support (if available)
5. Installs all dependencies with version constraints
6. Verifies critical dependency versions
7. Installs JENOVA AI in development mode
8. Compiles Protocol Buffers
9. Sets up models directory
10. Runs compatibility tests

**Time required:** 5-15 minutes depending on your system

---

### Method 2: Manual Installation

For users who want more control:

#### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

#### Step 2: Install llama-cpp-python

**CPU-only:**
```bash
pip install 'llama-cpp-python>=0.3.0,<0.4.0'
```

**With CUDA support:**
```bash
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUBLAS=on" \
FORCE_CMAKE=1 \
pip install 'llama-cpp-python>=0.3.0,<0.4.0' --no-cache-dir
```

#### Step 3: Install Core Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Install JENOVA AI

```bash
pip install -e .
```

#### Step 5: Compile Protocol Buffers

```bash
python3 build_proto.py
```

#### Step 6: Verify Installation

```bash
python3 test_compatibility.py
```

---

### Method 3: PyPI Installation (Future)

Once published to PyPI:

```bash
pip install jenova-ai

# With optional dependencies
pip install jenova-ai[web,browser]
pip install jenova-ai[all]
```

---

## Troubleshooting

### Issue: llama-cpp-python CUDA build fails

**Symptoms:**
```
error: command 'gcc' failed
CMake Error: CUDA not found
```

**Solutions:**

1. **Verify CUDA installation:**
   ```bash
   nvcc --version
   ldconfig -p | grep cuda
   ```

2. **Install missing packages:**
   ```bash
   sudo dnf install cuda-toolkit gcc-c++ cmake python3-devel
   ```

3. **Set environment variables:**
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

4. **Fallback to CPU:**
   ```bash
   pip install 'llama-cpp-python>=0.3.0,<0.4.0'
   ```

---

### Issue: Protocol Buffer compilation fails

**Symptoms:**
```
ModuleNotFoundError: No module named 'grpc_tools'
FileNotFoundError: jenova/proto/cognitive_pb2.py not found
```

**Solutions:**

1. **Ensure grpcio-tools is installed:**
   ```bash
   pip install 'grpcio-tools>=1.69.0,<1.70.0'
   ```

2. **Manually compile protos:**
   ```bash
   python3 build_proto.py
   ```

3. **Check proto files exist:**
   ```bash
   ls -la src/jenova/proto/*.proto
   ```

**Note:** Proto compilation failures are non-fatal. Distributed features will compile protos on first import if needed.

---

### Issue: ChromaDB initialization fails

**Symptoms:**
```
TypeError: 'NoneType' object is not subscriptable
sqlite3.OperationalError: database disk image is malformed
```

**Solutions:**

1. **Clear ChromaDB cache:**
   ```bash
   rm -rf ~/.jenova-ai/users/*/semantic_memory/chroma/
   ```

2. **Verify numpy version:**
   ```bash
   python3 -c "import numpy; print(numpy.__version__)"
   # Should be < 2.0.0
   ```

3. **Reinstall chromadb:**
   ```bash
   pip install 'chromadb>=0.5.20,<0.6.0' --force-reinstall
   ```

---

### Issue: gRPC import errors

**Symptoms:**
```
ImportError: cannot import name 'builder' from 'google.protobuf.internal'
TypeError: Couldn't build proto file into descriptor pool
```

**Root cause:** protobuf >= 5.0.0 installed

**Solution:**
```bash
pip install 'protobuf>=4.25.2,<5.0.0' --force-reinstall
python3 -c "from google.protobuf import __version__; print(__version__)"
# Should print 4.x.x
```

---

### Issue: NumPy compatibility errors

**Symptoms:**
```
AttributeError: module 'numpy' has no attribute 'float'
AttributeError: module 'numpy' has no attribute 'int'
```

**Root cause:** numpy >= 2.0.0 installed

**Solution:**
```bash
pip install 'numpy>=1.26.4,<2.0.0' --force-reinstall
python3 -c "import numpy; print(numpy.__version__)"
# Should print 1.x.x
```

---

### Issue: Dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account...
ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/...
```

**Solutions:**

1. **Run pip check:**
   ```bash
   pip check
   ```

2. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

3. **Recreate virtual environment:**
   ```bash
   deactivate
   rm -rf venv
   ./install.sh
   ```

4. **Install dependencies in order:**
   ```bash
   # Critical dependencies first
   pip install 'numpy>=1.26.4,<2.0.0'
   pip install 'protobuf>=4.25.2,<5.0.0'
   pip install 'torch>=2.5.1,<2.6.0'

   # Then remaining dependencies
   pip install -r requirements.txt
   ```

---

### Issue: Python version incompatibility

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch>=2.5.1
```

**Solutions:**

1. **Check Python version:**
   ```bash
   python3 --version
   # Must be 3.10, 3.11, 3.12, or 3.13
   ```

2. **Install correct Python version:**
   ```bash
   # Fedora
   sudo dnf install python3.11

   # Ubuntu
   sudo apt install python3.11 python3.11-venv python3.11-dev
   ```

3. **Use specific Python version:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

---

## Verification

After installation, verify everything works:

### Run Compatibility Test

```bash
python3 test_compatibility.py
```

This checks:
- ✓ All critical dependencies installed
- ✓ protobuf < 5.0.0
- ✓ numpy < 2.0.0
- ✓ torch + chromadb compatibility
- ✓ No dependency conflicts

### Manual Verification

```bash
# Activate virtual environment
source venv/bin/activate

# Check versions
python3 -c "
import torch
import numpy as np
import chromadb
from google.protobuf import __version__ as pb_ver

print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'ChromaDB: {chromadb.__version__}')
print(f'Protobuf: {pb_ver}')

# Verify constraints
assert np.__version__.split('.')[0] < '2', 'numpy must be <2.0.0'
assert pb_ver.split('.')[0] < '5', 'protobuf must be <5.0.0'
print('✓ All version constraints satisfied!')
"
```

### Test Import

```bash
python3 -c "from jenova.main import main; print('✓ JENOVA AI imports successfully')"
```

---

## Optional Dependencies

### Web Tools (Selenium)

```bash
pip install jenova-ai[web]
```

Installs:
- `requests` - HTTP client
- `beautifulsoup4` - HTML parsing
- `selenium` - Browser automation
- `webdriver-manager` - WebDriver management

### Browser Automation (Playwright)

```bash
pip install jenova-ai[browser]
playwright install  # Downloads browser binaries
```

### Development Tools

```bash
pip install jenova-ai[dev]
```

Installs:
- Code formatters (autopep8, black, isort)
- Testing framework (pytest)
- Type checking (mypy)
- Security scanning (bandit, safety)
- Documentation (sphinx)

### All Optional Dependencies

```bash
pip install jenova-ai[all]
```

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Run diagnostics:**
   ```bash
   python3 test_compatibility.py > diagnostics.txt 2>&1
   pip list > installed_packages.txt
   pip check > conflicts.txt 2>&1
   ```

2. **Check existing issues:**
   - GitHub: https://github.com/orpheus497/jenova-ai/issues

3. **Create a new issue** with:
   - Output of `python3 test_compatibility.py`
   - Python version: `python3 --version`
   - OS version: `cat /etc/os-release`
   - Installation method used
   - Complete error messages

---

## License

JENOVA AI is licensed under the MIT License.

Copyright (c) 2024-2025, orpheus497. All rights reserved.

All dependencies are FOSS (Free and Open Source Software) with compatible licenses. See `requirements.txt` for individual package licenses.
