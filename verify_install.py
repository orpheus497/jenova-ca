#!/usr/bin/env python3
# The JENOVA Cognitive Architecture - Installation Verification Script
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Installation verification script for JENOVA AI.

This script checks that all critical dependencies are installed correctly
and that the system is ready to run JENOVA.
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_check(name, status, details=""):
    """Print a check result."""
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {name}")
    if details:
        print(f"  → {details}")


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 10 and version.minor < 14:
        print_check(f"Python version ({version_str})", True, "Compatible")
        return True
    else:
        print_check(f"Python version ({version_str})", False,
                    "Requires Python 3.10-3.13")
        return False


def check_import(module_name, package_name=None, optional=False):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print_check(f"{package_name}", True, "Installed")
        return True
    except ImportError as e:
        if optional:
            print_check(f"{package_name} (optional)", False,
                        f"Not installed - {e}")
        else:
            print_check(f"{package_name}", False,
                        f"Import failed: {e}")
        return False


def check_llama_cpp():
    """Check llama-cpp-python and CUDA support."""
    try:
        import llama_cpp
        print_check("llama-cpp-python", True, "Installed")

        # Check version
        if hasattr(llama_cpp, '__version__'):
            print(f"  → Version: {llama_cpp.__version__}")

        # Check for CUDA support (best effort)
        try:
            # Try to check if GPU offload is supported
            from llama_cpp import llama_cpp as llama_c
            has_gpu = llama_c.llama_supports_gpu_offload()
            if has_gpu:
                print("  → GPU offload: Supported")
            else:
                print("  → GPU offload: Not available (CPU-only build)")
        except (AttributeError, ImportError):
            print("  → GPU offload: Unknown (check manually)")

        return True
    except ImportError as e:
        print_check("llama-cpp-python", False, f"Import failed: {e}")
        return False


def check_torch_cuda():
    """Check PyTorch and CUDA support."""
    try:
        import torch
        print_check("PyTorch", True, f"Version {torch.__version__}")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"  → CUDA available: Yes")
            print(f"  → GPU devices: {device_count}")
            print(f"  → Primary GPU: {device_name}")
            return True
        else:
            print(f"  → CUDA available: No (CPU-only)")
            return True
    except ImportError as e:
        print_check("PyTorch", False, f"Import failed: {e}")
        return False


def check_chromadb():
    """Check ChromaDB installation."""
    try:
        import chromadb
        print_check("ChromaDB", True, f"Version {chromadb.__version__}")
        return True
    except ImportError as e:
        print_check("ChromaDB", False, f"Import failed: {e}")
        return False
    except AttributeError:
        print_check("ChromaDB", True, "Installed (version unknown)")
        return True


def check_proto_files():
    """Check if Protocol Buffer files are compiled."""
    proto_dir = Path("src/jenova/network/proto")
    pb2_file = proto_dir / "jenova_pb2.py"
    pb2_grpc_file = proto_dir / "jenova_pb2_grpc.py"

    if pb2_file.exists() and pb2_grpc_file.exists():
        print_check("Protocol Buffers", True, "Compiled")
        return True
    else:
        print_check("Protocol Buffers", False,
                    "Not compiled (run: python build_proto.py)")
        return False


def check_models_directory():
    """Check if models directory exists and has models."""
    system_dir = Path("/usr/local/share/models")
    local_dir = Path("models")

    models_found = []

    if system_dir.exists():
        gguf_files = list(system_dir.glob("*.gguf"))
        if gguf_files:
            models_found.extend([str(f) for f in gguf_files])

    if local_dir.exists():
        gguf_files = list(local_dir.glob("*.gguf"))
        if gguf_files:
            models_found.extend([str(f) for f in gguf_files])

    if models_found:
        print_check("GGUF Models", True, f"Found {len(models_found)} model(s)")
        for model in models_found[:3]:  # Show first 3
            print(f"  → {model}")
        return True
    else:
        print_check("GGUF Models", False,
                    "No models found. Download a model to get started.")
        print("  → See: https://huggingface.co/TheBloke")
        return False


def main():
    """Run all verification checks."""
    print_header("JENOVA AI Installation Verification")

    results = {}

    # Core Python
    print_header("Core Python Environment")
    results['python'] = check_python_version()

    # Critical dependencies
    print_header("Critical Dependencies")
    results['llama'] = check_llama_cpp()
    results['torch'] = check_torch_cuda()
    results['chromadb'] = check_chromadb()
    results['sentence_transformers'] = check_import(
        'sentence_transformers', 'sentence-transformers')
    results['rich'] = check_import('rich')
    results['prompt_toolkit'] = check_import('prompt_toolkit', 'prompt-toolkit')
    results['yaml'] = check_import('yaml', 'PyYAML')

    # Configuration and utilities
    print_header("Configuration & Utilities")
    results['pydantic'] = check_import('pydantic')
    results['numpy'] = check_import('numpy')
    results['tenacity'] = check_import('tenacity')
    results['psutil'] = check_import('psutil')
    results['filelock'] = check_import('filelock')

    # Distributed computing
    print_header("Distributed Computing")
    results['zeroconf'] = check_import('zeroconf')
    results['grpc'] = check_import('grpc', 'grpcio')
    results['grpc_tools'] = check_import('grpc_tools', 'grpcio-tools')
    results['protobuf'] = check_import('google.protobuf', 'protobuf')
    results['jwt'] = check_import('jwt', 'PyJWT')

    # CLI Enhancement Tools
    print_header("CLI Enhancement Tools")
    results['git'] = check_import('git', 'gitpython')
    results['pygments'] = check_import('pygments')
    results['rope'] = check_import('rope')
    results['tree_sitter'] = check_import('tree_sitter', 'tree-sitter')
    results['jsonschema'] = check_import('jsonschema')
    results['radon'] = check_import('radon')
    results['bandit'] = check_import('bandit')

    # Optional dependencies
    print_header("Optional Dependencies")
    results['selenium'] = check_import('selenium', optional=True)
    results['playwright'] = check_import('playwright', optional=True)
    results['requests'] = check_import('requests', optional=True)

    # System checks
    print_header("System Configuration")
    results['proto'] = check_proto_files()
    results['models'] = check_models_directory()

    # Summary
    print_header("Verification Summary")

    critical_deps = [
        'python', 'llama', 'torch', 'chromadb', 'sentence_transformers',
        'rich', 'prompt_toolkit', 'yaml', 'pydantic', 'numpy', 'tenacity',
        'psutil', 'filelock', 'zeroconf', 'grpc', 'grpc_tools', 'protobuf',
        'jwt', 'git', 'pygments', 'rope', 'tree_sitter', 'jsonschema',
        'radon', 'bandit'
    ]

    critical_passed = sum(1 for dep in critical_deps if results.get(dep, False))
    critical_total = len(critical_deps)

    print(f"\nCritical Dependencies: {critical_passed}/{critical_total} passed")

    if results.get('proto', False):
        print("✓ Protocol Buffers compiled")
    else:
        print("⚠ Protocol Buffers not compiled (distributed features unavailable)")

    if results.get('models', False):
        print("✓ GGUF models found")
    else:
        print("⚠ No GGUF models found (download required to run JENOVA)")

    print()

    if critical_passed == critical_total and results.get('proto', False):
        print("\033[92m✓ Installation verification PASSED\033[0m")
        print("\nYou're ready to run JENOVA!")
        print("  python -m jenova.main")

        if not results.get('models', False):
            print("\nNote: Download a GGUF model first:")
            print("  See models/README.md for instructions")

        return 0
    else:
        print("\033[91m✗ Installation verification FAILED\033[0m")
        print("\nSome dependencies are missing or incorrectly installed.")
        print("Please review the errors above and:")
        print("  1. Ensure you're in the virtual environment: source venv/bin/activate")
        print("  2. Try reinstalling: pip install -r requirements.txt")
        print("  3. Compile proto files: python build_proto.py")
        print("  4. For detailed logs, check the output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
