#!/usr/bin/env python3
"""
Test critical dependency compatibility for JENOVA AI.

This script verifies that all critical dependencies are installed correctly
and are compatible with each other, particularly focusing on:
- torch, chromadb, and sentence-transformers compatibility
- protobuf version (must be <5.0.0 for grpcio compatibility)
- numpy version (must be <2.0.0 for ML package compatibility)
- pydantic v2 compatibility

Copyright (c) 2024-2025, orpheus497. All rights reserved.
Licensed under the MIT License
"""

import sys
import subprocess
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{title:^70}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.NC} {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.NC} {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.NC}  {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ{Colors.NC}  {message}")


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str, List[str]]:
    """
    Test if a module can be imported and get its version.

    Args:
        module_name: Name of the module to import
        package_name: Display name of the package (defaults to module_name)

    Returns:
        Tuple of (success, version, issues)
    """
    if package_name is None:
        package_name = module_name

    issues = []

    try:
        module = __import__(module_name)

        # Try to get version
        version = "unknown"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        elif hasattr(module, 'version'):
            if callable(module.version):
                version = module.version()
            else:
                version = module.version

        return True, version, issues

    except ImportError as e:
        issues.append(f"Import failed: {e}")
        return False, "not installed", issues
    except Exception as e:
        issues.append(f"Unexpected error: {e}")
        return False, "error", issues


def check_version_constraint(version: str, min_ver: str = None, max_ver: str = None,
                             exclude_major: int = None) -> Tuple[bool, List[str]]:
    """
    Check if a version satisfies constraints.

    Args:
        version: The version string to check
        min_ver: Minimum version (inclusive)
        max_ver: Maximum version (exclusive)
        exclude_major: Major version to exclude (e.g., 2 for numpy 2.x)

    Returns:
        Tuple of (passes, issues)
    """
    issues = []

    if version in ["unknown", "not installed", "error"]:
        return False, ["Version could not be determined"]

    try:
        # Parse version (basic parsing for major.minor.patch)
        parts = version.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0

        # Check excluded major version
        if exclude_major is not None and major >= exclude_major:
            issues.append(f"Version {version} has breaking changes (major version {major} excluded)")
            return False, issues

        # For more complex version comparisons, we'd need packaging library
        # but keeping dependencies minimal for this test script

        return True, issues

    except Exception as e:
        issues.append(f"Version parsing error: {e}")
        return False, issues


def test_core_dependencies() -> int:
    """Test core Python dependencies. Returns number of failures."""
    print_section("CORE DEPENDENCIES TEST")

    failures = 0
    warnings = 0

    # Test Python version
    print_info(f"Python version: {sys.version.split()[0]}")
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor

    if py_major == 3 and 10 <= py_minor <= 13:
        print_success(f"Python {py_major}.{py_minor} is supported")
    else:
        print_error(f"Python {py_major}.{py_minor} is not supported (requires 3.10-3.13)")
        failures += 1

    print()

    # Define critical dependencies with their constraints
    dependencies = [
        # (module_name, display_name, excluded_major_version, is_critical)
        ("torch", "PyTorch", None, True),
        ("numpy", "NumPy", 2, True),  # Exclude numpy 2.x
        ("chromadb", "ChromaDB", None, True),
        ("sentence_transformers", "Sentence Transformers", None, True),
        ("grpc", "gRPC", None, True),
        ("pydantic", "Pydantic", 3, True),  # Exclude pydantic 3.x
        ("rich", "Rich", None, False),
        ("prompt_toolkit", "Prompt Toolkit", None, False),
        ("yaml", "PyYAML", None, False),
        ("tenacity", "Tenacity", None, False),
        ("psutil", "psutil", None, False),
        ("filelock", "filelock", None, False),
        ("zeroconf", "Zeroconf", None, False),
        ("jwt", "PyJWT", None, False),
        ("git", "GitPython", None, False),
        ("pygments", "Pygments", None, False),
        ("rope", "Rope", None, False),
        ("tree_sitter", "Tree-sitter", None, False),
        ("jsonschema", "JSON Schema", None, False),
        ("radon", "Radon", None, False),
        ("bandit", "Bandit", None, False),
    ]

    for module_name, display_name, exclude_major, is_critical in dependencies:
        success, version, issues = test_import(module_name, display_name)

        if success:
            # Check version constraints
            version_ok, version_issues = check_version_constraint(
                version, exclude_major=exclude_major
            )

            if version_ok:
                print_success(f"{display_name:25s} {version}")
            else:
                print_warning(f"{display_name:25s} {version} - {version_issues[0]}")
                if is_critical:
                    failures += 1
                else:
                    warnings += 1
        else:
            if is_critical:
                print_error(f"{display_name:25s} NOT INSTALLED - {issues[0] if issues else 'Import failed'}")
                failures += 1
            else:
                print_warning(f"{display_name:25s} NOT INSTALLED (optional)")
                warnings += 1

    return failures


def test_critical_compatibility() -> int:
    """Test critical compatibility constraints. Returns number of failures."""
    print_section("CRITICAL COMPATIBILITY CHECKS")

    failures = 0

    # Check protobuf version (CRITICAL: must be <5.0.0)
    print_info("Checking protobuf version (CRITICAL: must be <5.0.0)...")
    try:
        from google.protobuf import __version__ as pb_version
        print_info(f"  protobuf version: {pb_version}")

        if pb_version.startswith('5.') or pb_version.startswith('6.'):
            print_error(f"  protobuf {pb_version} is incompatible with grpcio!")
            print_error(f"  SOLUTION: pip install 'protobuf>=4.25.2,<5.0.0'")
            failures += 1
        else:
            print_success(f"  protobuf {pb_version} is compatible")
    except ImportError:
        print_error("  protobuf not installed")
        failures += 1

    print()

    # Check numpy version (CRITICAL: must be <2.0.0)
    print_info("Checking numpy version (CRITICAL: must be <2.0.0)...")
    try:
        import numpy as np
        print_info(f"  numpy version: {np.__version__}")

        if np.__version__.startswith('2.'):
            print_error(f"  numpy {np.__version__} breaks many ML packages!")
            print_error(f"  SOLUTION: pip install 'numpy>=1.26.4,<2.0.0'")
            failures += 1
        else:
            print_success(f"  numpy {np.__version__} is compatible")
    except ImportError:
        print_error("  numpy not installed")
        failures += 1

    print()

    # Check torch and chromadb compatibility
    print_info("Checking torch + chromadb compatibility...")
    try:
        import torch
        import chromadb
        print_info(f"  torch version: {torch.__version__}")
        print_info(f"  chromadb version: {chromadb.__version__}")

        # Try to create a simple ChromaDB client
        try:
            client = chromadb.Client()
            print_success("  ChromaDB client initialization successful")
        except Exception as e:
            print_warning(f"  ChromaDB client initialization warning: {e}")
    except ImportError as e:
        print_warning(f"  Skipping (missing dependencies): {e}")

    print()

    # Check llama-cpp-python
    print_info("Checking llama-cpp-python...")
    try:
        import llama_cpp
        version = llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else "unknown"
        print_success(f"  llama-cpp-python {version} installed")

        # Check for CUDA support
        try:
            # This is a simple check - actual CUDA availability depends on compilation
            print_info("  Checking CUDA support...")
            # We can't easily test CUDA without loading a model, so just report installation
            print_success("  llama-cpp-python is installed (CUDA support depends on build flags)")
        except Exception as e:
            print_info(f"  CUDA check skipped: {e}")
    except ImportError:
        print_error("  llama-cpp-python not installed")
        failures += 1

    return failures


def test_optional_dependencies() -> int:
    """Test optional dependencies. Returns number of warnings."""
    print_section("OPTIONAL DEPENDENCIES")

    warnings = 0

    optional_deps = [
        ("requests", "Requests (web tools)"),
        ("bs4", "BeautifulSoup4 (web parsing)"),
        ("selenium", "Selenium (browser automation)"),
        ("playwright", "Playwright (modern browser automation)"),
    ]

    for module_name, display_name in optional_deps:
        success, version, issues = test_import(module_name, display_name)

        if success:
            print_success(f"{display_name:40s} {version}")
        else:
            print_info(f"{display_name:40s} NOT INSTALLED (optional)")
            warnings += 1

    return warnings


def check_pip_dependencies() -> int:
    """Check for dependency conflicts using pip check. Returns number of issues."""
    print_section("PIP DEPENDENCY CHECK")

    print_info("Running 'pip check' to detect dependency conflicts...")
    print()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print_success("No dependency conflicts detected")
            return 0
        else:
            print_error("Dependency conflicts detected:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return 1

    except subprocess.TimeoutExpired:
        print_warning("pip check timed out")
        return 0
    except Exception as e:
        print_warning(f"Could not run pip check: {e}")
        return 0


def main() -> int:
    """Main test function. Returns exit code."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{'JENOVA AI - Dependency Compatibility Test':^70}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}")

    total_failures = 0
    total_warnings = 0

    # Run all tests
    total_failures += test_core_dependencies()
    total_failures += test_critical_compatibility()
    total_warnings += test_optional_dependencies()
    total_failures += check_pip_dependencies()

    # Print summary
    print_section("SUMMARY")

    if total_failures == 0:
        print_success(f"All critical tests passed!")
        if total_warnings > 0:
            print_warning(f"{total_warnings} optional dependencies not installed")
            print_info("Install with: pip install jenova-ai[web,browser]")
        print()
        print(f"{Colors.GREEN}✅ Dependencies are compatible - installation is good!{Colors.NC}")
        return 0
    else:
        print_error(f"{total_failures} critical issues found")
        if total_warnings > 0:
            print_warning(f"{total_warnings} optional dependencies not installed")
        print()
        print(f"{Colors.RED}❌ Please fix the issues above before using JENOVA AI{Colors.NC}")
        print()
        print("Common fixes:")
        print(f"  1. Reinstall with constraints: pip install -r requirements.txt --upgrade")
        print(f"  2. Fix protobuf: pip install 'protobuf>=4.25.2,<5.0.0' --upgrade")
        print(f"  3. Fix numpy: pip install 'numpy>=1.26.4,<2.0.0' --upgrade")
        return 1


if __name__ == "__main__":
    sys.exit(main())
