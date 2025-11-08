#!/usr/bin/env python3
# The JENOVA Cognitive Architecture - Build Verification Script
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Build verification script for JENOVA Phase 8 distributed features.

Verifies that all required components are properly built and importable:
- Protocol Buffer compiled files
- Network layer modules
- Distributed LLM and memory interfaces
"""

import sys
from pathlib import Path


def verify_proto_files():
    """Verify Protocol Buffer compiled files exist and are importable."""
    print("Checking Protocol Buffer compiled files...")

    proto_dir = Path("src/jenova/network/proto")

    # Check files exist
    pb2_file = proto_dir / "jenova_pb2.py"
    pb2_grpc_file = proto_dir / "jenova_pb2_grpc.py"

    if not pb2_file.exists():
        print(f"  ✗ FAIL: Missing {pb2_file}")
        return False

    if not pb2_grpc_file.exists():
        print(f"  ✗ FAIL: Missing {pb2_grpc_file}")
        return False

    # Try importing
    try:
        sys.path.insert(0, str(Path("src").absolute()))
        import jenova.network.proto.jenova_pb2 as pb2
        import jenova.network.proto.jenova_pb2_grpc as pb2_grpc
        print("  ✓ Protocol Buffer files compiled and importable")
        return True
    except ImportError as e:
        print(f"  ✗ FAIL: Cannot import protobuf modules: {e}")
        return False


def verify_network_modules():
    """Verify network layer modules are importable."""
    print("Checking network layer modules...")

    modules = [
        "jenova.network.discovery",
        "jenova.network.peer_manager",
        "jenova.network.rpc_service",
        "jenova.network.rpc_client",
        "jenova.network.security",
        "jenova.network.metrics",
    ]

    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ FAIL: {module_name} - {e}")
            all_ok = False

    return all_ok


def verify_distributed_interfaces():
    """Verify distributed LLM and memory interfaces are importable."""
    print("Checking distributed interfaces...")

    modules = [
        "jenova.llm.distributed_llm_interface",
        "jenova.memory.distributed_memory_search",
    ]

    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ FAIL: {module_name} - {e}")
            all_ok = False

    return all_ok


def verify_dependencies():
    """Verify required dependencies are installed."""
    print("Checking network dependencies...")

    dependencies = [
        ("zeroconf", "0.132.2"),
        ("grpcio", "1.60.1"),
        ("grpcio_tools", "1.60.1"),
        ("google.protobuf", "4.25.2"),
        ("jwt", "PyJWT 2.8.0"),
    ]

    all_ok = True
    for module_name, expected_version in dependencies:
        try:
            if module_name == "jwt":
                import jwt
                version = jwt.__version__
                print(f"  ✓ PyJWT {version}")
            elif module_name == "google.protobuf":
                from google import protobuf
                # Protobuf version is in __version__
                import google.protobuf
                version = google.protobuf.__version__
                print(f"  ✓ protobuf {version}")
            else:
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")
                print(f"  ✓ {module_name} {version}")
        except ImportError as e:
            print(f"  ✗ FAIL: {module_name} - {e}")
            all_ok = False

    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("JENOVA Phase 8 Build Verification")
    print("=" * 60)
    print()

    checks = [
        ("Protocol Buffer Compilation", verify_proto_files),
        ("Network Layer Modules", verify_network_modules),
        ("Distributed Interfaces", verify_distributed_interfaces),
        ("Network Dependencies", verify_dependencies),
    ]

    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False

    print()

    if all_passed:
        print("✓ All verification checks passed!")
        print("Phase 8 distributed computing layer is ready.")
        return 0
    else:
        print("✗ Some verification checks failed.")
        print("Please review errors above and rebuild.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
