#!/usr/bin/env python3
# The JENOVA Cognitive Architecture - Protocol Buffer Build Script
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Build script for compiling Protocol Buffer definitions.

This script compiles the JENOVA RPC service definitions into Python code
using the Protocol Buffer compiler (protoc) via grpcio-tools.
"""

import sys
from pathlib import Path
from grpc_tools import protoc


def compile_protos():
    """
    Compile Protocol Buffer definitions to Python code.

    Generates:
        - jenova_pb2.py: Message class definitions
        - jenova_pb2_grpc.py: gRPC service stubs and servicers

    Returns:
        int: Exit code (0 = success, 1 = failure)
    """
    # Paths
    repo_root = Path(__file__).parent
    proto_dir = repo_root / "src" / "jenova" / "network" / "proto"
    proto_file = proto_dir / "jenova.proto"

    # Verify proto file exists
    if not proto_file.exists():
        print(f"ERROR: Proto file not found: {proto_file}", file=sys.stderr)
        return 1

    print(f"Compiling Protocol Buffers from {proto_file}...")

    # Build protoc arguments
    # We need to:
    # 1. Set proto_path to the proto directory
    # 2. Set python_out to the proto directory
    # 3. Set grpc_python_out to the proto directory
    # 4. Specify the proto file
    proto_include = str(proto_dir)
    python_out = str(proto_dir)

    command = [
        'grpc_tools.protoc',
        f'--proto_path={proto_include}',
        f'--python_out={python_out}',
        f'--grpc_python_out={python_out}',
        str(proto_file.name)
    ]

    print(f"Running: {' '.join(command)}")

    # Run protoc
    result = protoc.main(command)

    if result != 0:
        print(f"ERROR: protoc compilation failed with exit code {result}", file=sys.stderr)
        return 1

    # Verify generated files
    pb2_file = proto_dir / "jenova_pb2.py"
    pb2_grpc_file = proto_dir / "jenova_pb2_grpc.py"

    if not pb2_file.exists():
        print(f"ERROR: Expected generated file not found: {pb2_file}", file=sys.stderr)
        return 1

    if not pb2_grpc_file.exists():
        print(f"ERROR: Expected generated file not found: {pb2_grpc_file}", file=sys.stderr)
        return 1

    print("✓ Protocol Buffer compilation successful!")
    print(f"  Generated: {pb2_file.name}")
    print(f"  Generated: {pb2_grpc_file.name}")

    # Create __init__.py if it doesn't exist
    init_file = proto_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            "# Generated Protocol Buffer package\n"
            "# This package contains compiled protobuf definitions for JENOVA RPC\n"
        )
        print(f"  Created: {init_file.name}")

    return 0


if __name__ == "__main__":
    sys.exit(compile_protos())
