from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
import subprocess
import sys


class BuildProtoCommand(build_py):
    """Custom build command that compiles Protocol Buffers."""

    def run(self):
        """Run proto compilation before standard build."""
        # Check if grpcio-tools is available (needed for proto compilation)
        try:
            import grpc_tools
            print("Compiling Protocol Buffers...")
            result = subprocess.run([sys.executable, "build_proto.py"], cwd=".")
            if result.returncode != 0:
                print("WARNING: Protocol Buffer compilation failed.")
                print("         Proto files will be compiled on first import if needed.")
                print("         This is not a fatal error for installation.")
            else:
                print("✓ Protocol Buffers compiled successfully")
        except ImportError:
            print("INFO: grpcio-tools not yet installed, skipping proto compilation.")
            print("      Proto files will be compiled automatically on first import.")
            print("      This is normal during initial installation.")

        # Continue with standard build regardless of proto compilation result
        build_py.run(self)


class DevelopProtoCommand(develop):
    """Custom develop command that compiles Protocol Buffers."""

    def run(self):
        """Run proto compilation before development install."""
        # Check if grpcio-tools is available (needed for proto compilation)
        try:
            import grpc_tools
            print("Compiling Protocol Buffers for development...")
            result = subprocess.run([sys.executable, "build_proto.py"], cwd=".")
            if result.returncode != 0:
                print("WARNING: Protocol Buffer compilation failed.")
                print("         Proto files will be compiled on first import if needed.")
                print("         This is not a fatal error for installation.")
            else:
                print("✓ Protocol Buffers compiled successfully")
        except ImportError:
            print("INFO: grpcio-tools not yet installed, skipping proto compilation.")
            print("      Proto files will be compiled automatically on first import.")
            print("      This is normal during initial installation.")

        # Continue with development install regardless of proto compilation result
        develop.run(self)


# Configuration is now primarily in pyproject.toml
# This file is kept for backward compatibility with tools that don't yet support PEP 517
# Custom commands are added to automatically compile Protocol Buffers
setup(
    cmdclass={
        'build_py': BuildProtoCommand,
        'develop': DevelopProtoCommand,
    }
)