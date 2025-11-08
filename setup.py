from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
import subprocess
import sys


class BuildProtoCommand(build_py):
    """Custom build command that compiles Protocol Buffers."""

    def run(self):
        """Run proto compilation before standard build."""
        print("Compiling Protocol Buffers...")
        result = subprocess.run([sys.executable, "build_proto.py"], cwd=".")
        if result.returncode != 0:
            sys.exit(result.returncode)
        build_py.run(self)


class DevelopProtoCommand(develop):
    """Custom develop command that compiles Protocol Buffers."""

    def run(self):
        """Run proto compilation before development install."""
        print("Compiling Protocol Buffers for development...")
        result = subprocess.run([sys.executable, "build_proto.py"], cwd=".")
        if result.returncode != 0:
            sys.exit(result.returncode)
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