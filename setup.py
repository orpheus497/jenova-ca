##Script function and purpose: Setup script for The JENOVA Cognitive Architecture
##This script provides backward compatibility with tools that don't support PEP 517
##Primary configuration is in pyproject.toml

from setuptools import setup

##Block purpose: Invoke setup with configuration from pyproject.toml
# Configuration is now primarily in pyproject.toml
# This file is kept for backward compatibility with tools that don't yet support PEP 517
setup()