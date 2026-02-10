#!/bin/bash
# Install script for JENOVA

echo "Installing JENOVA dependencies and package in editable mode..."
pip install -e ".[dev]"

echo ""
echo "Installation complete!"
echo "To run JENOVA, type: jenova"
