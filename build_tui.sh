#!/bin/bash
# Build script for JENOVA Bubble Tea UI

set -e

echo "Building JENOVA Bubble Tea TUI..."

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed. Please install Go 1.20 or later."
    exit 1
fi

# Navigate to tui directory
cd "$(dirname "$0")/tui"

# Ensure dependencies are up to date
echo "Fetching Go dependencies..."
go mod download
go mod tidy

# Build the TUI
echo "Compiling TUI binary..."
go build -o jenova-tui main.go

if [ -f jenova-tui ]; then
    echo "✓ TUI binary built successfully: tui/jenova-tui"
    echo ""
    echo "To use the Bubble Tea UI, run:"
    echo "  export JENOVA_UI=bubbletea"
    echo "  ./jenova"
    echo ""
    echo "Or run directly:"
    echo "  JENOVA_UI=bubbletea ./jenova"
else
    echo "✗ Failed to build TUI binary"
    exit 1
fi
