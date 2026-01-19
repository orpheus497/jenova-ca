# JENOVA

##Script function and purpose: Project README - Provides overview, installation, and usage instructions.

> Self-aware AI cognitive architecture with graph-based memory and RAG-based response generation.

## Overview

JENOVA is a cognitive AI system featuring:

- **Multi-layered Memory**: Unified ChromaDB-based memory system
- **Graph-based Cognition**: Dict-based cognitive graph for relationship tracking
- **RAG Response Generation**: Context-aware response generation via LLM
- **Fine-tunable Embeddings**: Custom embedding model that evolves with usage

## Installation

```bash
# Install in development mode
pip install -e ".[dev]"

# Install with fine-tuning support
pip install -e ".[dev,finetune]"
```

## Usage

```bash
# Show help
jenova --help

# Run with TUI
jenova

# Run without TUI (CLI mode)
jenova --no-tui

# Run with custom config
jenova --config path/to/config.yaml

# Run in debug mode
jenova --debug
```

## Development

```bash
# Run tests
pytest -v --tb=short

# Run unit tests only
pytest -v tests/unit/

# Run integration tests
pytest -v tests/integration/ -m integration

# Type checking
mypy src/jenova/

# Linting
ruff check src/
```

## Project Structure

```
jenova-ca/
├── src/jenova/          # Main package
│   ├── config/          # Configuration (Pydantic models)
│   ├── core/            # Cognitive engine and knowledge store
│   ├── memory/          # Unified memory system
│   ├── graph/           # Dict-based cognitive graph
│   ├── llm/             # LLM interface
│   ├── embeddings/      # JENOVA embedding model
│   ├── ui/              # Textual TUI
│   └── utils/           # Logging, migrations
├── finetune/            # Embedding model fine-tuning
├── tests/               # Unit and integration tests
└── pyproject.toml       # Project configuration
```

## Platform Support

JENOVA is designed for **native, out-of-the-box support** on:

| Platform | Status | Notes |
|----------|--------|-------|
| **FreeBSD** | ✅ Fully Supported | Tested on FreeBSD 13.x, 14.x |
| **Linux** | ✅ Fully Supported | Tested on Ubuntu 22.04+, Debian 12+, Fedora 38+ |

### Platform Requirements

- **Python**: 3.10+ (native packages available on both platforms)
- **ChromaDB**: Uses SQLite backend (native on FreeBSD/Linux)
- **llama-cpp-python**: Compiles natively with system compilers
- **No Windows-specific dependencies**: Pure POSIX-compliant codebase

### FreeBSD-Specific Notes

```bash
# Install Python and pip
pkg install python311 py311-pip

# Optional: Install with GPU support (if available)
pkg install py311-pytorch  # For fine-tuning
```

### Linux-Specific Notes

```bash
# Debian/Ubuntu
apt install python3.11 python3.11-venv python3-pip

# Fedora/RHEL
dnf install python3.11 python3-pip

# Arch Linux
pacman -S python python-pip
```

### Cross-Platform Compatibility

- All file paths use POSIX conventions (`/` separators)
- Line endings enforced as LF via `.editorconfig`
- No platform-specific code paths or conditionals
- Atomic file operations use POSIX `rename()` semantics

## License

AGPL-3.0 - See LICENSE file for details.
