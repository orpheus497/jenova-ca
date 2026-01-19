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

### GPU/CUDA Support

JENOVA supports GPU acceleration via llama-cpp-python for faster inference.

**For NVIDIA GPUs (CUDA):**
```bash
# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
```

**For Apple Silicon (Metal):**
```bash
# Metal support is usually enabled by default on macOS
pip install llama-cpp-python
```

**Configuration:**
```yaml
hardware:
  gpu_layers: all  # Use all GPU layers (or "none" for CPU only)
```

The system automatically detects GPU availability and falls back to CPU if GPU is unavailable. See `.devdocs/builders/logic_engineer/CUDA_SUPPORT.md` for detailed documentation.

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

JENOVA is designed for **100% POSIX/UNIX compliance** and supports:

| Platform | Status | Notes |
|----------|--------|-------|
| **FreeBSD** | ✅ Fully Supported | Tested on FreeBSD 13.x, 14.x |
| **Linux** | ✅ Fully Supported | Tested on Ubuntu 22.04+, Debian 12+, Fedora 38+, Arch Linux |
| **macOS** | ✅ Fully Supported | Tested on macOS 11.0+ (Apple Silicon and Intel) |

### Platform Requirements

- **Python**: 3.10+ (native packages available on all POSIX platforms)
- **ChromaDB**: Uses SQLite backend (native on all POSIX systems)
- **llama-cpp-python**: Compiles natively with system compilers
- **POSIX Compliance**: 100% POSIX/UNIX compliant codebase
- **No Windows Support**: This is a POSIX-first architecture

### POSIX/UNIX Standards

JENOVA follows strict POSIX/UNIX standards:
- ✅ POSIX path conventions (`/` separator only, no backslashes)
- ✅ POSIX file operations (atomic writes, octal permissions)
- ✅ POSIX shell commands only (no Windows commands)
- ✅ LF line endings (enforced via `.editorconfig`)
- ✅ UTF-8 encoding throughout
- ✅ POSIX-compliant subprocess execution

See `.devdocs/builders/logic_engineer/POSIX_COMPLIANCE.md` for detailed compliance documentation.

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
