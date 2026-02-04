##Script function and purpose: Pytest configuration and shared fixtures
"""
Test Configuration

Shared fixtures and configuration for JENOVA tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

##Fix: Fail fast on Python 3.14+ so agents/users get a clear message instead of pip resolution
##Note: chromadb requires onnxruntime; onnxruntime has no wheels for Python 3.14, so
##pip install -e ".[dev]" fails. Use Python 3.10, 3.11, 3.12, or 3.13 (e.g. pyenv, or CI uses 3.11).
if sys.version_info >= (3, 14):
    raise RuntimeError(
        "Python 3.14+ is not supported for development: chromadb depends on "
        "onnxruntime, which has no wheels for 3.14. Use Python 3.10, 3.11, "
        "3.12, or 3.13 (e.g. pyenv install 3.11; pyenv local 3.11)."
    )

##Mock purpose: Mock onnxruntime before chromadb imports it (not available on FreeBSD)
##Note: This mock is required because chromadb creates DefaultEmbeddingFunction at class
##definition time, which requires onnxruntime. JENOVA uses its own embedding functions.
if "onnxruntime" not in sys.modules:
    onnxruntime_mock = MagicMock()
    onnxruntime_mock.__version__ = "1.14.1"
    sys.modules["onnxruntime"] = onnxruntime_mock

##Fix: Mock llama_cpp when not installed or when native lib fails so CI can run
##Note: Catch ImportError, OSError, RuntimeError so native-library failures use mock
if "llama_cpp" not in sys.modules:
    try:
        import llama_cpp  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        _llama_mock = MagicMock()
        _llama_mock.LlamaGrammar = MagicMock()
        _llama_mock.Llama = MagicMock()
        sys.modules["llama_cpp"] = _llama_mock

import pytest

from jenova.config.models import GraphConfig, JenovaConfig, MemoryConfig
from jenova.memory import Memory, MemoryType

##Fix: Avoid ChromaDB default ONNX embedding in unit tests (numpy broadcast errors in CI)
##Note: Custom embedding returns fixed-dimension vectors so add/search work without ONNX
_TEST_EMBED_DIM = 384


class _TestEmbedding:
    """Minimal embedding for unit tests; avoids chromadb default ONNX."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * _TEST_EMBED_DIM for _ in texts]


##Function purpose: Provide temporary directory fixture
@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Provide a temporary storage directory."""
    storage = tmp_path / "storage"
    storage.mkdir()
    return storage


##Function purpose: Provide default config fixture
@pytest.fixture
def config(tmp_storage: Path) -> JenovaConfig:
    """Provide a test configuration."""
    return JenovaConfig(
        memory=MemoryConfig(storage_path=tmp_storage / "memory"),
        graph=GraphConfig(storage_path=tmp_storage / "graph"),
        debug=True,
    )


##Function purpose: Provide episodic memory fixture
@pytest.fixture
def episodic_memory(tmp_storage: Path) -> Memory:
    """Provide an episodic memory instance."""
    return Memory(
        memory_type=MemoryType.EPISODIC,
        storage_path=tmp_storage / "episodic",
        embedding_function=_TestEmbedding(),
    )


##Function purpose: Provide semantic memory fixture
@pytest.fixture
def semantic_memory(tmp_storage: Path) -> Memory:
    """Provide a semantic memory instance."""
    return Memory(
        memory_type=MemoryType.SEMANTIC,
        storage_path=tmp_storage / "semantic",
        embedding_function=_TestEmbedding(),
    )
