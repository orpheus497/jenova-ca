##Script function and purpose: Pytest configuration and shared fixtures
"""
Test Configuration

Shared fixtures and configuration for JENOVA tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

##Fix: Import Python 3.14 compatibility patch BEFORE any ChromaDB usage (D3-2026-02-11T07:36:09Z)
##Note: This must be imported before jenova.memory or any ChromaDB-dependent modules
import jenova.compat_py314  # noqa: F401  # Monkey-patches Pydantic V1 for Python 3.14

##Mock purpose: Mock onnxruntime before chromadb imports it (not available on FreeBSD)
##Note: This mock is required because chromadb creates DefaultEmbeddingFunction at class
##definition time, which requires onnxruntime. JENOVA uses its own embedding functions.
if "onnxruntime" not in sys.modules:
    onnxruntime_mock = MagicMock()
    onnxruntime_mock.__version__ = "1.14.1"
    sys.modules["onnxruntime"] = onnxruntime_mock

import pytest
from chromadb.api.types import EmbeddingFunction

from jenova.config.models import GraphConfig, JenovaConfig, MemoryConfig
from jenova.memory import Memory, MemoryType


##Class purpose: Shared mock embedding function for test fixtures
class MockEmbeddingFunction(EmbeddingFunction):
    """Mock embedding function returning fixed-dimension vectors."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in input]


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
        embedding_function=MockEmbeddingFunction(),
    )


##Function purpose: Provide semantic memory fixture
@pytest.fixture
def semantic_memory(tmp_storage: Path) -> Memory:
    """Provide a semantic memory instance."""
    return Memory(
        memory_type=MemoryType.SEMANTIC,
        storage_path=tmp_storage / "semantic",
        embedding_function=MockEmbeddingFunction(),
    )
