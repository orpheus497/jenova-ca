##Script function and purpose: Pytest configuration and shared fixtures
"""
Test Configuration

Shared fixtures and configuration for JENOVA tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

##Mock purpose: Mock onnxruntime before chromadb imports it (not available on FreeBSD)
##Note: This mock is required because chromadb creates DefaultEmbeddingFunction at class
##definition time, which requires onnxruntime. JENOVA uses its own embedding functions.
if "onnxruntime" not in sys.modules:
    onnxruntime_mock = MagicMock()
    onnxruntime_mock.__version__ = "1.14.1"
    sys.modules["onnxruntime"] = onnxruntime_mock

import pytest

from jenova.config.models import GraphConfig, JenovaConfig, MemoryConfig
from jenova.memory import Memory, MemoryType


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


##Class purpose: Mock embedding function for testing
class MockEmbeddingFunction:
    """Mock embedding function for tests to avoid external dependencies."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate fake embeddings."""
        # Return constant vector of dimension 3 for predictable testing
        return [[0.1, 0.2, 0.3] for _ in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Alias for __call__ to satisfy ChromaDB interface."""
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Alias for __call__ to satisfy ChromaDB interface."""
        return self(input)

    def name(self) -> str:
        """Return name of embedding function."""
        return "mock_embedding"


##Function purpose: Provide episodic memory fixture
@pytest.fixture
def episodic_memory(tmp_storage: Path) -> Memory:
    """Provide an episodic memory instance."""
    return Memory(
        memory_type=MemoryType.EPISODIC,
        storage_path=tmp_storage / "episodic",
        embedding_function=MockEmbeddingFunction(),  # type: ignore
    )


##Function purpose: Provide semantic memory fixture
@pytest.fixture
def semantic_memory(tmp_storage: Path) -> Memory:
    """Provide a semantic memory instance."""
    return Memory(
        memory_type=MemoryType.SEMANTIC,
        storage_path=tmp_storage / "semantic",
        embedding_function=MockEmbeddingFunction(),  # type: ignore
    )
