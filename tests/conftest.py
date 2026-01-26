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


##Function purpose: Provide episodic memory fixture
@pytest.fixture
def episodic_memory(tmp_storage: Path) -> Memory:
    """Provide an episodic memory instance."""
    return Memory(
        memory_type=MemoryType.EPISODIC,
        storage_path=tmp_storage / "episodic",
    )


##Function purpose: Provide semantic memory fixture
@pytest.fixture
def semantic_memory(tmp_storage: Path) -> Memory:
    """Provide a semantic memory instance."""
    return Memory(
        memory_type=MemoryType.SEMANTIC,
        storage_path=tmp_storage / "semantic",
    )
