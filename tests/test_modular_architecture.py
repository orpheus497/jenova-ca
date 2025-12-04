# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Tests for the modular JENOVA Cognitive Architecture.

These tests verify that the pluggable interfaces and CognitiveArchitecture
work correctly with both default and custom implementations.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Optional, Dict, Any

from jenova.core.interfaces.base import (
    LLMAdapter,
    EmbeddingProvider,
    MemoryBackend,
    MemoryEntry,
    MemoryType,
    SearchResult,
    Logger,
    CognitiveComponent,
    KnowledgeGraph,
)
from jenova.core.architecture import CognitiveArchitecture, CognitiveConfig


# =============================================================================
# Mock Implementations for Testing
# =============================================================================


class MockLLM:
    """Mock LLM adapter for testing."""

    def __init__(self):
        self.generate_calls = []
        self.responses = {}

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        self.generate_calls.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        # Return a canned response or custom response if set
        return self.responses.get(prompt, f"Mock response to: {prompt[:50]}...")

    def generate_with_context(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        self.generate_calls.append({
            "prompt": prompt,
            "context": context,
            "temperature": temperature,
        })
        return f"Mock RAG response. Context items: {len(context)}"

    def set_response(self, prompt: str, response: str):
        """Set a specific response for a prompt."""
        self.responses[prompt] = response


class MockEmbedding:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self.embed_calls = []

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        self.embed_calls.append(texts)
        # Return dummy embeddings
        return [[0.1] * self._dimension for _ in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> List[float]:
        return self.embed(query)[0]


class MockMemoryBackend:
    """Mock memory backend for testing."""

    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}
        self.store_calls = []
        self.search_calls = []

    def store(self, entry: MemoryEntry) -> str:
        self.entries[entry.id] = entry
        self.store_calls.append(entry)
        return entry.id

    def search(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[MemoryType] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        self.search_calls.append({
            "query": query,
            "n_results": n_results,
            "memory_type": memory_type,
            "user": user,
        })

        # Return matching entries
        results = []
        for entry in self.entries.values():
            if memory_type and entry.memory_type != memory_type:
                continue
            if user and entry.user != user:
                continue
            results.append(SearchResult(
                entry=entry,
                score=0.8,
                distance=0.2,
            ))
            if len(results) >= n_results:
                break

        return results

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            del self.entries[entry_id]
            return True
        return False

    def count(
        self,
        memory_type: Optional[MemoryType] = None,
        user: Optional[str] = None
    ) -> int:
        count = 0
        for entry in self.entries.values():
            if memory_type and entry.memory_type != memory_type:
                continue
            if user and entry.user != user:
                continue
            count += 1
        return count


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.logs = []

    def info(self, message: str, **kwargs):
        self.logs.append(("INFO", message))

    def warning(self, message: str, **kwargs):
        self.logs.append(("WARNING", message))

    def error(self, message: str, error=None, **kwargs):
        self.logs.append(("ERROR", message))

    def debug(self, message: str, **kwargs):
        self.logs.append(("DEBUG", message))


# =============================================================================
# Interface Tests
# =============================================================================


class TestLLMAdapter:
    """Tests for LLM adapter interface compliance."""

    def test_mock_llm_implements_protocol(self):
        """Test that MockLLM implements LLMAdapter protocol."""
        llm = MockLLM()

        # Test generate method
        response = llm.generate("Hello")
        assert isinstance(response, str)

        # Test generate_with_context method
        response = llm.generate_with_context("Hello", ["context1", "context2"])
        assert isinstance(response, str)

    def test_llm_generate_tracking(self):
        """Test that LLM calls are tracked."""
        llm = MockLLM()

        llm.generate("Test prompt", temperature=0.5)

        assert len(llm.generate_calls) == 1
        assert llm.generate_calls[0]["prompt"] == "Test prompt"
        assert llm.generate_calls[0]["temperature"] == 0.5

    def test_llm_custom_response(self):
        """Test custom response setting."""
        llm = MockLLM()
        llm.set_response("specific prompt", "specific response")

        response = llm.generate("specific prompt")

        assert response == "specific response"


class TestEmbeddingProvider:
    """Tests for embedding provider interface compliance."""

    def test_mock_embedding_implements_protocol(self):
        """Test that MockEmbedding implements EmbeddingProvider protocol."""
        embedding = MockEmbedding(dimension=768)

        # Test embed method
        embeddings = embedding.embed("test text")
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768

        # Test dimension property
        assert embedding.dimension == 768

        # Test embed_query method
        query_embedding = embedding.embed_query("query")
        assert len(query_embedding) == 768

    def test_batch_embedding(self):
        """Test batch embedding."""
        embedding = MockEmbedding()

        texts = ["text1", "text2", "text3"]
        embeddings = embedding.embed(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)


class TestMemoryBackend:
    """Tests for memory backend interface compliance."""

    def test_mock_memory_implements_interface(self):
        """Test that MockMemoryBackend implements MemoryBackend interface."""
        memory = MockMemoryBackend()

        # Test store
        entry = MemoryEntry(
            id="test_1",
            content="Test content",
            memory_type=MemoryType.SEMANTIC,
            user="alice",
        )
        entry_id = memory.store(entry)
        assert entry_id == "test_1"

        # Test get
        retrieved = memory.get("test_1")
        assert retrieved is not None
        assert retrieved.content == "Test content"

        # Test search
        results = memory.search("test", user="alice")
        assert len(results) == 1

        # Test count
        assert memory.count() == 1
        assert memory.count(user="alice") == 1
        assert memory.count(user="bob") == 0

        # Test delete
        assert memory.delete("test_1") is True
        assert memory.get("test_1") is None

    def test_memory_type_filtering(self):
        """Test filtering by memory type."""
        memory = MockMemoryBackend()

        # Add entries of different types
        memory.store(MemoryEntry(
            id="semantic_1",
            content="A fact",
            memory_type=MemoryType.SEMANTIC,
            user="alice",
        ))
        memory.store(MemoryEntry(
            id="episodic_1",
            content="A conversation",
            memory_type=MemoryType.EPISODIC,
            user="alice",
        ))

        # Search by type
        semantic_results = memory.search("query", memory_type=MemoryType.SEMANTIC)
        assert len(semantic_results) == 1
        assert semantic_results[0].entry.memory_type == MemoryType.SEMANTIC

        episodic_results = memory.search("query", memory_type=MemoryType.EPISODIC)
        assert len(episodic_results) == 1
        assert episodic_results[0].entry.memory_type == MemoryType.EPISODIC


# =============================================================================
# CognitiveArchitecture Tests
# =============================================================================


class TestCognitiveArchitecture:
    """Tests for the main CognitiveArchitecture class."""

    @pytest.fixture
    def mock_arch(self):
        """Create a CognitiveArchitecture with mock components."""
        return CognitiveArchitecture(
            llm=MockLLM(),
            memory_backend=MockMemoryBackend(),
            embedding_provider=MockEmbedding(),
            logger=MockLogger(),
        )

    def test_architecture_initialization(self, mock_arch):
        """Test that architecture initializes correctly."""
        assert mock_arch.llm is not None
        assert mock_arch.memory is not None
        assert mock_arch.embedding is not None
        assert mock_arch.logger is not None

    def test_think_basic(self, mock_arch):
        """Test basic think operation."""
        response = mock_arch.think("Hello", user="alice")

        assert isinstance(response, str)
        assert "Mock" in response or "response" in response.lower()

    def test_think_stores_in_episodic_memory(self, mock_arch):
        """Test that think stores conversation in episodic memory."""
        mock_arch.think("Hello", user="alice")

        # Check memory was stored
        assert mock_arch.memory.count(memory_type=MemoryType.EPISODIC, user="alice") == 1

    def test_think_updates_history(self, mock_arch):
        """Test that think updates conversation history."""
        mock_arch.think("Hello", user="alice")
        mock_arch.think("How are you?", user="alice")

        history = mock_arch.get_history("alice")
        assert len(history) == 4  # 2 user messages + 2 responses

    def test_think_tracks_turn_count(self, mock_arch):
        """Test that think tracks turn count."""
        mock_arch.think("Hello", user="alice")
        mock_arch.think("Question 2", user="alice")
        mock_arch.think("Question 3", user="alice")

        assert mock_arch.get_turn_count("alice") == 3
        assert mock_arch.get_turn_count("bob") == 0

    def test_retrieve(self, mock_arch):
        """Test memory retrieval."""
        # Store some memories first
        mock_arch.remember("Python is a programming language", user="alice")
        mock_arch.remember("JavaScript is used for web development", user="alice")

        # Retrieve
        context = mock_arch.retrieve("programming", user="alice")

        assert len(context) > 0

    def test_remember(self, mock_arch):
        """Test storing memories."""
        entry_id = mock_arch.remember(
            "Important fact",
            user="alice",
            memory_type=MemoryType.SEMANTIC,
        )

        assert entry_id is not None
        assert mock_arch.memory.count(memory_type=MemoryType.SEMANTIC) == 1

    def test_remember_different_types(self, mock_arch):
        """Test storing different memory types."""
        mock_arch.remember("A fact", user="alice", memory_type="semantic")
        mock_arch.remember("A procedure", user="alice", memory_type="procedural")

        assert mock_arch.memory.count(memory_type=MemoryType.SEMANTIC) == 1
        assert mock_arch.memory.count(memory_type=MemoryType.PROCEDURAL) == 1

    def test_clear_history(self, mock_arch):
        """Test clearing conversation history."""
        mock_arch.think("Hello", user="alice")
        mock_arch.think("Hello again", user="alice")

        assert len(mock_arch.get_history("alice")) > 0
        assert mock_arch.get_turn_count("alice") > 0

        mock_arch.clear_history("alice")

        assert len(mock_arch.get_history("alice")) == 0
        assert mock_arch.get_turn_count("alice") == 0

    def test_health_check(self, mock_arch):
        """Test health check functionality."""
        health = mock_arch.health_check()

        assert health["healthy"] is True
        assert "components" in health
        assert "llm" in health["components"]
        assert "memory" in health["components"]

    def test_lifecycle_methods(self, mock_arch):
        """Test lifecycle methods don't raise errors."""
        mock_arch.initialize()
        mock_arch.start()
        mock_arch.stop()
        mock_arch.dispose()

        # Verify history was cleared on dispose
        assert len(mock_arch._history) == 0

    def test_user_isolation(self, mock_arch):
        """Test that different users are isolated."""
        mock_arch.think("Hello from Alice", user="alice")
        mock_arch.think("Hello from Bob", user="bob")

        assert mock_arch.get_turn_count("alice") == 1
        assert mock_arch.get_turn_count("bob") == 1

        alice_history = mock_arch.get_history("alice")
        bob_history = mock_arch.get_history("bob")

        assert "Alice" in str(alice_history) or "alice" in str(alice_history).lower()
        assert "Bob" in str(bob_history) or "bob" in str(bob_history).lower()


class TestCognitiveConfig:
    """Tests for CognitiveConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CognitiveConfig()

        assert config.memory_cache_size == 100
        assert config.insight_interval == 5
        assert config.assumption_interval == 7
        assert config.max_context_items == 10
        assert config.rerank_enabled is True
        assert config.llm_timeout == 120

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CognitiveConfig(
            memory_cache_size=200,
            insight_interval=10,
            llm_timeout=60,
        )

        assert config.memory_cache_size == 200
        assert config.insight_interval == 10
        assert config.llm_timeout == 60


class TestArchitectureWithoutComponents:
    """Tests for architecture behavior when components are missing."""

    def test_think_without_llm_raises_error(self):
        """Test that think raises error without LLM."""
        arch = CognitiveArchitecture()

        with pytest.raises(RuntimeError, match="No LLM adapter"):
            arch.think("Hello", user="alice")

    def test_remember_without_memory_raises_error(self):
        """Test that remember raises error without memory backend."""
        arch = CognitiveArchitecture(llm=MockLLM())

        with pytest.raises(RuntimeError, match="No memory backend"):
            arch.remember("fact", user="alice")

    def test_retrieve_without_memory_returns_empty(self):
        """Test that retrieve returns empty list without memory backend."""
        arch = CognitiveArchitecture(llm=MockLLM())

        result = arch.retrieve("query", user="alice")

        assert result == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestArchitectureIntegration:
    """Integration tests for the architecture."""

    def test_full_cognitive_cycle(self):
        """Test a full cognitive cycle with mock components."""
        llm = MockLLM()
        memory = MockMemoryBackend()

        # Pre-populate memory
        memory.store(MemoryEntry(
            id="fact_1",
            content="Python was created by Guido van Rossum",
            memory_type=MemoryType.SEMANTIC,
            user="alice",
        ))

        arch = CognitiveArchitecture(
            llm=llm,
            memory_backend=memory,
            logger=MockLogger(),
        )

        # Think with pre-existing memory
        response = arch.think("Tell me about Python", user="alice")

        # Verify LLM was called
        assert len(llm.generate_calls) > 0

        # Verify memory was searched
        assert len(memory.search_calls) > 0

    def test_insight_generation_trigger(self):
        """Test that insight generation is triggered at configured intervals."""
        config = CognitiveConfig(insight_interval=2)

        # Create mock insight generator
        mock_insight_gen = Mock()
        mock_insight_gen.generate_insight = Mock(return_value={
            "topic": "test",
            "insight": "test insight"
        })
        mock_insight_gen.store_insight = Mock(return_value="insight_1")

        arch = CognitiveArchitecture(
            llm=MockLLM(),
            memory_backend=MockMemoryBackend(),
            insight_generator=mock_insight_gen,
            config=config,
        )

        # First turn - no insight
        arch.think("Hello", user="alice")
        assert mock_insight_gen.generate_insight.call_count == 0

        # Second turn - should trigger insight
        arch.think("Hello again", user="alice")
        assert mock_insight_gen.generate_insight.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
