# The JENOVA Cognitive Architecture - Context Window Tests
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 24: Tests for Adaptive Context Window Management.

Tests context window manager and context compression with:
- Priority queue operations
- Relevance scoring
- Token limit enforcement
- Compression strategies
"""

import pytest
import time
from jenova.memory.context_window_manager import ContextWindowManager, ContextItem
from jenova.memory.context_compression import ContextCompressor


class TestContextWindowManager:
    """Test suite for ContextWindowManager."""

    def test_initialization(self):
        """Test manager initialization with default and custom parameters."""
        # Default initialization
        manager = ContextWindowManager()
        assert manager.max_tokens == 4096
        assert manager.compression_threshold == 0.8
        assert manager.min_priority_score == 0.3

        # Custom initialization
        manager = ContextWindowManager(
            max_tokens=2048, compression_threshold=0.7, min_priority_score=0.4
        )
        assert manager.max_tokens == 2048
        assert manager.compression_threshold == 0.7
        assert manager.min_priority_score == 0.4

    def test_add_context(self):
        """Test adding context items."""
        manager = ContextWindowManager(max_tokens=1000)

        manager.add_context("Python is great", "semantic", {"confidence": 0.9})
        stats = manager.get_stats()

        assert stats["item_count"] == 1
        assert stats["total_tokens"] > 0

    def test_deduplication(self):
        """Test that duplicate content is not added twice."""
        manager = ContextWindowManager()

        manager.add_context("Same content", "semantic")
        manager.add_context("Same content", "semantic")  # Duplicate

        stats = manager.get_stats()
        assert stats["item_count"] == 1  # Only one item

    def test_token_counting(self):
        """Test token counting approximation."""
        manager = ContextWindowManager()

        # Approximate: 4 chars per token
        text = "a" * 400  # Should be ~100 tokens
        manager.add_context(text, "semantic")

        stats = manager.get_stats()
        assert 90 <= stats["total_tokens"] <= 110  # Allow some tolerance

    def test_priority_eviction(self):
        """Test that low-priority items are evicted when over limit."""
        manager = ContextWindowManager(max_tokens=100)

        # Add items that exceed limit
        manager.add_context("High priority content", "insight", {"priority": 0.9})
        manager.add_context("Low priority content", "episodic", {"priority": 0.1})
        manager.add_context(
            "Very long text " * 50, "episodic", {"priority": 0.2}
        )  # Push over limit

        stats = manager.get_stats()
        assert stats["total_tokens"] <= 100  # Should respect limit

    def test_get_optimal_context(self):
        """Test context retrieval optimized for query."""
        manager = ContextWindowManager()

        manager.add_context("Python is great for ML", "semantic")
        manager.add_context("JavaScript runs in browsers", "semantic")
        manager.add_context("Machine learning uses Python", "semantic")

        context = manager.get_optimal_context("machine learning")

        # Should prioritize ML-related content
        assert "machine learning" in context.lower() or "ml" in context.lower()

    def test_relevance_calculation(self):
        """Test relevance score calculation."""
        manager = ContextWindowManager()

        # High relevance: contains query keywords
        score1 = manager.calculate_relevance(
            "Python is used for machine learning",
            "machine learning with Python",
            {"timestamp": time.time()},
        )

        # Low relevance: no keyword overlap
        score2 = manager.calculate_relevance(
            "JavaScript runs in browsers",
            "machine learning with Python",
            {"timestamp": time.time()},
        )

        assert score1 > score2

    def test_recency_scoring(self):
        """Test that recent items score higher."""
        manager = ContextWindowManager()

        # Recent timestamp
        recent_score = manager.calculate_relevance(
            "Test content",
            "test",
            {"timestamp": time.time()},  # Now
        )

        # Old timestamp (30 days ago)
        old_score = manager.calculate_relevance(
            "Test content", "test", {"timestamp": time.time() - (30 * 86400)}
        )

        assert recent_score > old_score

    def test_frequency_scoring(self):
        """Test that frequently accessed items score higher."""
        manager = ContextWindowManager()

        content = "Frequently accessed content"
        manager.add_context(content, "semantic")

        # Access multiple times
        for _ in range(5):
            manager.get_optimal_context("content")

        # Frequency score should increase
        freq_score = manager._calculate_frequency_score(content)
        assert freq_score > 0.0

    def test_clear_context(self):
        """Test clearing all context."""
        manager = ContextWindowManager()

        manager.add_context("Item 1", "semantic")
        manager.add_context("Item 2", "semantic")

        manager.clear_context()

        stats = manager.get_stats()
        assert stats["item_count"] == 0
        assert stats["total_tokens"] == 0

    def test_context_grouping_by_type(self):
        """Test that context is grouped by type in output."""
        manager = ContextWindowManager()

        manager.add_context("Semantic fact", "semantic")
        manager.add_context("Procedural step", "procedural")
        manager.add_context("Episodic memory", "episodic")

        context = manager.get_optimal_context("test")

        # Should contain type headers
        assert "[SEMANTIC]" in context or "[PROCEDURAL]" in context

    def test_stats_reporting(self):
        """Test statistics reporting."""
        manager = ContextWindowManager(max_tokens=1000)

        manager.add_context("Test content", "semantic")

        stats = manager.get_stats()

        assert "item_count" in stats
        assert "total_tokens" in stats
        assert "max_tokens" in stats
        assert "utilization" in stats
        assert stats["max_tokens"] == 1000


class TestContextCompressor:
    """Test suite for ContextCompressor."""

    def test_initialization(self):
        """Test compressor initialization."""
        compressor = ContextCompressor()
        assert compressor.stop_words is not None
        assert len(compressor.stop_words) > 0

    def test_extractive_compression(self):
        """Test extractive compression using TF-IDF."""
        compressor = ContextCompressor()

        text = """
        Python is a programming language.
        It is widely used for machine learning.
        Machine learning is a subset of AI.
        AI systems can learn from data.
        """

        compressed = compressor.compress_context(text, target_ratio=0.5, strategy="extractive")

        assert len(compressed) < len(text)
        assert compressed  # Not empty

    def test_extractive_sentence_selection(self):
        """Test that extractive compression selects important sentences."""
        compressor = ContextCompressor()

        text = """
        Machine learning is important.
        The weather is nice today.
        Neural networks are used in ML.
        I like pizza.
        Deep learning is a subset of machine learning.
        """

        compressed = compressor.compress_context(
            text, target_ratio=0.4, strategy="extractive"
        )

        # Should select ML-related sentences (higher TF-IDF scores)
        assert "machine learning" in compressed.lower() or "ml" in compressed.lower()

    def test_compression_ratio(self):
        """Test that compression achieves approximate target ratio."""
        compressor = ContextCompressor()

        text = "This is a test sentence. " * 50

        compressed = compressor.compress_context(text, target_ratio=0.3, strategy="extractive")

        ratio = len(compressed) / len(text)
        assert 0.2 <= ratio <= 0.5  # Allow some tolerance

    def test_compression_stats(self):
        """Test compression statistics calculation."""
        compressor = ContextCompressor()

        original = "This is a test sentence. " * 20
        compressed = compressor.compress_context(original, target_ratio=0.5, strategy="extractive")

        stats = compressor.get_compression_stats(original, compressed)

        assert "original_chars" in stats
        assert "compressed_chars" in stats
        assert "char_ratio" in stats
        assert "reduction_percent" in stats
        assert stats["original_chars"] > stats["compressed_chars"]

    def test_sentence_splitting(self):
        """Test sentence splitting."""
        compressor = ContextCompressor()

        text = "First sentence. Second sentence! Third sentence?"
        sentences = compressor._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences

    def test_tokenization(self):
        """Test word tokenization and stop word removal."""
        compressor = ContextCompressor()

        text = "The quick brown fox jumps over the lazy dog"
        tokens = compressor._tokenize(text)

        # Stop words like "the" should be removed
        assert "the" not in tokens
        assert "quick" in tokens
        assert "fox" in tokens

    def test_tfidf_scoring(self):
        """Test TF-IDF importance scoring."""
        compressor = ContextCompressor()

        sentences = [
            "Machine learning is important",
            "The weather is nice",
            "Machine learning uses data",
        ]

        scores = compressor._calculate_sentence_importance(sentences)

        assert len(scores) == 3
        # Sentences 0 and 2 mention "machine learning" so should have higher scores
        assert scores[0] > scores[1] or scores[2] > scores[1]

    def test_empty_content(self):
        """Test handling of empty content."""
        compressor = ContextCompressor()

        compressed = compressor.compress_context("", target_ratio=0.5, strategy="extractive")

        assert compressed == ""

    def test_no_compression_needed(self):
        """Test that ratio >= 1.0 returns original content."""
        compressor = ContextCompressor()

        original = "This is test content"
        compressed = compressor.compress_context(original, target_ratio=1.0, strategy="extractive")

        assert compressed == original


class TestContextIntegration:
    """Integration tests for context window and compression together."""

    def test_window_with_compression(self):
        """Test using compression within context window."""
        manager = ContextWindowManager(max_tokens=100)
        compressor = ContextCompressor()

        # Add long content
        long_content = "This is a very long piece of content. " * 20

        # Compress before adding
        compressed = compressor.compress_context(
            long_content, target_ratio=0.3, strategy="extractive"
        )

        manager.add_context(compressed, "semantic")

        stats = manager.get_stats()
        assert stats["total_tokens"] <= 100

    def test_full_workflow(self):
        """Test complete workflow: add, retrieve, compress."""
        manager = ContextWindowManager(max_tokens=500)
        compressor = ContextCompressor()

        # Add various types of context
        manager.add_context(
            "Python is used for machine learning", "semantic", {"confidence": 0.9}
        )
        manager.add_context(
            "User asked about decorators yesterday", "episodic", {"timestamp": time.time()}
        )
        manager.add_context(
            "To create a decorator, use @decorator_name", "procedural"
        )

        # Get context for query
        context = manager.get_optimal_context("Python decorators")

        assert context  # Should return something
        assert len(context) > 0

        # Compress if needed
        if len(context) > 200:
            compressed = compressor.compress_context(
                context, target_ratio=0.5, strategy="extractive"
            )
            assert len(compressed) < len(context)


# Fixtures
@pytest.fixture
def sample_manager():
    """Fixture providing a ContextWindowManager with sample data."""
    manager = ContextWindowManager(max_tokens=1000)

    manager.add_context("Python is a programming language", "semantic")
    manager.add_context("JavaScript runs in browsers", "semantic")
    manager.add_context("User prefers functional programming", "episodic")

    return manager


@pytest.fixture
def sample_compressor():
    """Fixture providing a ContextCompressor."""
    return ContextCompressor()


def test_with_fixtures(sample_manager, sample_compressor):
    """Test using pytest fixtures."""
    stats = sample_manager.get_stats()
    assert stats["item_count"] == 3

    context = sample_manager.get_optimal_context("programming")
    assert "programming" in context.lower()
