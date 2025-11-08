# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for memory systems in the JENOVA Cognitive Architecture.

Tests all memory layers:
- Episodic Memory (conversation history)
- Semantic Memory (factual knowledge)
- Procedural Memory (how-to information)
- Memory Manager (coordination and search)
- Distributed Memory Search

This module ensures robust operation of the cognitive architecture's memory foundation.
"""

import os
import pytest
import tempfile
import shutil
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Import memory system components
from jenova.memory.base_memory import BaseMemory
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.semantic import SemanticMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.memory.memory_manager import MemoryManager


class TestEpisodicMemory:
    """Test suite for episodic memory (conversation history)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = Mock()
        manager.encode = Mock(return_value=[[0.1] * 384])  # Mock embedding
        manager.is_ready = Mock(return_value=True)
        return manager

    @pytest.fixture
    def episodic_memory(self, temp_dir, mock_file_logger, mock_embedding_manager):
        """Create episodic memory instance for testing."""
        db_path = os.path.join(temp_dir, "episodic_test")
        memory = EpisodicMemory(
            db_path=db_path,
            collection_name="test_episodic",
            embedding_manager=mock_embedding_manager,
            file_logger=mock_file_logger
        )
        return memory

    def test_episodic_memory_initialization(self, episodic_memory):
        """Test episodic memory initializes correctly."""
        assert episodic_memory is not None
        assert episodic_memory.collection is not None
        assert episodic_memory.collection_name == "test_episodic"

    def test_add_episode(self, episodic_memory):
        """Test adding an episode to episodic memory."""
        episode_data = {
            "user": "Tell me about Python",
            "assistant": "Python is a high-level programming language",
            "entities": ["Python", "programming"],
            "emotion": "neutral"
        }

        result = episodic_memory.add(episode_data)
        assert result is True

    def test_search_episodes(self, episodic_memory):
        """Test searching for episodes."""
        # Add test episodes
        episodes = [
            {
                "user": "What is Python?",
                "assistant": "Python is a programming language",
                "entities": ["Python"],
                "emotion": "neutral"
            },
            {
                "user": "How do I learn JavaScript?",
                "assistant": "Start with basic syntax",
                "entities": ["JavaScript"],
                "emotion": "curious"
            }
        ]

        for episode in episodes:
            episodic_memory.add(episode)

        # Search for Python-related episodes
        results = episodic_memory.search("Python programming", n_results=5)
        assert len(results) > 0
        assert "Python" in str(results)

    def test_episodic_memory_metadata(self, episodic_memory):
        """Test that episodes include proper metadata."""
        episode = {
            "user": "Test question",
            "assistant": "Test answer",
            "entities": ["test"],
            "emotion": "neutral"
        }

        episodic_memory.add(episode)
        results = episodic_memory.search("test", n_results=1)

        assert len(results) > 0
        # Check metadata fields
        result = results[0]
        assert "timestamp" in result["metadata"]
        assert "entities" in result["metadata"]
        assert "emotion" in result["metadata"]

    def test_episode_count(self, episodic_memory):
        """Test getting episode count."""
        # Add multiple episodes
        for i in range(5):
            episodic_memory.add({
                "user": f"Question {i}",
                "assistant": f"Answer {i}",
                "entities": [],
                "emotion": "neutral"
            })

        count = episodic_memory.count()
        assert count == 5


class TestSemanticMemory:
    """Test suite for semantic memory (factual knowledge)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = Mock()
        manager.encode = Mock(return_value=[[0.1] * 384])
        manager.is_ready = Mock(return_value=True)
        return manager

    @pytest.fixture
    def semantic_memory(self, temp_dir, mock_file_logger, mock_embedding_manager):
        """Create semantic memory instance for testing."""
        db_path = os.path.join(temp_dir, "semantic_test")
        memory = SemanticMemory(
            db_path=db_path,
            collection_name="test_semantic",
            embedding_manager=mock_embedding_manager,
            file_logger=mock_file_logger
        )
        return memory

    def test_semantic_memory_initialization(self, semantic_memory):
        """Test semantic memory initializes correctly."""
        assert semantic_memory is not None
        assert semantic_memory.collection is not None

    def test_add_fact(self, semantic_memory):
        """Test adding a fact to semantic memory."""
        fact_data = {
            "fact": "The Earth orbits the Sun",
            "source": "astronomy textbook",
            "confidence": 1.0,
            "temporal_validity": "permanent"
        }

        result = semantic_memory.add(fact_data)
        assert result is True

    def test_search_facts(self, semantic_memory):
        """Test searching for facts."""
        # Add test facts
        facts = [
            {
                "fact": "Python was created by Guido van Rossum",
                "source": "Python history",
                "confidence": 1.0,
                "temporal_validity": "permanent"
            },
            {
                "fact": "JavaScript runs in web browsers",
                "source": "web development guide",
                "confidence": 1.0,
                "temporal_validity": "permanent"
            }
        ]

        for fact in facts:
            semantic_memory.add(fact)

        # Search for Python facts
        results = semantic_memory.search("Python creator", n_results=5)
        assert len(results) > 0

    def test_fact_confidence_levels(self, semantic_memory):
        """Test facts with different confidence levels."""
        low_confidence_fact = {
            "fact": "Uncertain information",
            "source": "speculation",
            "confidence": 0.3,
            "temporal_validity": "uncertain"
        }

        high_confidence_fact = {
            "fact": "Well-established fact",
            "source": "verified source",
            "confidence": 1.0,
            "temporal_validity": "permanent"
        }

        semantic_memory.add(low_confidence_fact)
        semantic_memory.add(high_confidence_fact)

        count = semantic_memory.count()
        assert count == 2

    def test_update_fact_confidence(self, semantic_memory):
        """Test updating fact confidence (Phase 19 Feature 6)."""
        # This test prepares for confidence tracking feature
        fact = {
            "fact": "Test fact for confidence update",
            "source": "test",
            "confidence": 0.5,
            "temporal_validity": "uncertain"
        }

        semantic_memory.add(fact)
        # Future: Test confidence update method when Feature 6 is implemented


class TestProceduralMemory:
    """Test suite for procedural memory (how-to information)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = Mock()
        manager.encode = Mock(return_value=[[0.1] * 384])
        manager.is_ready = Mock(return_value=True)
        return manager

    @pytest.fixture
    def procedural_memory(self, temp_dir, mock_file_logger, mock_embedding_manager):
        """Create procedural memory instance for testing."""
        db_path = os.path.join(temp_dir, "procedural_test")
        memory = ProceduralMemory(
            db_path=db_path,
            collection_name="test_procedural",
            embedding_manager=mock_embedding_manager,
            file_logger=mock_file_logger
        )
        return memory

    def test_procedural_memory_initialization(self, procedural_memory):
        """Test procedural memory initializes correctly."""
        assert procedural_memory is not None
        assert procedural_memory.collection is not None

    def test_add_procedure(self, procedural_memory):
        """Test adding a procedure to procedural memory."""
        procedure_data = {
            "name": "Install Python package",
            "steps": ["Open terminal", "Run pip install <package>", "Verify installation"],
            "goal": "Install a Python package using pip",
            "context": "Python development"
        }

        result = procedural_memory.add(procedure_data)
        assert result is True

    def test_search_procedures(self, procedural_memory):
        """Test searching for procedures."""
        # Add test procedures
        procedures = [
            {
                "name": "Create virtual environment",
                "steps": ["python -m venv venv", "activate venv"],
                "goal": "Isolate Python dependencies",
                "context": "Python development"
            },
            {
                "name": "Run pytest",
                "steps": ["Install pytest", "Run pytest command"],
                "goal": "Execute unit tests",
                "context": "Testing"
            }
        ]

        for procedure in procedures:
            procedural_memory.add(procedure)

        # Search for Python procedures
        results = procedural_memory.search("Python virtual environment", n_results=5)
        assert len(results) > 0

    def test_procedure_with_steps(self, procedural_memory):
        """Test that procedures contain step information."""
        procedure = {
            "name": "Git commit",
            "steps": ["git add .", "git commit -m 'message'", "git push"],
            "goal": "Commit and push code",
            "context": "Version control"
        }

        procedural_memory.add(procedure)
        results = procedural_memory.search("git commit", n_results=1)

        assert len(results) > 0
        result = results[0]
        assert "steps" in result["metadata"]


class TestMemoryManager:
    """Test suite for memory manager (coordination layer)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        return {
            "memory": {
                "episodic_db_path": os.path.join(temp_dir, "episodic"),
                "semantic_db_path": os.path.join(temp_dir, "semantic"),
                "procedural_db_path": os.path.join(temp_dir, "procedural"),
                "preload_memories": False
            }
        }

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = Mock()
        manager.encode = Mock(return_value=[[0.1] * 384])
        manager.is_ready = Mock(return_value=True)
        return manager

    @pytest.fixture
    def memory_manager(self, mock_config, temp_dir, mock_file_logger, mock_embedding_manager):
        """Create memory manager instance for testing."""
        manager = MemoryManager(
            config=mock_config,
            user_data_root=temp_dir,
            embedding_manager=mock_embedding_manager,
            file_logger=mock_file_logger
        )
        return manager

    def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initializes all memory layers."""
        assert memory_manager is not None
        assert memory_manager.episodic_memory is not None
        assert memory_manager.semantic_memory is not None
        assert memory_manager.procedural_memory is not None

    def test_add_to_episodic(self, memory_manager):
        """Test adding to episodic memory via manager."""
        episode = {
            "user": "Test question",
            "assistant": "Test answer",
            "entities": ["test"],
            "emotion": "neutral"
        }

        result = memory_manager.add_to_episodic(episode)
        assert result is True

    def test_add_to_semantic(self, memory_manager):
        """Test adding to semantic memory via manager."""
        fact = {
            "fact": "Test fact",
            "source": "test",
            "confidence": 1.0,
            "temporal_validity": "permanent"
        }

        result = memory_manager.add_to_semantic(fact)
        assert result is True

    def test_add_to_procedural(self, memory_manager):
        """Test adding to procedural memory via manager."""
        procedure = {
            "name": "Test procedure",
            "steps": ["step 1", "step 2"],
            "goal": "test goal",
            "context": "testing"
        }

        result = memory_manager.add_to_procedural(procedure)
        assert result is True

    def test_search_all_memories(self, memory_manager):
        """Test searching across all memory layers."""
        # Add data to all layers
        memory_manager.add_to_episodic({
            "user": "Python question",
            "assistant": "Python answer",
            "entities": ["Python"],
            "emotion": "neutral"
        })

        memory_manager.add_to_semantic({
            "fact": "Python is a programming language",
            "source": "test",
            "confidence": 1.0,
            "temporal_validity": "permanent"
        })

        memory_manager.add_to_procedural({
            "name": "Install Python",
            "steps": ["Download Python", "Run installer"],
            "goal": "Install Python",
            "context": "setup"
        })

        # Search all memories
        results = memory_manager.search_all_memories("Python", n_results=5)
        assert "episodic" in results
        assert "semantic" in results
        assert "procedural" in results

    def test_memory_stats(self, memory_manager):
        """Test getting memory statistics."""
        # Add some data
        for i in range(3):
            memory_manager.add_to_episodic({
                "user": f"Question {i}",
                "assistant": f"Answer {i}",
                "entities": [],
                "emotion": "neutral"
            })

        stats = memory_manager.get_stats()
        assert "episodic_count" in stats
        assert stats["episodic_count"] == 3


class TestMemoryErrorHandling:
    """Test suite for memory error handling and edge cases."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = Mock()
        manager.encode = Mock(return_value=[[0.1] * 384])
        manager.is_ready = Mock(return_value=True)
        return manager

    def test_invalid_episode_data(self, temp_dir, mock_file_logger, mock_embedding_manager):
        """Test handling of invalid episode data."""
        db_path = os.path.join(temp_dir, "episodic_error_test")
        memory = EpisodicMemory(
            db_path=db_path,
            collection_name="test_episodic",
            embedding_manager=mock_embedding_manager,
            file_logger=mock_file_logger
        )

        # Try adding invalid data
        invalid_episode = {}  # Missing required fields
        result = memory.add(invalid_episode)
        # Should handle gracefully (implementation-dependent)

    def test_search_empty_memory(self, temp_dir, mock_file_logger, mock_embedding_manager):
        """Test searching in empty memory."""
        db_path = os.path.join(temp_dir, "empty_test")
        memory = EpisodicMemory(
            db_path=db_path,
            collection_name="test_empty",
            embedding_manager=mock_embedding_manager,
            file_logger=mock_file_logger
        )

        results = memory.search("test query", n_results=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_embedding_manager_failure(self, temp_dir, mock_file_logger):
        """Test handling of embedding manager failures."""
        # Create failing embedding manager
        failing_manager = Mock()
        failing_manager.encode = Mock(side_effect=Exception("Embedding failed"))
        failing_manager.is_ready = Mock(return_value=False)

        db_path = os.path.join(temp_dir, "failure_test")

        # Should handle initialization gracefully
        try:
            memory = EpisodicMemory(
                db_path=db_path,
                collection_name="test_failure",
                embedding_manager=failing_manager,
                file_logger=mock_file_logger
            )
            # If it doesn't raise, test passes
        except Exception as e:
            # If it raises, ensure it's handled properly
            assert mock_file_logger.log_error.called


# Run tests with: pytest tests/test_memory.py -v
