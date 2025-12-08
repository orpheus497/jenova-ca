##Script function and purpose: Unit tests for memory systems
##Tests for EpisodicMemory, SemanticMemory, and ProceduralMemory classes

import pytest
from unittest.mock import Mock, MagicMock
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.semantic import SemanticMemory
from jenova.memory.procedural import ProceduralMemory

##Class purpose: Test suite for EpisodicMemory
class TestEpisodicMemory:
    ##Function purpose: Test episodic memory initialization
    def test_init(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface):
        """Test that EpisodicMemory initializes correctly."""
        memory = EpisodicMemory(
            mock_config, mock_ui_logger, mock_file_logger,
            mock_config['memory']['episodic_db_path'], mock_llm_interface
        )
        assert memory is not None
        assert memory.collection is not None

    ##Function purpose: Test adding an episode to episodic memory
    def test_add_episode(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface):
        """Test adding an episode to episodic memory."""
        memory = EpisodicMemory(
            mock_config, mock_ui_logger, mock_file_logger,
            mock_config['memory']['episodic_db_path'], mock_llm_interface
        )
        memory.add_episode("Test conversation", "test_user")
        assert memory.collection.count() > 0

    ##Function purpose: Test recalling relevant episodes
    def test_recall_relevant_episodes(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface):
        """Test recalling relevant episodes from memory."""
        memory = EpisodicMemory(
            mock_config, mock_ui_logger, mock_file_logger,
            mock_config['memory']['episodic_db_path'], mock_llm_interface
        )
        memory.add_episode("Test conversation about Python", "test_user")
        results = memory.recall_relevant_episodes("Python", "test_user", n_results=1)
        assert isinstance(results, list)

##Class purpose: Test suite for SemanticMemory
class TestSemanticMemory:
    ##Function purpose: Test semantic memory initialization
    @pytest.mark.skipif(
        True,  # Skip by default - requires sentence-transformers model download
        reason="Requires sentence-transformers model download"
    )
    def test_init(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface):
        """Test that SemanticMemory initializes correctly."""
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            memory = SemanticMemory(
                mock_config, mock_ui_logger, mock_file_logger,
                mock_config['memory']['semantic_db_path'], mock_llm_interface, embedding_model
            )
            assert memory is not None
            assert memory.collection is not None
        except ImportError:
            pytest.skip("sentence-transformers not available")

    ##Function purpose: Test adding a fact to semantic memory
    @pytest.mark.skipif(
        True,  # Skip by default - requires sentence-transformers model download
        reason="Requires sentence-transformers model download"
    )
    def test_add_fact(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface):
        """Test adding a fact to semantic memory."""
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            memory = SemanticMemory(
                mock_config, mock_ui_logger, mock_file_logger,
                mock_config['memory']['semantic_db_path'], mock_llm_interface, embedding_model
            )
            memory.add_fact("Python is a programming language", "test_user")
            assert memory.collection.count() > 0
        except ImportError:
            pytest.skip("sentence-transformers not available")

##Class purpose: Test suite for ProceduralMemory
class TestProceduralMemory:
    ##Function purpose: Test procedural memory initialization
    def test_init(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface):
        """Test that ProceduralMemory initializes correctly."""
        memory = ProceduralMemory(
            mock_config, mock_ui_logger, mock_file_logger,
            mock_config['memory']['procedural_db_path'], mock_llm_interface
        )
        assert memory is not None
        assert memory.collection is not None

    ##Function purpose: Test adding a procedure to procedural memory
    def test_add_procedure(self, mock_config, mock_ui_logger, mock_file_logger, mock_llm_interface):
        """Test adding a procedure to procedural memory."""
        memory = ProceduralMemory(
            mock_config, mock_ui_logger, mock_file_logger,
            mock_config['memory']['procedural_db_path'], mock_llm_interface
        )
        memory.add_procedure(
            "How to test code",
            "test_user",
            goal="Test code properly",
            steps=["Write tests", "Run tests"],
            context="Testing context"
        )
        assert memory.collection.count() > 0

