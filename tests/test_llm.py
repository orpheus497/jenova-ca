# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for LLM interfaces in the JENOVA Cognitive Architecture.

Tests all LLM components:
- LLM Interface (generation, timeout handling)
- Model Manager (lifecycle management)
- Embedding Manager (vector embeddings)
- CUDA Manager (GPU detection and management)
- Distributed LLM Interface (parallel inference)

This module ensures robust operation of the cognitive architecture's LLM foundation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import time

# Import LLM components
from jenova.llm.llm_interface import LLMInterface
from jenova.llm.cuda_manager import CUDAManager


class TestLLMInterface:
    """Test suite for LLM interface."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "model": {
                "model_path": "/fake/path/to/model.gguf",
                "threads": 4,
                "gpu_layers": 0,  # CPU only for testing
                "mlock": False,
                "n_batch": 256,
                "context_size": 2048,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95,
                "timeout_seconds": 120
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
    @patch('jenova.llm.llm_interface.Llama')
    def llm_interface(self, mock_llama, mock_config, mock_file_logger):
        """Create LLM interface instance with mocked Llama."""
        # Mock the Llama model
        mock_model = Mock()
        mock_model.create_completion = Mock(return_value={
            "choices": [{"text": "Test response"}]
        })
        mock_model.tokenize = Mock(return_value=[1, 2, 3])
        mock_model.detokenize = Mock(return_value=b"Test")
        mock_llama.return_value = mock_model

        interface = LLMInterface(
            config=mock_config,
            file_logger=mock_file_logger
        )
        return interface

    def test_llm_interface_initialization(self, llm_interface):
        """Test LLM interface initializes correctly."""
        assert llm_interface is not None
        assert llm_interface.model is not None

    def test_generate_text(self, llm_interface):
        """Test text generation."""
        prompt = "What is Python?"
        response = llm_interface.generate(prompt, max_tokens=100)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_stop_tokens(self, llm_interface):
        """Test generation with stop tokens."""
        prompt = "Count to three:"
        response = llm_interface.generate(
            prompt,
            max_tokens=50,
            stop=["three", "\n\n"]
        )

        assert response is not None

    def test_generate_with_temperature(self, llm_interface):
        """Test generation with different temperatures."""
        prompt = "Write a creative sentence."

        # Low temperature (deterministic)
        response_low = llm_interface.generate(
            prompt,
            max_tokens=50,
            temperature=0.1
        )

        # High temperature (creative)
        response_high = llm_interface.generate(
            prompt,
            max_tokens=50,
            temperature=0.9
        )

        assert response_low is not None
        assert response_high is not None

    def test_tokenization(self, llm_interface):
        """Test tokenization methods."""
        text = "Hello, world!"
        tokens = llm_interface.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Test detokenization
        detokenized = llm_interface.detokenize(tokens)
        assert isinstance(detokenized, (str, bytes))

    @patch('jenova.llm.llm_interface.with_timeout')
    def test_timeout_handling(self, mock_timeout, llm_interface):
        """Test timeout protection for generation."""
        # Simulate timeout
        mock_timeout.side_effect = TimeoutError("Generation timeout")

        prompt = "Generate a very long response"

        with pytest.raises(TimeoutError):
            llm_interface.generate(prompt, max_tokens=1000)

    def test_generate_json_mode(self, llm_interface):
        """Test JSON-structured generation."""
        prompt = "Generate a JSON object with name and age fields"
        response = llm_interface.generate(
            prompt,
            max_tokens=100,
            response_format="json"
        )

        # Response should be JSON-like (implementation dependent)
        assert response is not None

    def test_error_handling_invalid_prompt(self, llm_interface):
        """Test error handling for invalid prompts."""
        # Empty prompt
        response = llm_interface.generate("", max_tokens=50)
        # Should handle gracefully

        # None prompt
        try:
            response = llm_interface.generate(None, max_tokens=50)
        except (TypeError, ValueError):
            pass  # Expected behavior


class TestCUDAManager:
    """Test suite for CUDA manager."""

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
    def cuda_manager(self, mock_file_logger):
        """Create CUDA manager instance."""
        manager = CUDAManager(file_logger=mock_file_logger)
        return manager

    def test_cuda_manager_initialization(self, cuda_manager):
        """Test CUDA manager initializes."""
        assert cuda_manager is not None

    @patch('torch.cuda.is_available')
    def test_cuda_availability_detection(self, mock_cuda_available, cuda_manager):
        """Test CUDA availability detection."""
        # Test CUDA available
        mock_cuda_available.return_value = True
        is_available = cuda_manager.is_cuda_available()
        assert is_available is True

        # Test CUDA not available
        mock_cuda_available.return_value = False
        is_available = cuda_manager.is_cuda_available()
        assert is_available is False

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_gpu_count_detection(self, mock_device_count, mock_cuda_available, cuda_manager):
        """Test GPU count detection."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2

        count = cuda_manager.get_device_count()
        assert count == 2

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    def test_gpu_name_detection(self, mock_get_name, mock_cuda_available, cuda_manager):
        """Test GPU name detection."""
        mock_cuda_available.return_value = True
        mock_get_name.return_value = "NVIDIA GeForce GTX 1650 Ti"

        name = cuda_manager.get_device_name(0)
        assert "GTX 1650" in name or name == "NVIDIA GeForce GTX 1650 Ti"

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.mem_get_info')
    def test_memory_info(self, mock_mem_info, mock_cuda_available, cuda_manager):
        """Test GPU memory info."""
        mock_cuda_available.return_value = True
        mock_mem_info.return_value = (2 * 1024**3, 4 * 1024**3)  # 2GB free, 4GB total

        free, total = cuda_manager.get_memory_info(0)
        assert free == 2 * 1024**3
        assert total == 4 * 1024**3

    def test_cuda_not_available_graceful_handling(self, cuda_manager):
        """Test graceful handling when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            is_available = cuda_manager.is_cuda_available()
            assert is_available is False

            # Should not crash when querying devices
            count = cuda_manager.get_device_count()
            assert count == 0


class TestEmbeddingManager:
    """Test suite for embedding manager."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "model": {
                "embedding_model": "all-MiniLM-L6-v2"
            },
            "hardware": {
                "pytorch_gpu_enabled": False  # CPU only for testing
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
    @patch('jenova.llm.embedding_manager.SentenceTransformer')
    def embedding_manager(self, mock_transformer, mock_config, mock_file_logger):
        """Create embedding manager with mocked transformer."""
        from jenova.llm.embedding_manager import EmbeddingManager

        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[0.1] * 384])  # Mock embedding
        mock_transformer.return_value = mock_model

        manager = EmbeddingManager(
            config=mock_config,
            file_logger=mock_file_logger
        )
        return manager

    def test_embedding_manager_initialization(self, embedding_manager):
        """Test embedding manager initializes."""
        assert embedding_manager is not None
        assert embedding_manager.model is not None

    def test_encode_single_text(self, embedding_manager):
        """Test encoding single text."""
        text = "This is a test sentence"
        embedding = embedding_manager.encode(text)

        assert embedding is not None
        assert len(embedding) > 0
        assert len(embedding[0]) == 384  # Typical embedding dimension

    def test_encode_multiple_texts(self, embedding_manager):
        """Test encoding multiple texts."""
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        embeddings = embedding_manager.encode(texts)

        assert embeddings is not None
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_encode_empty_text(self, embedding_manager):
        """Test encoding empty text."""
        embedding = embedding_manager.encode("")
        assert embedding is not None  # Should handle gracefully

    def test_embedding_consistency(self, embedding_manager):
        """Test that same text produces consistent embeddings."""
        text = "Consistent test"
        embedding1 = embedding_manager.encode(text)
        embedding2 = embedding_manager.encode(text)

        # Embeddings should be identical for deterministic models
        assert embedding1 == embedding2

    def test_is_ready(self, embedding_manager):
        """Test is_ready check."""
        ready = embedding_manager.is_ready()
        assert ready is True


class TestModelManager:
    """Test suite for model manager."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "model": {
                "model_path": "/fake/path/model.gguf",
                "threads": 4,
                "gpu_layers": 0,
                "mlock": False
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

    def test_model_path_validation(self, mock_config, mock_file_logger):
        """Test model path validation."""
        from jenova.llm.model_manager import ModelManager

        # Test with valid path format
        manager = ModelManager(
            config=mock_config,
            file_logger=mock_file_logger
        )
        assert manager is not None

    def test_config_validation(self, mock_config, mock_file_logger):
        """Test configuration validation."""
        from jenova.llm.model_manager import ModelManager

        # Test with valid config
        manager = ModelManager(
            config=mock_config,
            file_logger=mock_file_logger
        )

        # Test invalid config
        invalid_config = {"model": {}}  # Missing required fields
        try:
            manager = ModelManager(
                config=invalid_config,
                file_logger=mock_file_logger
            )
        except (KeyError, ValueError):
            pass  # Expected behavior


class TestDistributedLLM:
    """Test suite for distributed LLM interface."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "model": {
                "model_path": "/fake/path/model.gguf",
                "timeout_seconds": 120
            },
            "network": {
                "enabled": True,
                "peer_selection": {
                    "strategy": "load_balanced",
                    "timeout_ms": 5000
                }
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
    def mock_peer_manager(self):
        """Create mock peer manager."""
        manager = Mock()
        manager.get_active_peers = Mock(return_value=[])
        manager.get_peer_load = Mock(return_value=0.5)
        return manager

    @pytest.fixture
    def mock_rpc_client(self):
        """Create mock RPC client."""
        client = Mock()
        client.call_peer_llm = Mock(return_value="Distributed response")
        return client

    def test_distributed_llm_fallback_to_local(
        self, mock_config, mock_file_logger, mock_peer_manager, mock_rpc_client
    ):
        """Test fallback to local generation when no peers available."""
        from jenova.llm.distributed_llm_interface import DistributedLLMInterface

        mock_peer_manager.get_active_peers.return_value = []  # No peers

        with patch('jenova.llm.distributed_llm_interface.LLMInterface') as mock_llm:
            mock_local = Mock()
            mock_local.generate = Mock(return_value="Local response")
            mock_llm.return_value = mock_local

            interface = DistributedLLMInterface(
                config=mock_config,
                file_logger=mock_file_logger,
                peer_manager=mock_peer_manager,
                rpc_client=mock_rpc_client
            )

            response = interface.generate("Test prompt", max_tokens=50)
            assert response == "Local response"

    def test_parallel_voting_strategy(
        self, mock_config, mock_file_logger, mock_peer_manager, mock_rpc_client
    ):
        """Test parallel voting strategy with multiple peers."""
        from jenova.llm.distributed_llm_interface import DistributedLLMInterface

        # Mock multiple peers
        mock_peer_manager.get_active_peers.return_value = [
            {"id": "peer1", "host": "192.168.1.1"},
            {"id": "peer2", "host": "192.168.1.2"},
            {"id": "peer3", "host": "192.168.1.3"}
        ]

        mock_config["network"]["peer_selection"]["strategy"] = "parallel_voting"

        # Future: Test parallel voting when fully implemented


# Run tests with: pytest tests/test_llm.py -v
