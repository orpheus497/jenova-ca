# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Embedding model management for JENOVA.

This module handles loading and using embedding models for semantic search
and vector operations with proper resource management.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

from jenova.llm.cuda_manager import CUDAManager
from jenova.infrastructure.timeout_manager import timeout, TimeoutError


class EmbeddingLoadError(Exception):
    """Raised when embedding model loading fails."""

    pass


class EmbeddingManager:
    """
    Manages embedding model lifecycle and operations.

    Features:
    - Safe embedding model loading
    - Batch embedding generation
    - CUDA-aware device selection
    - Proper resource cleanup
    """

    def __init__(self, config: Dict[str, Any], file_logger=None, ui_logger=None):
        self.config = config
        self.file_logger = file_logger
        self.ui_logger = ui_logger
        self.cuda_manager = CUDAManager(file_logger)
        self.embedding_model = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None

    def find_embedding_model(self) -> str:
        """
        Determine which embedding model to use.

        Returns:
            Model name/path for sentence-transformers
        """
        # Check if user specified a model
        embedding_config = self.config.get("embedding", {})
        if "model_name" in embedding_config:
            return embedding_config["model_name"]

        # Default to a reliable, lightweight model
        return "all-MiniLM-L6-v2"  # 384 dimensions, 80MB, fast

    def determine_device(self) -> str:
        """
        Determine optimal device for embedding model.

        NOTE: Embeddings ALWAYS use CPU to preserve GPU VRAM for the main LLM model.
        This is by design - embedding operations are lightweight and don't benefit
        significantly from GPU acceleration, while the LLM needs all available VRAM.

        Returns:
            Device string (always 'cpu')
        """
        # ALWAYS use CPU for embeddings to preserve GPU memory for main LLM
        # This is documented in main_config.yaml and is intentional
        return "cpu"

        # Original logic kept for reference but disabled:
        # hardware_config = self.config.get('hardware', {})
        # prefer_device = hardware_config.get('prefer_device', 'cpu')
        #
        # if prefer_device == 'cuda':
        #     cuda_info = self.cuda_manager.detect_cuda()
        #     if cuda_info.available:
        #         # Check if we have enough free VRAM (need ~500MB for embedding model)
        #         if cuda_info.free_memory and cuda_info.free_memory > 500:
        #             return 'cuda'
        #         elif self.file_logger:
        #             self.file_logger.log_warning(
        #                 f"CUDA available but insufficient free VRAM "
        #                 f"({cuda_info.free_memory}MB), using CPU for embeddings"
        #             )

        # Check for Apple Silicon
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except (ImportError, AttributeError):
            pass

        return "cpu"

    def load_model(self):
        """
        Load the embedding model.

        Raises:
            EmbeddingLoadError: If loading fails
        """
        if self.embedding_model is not None:
            if self.file_logger:
                self.file_logger.log_warning("Embedding model already loaded")
            return

        try:
            # Import sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise EmbeddingLoadError(
                    "sentence-transformers not installed. Install with:\n"
                    "pip install sentence-transformers"
                )

            # Determine model and device
            model_name = self.find_embedding_model()
            device = self.determine_device()

            self._model_name = model_name
            self._device = device

            if self.ui_logger:
                self.ui_logger.info(
                    f"Loading embedding model: {model_name} on {device}..."
                )

            # Load with timeout (2 minutes for first download)
            try:
                with timeout(120, "Embedding model loading timed out"):
                    self.embedding_model = SentenceTransformer(
                        model_name, device=device
                    )
            except TimeoutError as e:
                raise EmbeddingLoadError(
                    f"{e}\n\n"
                    "Try:\n"
                    "1. Check your internet connection (first load downloads model)\n"
                    "2. Manually download model to ~/.cache/torch/sentence_transformers/"
                )

            if self.ui_logger:
                self.ui_logger.success(f"Embedding model loaded on {device}")

            if self.file_logger:
                self.file_logger.log_info(
                    f"Embedding model loaded: {model_name} on {device}"
                )

        except EmbeddingLoadError:
            raise
        except Exception as e:
            raise EmbeddingLoadError(
                f"Failed to load embedding model: {e}\n\n"
                "Try:\n"
                "1. Check internet connection (for first download)\n"
                "2. Ensure sufficient disk space for model cache\n"
                "3. Set prefer_device: 'cpu' if GPU issues"
            )

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            EmbeddingLoadError: If model not loaded
            ValueError: If texts is empty
        """
        if self.embedding_model is None:
            raise EmbeddingLoadError(
                "Embedding model not loaded. Call load_model() first."
            )

        if not texts:
            raise ValueError("Cannot embed empty text list")

        try:
            # Generate embeddings with timeout (30s per batch)
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                try:
                    with timeout(
                        30, f"Embedding batch {i // batch_size + 1} timed out"
                    ):
                        batch_embeddings = self.embedding_model.encode(
                            batch, convert_to_numpy=True, show_progress_bar=False
                        )
                        embeddings.extend(batch_embeddings.tolist())
                except TimeoutError:
                    if self.file_logger:
                        self.file_logger.log_error(
                            f"Embedding timeout for batch starting at {i}"
                        )
                    # Use zero vectors as fallback
                    dim = self.embedding_model.get_sentence_embedding_dimension()
                    embeddings.extend([[0.0] * dim] * len(batch))

            return embeddings

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Embedding generation failed: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            query: Query string to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingLoadError: If model not loaded
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Cannot embed empty query")

        embeddings = self.embed_texts([query], batch_size=1)
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 384 for MiniLM)

        Raises:
            EmbeddingLoadError: If model not loaded
        """
        if self.embedding_model is None:
            raise EmbeddingLoadError("Embedding model not loaded")

        return self.embedding_model.get_sentence_embedding_dimension()

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def unload_model(self):
        """Unload the embedding model and free resources."""
        if self.embedding_model is not None:
            try:
                del self.embedding_model
                self.embedding_model = None

                if self.file_logger:
                    self.file_logger.log_info("Embedding model unloaded")

                # Force garbage collection
                import gc

                gc.collect()

                # Clear CUDA cache if used
                if self._device == "cuda":
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception as e:
                        # Unable to clear CUDA cache, not critical
                        pass

            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Error unloading embedding model: {e}")

    def is_loaded(self) -> bool:
        """Check if embedding model is loaded."""
        return self.embedding_model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded embedding model."""
        if not self.is_loaded():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_name": self._model_name,
            "device": self._device,
            "dimension": self.get_embedding_dimension() if self.is_loaded() else None,
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.unload_model()
