##Script function and purpose: JENOVA embedding model wrapper for sentence transformers
"""
JENOVA Embedding Model

Wrapper around sentence-transformers for generating embeddings.
This is the fine-tunable component of JENOVA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jenova.embeddings.types import Embedding, EmbeddingResult
from jenova.exceptions import EmbeddingLoadError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


##Class purpose: JENOVA embedding model wrapper
class JenovaEmbedding:
    """
    JENOVA's embedding model.

    Wraps sentence-transformers for generating embeddings.
    This model can be fine-tuned on JENOVA's learned insights.
    """

    ##Method purpose: Initialize with model name
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize embedding model.

        Args:
            model_name: Name of sentence-transformer model
        """
        ##Step purpose: Store configuration
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._dimensions: int = 0

    ##Method purpose: Load the embedding model
    def load(self) -> None:
        """
        Load the embedding model.

        Raises:
            EmbeddingLoadError: If loading fails
        """
        ##Error purpose: Catch model loading errors
        try:
            from sentence_transformers import SentenceTransformer

            ##Action purpose: Load the model
            self._model = SentenceTransformer(self.model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            raise EmbeddingLoadError(self.model_name, str(e)) from e

    ##Method purpose: Check if model is loaded
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    ##Method purpose: Get embedding dimensions
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    ##Method purpose: Embed a single text
    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        ##Condition purpose: Ensure model is loaded
        if self._model is None:
            self.load()

        ##Action purpose: Generate embedding
        embedding = self._model.encode(text, convert_to_numpy=True)

        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model_name=self.model_name,
            dimensions=self._dimensions,
        )

    ##Method purpose: Embed multiple texts efficiently
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Embed multiple texts in a batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        ##Condition purpose: Ensure model is loaded
        if self._model is None:
            self.load()

        ##Action purpose: Generate embeddings for all texts
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        ##Step purpose: Convert to results
        return [
            EmbeddingResult(
                text=text,
                embedding=emb.tolist(),
                model_name=self.model_name,
                dimensions=self._dimensions,
            )
            for text, emb in zip(texts, embeddings, strict=False)
        ]

    ##Method purpose: Get raw embedding vector
    def embed_raw(self, text: str) -> Embedding:
        """
        Get raw embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        result = self.embed(text)
        return result.embedding

    ##Method purpose: Factory method
    @classmethod
    def create(cls, model_name: str = "all-MiniLM-L6-v2") -> JenovaEmbedding:
        """
        Create and load embedding model.

        Args:
            model_name: Name of model to use

        Returns:
            Loaded JenovaEmbedding instance
        """
        model = cls(model_name)
        model.load()
        return model
