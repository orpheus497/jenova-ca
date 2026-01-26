##Script function and purpose: Embedding type definitions
"""
Embedding Types

Type definitions for embedding operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np

##Step purpose: Define type alias for embedding vectors
Embedding: TypeAlias = list[float]


##Class purpose: Result from embedding operation
@dataclass
class EmbeddingResult:
    """Result from embedding a text."""

    text: str
    """Original text that was embedded."""

    embedding: Embedding
    """The embedding vector."""

    model_name: str
    """Name of model used for embedding."""

    dimensions: int
    """Number of dimensions in embedding."""

    ##Method purpose: Get embedding as numpy array if available
    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        """Convert to numpy array (requires numpy)."""
        import numpy as np

        return np.array(self.embedding, dtype=np.float32)
