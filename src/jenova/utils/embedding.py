##Script function and purpose: Custom Embedding Function for The JENOVA Cognitive Architecture
##This module provides a ChromaDB-compatible embedding function using SentenceTransformer

##Block purpose: Import Pydantic compatibility fix before ChromaDB import
from jenova.utils.pydantic_compat import *  # noqa: F401, F403

##Block purpose: Import ChromaDB types - compatibility for different ChromaDB versions
##ChromaDB 0.3.23 uses chromadb.api.types, newer versions may use chromadb directly
try:
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except ImportError:
    # Fallback for newer ChromaDB versions
    from chromadb import Documents, EmbeddingFunction, Embeddings

from sentence_transformers import SentenceTransformer

##Class purpose: Wraps SentenceTransformer to provide ChromaDB-compatible embedding interface
class CustomEmbeddingFunction(EmbeddingFunction):
    ##Function purpose: Initialize embedding function with model and name
    def __init__(self, model: SentenceTransformer, model_name: str):
        self._model = model
        self._name = model_name

    ##Function purpose: Generate embeddings for input documents
    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), convert_to_numpy=True).tolist()

    ##Function purpose: Return the model name for identification
    def name(self) -> str:
        return self._name
