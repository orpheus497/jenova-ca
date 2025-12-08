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

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: SentenceTransformer, model_name: str):
        self._model = model
        self._name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), convert_to_numpy=True).tolist()

    def name(self) -> str:
        return self._name
