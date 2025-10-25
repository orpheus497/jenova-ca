from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

class CustomEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that uses a pre-loaded SentenceTransformer model.
    This allows for GPU acceleration and sharing the same model instance across
    multiple ChromaDB collections, improving memory efficiency and performance.
    """
    def __init__(self, model: SentenceTransformer, model_name: str):
        self._model = model
        self._name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.
        
        Args:
            input: List of document strings to embed
            
        Returns:
            List of embedding vectors as lists of floats
        """
        # Ensure input is a list and encode using the model
        # convert_to_numpy=True then .tolist() ensures compatibility with ChromaDB
        embeddings = self._model.encode(list(input), convert_to_numpy=True).tolist()
        return embeddings

    def name(self) -> str:
        """Return the name of the embedding model."""
        return self._name
