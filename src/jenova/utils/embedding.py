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
