from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name: str, device: str = 'cpu'):
    """
    Loads the SentenceTransformer model with specified device.
    """
    try:
        print(f"Loading embedding model '{model_name}' on device '{device}'...")
        model = SentenceTransformer(model_name, device=device)
        print("Embedding model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None
