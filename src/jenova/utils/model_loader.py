##Script function and purpose: Model Loader Utility for The JENOVA Cognitive Architecture
##This module loads SentenceTransformer embedding models with automatic device detection

from sentence_transformers import SentenceTransformer
import torch

##Function purpose: Load SentenceTransformer model with automatic GPU/CPU detection
def load_embedding_model(model_name: str, device: str = None):
    """
    Loads the SentenceTransformer model with automatic device detection.
    
    Args:
        model_name: Name of the model to load
        device: Device to use ('cpu', 'cuda', or None for auto-detection)
    """
    try:
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🎮 GPU detected: Using CUDA for embedding model '{model_name}'")
            else:
                device = 'cpu'
                print(f"💻 Loading embedding model '{model_name}' on CPU (no GPU detected)")
        else:
            print(f"Loading embedding model '{model_name}' on device '{device}'...")
        
        model = SentenceTransformer(model_name, device=device)
        print(f"✓ Embedding model loaded successfully on {device.upper()}")
        return model
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None
