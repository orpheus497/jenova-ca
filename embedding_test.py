from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Model loaded.")

sentences = ["This is an example sentence", "Each sentence is converted"]

print("Encoding sentences...")
embeddings = model.encode(sentences)
print("Sentences encoded.")

print(f"Embeddings shape: {embeddings.shape}")
