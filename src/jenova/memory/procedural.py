import os
import chromadb
from chromadb.utils import embedding_functions

class ProceduralMemory:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = os.path.join(config['user_data_root'], config['memory']['procedural_db_path'])
        os.makedirs(self.db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config['model']['embedding_model'])
        self.collection = client.get_or_create_collection(name="procedural_steps", embedding_function=self.embedding_function)
        self.next_id = self.collection.count()

    def add_procedure(self, procedure: str, metadata: dict = None):
        if metadata is None: metadata = {}
        doc_id = f"proc_{self.next_id}"
        self.collection.add(ids=[doc_id], documents=[procedure], metadatas=[metadata])
        self.next_id += 1
        self.file_logger.log_info(f"Added procedure {doc_id} to procedural memory.")

    def search(self, query: str, n_results: int = 2) -> list[tuple[str, float]]:
        if self.collection.count() == 0: return []
        n_results = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results['documents']: return []
        return list(zip(results['documents'][0], results['distances'][0]))