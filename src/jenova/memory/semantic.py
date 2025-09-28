import os
import chromadb
from chromadb.utils import embedding_functions
from importlib import resources

class SemanticMemory:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = os.path.join(config['user_data_root'], config['memory']['semantic_db_path'])
        os.makedirs(self.db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config['model']['embedding_model'])
        self.collection = client.get_or_create_collection(name="semantic_facts", embedding_function=self.embedding_function)
        
        self._load_initial_facts(config['persona']['initial_facts'])

    def _load_initial_facts(self, initial_facts):
        if not initial_facts: return
        ids = [f.split(':', 1)[0].strip() for f in initial_facts]
        existing = self.collection.get(ids=ids)
        existing_ids = existing['ids'] if existing and existing['ids'] else []
        new_facts_to_add = [f for f in initial_facts if f.split(':', 1)[0].strip() not in existing_ids]

        if new_facts_to_add:
            self.ui_logger.info(f"Loading {len(new_facts_to_add)} new initial facts into Semantic Memory...")
            self.file_logger.log_info(f"Loading {len(new_facts_to_add)} new initial facts into Semantic Memory...")
            new_ids = [f.split(':', 1)[0].strip() for f in new_facts_to_add]
            documents = [f.split(':', 1)[1].strip() for f in new_facts_to_add]
            self.collection.add(ids=new_ids, documents=documents)

    def load_rag_document(self):
        try:
            self.ui_logger.info("Loading RAG document into Semantic Memory...")
            self.file_logger.log_info("Loading RAG document into Semantic Memory...")
            with resources.open_text("jenova.docs", "RAG.md") as f:
                content = f.read()
            chunks = [content[i:i+500] for i in range(0, len(content), 450)]
            chunk_ids = [f"rag_doc_{i}" for i in range(len(chunks))]
            if not chunk_ids: return
            existing = self.collection.get(ids=chunk_ids)
            existing_ids = existing['ids'] if existing and existing['ids'] else []
            new_chunk_ids = [cid for cid in chunk_ids if cid not in existing_ids]
            new_chunks = [chunks[i] for i, cid in enumerate(chunk_ids) if cid not in existing_ids]
            if new_chunks:
                self.collection.add(ids=new_chunk_ids, documents=new_chunks)
                self.ui_logger.info(f"Ingested {len(new_chunks)} chunks from RAG.md into Semantic Memory.")
                self.file_logger.log_info(f"Ingested {len(new_chunks)} chunks from RAG.md into Semantic Memory.")
        except FileNotFoundError:
            self.ui_logger.error("jenova/docs/RAG.md not found. Skipping RAG doc ingestion.")
            self.file_logger.log_error("jenova/docs/RAG.md not found. Skipping RAG doc ingestion.")

    def search(self, query: str, n_results: int = 3) -> list[tuple[str, float]]:
        if self.collection.count() == 0: return []
        n_results = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results['documents']: return []
        return list(zip(results['documents'][0], results['distances'][0]))