import os
import json
import chromadb
from datetime import datetime
from chromadb.utils import embedding_functions
from importlib import resources

class SemanticMemory:
    def __init__(self, config, ui_logger, file_logger, db_path, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = db_path
        self.llm = llm
        os.makedirs(self.db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config['model']['embedding_model'])
        self.collection = client.get_or_create_collection(name="semantic_facts", embedding_function=self.embedding_function)
        
        self._load_initial_facts(config['persona']['initial_facts'])

    def _load_initial_facts(self, initial_facts):
        if not initial_facts: return
        for fact in initial_facts:
            self.add_fact(fact, source="initial_persona", confidence=1.0)


    def add_fact(self, fact: str, source: str = None, confidence: float = None, temporal_validity: str = None, doc_id: str = None):
        if not source or not confidence or not temporal_validity:
            prompt = f'''Analyze the following fact and extract the source, confidence level (a float between 0 and 1), and temporal validity (e.g., "timeless", "until 2025", "for the next 2 hours"). Respond with a JSON object containing "source", "confidence", and "temporal_validity".

Fact: "{fact}"

JSON Response:'''
            response_str = self.llm.generate(prompt, temperature=0.2)
            try:
                response_data = json.loads(response_str)
                source = response_data.get('source', 'unknown')
                confidence = response_data.get('confidence', 0.5)
                temporal_validity = response_data.get('temporal_validity', 'unknown')
            except (json.JSONDecodeError, KeyError):
                source = 'unknown'
                confidence = 0.5
                temporal_validity = 'unknown'

        metadata = {
            "source": source,
            "confidence": confidence,
            "temporal_validity": temporal_validity,
            "timestamp": datetime.now().isoformat()
        }
        
        if not doc_id:
            doc_id = f"fact_{self.collection.count()}"

        self.collection.add(ids=[doc_id], documents=[fact], metadatas=[metadata])
        self.file_logger.log_info(f"Added fact {doc_id} to semantic memory.")

    def search(self, query: str, n_results: int = 3, documents: list[str] = None) -> list[tuple[str, float]]:
        if self.collection.count() == 0: return []
        n_results = min(n_results, self.collection.count())
        if documents:
            results = self.collection.query(query_texts=documents, n_results=n_results)
        else:
            results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results['documents']: return []
        
        if documents:
            # When querying with multiple documents, the result is a list of lists.
            # We need to flatten it and return the top n_results.
            all_results = []
            for i in range(len(results['documents'])):
                all_results.extend(list(zip(results['documents'][i], results['distances'][i])))
            all_results.sort(key=lambda x: x[1])
            return all_results[:n_results]
        else:
            return list(zip(results['documents'][0], results['distances'][0]))