import os
import json
import chromadb
from datetime import datetime
from chromadb.utils import embedding_functions

import uuid

from jenova.utils.json_parser import extract_json

class EpisodicMemory:
    def __init__(self, config, ui_logger, file_logger, db_path, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = db_path
        self.llm = llm
        os.makedirs(self.db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config['model']['embedding_model'])
        self.collection = client.get_or_create_collection(name="episodic_episodes", embedding_function=self.embedding_function)

    def add_episode(self, summary: str, username: str, entities: list = None, emotion: str = None, timestamp: str = None):
        if not timestamp:
            timestamp = datetime.now().isoformat()

        if not entities or not emotion:
            prompt = f'''Analyze the following summary and extract the key entities (people, places, things) and the primary emotion. Respond with a JSON object containing "entities" (a list of strings) and "emotion" (a single string).

Ensure your response is a single, valid JSON object and nothing else.

Summary: "{summary}"

JSON Response:'''
            try:
                response_str = self.llm.generate(prompt, temperature=0.2)
                response_data = extract_json(response_str)
                entities = response_data.get('entities', [])
                emotion = response_data.get('emotion')
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                entities = None
                emotion = None
            except Exception as e:
                self.file_logger.log_error(f"Error during episode metadata extraction: {e}")
                entities = None
                emotion = None

        metadata = {
            "username": username,
            "entities": json.dumps(entities), # ChromaDB metadata values must be strings, numbers, or booleans
            "emotion": emotion,
            "timestamp": timestamp
        }
        
        doc_id = f"ep_{uuid.uuid4()}"
        self.collection.add(ids=[doc_id], documents=[summary], metadatas=[metadata])
        self.file_logger.log_info(f"Added episode {doc_id} to episodic memory.")

    def recall_relevant_episodes(self, query: str, username: str, n_results: int = 3) -> list[tuple[str, float]]:
        if self.collection.count() == 0: return []
        n_results = min(n_results, self.collection.count())
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results, where={"username": username})
            if not results['documents']: return []
            return list(zip(results['documents'][0], results['distances'][0]))
        except Exception as e:
            self.file_logger.log_error(f"Error during episodic memory recall: {e}")
            return []