##Script function and purpose: Semantic Memory for The JENOVA Cognitive Architecture
##This module manages fact-based knowledge storage with confidence levels and temporal validity

import os
import json
##Block purpose: Import Pydantic compatibility fix before ChromaDB import
from jenova.utils.pydantic_compat import *  # noqa: F401, F403
from jenova.utils.pydantic_compat import create_chromadb_client, get_or_create_collection_with_embedding

import chromadb
from datetime import datetime
from importlib import resources
import uuid

from jenova.utils.json_parser import extract_json
from jenova.utils.embedding import CustomEmbeddingFunction

##Class purpose: Manages semantic memory storing facts with confidence and validity metadata
class SemanticMemory:
    ##Function purpose: Initialize semantic memory with database, embedding model, and initial facts
    def __init__(self, config, ui_logger, file_logger, db_path, llm, embedding_model):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = db_path
        self.llm = llm
        os.makedirs(self.db_path, exist_ok=True)
        
        client = create_chromadb_client(path=self.db_path)
        self.embedding_function = CustomEmbeddingFunction(model=embedding_model, model_name=config['model']['embedding_model'])
        
        # Use the compatibility helper which handles embedding function issues
        self.collection = get_or_create_collection_with_embedding(client, name="semantic_facts", embedding_function=self.embedding_function)
        
        try:
            self._load_initial_facts(config['persona']['initial_facts'])
        except Exception as e:
            self.file_logger.log_error(f"Failed to load initial facts into semantic memory: {e}")
            self.ui_logger.system_message("Warning: Could not load initial facts into semantic memory. The AI's persona may be incomplete.")

    ##Function purpose: Load predefined facts into semantic memory during initialization
    def _load_initial_facts(self, initial_facts):
        if not initial_facts: return
        self.ui_logger.info("Loading initial facts into semantic memory...")
        for fact in initial_facts:
            self.add_fact(fact, "jenova", source="initial_persona", confidence=1.0)
        self.ui_logger.info("Initial facts loaded.")


    ##Function purpose: Add a new fact to semantic memory with automatic metadata extraction
    def add_fact(self, fact: str, username: str, source: str = None, confidence: float = None, temporal_validity: str = None, doc_id: str = None):
        if not source or not confidence or not temporal_validity:
            prompt = f'''Analyze the following fact and extract the source, confidence level (a float between 0 and 1), and temporal validity (e.g., "timeless", "until 2025", "for the next 2 hours"). Respond with a JSON object containing "source", "confidence", and "temporal_validity".

Fact: "{fact}"

JSON Response:'''
            response_str = self.llm.generate(prompt, temperature=0.2)
            try:
                response_data = extract_json(response_str)
                source = response_data.get('source', 'unknown')
                confidence = response_data.get('confidence', 0.5)
                temporal_validity = response_data.get('temporal_validity', 'unknown')
            except (json.JSONDecodeError, KeyError, ValueError):
                source = None
                confidence = None
                temporal_validity = None

        metadata = {
            "username": username,
            "source": source,
            "confidence": confidence,
            "temporal_validity": temporal_validity,
            "timestamp": datetime.now().isoformat()
        }
        
        if not doc_id:
            doc_id = f"fact_{uuid.uuid4()}"

        try:
            self.collection.add(ids=[doc_id], documents=[fact], metadatas=[metadata])
            self.file_logger.log_info(f"Added fact {doc_id} to semantic memory.")
        except Exception as e:
            self.file_logger.log_error(f"Failed to add fact to semantic memory: {e}")


    ##Function purpose: Search semantic memory collection for facts matching the query
    def search_collection(self, query: str, username: str, n_results: int = 3) -> list[tuple[str, float]]:
        if self.collection.count() == 0: return []
        n_results = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_results, where={"username": username})
        if not results['documents']: return []
        return list(zip(results['documents'][0], results['distances'][0]))

    ##Function purpose: Search semantic memory using provided document list for cross-referencing
    def search_documents(self, query: str, documents: list[str], n_results: int = 3) -> list[tuple[str, float]]:
        if not documents: return []
        n_results = min(n_results, len(documents))
        results = self.collection.query(query_texts=documents, n_results=n_results)
        if not results['documents']: return []
        
        all_results = []
        for i in range(len(results['documents'])):
            all_results.extend(list(zip(results['documents'][i], results['distances'][i])))
        all_results.sort(key=lambda x: x[1])
        return all_results[:n_results]