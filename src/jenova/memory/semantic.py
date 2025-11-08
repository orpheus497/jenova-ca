# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for managing the semantic memory of the JENOVA Cognitive Architecture."""

import json
import os
import uuid
from datetime import datetime

import chromadb

from jenova.utils.data_sanitizer import sanitize_metadata
from jenova.utils.embedding import CustomEmbeddingFunction
from jenova.utils.json_parser import extract_json


class SemanticMemory:
    def __init__(self, config, ui_logger, file_logger, db_path, llm, embedding_model):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = db_path
        self.llm = llm
        os.makedirs(self.db_path, exist_ok=True)

        client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = CustomEmbeddingFunction(
            model=embedding_model, model_name=config["model"]["embedding_model"]
        )

        try:
            self.collection = client.get_or_create_collection(
                name="semantic_facts", embedding_function=self.embedding_function
            )
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                if self.ui_logger:
                    self.ui_logger.system_message(
                        "Embedding function conflict detected. Recreating collection and migrating data..."
                    )
                if self.file_logger:
                    self.file_logger.log_warning(
                        "Embedding function conflict detected. Recreating collection 'semantic_facts' and migrating data."
                    )

                try:
                    old_collection = client.get_collection(name="semantic_facts")
                    data = old_collection.get(include=["documents", "metadatas"])

                    client.delete_collection(name="semantic_facts")
                    self.collection = client.get_or_create_collection(
                        name="semantic_facts",
                        embedding_function=self.embedding_function,
                    )

                    if data["ids"]:
                        self.collection.add(
                            ids=data["ids"],
                            documents=data["documents"],
                            metadatas=data["metadatas"],
                        )
                    if self.ui_logger:
                        self.ui_logger.system_message("Data migration successful.")
                    if self.file_logger:
                        self.file_logger.log_info(
                            "Data migration to new collection successful."
                        )

                except Exception as migration_error:
                    if self.file_logger:
                        self.file_logger.log_error(
                            f"Error during data migration: {migration_error}"
                        )
                    if self.ui_logger:
                        self.ui_logger.system_message(
                            "Error: Failed to migrate data during collection recreation. Data may be lost."
                        )

            else:
                raise e

        try:
            self._load_initial_facts(config["persona"]["initial_facts"])
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Failed to load initial facts into semantic memory: {e}"
                )
            if self.ui_logger:
                self.ui_logger.system_message(
                    "Warning: Could not load initial facts into semantic memory. The AI's persona may be incomplete."
                )

    def _load_initial_facts(self, initial_facts):
        """
        Load initial persona facts into semantic memory.

        Populates the semantic memory with foundational facts about the AI's
        identity, capabilities, and persona. These facts are marked with high
        confidence and attributed to the "jenova" user.

        Args:
            initial_facts: List of fact strings to load, or None

        Example:
            >>> facts = ["I am JENOVA, a cognitive AI assistant"]
            >>> semantic._load_initial_facts(facts)
        """
        if not initial_facts:
            return
        if self.ui_logger:
            self.ui_logger.info("Loading initial facts into semantic memory...")
        for fact in initial_facts:
            self.add_fact(fact, "jenova", source="initial_persona", confidence=1.0)
        if self.ui_logger:
            self.ui_logger.info("Initial facts loaded.")

    def add_fact(
        self,
        fact: str,
        username: str,
        source: str = None,
        confidence: float = None,
        temporal_validity: str = None,
        doc_id: str = None,
    ):
        """
        Add a fact to semantic memory with metadata.

        Stores a fact in the ChromaDB collection with associated metadata
        including source, confidence level, and temporal validity. If metadata
        is not provided, the LLM analyzes the fact to extract it.

        Args:
            fact: The fact statement to store
            username: Username associated with this fact
            source: Optional source of the fact (e.g., "user", "web", "inference")
            confidence: Optional confidence level (0.0 to 1.0)
            temporal_validity: Optional temporal scope (e.g., "timeless", "until 2025")
            doc_id: Optional document ID (auto-generated if not provided)

        Example:
            >>> semantic.add_fact(
            ...     "Paris is the capital of France",
            ...     "user123",
            ...     source="user",
            ...     confidence=1.0,
            ...     temporal_validity="timeless"
            ... )
        """
        if not source or not confidence or not temporal_validity:
            prompt = f"""Analyze the following fact and extract the source, confidence level (a float between 0 and 1), and temporal validity (e.g., "timeless", "until 2025", "for the next 2 hours"). Respond with a JSON object containing "source", "confidence", and "temporal_validity".

Fact: "{fact}"

JSON Response:"""
            response_str = self.llm.generate(prompt, temperature=0.2)
            try:
                response_data = extract_json(response_str)
                source = response_data.get("source", "unknown")
                confidence = response_data.get("confidence", 0.5)
                temporal_validity = response_data.get("temporal_validity", "unknown")
            except (json.JSONDecodeError, KeyError, ValueError):
                source = None
                confidence = None
                temporal_validity = None

        metadata = {
            "username": username,
            "source": source,
            "confidence": confidence,
            "temporal_validity": temporal_validity,
            "timestamp": datetime.now().isoformat(),
        }

        # Sanitize metadata to remove None values before passing to ChromaDB
        metadata = sanitize_metadata(metadata)

        if not doc_id:
            doc_id = f"fact_{uuid.uuid4()}"

        try:
            self.collection.add(ids=[doc_id], documents=[fact], metadatas=[metadata])
            if self.file_logger:
                self.file_logger.log_info(f"Added fact {doc_id} to semantic memory.")
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Failed to add fact to semantic memory: {e}"
                )

    def search_collection(
        self, query: str, username: str, n_results: int = 3
    ) -> list[tuple[str, float]]:
        """
        Search semantic memory for facts relevant to a query.

        Performs vector similarity search in the ChromaDB collection to find
        facts most relevant to the query, filtered by username.

        Args:
            query: Search query string
            username: Filter results to this username
            n_results: Maximum number of results to return (default: 3)

        Returns:
            List of tuples containing (fact_text, distance_score)
            where lower distance indicates higher similarity

        Example:
            >>> results = semantic.search_collection("capital of France", "user123", n_results=5)
            >>> for fact, distance in results:
            ...     print(f"{fact} (score: {distance})")
        """
        if self.collection.count() == 0:
            return []
        n_results = min(n_results, self.collection.count())
        results = self.collection.query(
            query_texts=[query], n_results=n_results, where={"username": username}
        )
        if not results["documents"]:
            return []
        return list(zip(results["documents"][0], results["distances"][0]))

    def search_documents(
        self, query: str, documents: list[str], n_results: int = 3
    ) -> list[tuple[str, float]]:
        """
        Search a specific list of documents for relevance to queries.

        Performs vector similarity search on a provided set of documents
        rather than the entire collection. Useful for filtering or
        re-ranking a subset of facts.

        Args:
            query: Search query string
            documents: List of document texts to search within
            n_results: Maximum number of results to return (default: 3)

        Returns:
            List of tuples containing (document_text, distance_score)
            sorted by relevance (lowest distance first)

        Example:
            >>> docs = ["Paris is in France", "London is in England"]
            >>> results = semantic.search_documents("French capital", docs, n_results=1)
            >>> print(results[0][0])  # "Paris is in France"
        """
        if not documents:
            return []
        n_results = min(n_results, len(documents))
        results = self.collection.query(query_texts=documents, n_results=n_results)
        if not results["documents"]:
            return []

        all_results = []
        for i in range(len(results["documents"])):
            all_results.extend(
                list(zip(results["documents"][i], results["distances"][i]))
            )
        all_results.sort(key=lambda x: x[1])
        return all_results[:n_results]
