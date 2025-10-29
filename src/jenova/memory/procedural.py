# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for managing the procedural memory of the JENOVA Cognitive Architecture.
"""

import json
import os
import uuid

import chromadb

from jenova.utils.data_sanitizer import sanitize_metadata
from jenova.utils.embedding import CustomEmbeddingFunction
from jenova.utils.json_parser import extract_json


class ProceduralMemory:
    def __init__(self, config, ui_logger, file_logger, db_path, llm, embedding_model):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = db_path
        self.llm = llm
        os.makedirs(self.db_path, exist_ok=True)

        client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = CustomEmbeddingFunction(
            model=embedding_model, model_name=config['model']['embedding_model'])

        try:
            self.collection = client.get_or_create_collection(
                name="procedural_steps", embedding_function=self.embedding_function)
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                if self.ui_logger:
                    self.ui_logger.system_message(
                        "Embedding function conflict detected in procedural memory. Recreating collection and migrating data...")
                if self.file_logger:
                    self.file_logger.log_warning(
                        "Embedding function conflict detected. Recreating collection 'procedural_steps' and migrating data.")

                try:
                    old_collection = client.get_collection(
                        name="procedural_steps")
                    data = old_collection.get(
                        include=["documents", "metadatas"])

                    client.delete_collection(name="procedural_steps")
                    self.collection = client.get_or_create_collection(
                        name="procedural_steps", embedding_function=self.embedding_function)

                    if data['ids']:
                        self.collection.add(
                            ids=data['ids'],
                            documents=data['documents'],
                            metadatas=data['metadatas']
                        )
                    if self.ui_logger:
                        self.ui_logger.system_message(
                            "Procedural memory data migration successful.")
                    if self.file_logger:
                        self.file_logger.log_info(
                            "Procedural memory data migration to new collection successful.")

                except Exception as migration_error:
                    if self.file_logger:
                        self.file_logger.log_error(
                            f"Error during procedural memory data migration: {migration_error}")
                    if self.ui_logger:
                        self.ui_logger.system_message(
                            "Warning: Failed to migrate procedural memory data during collection recreation.")

            else:
                raise e

    def add_procedure(self, procedure: str, username: str, goal: str = None, steps: list = None, context: str = None):
        if not goal or not steps or not context:
            prompt = f'''Analyze the following procedure and extract the goal, the steps, and the context. Respond with a JSON object containing "goal" (a string), "steps" (a list of strings), and "context" (a string).

Ensure your response is a single, valid JSON object and nothing else.

Procedure: "{procedure}"

JSON Response:'''
            try:
                response_str = self.llm.generate(prompt, temperature=0.2)
                response_data = extract_json(response_str)
                goal = response_data.get('goal')
                steps = response_data.get('steps', [])
                context = response_data.get('context')
            except (json.JSONDecodeError, KeyError, ValueError):
                goal = None
                steps = None
                context = None
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(
                        f"Error during procedure metadata extraction: {e}")
                goal = None
                steps = None
                context = None

        metadata = {
            "username": username,
            "goal": goal,
            "steps": json.dumps(steps),
            "context": context
        }

        # Sanitize metadata to remove None values before passing to ChromaDB
        metadata = sanitize_metadata(metadata)

        doc_id = f"proc_{uuid.uuid4()}"
        self.collection.add(ids=[doc_id], documents=[
                            procedure], metadatas=[metadata])
        if self.file_logger:
            self.file_logger.log_info(
                f"Added procedure {doc_id} to procedural memory.")

    def search(self, query: str, username: str, n_results: int = 2) -> list[tuple[str, float]]:
        if self.collection.count() == 0:
            return []
        n_results = min(n_results, self.collection.count())
        try:
            results = self.collection.query(
                query_texts=[query], n_results=n_results, where={"username": username})
            if not results['documents']:
                return []
            return list(zip(results['documents'][0], results['distances'][0]))
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error during procedural memory search: {e}")
            return []
