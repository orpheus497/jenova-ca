# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""The MemorySearch class is responsible for searching all memory systems.
"""

import json

from jenova.memory.episodic import EpisodicMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.memory.semantic import SemanticMemory


class MemorySearch:
    def __init__(self, semantic_memory: SemanticMemory, episodic_memory: EpisodicMemory, procedural_memory: ProceduralMemory, config, file_logger):
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.insight_manager = None  # Will be set later
        self.config = config
        self.file_logger = file_logger
        self.llm = self.semantic_memory.llm  # Get LLM from memory module

        if self.config.get('memory', {}).get('preload_memories', False):
            self._preload_memories()

    def _preload_memories(self):
        self.file_logger.log_info("Pre-loading memories into RAM...")
        try:
            import threading
            threads = []
            collections = [self.semantic_memory.collection,
                           self.episodic_memory.collection, self.procedural_memory.collection]
            for collection in collections:
                thread = threading.Thread(target=collection.get)
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            self.file_logger.log_info("Memories pre-loaded successfully.")
        except Exception as e:
            self.file_logger.log_error(f"Error pre-loading memories: {e}")

    def _rerank_context(self, query: str, documents: list[str]) -> list[str]:
        """Uses an LLM call to re-rank retrieved documents for relevance."""
        if not documents:
            return []

        # Limit to a reasonable number of documents to avoid large prompt
        documents_to_rank = documents[:15]

        numbered_docs = "\n".join(
            [f"{i+1}. {doc}" for i, doc in enumerate(documents_to_rank)])

        prompt = f"""Given the user's query, re-rank the following documents from most to least relevant. Your output should be a JSON object containing a single key "ranked_indices" which is a list of integers representing the new order of the documents.

User Query: "{query}"

Documents:
{numbered_docs}

JSON Response:"""

        try:
            response_str = self.llm.generate(prompt, temperature=0.1)
            result = json.loads(response_str)
            ranked_indices = result.get("ranked_indices")

            if not ranked_indices or not all(isinstance(i, int) for i in ranked_indices):
                return documents_to_rank  # Fallback to original order

            # Create the new list based on ranked indices (1-based to 0-based)
            ranked_docs = [documents_to_rank[i-1]
                           for i in ranked_indices if 0 < i <= len(documents_to_rank)]
            return ranked_docs
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.file_logger.log_error(
                f"Error re-ranking context: {e}. Falling back to original order.")
            return documents_to_rank  # Fallback

    def _search_memory(self, memory_type, search_func, query, username, n_results):
        try:
            return search_func(query, username, n_results=n_results)
        except Exception as e:
            self.file_logger.log_error(
                f"Error during {memory_type} memory search: {e}")
            return []

    def search_all(self, query: str, username: str) -> list[str]:
        self.file_logger.log_info(
            f"Searching all memories for user '{username}' with query: '{query}'")
        memory_search_config = self.config.get('memory_search', {})
        semantic_n_results = memory_search_config.get('semantic_n_results', 5)
        episodic_n_results = memory_search_config.get('episodic_n_results', 3)
        procedural_n_results = memory_search_config.get(
            'procedural_n_results', 3)
        insight_n_results = memory_search_config.get('insight_n_results', 5)

        # 1. Retrieve from all sources
        semantic_results = self._search_memory(
            "semantic", self.semantic_memory.search_collection, query, username, semantic_n_results)
        episodic_results = self._search_memory(
            "episodic", self.episodic_memory.recall_relevant_episodes, query, username, episodic_n_results)
        procedural_results = self._search_memory(
            "procedural", self.procedural_memory.search, query, username, procedural_n_results)
        insight_docs = self._search_memory(
            "insight", self.search_insights, query, username, insight_n_results)

        # 2. Combine all retrieved documents
        # Combine and sort by distance initially
        vector_results = semantic_results + episodic_results + procedural_results
        vector_results.sort(key=lambda x: x[1])
        combined_docs = [doc for doc, dist in vector_results]

        # Prepend insights as they are often high-signal
        initial_context = insight_docs + combined_docs

        # 3. Re-rank the combined context for relevance
        self.file_logger.log_info(
            f"Re-ranking {len(initial_context)} context documents.")
        final_context = self._rerank_context(query, initial_context)

        self.file_logger.log_info(
            f"Final context length after re-ranking: {len(final_context)}")
        return final_context[:10]  # Return top 10 most relevant documents

    def search_insights(self, query: str, username: str, max_insights: int = 3) -> list[str]:
        """Uses semantic search to find the most relevant insights for a given query."""
        self.file_logger.log_info(
            f"Searching insights for user '{username}' with query: '{query}'")
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            self.file_logger.log_info("No insights found for user.")
            return []
        self.file_logger.log_info(
            f"Found {len(all_insights)} total insights for user.")

        insight_contents = [
            f"Learned Insight on '{insight['topic']}': {insight['content']}" for insight in all_insights]

        try:
            # This uses the embedding model to find relevant insights
            relevant_insights = self.semantic_memory.search_documents(
                query, documents=insight_contents, n_results=max_insights)
            self.file_logger.log_info(
                f"Found {len(relevant_insights)} relevant insights after semantic search.")
            return [doc for doc, dist in relevant_insights]
        except Exception as e:
            self.file_logger.log_error(
                f"Error during insight semantic search: {e}")
            return []
