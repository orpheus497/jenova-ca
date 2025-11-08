# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 5: Enhanced Memory Search
Improvements:
- Optional/configurable re-ranking (can be disabled for performance)
- Timeout protection on all operations
- Better error handling
- Simplified fallback strategies
"""

import json

from jenova.memory.episodic import EpisodicMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.memory.semantic import SemanticMemory
from jenova.infrastructure.timeout_manager import timeout, TimeoutError


class MemorySearch:
    def __init__(
        self,
        semantic_memory: SemanticMemory,
        episodic_memory: EpisodicMemory,
        procedural_memory: ProceduralMemory,
        config,
        file_logger,
    ):
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.insight_manager = None  # Will be set later
        self.config = config
        self.file_logger = file_logger
        self.llm = self.semantic_memory.llm  # Get LLM from memory module

        # Memory search configuration
        memory_config = self.config.get("memory", {})
        self.preload_enabled = memory_config.get("preload_memories", False)

        # Re-ranking configuration
        memory_search_config = self.config.get("memory_search", {})
        self.rerank_enabled = memory_search_config.get("rerank_enabled", True)
        self.rerank_timeout = memory_search_config.get("rerank_timeout", 15)

        if self.file_logger:
            self.file_logger.log_info(
                f"MemorySearch initialized: preload={self.preload_enabled}, "
                f"rerank={self.rerank_enabled}, rerank_timeout={self.rerank_timeout}s"
            )

        if self.preload_enabled:
            self._preload_memories()

    def _preload_memories(self):
        """Pre-load memories into RAM for faster access."""
        self.file_logger.log_info("Pre-loading memories into RAM...")
        try:
            import threading

            threads = []
            collections = [
                self.semantic_memory.collection,
                self.episodic_memory.collection,
                self.procedural_memory.collection,
            ]
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
        """
        Uses an LLM call to re-rank retrieved documents for relevance.
        Falls back to original order on any error.

        Args:
            query: User query
            documents: List of document strings to re-rank

        Returns:
            list[str]: Re-ranked documents (or original order on failure)
        """
        if not documents:
            return []

        # Disable re-ranking if configured
        if not self.rerank_enabled:
            return documents

        # Limit to a reasonable number of documents to avoid large prompt
        documents_to_rank = documents[:15]

        numbered_docs = "\n".join(
            [f"{i+1}. {doc}" for i, doc in enumerate(documents_to_rank)]
        )

        prompt = f"""Given the user's query, re-rank the following documents from most to least relevant. Your output should be a JSON object containing a single key "ranked_indices" which is a list of integers representing the new order of the documents.

User Query: "{query}"

Documents:
{numbered_docs}

JSON Response:"""

        try:
            with timeout(self.rerank_timeout):
                response_str = self.llm.generate(prompt, temperature=0.1)
                result = json.loads(response_str)

                # Handle both list and dict responses from LLM
                if isinstance(result, list):
                    ranked_indices = result
                elif isinstance(result, dict):
                    ranked_indices = result.get("ranked_indices")
                else:
                    if self.file_logger:
                        self.file_logger.log_warning(
                            "Re-ranking returned unexpected format, using original order"
                        )
                    return documents_to_rank

                if not ranked_indices or not all(
                    isinstance(i, int) for i in ranked_indices
                ):
                    if self.file_logger:
                        self.file_logger.log_warning(
                            "Re-ranking indices invalid, using original order"
                        )
                    return documents_to_rank

                # Create the new list based on ranked indices (1-based to 0-based)
                ranked_docs = [
                    documents_to_rank[i - 1]
                    for i in ranked_indices
                    if 0 < i <= len(documents_to_rank)
                ]

                if self.file_logger:
                    self.file_logger.log_info(
                        f"Successfully re-ranked {len(ranked_docs)} documents"
                    )

                return ranked_docs

        except TimeoutError:
            if self.file_logger:
                self.file_logger.log_warning(
                    f"Re-ranking timeout after {self.rerank_timeout}s, using original order"
                )
            return documents_to_rank
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error re-ranking context: {e}. Falling back to original order."
                )
            return documents_to_rank
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Unexpected error in re-ranking: {e}. Falling back to original order."
                )
            return documents_to_rank

    def _search_memory(self, memory_type, search_func, query, username, n_results):
        """
        Search a specific memory type with error handling.

        Args:
            memory_type: Name of memory type for logging
            search_func: Function to call for search
            query: Search query
            username: Username
            n_results: Number of results to return

        Returns:
            list: Search results (empty list on error)
        """
        try:
            with timeout(10):  # 10s timeout per memory search
                return search_func(query, username, n_results=n_results)
        except TimeoutError:
            if self.file_logger:
                self.file_logger.log_warning(f"{memory_type} memory search timeout")
            return []
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error during {memory_type} memory search: {e}"
                )
            return []

    def search_all(self, query: str, username: str) -> list[str]:
        """
        Search all memory sources and return ranked results.

        Args:
            query: Search query
            username: Username

        Returns:
            list[str]: Top 10 most relevant documents across all memories
        """
        self.file_logger.log_info(
            f"Searching all memories for user '{username}' with query: '{query}'"
        )

        memory_search_config = self.config.get("memory_search", {})
        semantic_n_results = memory_search_config.get("semantic_n_results", 5)
        episodic_n_results = memory_search_config.get("episodic_n_results", 3)
        procedural_n_results = memory_search_config.get("procedural_n_results", 3)
        insight_n_results = memory_search_config.get("insight_n_results", 5)

        try:
            # 1. Retrieve from all sources in parallel (with individual timeouts)
            semantic_results = self._search_memory(
                "semantic",
                self.semantic_memory.search_collection,
                query,
                username,
                semantic_n_results,
            )
            episodic_results = self._search_memory(
                "episodic",
                self.episodic_memory.recall_relevant_episodes,
                query,
                username,
                episodic_n_results,
            )
            procedural_results = self._search_memory(
                "procedural",
                self.procedural_memory.search,
                query,
                username,
                procedural_n_results,
            )
            # search_insights uses max_insights parameter, not n_results
            insight_docs = self.search_insights(
                query, username, max_insights=insight_n_results
            )

            # 2. Combine all retrieved documents
            # Combine and sort by distance initially
            vector_results = semantic_results + episodic_results + procedural_results
            vector_results.sort(
                key=lambda x: x[1]
            )  # Sort by distance (lower is better)
            combined_docs = [doc for doc, dist in vector_results]

            # Prepend insights as they are often high-signal
            initial_context = insight_docs + combined_docs

            self.file_logger.log_info(
                f"Combined {len(initial_context)} documents before re-ranking"
            )

            # 3. Re-rank the combined context for relevance (if enabled)
            if self.rerank_enabled and initial_context:
                final_context = self._rerank_context(query, initial_context)
            else:
                final_context = initial_context

            self.file_logger.log_info(f"Final context length: {len(final_context)}")

            # Return top 10 most relevant documents
            return final_context[:10]

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error in search_all: {e}")
            return []  # Return empty list on catastrophic failure

    def search_insights(
        self, query: str, username: str, max_insights: int = 3
    ) -> list[str]:
        """
        Uses semantic search to find the most relevant insights for a given query.

        Args:
            query: Search query
            username: Username
            max_insights: Maximum number of insights to return (default: 3)

        Returns:
            list[str]: List of relevant insight strings
        """
        self.file_logger.log_info(
            f"Searching insights for user '{username}' with query: '{query}'"
        )

        try:
            all_insights = self.insight_manager.get_all_insights(username)
            if not all_insights:
                self.file_logger.log_info("No insights found for user.")
                return []

            self.file_logger.log_info(
                f"Found {len(all_insights)} total insights for user."
            )

            insight_contents = [
                f"Learned Insight on '{insight['topic']}': {insight['content']}"
                for insight in all_insights
            ]

            # This uses the embedding model to find relevant insights
            with timeout(10):  # 10s timeout for insight search
                relevant_insights = self.semantic_memory.search_documents(
                    query, documents=insight_contents, n_results=max_insights
                )
                self.file_logger.log_info(
                    f"Found {len(relevant_insights)} relevant insights after semantic search."
                )
                return [doc for doc, dist in relevant_insights]

        except TimeoutError:
            if self.file_logger:
                self.file_logger.log_warning("Insight search timeout")
            return []
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error during insight semantic search: {e}")
            return []

    def get_search_config(self) -> dict:
        """
        Get current search configuration.

        Returns:
            dict: Search configuration
        """
        return {
            "preload_enabled": self.preload_enabled,
            "rerank_enabled": self.rerank_enabled,
            "rerank_timeout": self.rerank_timeout,
            "search_limits": self.config.get("memory_search", {}),
        }
