# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 5: Enhanced RAG System
Improvements:
- Caching layer for frequently accessed queries
- Timeout protection on response generation
- Better error handling
- Cache statistics and management
"""

import hashlib
import re
from collections import OrderedDict
from typing import Optional

from jenova.infrastructure.timeout_manager import timeout, TimeoutError


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation."""

    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        """Get value from cache, return None if not found."""
        if key in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: str):
        """Put value in cache, evict oldest if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove oldest (first) item
            self.cache.popitem(last=False)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }


class RAGSystem:
    """The Enhanced Retrieval-Augmented Generation (RAG) system with caching."""

    def __init__(self, llm, memory_search, insight_manager, config):
        self.llm = llm
        self.memory_search = memory_search
        self.insight_manager = insight_manager
        self.config = config
        # Get ui_logger and file_logger from memory_search which has access to them
        self.ui_logger = memory_search.episodic_memory.ui_logger
        self.file_logger = memory_search.file_logger

        # Initialize response cache
        cache_config = config.get("rag_system", {})
        cache_size = cache_config.get("cache_size", 100)
        self.cache_enabled = cache_config.get("cache_enabled", True)
        self.response_cache = LRUCache(capacity=cache_size)
        self.generation_timeout = cache_config.get("generation_timeout", 120)

        if self.file_logger:
            self.file_logger.log_info(
                f"RAG System initialized with cache_enabled={self.cache_enabled}, "
                f"cache_size={cache_size}, generation_timeout={self.generation_timeout}s"
            )

    def _generate_cache_key(self, query: str, username: str, plan: str) -> str:
        """
        Generate a cache key from query, username, and plan.

        Args:
            query: User query
            username: Username
            plan: Execution plan

        Returns:
            str: MD5 hash as cache key
        """
        # Combine inputs and hash for cache key
        combined = f"{query}|{username}|{plan}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _format_search_results(self, search_results: list[dict] | None) -> str:
        """Format search results for inclusion in prompt."""
        if not search_results:
            return ""

        search_results_formatted = "== WEB SEARCH RESULTS ==\n"
        for i, res in enumerate(search_results):
            search_results_formatted += f"Result {i+1}:\n"
            search_results_formatted += f"Title: {res.get('title', 'N/A')}\n"
            search_results_formatted += f"Link: {res.get('link', 'N/A')}\n"
            search_results_formatted += f"Summary: {res.get('summary', 'N/A')}\n\n"
        return search_results_formatted

    def generate_response(
        self,
        query: str,
        username: str,
        history: list[str],
        plan: str,
        search_results: list[dict] | None = None,
    ) -> str:
        """
        Generates a response using the RAG process with caching.

        Args:
            query: The user's query
            username: The username
            history: Conversation history
            plan: The execution plan
            search_results: Optional web search results

        Returns:
            str: Generated response
        """
        # Check cache (only for non-search queries since search results vary)
        if self.cache_enabled and not search_results:
            cache_key = self._generate_cache_key(query, username, plan)
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                if self.file_logger:
                    self.file_logger.log_info(f"Cache hit for query: {query[:50]}...")
                return cached_response

        try:
            # 1. Retrieve context from all memory sources with timeout
            try:
                with timeout(30):  # 30s for context retrieval
                    context = self.memory_search.search_all(query, username)
            except TimeoutError:
                if self.file_logger:
                    self.file_logger.log_warning("Context retrieval timeout")
                context = []
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Context retrieval error: {e}")
                context = []

            # Safeguard: Ensure all context items are strings before joining
            safe_context = [str(c) for c in context]

            # 2. Generate a response using the context, history, and plan
            context_str = "\n".join(f"- {c}" for c in safe_context)
            history_str = "\n".join(history)

            persona = self.config.get("persona", {})
            identity = persona.get("identity", {})

            search_results_formatted = self._format_search_results(search_results)

            prompt = f"""== RESPONSE GUIDELINES ==
- Your response should be structured, logical, sensible, comprehensive, detailed, and grammatically sound.
- Ensure your response is coherent and easy to understand.
- Vary the length of your response based on the user's query and the available information. Do not feel obligated to use all available tokens.
- Be helpful and informative.

== YOUR KNOWLEDGE BASE (PRIORITIZED) ==
1.  **RETRIEVED CONTEXT (Your personal memories and learned insights):**
{context_str if context else "No context available."}

2.  **WEB SEARCH RESULTS (Real-time information):**
{search_results_formatted if search_results else "No web search performed."}

== CONVERSATION HISTORY ==
{history_str if history else "No history yet."}

== YOUR INTERNAL PLAN ==
{plan}

== TASK ==
Execute the plan to respond to the user's query. Your response MUST be primarily based on your KNOWLEDGE BASE, in the order of priority given (RETRIEVED CONTEXT first, then WEB SEARCH RESULTS). Only use your general knowledge if your KNOWLEDGE BASE does not contain the answer.

If WEB SEARCH RESULTS are present, you MUST follow these steps:
1. Start your response by stating that you have found some information on the web.
2. Present a summary of the search results to the user. For each result, include the title and a brief summary.
3. After presenting the results, provide a concise synthesis of the information.
4. Conclude by asking the user if they would like to perform a deeper search on any of the topics, or if they have any other questions.

If no WEB SEARCH RESULTS are present, integrate information from the RETRIEVED CONTEXT to provide a comprehensive and conversational answer.

Do not re-introduce yourself unless asked. Provide ONLY {identity.get('name', 'Jenova')}'s response.

User ({username}): \"{query}\"\n\n{identity.get('name', 'Jenova')}:"""

            # Generate response with timeout protection
            try:
                with timeout(self.generation_timeout):
                    response = self.llm.generate(prompt)
            except TimeoutError:
                if self.file_logger:
                    self.file_logger.log_error("Response generation timeout")
                response = "I apologize, but generating a response is taking longer than expected. Please try rephrasing your question or try again later."
                return response

            # Post-process the response to remove any RAG debug info
            response = re.sub(r"==.*?==", "", response).strip()

            # Cache the response (only for non-search queries)
            if self.cache_enabled and not search_results:
                cache_key = self._generate_cache_key(query, username, plan)
                self.response_cache.put(cache_key, response)
                if self.file_logger:
                    self.file_logger.log_info(
                        f"Cached response for query: {query[:50]}..."
                    )

            return response

        except Exception as e:
            if self.ui_logger:
                self.ui_logger.system_message(f"Error during response generation: {e}")
            if self.file_logger:
                self.file_logger.log_error(f"Error during response generation: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again later."

    def clear_cache(self):
        """Clear the response cache. Useful for testing or manual cache management."""
        self.response_cache.clear()
        if self.file_logger:
            self.file_logger.log_info("Response cache cleared")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            dict: Cache statistics including hits, misses, and hit rate
        """
        return self.response_cache.get_stats()
