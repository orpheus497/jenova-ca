from jenova.memory.semantic import SemanticMemory
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.insights.manager import InsightManager

class MemorySearch:
    def __init__(self, semantic_memory: SemanticMemory, episodic_memory: EpisodicMemory, procedural_memory: ProceduralMemory, config, file_logger):
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.insight_manager = None # Will be set later
        self.config = config
        self.file_logger = file_logger

        if self.config.get('memory', {}).get('preload_memories', False):
            self._preload_memories()

    def _preload_memories(self):
        self.file_logger.log_info("Pre-loading memories into RAM...")
        try:
            self.semantic_memory.collection.get()
            self.episodic_memory.collection.get()
            self.procedural_memory.collection.get()
            self.file_logger.log_info("Memories pre-loaded successfully.")
        except Exception as e:
            self.file_logger.log_error(f"Error pre-loading memories: {e}")

    def search_all(self, query: str, username: str) -> list[str]:
        self.file_logger.log_info(f"Searching all memories for user '{username}' with query: '{query}'")
        memory_search_config = self.config.get('memory_search', {})
        semantic_n_results = memory_search_config.get('semantic_n_results', 5)
        episodic_n_results = memory_search_config.get('episodic_n_results', 3)
        procedural_n_results = memory_search_config.get('procedural_n_results', 3)
        insight_n_results = memory_search_config.get('insight_n_results', 5)

        try:
            # Retrieve from structured memories
            semantic_results = self.semantic_memory.search_collection(query, username, n_results=semantic_n_results)
            self.file_logger.log_info(f"Found {len(semantic_results)} semantic results.")
        except Exception as e:
            self.file_logger.log_error(f"Error during semantic memory search: {e}")
            semantic_results = []

        try:
            episodic_results = self.episodic_memory.recall_relevant_episodes(query, n_results=episodic_n_results)
            self.file_logger.log_info(f"Found {len(episodic_results)} episodic results.")
        except Exception as e:
            self.file_logger.log_error(f"Error during episodic memory search: {e}")
            episodic_results = []

        try:
            procedural_results = self.procedural_memory.search(query, username, n_results=procedural_n_results)
            self.file_logger.log_info(f"Found {len(procedural_results)} procedural results.")
        except Exception as e:
            self.file_logger.log_error(f"Error during procedural memory search: {e}")
            procedural_results = []
        
        # Combine and re-rank vectorized results
        vector_results = semantic_results + episodic_results + procedural_results
        vector_results.sort(key=lambda x: x[1])
        ranked_docs = [doc for doc, dist in vector_results]

        # Retrieve relevant learned insights
        try:
            insight_docs = self.search_insights(query, username, max_insights=insight_n_results)
            self.file_logger.log_info(f"Found {len(insight_docs)} relevant insights.")
        except Exception as e:
            self.file_logger.log_error(f"Error during insight search: {e}")
            insight_docs = []

        # Prioritize newest insights, then add ranked documents
        final_context = insight_docs + ranked_docs
        self.file_logger.log_info(f"Final context length: {len(final_context)}")
        
        return final_context[:10] # Return a combined list of the most relevant context

    def search_insights(self, query: str, username: str, max_insights: int = 3) -> list[str]:
        """Uses semantic search to find the most relevant insights for a given query."""
        self.file_logger.log_info(f"Searching insights for user '{username}' with query: '{query}'")
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            self.file_logger.log_info("No insights found for user.")
            return []
        self.file_logger.log_info(f"Found {len(all_insights)} total insights for user.")

        insight_contents = [f"Learned Insight on '{insight['topic']}': {insight['content']}" for insight in all_insights]
        
        try:
            relevant_insights = self.semantic_memory.search_documents(query, documents=insight_contents, n_results=max_insights)
            self.file_logger.log_info(f"Found {len(relevant_insights)} relevant insights after semantic search.")
        except Exception as e:
            self.file_logger.log_error(f"Error during insight semantic search: {e}")
            return []
        
        return [doc for doc, dist in relevant_insights]
