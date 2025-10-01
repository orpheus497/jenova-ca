from jenova.memory.semantic import SemanticMemory
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.insights.manager import InsightManager

class MemorySearch:
    def __init__(self, semantic_memory: SemanticMemory, episodic_memory: EpisodicMemory, procedural_memory: ProceduralMemory, insight_manager: InsightManager):
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.insight_manager = insight_manager

    def search_all(self, query: str, username: str) -> list[str]:
        # Retrieve from structured memories
        semantic_results = self.semantic_memory.search(query, n_results=5)
        episodic_results = self.episodic_memory.recall_relevant_episodes(query, n_results=3)
        procedural_results = self.procedural_memory.search(query, n_results=3)
        
        # Combine and re-rank vectorized results
        vector_results = semantic_results + episodic_results + procedural_results
        vector_results.sort(key=lambda x: x[1])
        ranked_docs = [doc for doc, dist in vector_results]

        # Retrieve relevant learned insights
        insight_docs = self.search_insights(query, username, max_insights=5)

        # Prioritize newest insights, then add ranked documents
        final_context = insight_docs + ranked_docs
        
        return final_context[:10] # Return a combined list of the most relevant context

    def search_insights(self, query: str, username: str, max_insights: int = 3) -> list[str]:
        """Uses semantic search to find the most relevant insights for a given query."""
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            return []

        insight_contents = [f"Learned Insight on '{insight['topic']}': {insight['content']}" for insight in all_insights]
        
        # Assuming semantic_memory can search a list of documents
        relevant_insights = self.semantic_memory.search(query, documents=insight_contents, n_results=max_insights)
        
        return [doc for doc, dist in relevant_insights]