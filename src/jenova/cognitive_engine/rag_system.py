class RAGSystem:
    """The Retrieval-Augmented Generation (RAG) system. This system is responsible for querying the different memory sources, re-ranking the results, and generating a response."""
    def __init__(self, llm, memory_search, insight_manager):
        self.llm = llm
        self.memory_search = memory_search
        self.insight_manager = insight_manager

    def generate_response(self, query: str, username: str, history: list[str], plan: str) -> str:
        """Generates a response using the RAG process."""
        # 1. Retrieve context from all memory sources
        context = self.memory_search.search_all(query, username)

        # 2. Re-rank the context (already done in search_all)

        # 3. Generate a response using the context, history, and plan
        context_str = "\n".join(f"- {c}" for c in context)
        history_str = "\n".join(history)

        prompt = f"""== YOUR CORE INSTRUCTIONS ==
1. You are Jenova, a personalized AI assistant.
2. Your primary goal is to assist the user based on your understanding of them, which is derived from your own memories, insights, and assumptions.
3. The retrieved context below is your primary source of information. Ground your response in this context first and foremost.
4. Your own knowledge and experiences are more important than your general knowledge.

== RETRIEVED CONTEXT ==\n{context_str if context else "No context available."}\n\n== CONVERSATION HISTORY ==\n{history_str if history else "No history yet."}\n\n== YOUR INTERNAL PLAN ==\n{plan}\n\n== TASK ==
Execute the plan to respond to the user's query. Be direct and conversational. Do not re-introduce yourself unless asked. Provide ONLY Jenova's response.

User ({username}): \"{query}\"\n\nJenova:"""

        response = self.llm.generate(prompt)
        return response
