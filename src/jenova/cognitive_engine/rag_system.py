import re

class RAGSystem:
    """The Retrieval-Augmented Generation (RAG) system. This system is responsible for querying the different memory sources, re-ranking the results, and generating a response."""
    def __init__(self, llm, memory_search, insight_manager, config):
        self.llm = llm
        self.memory_search = memory_search
        self.insight_manager = insight_manager
        self.config = config
        # Get ui_logger and file_logger from memory_search which has access to them
        self.ui_logger = memory_search.episodic_memory.ui_logger
        self.file_logger = memory_search.file_logger

    def generate_response(self, query: str, username: str, history: list[str], plan: str, search_results: list[dict] | None = None, thinking_process=None) -> str:
        """Generates a response using the RAG process.
        
        Args:
            query: The user's query
            username: The username
            history: Conversation history
            plan: The execution plan
            search_results: Optional web search results
            thinking_process: Optional context manager for thinking status (avoids nested spinners)
        """
        # 1. Retrieve context from all memory sources
        context = self.memory_search.search_all(query, username)

        # Safeguard: Ensure all context items are strings before joining
        safe_context = [str(c) for c in context]

        # 2. Re-rank the context (already done in search_all)

        # 3. Generate a response using the context, history, and plan
        context_str = "\n".join(f"- {c}" for c in safe_context)
        history_str = "\n".join(history)

        persona = self.config.get('persona', {})
        identity = persona.get('identity', {})
        directives = persona.get('directives', [])

        search_results_formatted = ""
        if search_results:
            search_results_formatted = "== WEB SEARCH RESULTS ==\n"
            for i, res in enumerate(search_results):
                search_results_formatted += f"Result {i+1}:\n"
                search_results_formatted += f"Title: {res.get('title', 'N/A')}\n"
                search_results_formatted += f"Link: {res.get('link', 'N/A')}\n"
                search_results_formatted += f"Summary: {res.get('summary', 'N/A')}\n\n"

        prompt = f"""== YOUR CORE INSTRUCTIONS ==
1. You are {identity.get('name', 'Jenova')}, a {identity.get('type', 'personalized AI assistant')}.
2. Your origin story: {identity.get('origin_story', 'You are a helpful assistant.')}
3. Your creator is {identity.get('creator', 'a developer')}.
4. You must follow these directives:
{chr(10).join(f"    - {d}" for d in directives)}

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

        try:
            response = self.llm.generate(prompt, thinking_process=thinking_process)
        except Exception as e:
            self.ui_logger.system_message(f"Error during response generation: {e}")
            self.file_logger.log_error(f"Error during response generation: {e}")
            response = "I'm sorry, I'm having trouble generating a response right now. Please try again later."

        
        # Post-process the response to remove any RAG debug info
        response = re.sub(r"==.*?==", "", response).strip()
            
        return response
