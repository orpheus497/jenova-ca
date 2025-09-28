import json

class CognitiveEngine:
    """The Perfected Cognitive Engine. Manages the refined cognitive cycle."""
    def __init__(self, llm, memory_search, file_tools, insight_manager, config, ui_logger, file_logger):
        self.llm = llm
        self.memory_search = memory_search
        self.file_tools = file_tools
        self.insight_manager = insight_manager
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.history = []
        self.turn_count = 0
        self.MAX_HISTORY_TURNS = 10 # Keep the last 10 conversation turns

    def think(self, user_input: str, username: str) -> str:
        """Runs the full cognitive cycle: Retrieve, Plan, Execute, and Reflect."""
        with self.ui_logger.cognitive_process("Thinking..."):
            self.file_logger.log_info(f"New query received from {username}: {user_input}")
            self.turn_count += 1

            context = self.memory_search.search_all(user_input, username)
            plan = self._plan(user_input, context, username)
            response = self._execute(user_input, context, plan, username)

            self.history.append(f"{username}: {user_input}")
            self.history.append(f"Jenova: {response}")
            if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]

            # Reflective insight generation every 5 turns
            if self.turn_count % 5 == 0:
                self.generate_insight_from_history(username)

        return response

    def _plan(self, user_input: str, context: list[str], username: str) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        prompt = f"""== CONTEXT ==\n{context_str if context else "No context available."}\n\n== TASK ==\nBased on the user's query and the provided context, create a short, step-by-step plan to formulate a response. The plan can include generating a new insight if a novel conclusion is reached.\n\nUser ({username}): "{user_input}"\n\nPlan:"""
        with self.ui_logger.thinking_process("Formulating plan..."):
            plan = self.llm.generate(prompt, temperature=0.1, stop=["\n\n"])
        self.file_logger.log_info(f"Generated Plan: {plan}")
        return plan

    def _execute(self, user_input: str, context: list[str], plan: str, username: str) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        history_str = "\n".join(self.history)
        
        is_identity_query = "who are you" in user_input.lower() or "tell me about yourself" in user_input.lower()
        task_prompt = "Execute the plan to respond to the user's query. Your response must be grounded in your persona, the retrieved context, and the conversation history. Provide ONLY Jenova's response." if is_identity_query else "Execute the plan to respond to the user's query. Be direct and conversational. Do not re-introduce yourself unless asked. Provide ONLY Jenova's response."

        prompt = f"""== RETRIEVED CONTEXT ==\n{context_str if context else "No context available."}\n\n== CONVERSATION HISTORY ==\n{history_str if self.history else "No history yet."}\n\n== YOUR INTERNAL PLAN ==\n{plan}\n\n== TASK ==\n{task_prompt}\n\nUser ({username}): "{user_input}"\n\nJenova:"""
        with self.ui_logger.thinking_process("Executing plan..."):
            response = self.llm.generate(prompt)
        self.file_logger.log_info(f"Generated Response: {response}")
        return response

    def generate_insight_from_history(self, username: str):
        """Analyzes recent conversation history to generate and save a new, high-quality insight."""
        if len(self.history) < 2:
            return

        # Use the last 8 turns (4 user, 4 Jenova) to provide more context
        conversation_segment = "\n".join(self.history[-8:])

        prompt = f"""
Analyze the following conversation segment involving the user '{username}'. Your goal is to identify a single, high-quality insight that represents a new understanding, a significant user preference, a correction of a previous assumption, or a key takeaway that should be remembered for future interactions with '{username}'.

**Guidelines for a High-Quality Insight:**
- **Novelty:** It should be something new that wasn't known before this conversation.
- **Significance:** It should be important for future interactions (e.g., a user's goal, a core preference, a key fact about their project).
- **Conciseness:** The insight should be stated clearly and succinctly.
- **Not Obvious:** Avoid stating the obvious (e.g., "the user asked a question").

Based on these guidelines, analyze the conversation below and extract one such insight. Categorize this insight with a short, one-word topic (e.g., 'user_preference', 'project_goal', 'tech_stack', 'learning_style').

Format the output as a valid JSON object with "topic" and "insight" keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        
        with self.ui_logger.thinking_process("Generating insight from recent conversation..."):
            insight_json_str = self.llm.generate(prompt, temperature=0.2)
        
        self.insight_manager.save_insight_from_json(insight_json_str, username)

    def develop_insights_from_conversation(self, username: str):
        """Command to develop insights from the full current conversation history."""
        self.ui_logger.system_message(f"Developing new insights for user '{username}' from the current conversation...")
        
        conversation_segment = "\n".join(self.history)
        prompt = f"""
Analyze the following conversation involving '{username}'. Your goal is to identify up to three high-quality insights that represent new understandings, significant user preferences, or key takeaways that should be remembered for future interactions with this user.

**Guidelines for a High-Quality Insight:**
- **Novelty:** It should be something new that wasn't known before this conversation.
- **Significance:** It should be important for future interactions (e.g., a user's goal, a core preference, a key fact about their project).
- **Conciseness:** The insight should be stated clearly and succinctly.
- **Not Obvious:** Avoid stating the obvious (e.g., "the user asked a question").

Based on these guidelines, analyze the conversation below. For each insight, provide a one-word topic and the insight itself. Format the output as a valid JSON array of objects, where each object has 'topic' and 'insight' keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        
        with self.ui_logger.thinking_process("Generating insights..."):
            insights_json_str = self.llm.generate(prompt, temperature=0.3)
        
        try:
            insights = json.loads(insights_json_str)
            for insight in insights:
                self.insight_manager.save_insight_from_json(json.dumps(insight), username)
        except json.JSONDecodeError:
            self.ui_logger.system_message(f"Failed to decode insights from LLM response: {insights_json_str}")

    def reflect_on_insights(self, username: str):
        """Command to reflect on all existing insights and generate new, meta-insights."""
        self.ui_logger.system_message(f"Reflecting on existing insights for user '{username}' to generate meta-insights...")
        
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            self.ui_logger.system_message("No existing insights to reflect on.")
            return

        insights_str = "\n".join([f"- {i['topic']}: {i['content']}" for i in all_insights])
        prompt = f"""
Analyze the following collection of insights for user '{username}'. Your goal is to identify a new, higher-level 'meta-insight'.

**What is a Meta-Insight?**
A meta-insight is a conclusion drawn from analyzing existing insights. It's about finding the connections, patterns, or underlying themes that are not obvious from any single insight. It synthesizes multiple pieces of knowledge into a more profound understanding.

**Examples:**
- If you have insights about a user's preference for Python and their questions about data science, a meta-insight might be: "The user is likely a data scientist who prefers to work in Python."
- If you have insights about a project's tech stack and its performance issues, a meta-insight might be: "The project's performance issues may be related to the choice of database technology."

Based on this, analyze the collection below and generate a single, powerful meta-insight. Format the output as a valid JSON object with 'topic' and 'insight' keys. The topic should be 'meta' or a new, relevant topic that captures the essence of the meta-insight.

[EXISTING INSIGHTS]
{insights_str}

[JSON_OUTPUT]
"""

        with self.ui_logger.thinking_process("Generating meta-insight..."):
            meta_insight_json_str = self.llm.generate(prompt, temperature=0.4)
        
        self.insight_manager.save_insight_from_json(meta_insight_json_str, username)

    def develop_insights_from_memory(self, username: str):
        """Command to develop new insights from a broad search of long-term memory."""
        self.ui_logger.system_message(f"Developing new insights for user '{username}' from long-term memory...")
        
        # Use a broad query to get a wide range of memories
        context = self.memory_search.search_all("general knowledge and past experiences", username)
        context_str = "\n".join(f"- {c}" for c in context)

        prompt = f"""
Analyze the following context from long-term memory for user '{username}'. Your goal is to identify a single, high-quality insight that represents a new understanding or a key takeaway that should be remembered for future interactions.

**Guidelines for a High-Quality Insight:**
- **Novelty:** It should be something new that wasn't known before.
- **Significance:** It should be important for future interactions.
- **Conciseness:** The insight should be stated clearly and succinctly.
- **Not Obvious:** Avoid stating the obvious.

Based on these guidelines, analyze the context below and extract one such insight. Categorize this insight with a short, one-word topic.

Format the output as a valid JSON object with "topic" and "insight" keys.

[MEMORY CONTEXT]
{context_str}

[JSON_OUTPUT]
"""

        with self.ui_logger.thinking_process("Generating insight from memory..."):
            insight_json_str = self.llm.generate(prompt, temperature=0.3)
        
        self.insight_manager.save_insight_from_json(insight_json_str, username)