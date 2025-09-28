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
        self.MAX_HISTORY_TURNS = 5 # Keep the last 5 conversation turns (10 entries)

    def think(self, user_input: str) -> str:
        """Runs the full cognitive cycle: Retrieve, Plan, Execute, Reflect."""
        with self.ui_logger.cognitive_process(f"Cognitive cycle started for: '{user_input}'"):
            self.file_logger.log_info(f"New query received: {user_input}")

            context = self.memory_search.search_all(user_input)
            plan = self._plan(user_input, context)
            response = self._execute(user_input, context, plan)

            # Update history and manage its size
            self.history.append(f"User: {user_input}")
            self.history.append(f"Jenova: {response}")
            if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]

            self._reflect()
        
        return response

    def _plan(self, user_input: str, context: list[str]) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        prompt = f"""== CONTEXT ==\n{context_str if context else "No context available."}\n\n== TASK ==\nBased on the user's query and the provided context, create a short, step-by-step plan to formulate a response. The plan can include generating a new insight if a novel conclusion is reached.\n\nUser: "{user_input}"\n\nPlan:"""
        with self.ui_logger.thinking_process("Formulating plan..."):
            plan = self.llm.generate(prompt, temperature=0.1, stop=["\n\n"])
        self.file_logger.log_info(f"Generated Plan: {plan}")
        return plan

    def _execute(self, user_input: str, context: list[str], plan: str) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        history_str = "\n".join(self.history)
        
        is_identity_query = "who are you" in user_input.lower() or "tell me about yourself" in user_input.lower()
        task_prompt = "Execute the plan to respond to the user's query. Your response must be grounded in your persona, the retrieved context, and the conversation history. Provide ONLY Jenova's response." if is_identity_query else "Execute the plan to respond to the user's query. Be direct and conversational. Do not re-introduce yourself unless asked. Provide ONLY Jenova's response."

        prompt = f"""== RETRIEVED CONTEXT ==\n{context_str if context else "No context available."}\n\n== CONVERSATION HISTORY ==\n{history_str if self.history else "No history yet."}\n\n== YOUR INTERNAL PLAN ==\n{plan}\n\n== TASK ==\n{task_prompt}\n\nUser: "{user_input}"\n\nJenova:"""
        with self.ui_logger.thinking_process("Executing plan..."):
            response = self.llm.generate(prompt)
        self.file_logger.log_info(f"Generated Response: {response}")
        return response

    def _reflect(self):
        """Analyzes recent history to generate and save new insights."""
        if len(self.history) > 0 and len(self.history) % (self.config['memory']['reflection_interval'] * 2) == 0:
            self.ui_logger.reflection("Reflecting on recent conversation to generate new insights...")
            
            conversation_segment = "\n".join(self.history[-(self.config['memory']['reflection_interval'] * 2):])
            prompt = f"""Analyze the following conversation segment. Identify a single, novel insight or conclusion reached. Categorize this insight with a short, one-word topic (e.g., 'creator', 'linux', 'user_feedback'). Format the output as a valid JSON object with 'topic' and 'insight' keys.\n\n[CONVERSATION]\n{conversation_segment}\n\n[JSON_OUTPUT]"""
            
            with self.ui_logger.thinking_process("Generating insight..."):
                insight_json_str = self.llm.generate(prompt, temperature=0.2)
            
            self.insight_manager.save_insight_from_json(insight_json_str)