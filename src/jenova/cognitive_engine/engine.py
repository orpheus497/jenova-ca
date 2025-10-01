import json
from jenova.cortex.proactive_engine import ProactiveEngine

class CognitiveEngine:
    """The Perfected Cognitive Engine. Manages the refined cognitive cycle."""
    def __init__(self, llm, memory_search, file_tools, insight_manager, assumption_manager, config, ui_logger, file_logger, cortex, system_tools):
        self.llm = llm
        self.memory_search = memory_search
        self.file_tools = file_tools
        self.insight_manager = insight_manager
        self.assumption_manager = assumption_manager
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.cortex = cortex
        self.system_tools = system_tools
        self.proactive_engine = ProactiveEngine(cortex, llm, ui_logger)
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

            # Cognitive cycle hooks
            if self.turn_count % 5 == 0:
                self.generate_insight_from_history(username)
            if self.turn_count % 7 == 0:
                self.generate_assumption_from_history(username)
            if self.turn_count % 10 == 0:
                self.cortex.reflect(username)
            
            # Proactive suggestion
            if self.turn_count % 15 == 0:
                suggestion = self.proactive_engine.get_suggestion(username)
                if suggestion:
                    self.ui_logger.system_message(f"Jenova has a thought: {suggestion}")

        return response

    def _plan(self, user_input: str, context: list[str], username: str) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        prompt = f"""== CONTEXT ==\n{context_str if context else "No context available."}\n\n== TASK ==\nBased on the user's query and the provided context, create a short, step-by-step plan to formulate a response. The plan can include generating a new insight if a novel conclusion is reached.\n\nUser ({username}): \"{user_input}\"\n\nPlan:"""
        with self.ui_logger.thinking_process("Formulating plan..."):
            plan = self.llm.generate(prompt, temperature=0.1, stop=["\n\n"])
        self.file_logger.log_info(f"Generated Plan: {plan}")
        return plan

    def _execute(self, user_input: str, context: list[str], plan: str, username: str) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        history_str = "\n".join(self.history)
        
        is_identity_query = "who are you" in user_input.lower() or "tell me about yourself" in user_input.lower()
        task_prompt = "Execute the plan to respond to the user's query. Your response must be grounded in your persona, the retrieved context, and the conversation history. Provide ONLY Jenova's response." if is_identity_query else "Execute the plan to respond to the user's query. Be direct and conversational. Do not re-introduce yourself unless asked. Provide ONLY Jenova's response."

        prompt = f"""== RETRIEVED CONTEXT ==\n{context_str if context else "No context available."}\n\n== CONVERSATION HISTORY ==\n{history_str if self.history else "No history yet."}\n\n== YOUR INTERNAL PLAN ==\n{plan}\n\n== TASK ==\n{task_prompt}\n\nUser ({username}): \"{user_input}\"\n\nJenova:"""
        with self.ui_logger.thinking_process("Executing plan..."):
            response = self.llm.generate(prompt)
        self.file_logger.log_info(f"Generated Response: {response}")
        return response

    def generate_insight_from_history(self, username: str):
        """Analyzes recent conversation history to generate and save a new, high-quality insight."""
        if len(self.history) < 2:
            return

        conversation_segment = "\n".join(self.history[-8:])

        prompt = f"""
Analyze the following conversation segment involving the user '{username}'. Your goal is to identify a single, high-quality insight that represents a new understanding, a significant user preference, a correction of a previous assumption, or a key takeaway that should be remembered for future interactions with '{username}'.

Format the output as a valid JSON object with "topic" and "insight" keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        
        with self.ui_logger.thinking_process("Generating insight from recent conversation..."):
            insight_json_str = self.llm.generate(prompt, temperature=0.2)
        
        try:
            data = json.loads(insight_json_str)
            topic = data.get('topic')
            insight = data.get('insight')
            if topic and insight:
                self.insight_manager.save_insight(insight, username, topic=topic)
        except json.JSONDecodeError:
            self.ui_logger.system_message(f"Failed to decode insight from LLM response: {insight_json_str}")

    def generate_assumption_from_history(self, username: str):
        """Analyzes recent conversation history to generate a new assumption about the user."""
        if len(self.history) < 4:
            return

        conversation_segment = "\n".join(self.history[-8:])

        prompt = f"""
Analyze the following conversation with '{username}'. Your goal is to identify a single, non-trivial assumption about the user (their preferences, goals, knowledge level, etc.) that is implied but not explicitly stated. This assumption will be verified later.

Format the output as a valid JSON object with an "assumption" key.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        with self.ui_logger.thinking_process("Forming new assumption..."):
            assumption_json_str = self.llm.generate(prompt, temperature=0.3)

        try:
            data = json.loads(assumption_json_str)
            assumption = data.get('assumption')
            if assumption:
                self.assumption_manager.add_assumption(assumption, username)
        except json.JSONDecodeError:
            self.ui_logger.system_message(f"Failed to decode assumption from LLM response: {assumption_json_str}")

    def develop_insights_from_conversation(self, username: str):
        """Command to develop insights from the full current conversation history."""
        self.ui_logger.system_message(f"Developing new insights for user '{username}' from the current conversation...")
        
        conversation_segment = "\n".join(self.history)
        prompt = f"""
Analyze the following conversation involving '{username}'. Identify up to three high-quality insights.

Format the output as a valid JSON array of objects, where each object has 'topic' and 'insight' keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        
        with self.ui_logger.thinking_process("Generating insights..."):
            insights_json_str = self.llm.generate(prompt, temperature=0.3)
        
        try:
            insights = json.loads(insights_json_str)
            for insight_data in insights:
                topic = insight_data.get('topic')
                insight = insight_data.get('insight')
                if topic and insight:
                    self.insight_manager.save_insight(insight, username, topic=topic)
        except json.JSONDecodeError:
            self.ui_logger.system_message(f"Failed to decode insights from LLM response: {insights_json_str}")

    def reflect_on_insights(self, username: str):
        """Command to reorganize and interlink all existing insights."""
        self.insight_manager.reorganize_insights(username)

    def generate_meta_insight(self, username: str):
        """Command to generate a new, meta-insight from existing insights."""
        self.ui_logger.system_message(f"Reflecting on existing insights for user '{username}' to generate meta-insights...")
        
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            self.ui_logger.system_message("No existing insights to reflect on.")
            return

        insights_str = "\n".join([f"- {i['topic']}: {i['content']}" for i in all_insights])
        prompt = f"""
Analyze the following collection of insights for user '{username}'. Your goal is to identify a new, higher-level 'meta-insight'. This is a conclusion drawn from analyzing existing insights, finding connections, patterns, or underlying themes.

[EXISTING INSIGHTS]
{insights_str}

[JSON_OUTPUT]
"""

        with self.ui_logger.thinking_process("Generating meta-insight..."):
            meta_insight_json_str = self.llm.generate(prompt, temperature=0.4)
        
        try:
            data = json.loads(meta_insight_json_str)
            topic = data.get('topic', 'meta')
            insight = data.get('insight')
            if insight:
                self.insight_manager.save_insight(insight, username, topic=topic)
                self.ui_logger.system_message(f"New meta-insight generated under topic '{topic}': {insight}")
        except json.JSONDecodeError:
            self.ui_logger.system_message(f"Failed to decode meta-insight from LLM response: {meta_insight_json_str}")

    def develop_insights_from_memory(self, username: str):
        """Command to develop new insights or assumptions from a broad search of long-term memory."""
        self.ui_logger.system_message(f"Developing new insights for user '{username}' from long-term memory...")
        
        context = self.memory_search.search_all("general knowledge and past experiences", username)
        context_str = "\n".join(f"- {c}" for c in context)

        prompt = f"""
Analyze the following context from long-term memory for user '{username}'. Your goal is to extract a single, high-quality piece of information. Determine if it is a concrete insight or an unverified assumption about the user.

Format the output as a valid JSON object with one of two structures:
1. For an insight: {{"type": "insight", "topic": "<topic>", "content": "<insight_content>"}}
2. For an assumption: {{"type": "assumption", "content": "<assumption_content>"}}

[MEMORY CONTEXT]
{context_str}

[JSON_OUTPUT]
"""

        with self.ui_logger.thinking_process("Generating insight from memory..."):
            insight_json_str = self.llm.generate(prompt, temperature=0.3)
        
        try:
            data = json.loads(insight_json_str)
            if data.get('type') == 'insight':
                topic = data.get('topic')
                insight = data.get('content')
                if topic and insight:
                    self.insight_manager.save_insight(insight, username, topic=topic)
            elif data.get('type') == 'assumption':
                assumption = data.get('content')
                if assumption:
                    self.assumption_manager.add_assumption(assumption, username)
        except json.JSONDecodeError:
            self.ui_logger.system_message(f"Failed to decode insight from LLM response: {insight_json_str}")

    def verify_assumptions(self, username: str):
        """Command to verify unverified assumptions with the user."""
        return self.assumption_manager.get_assumption_to_verify(self.llm, username)

    def finetune(self):
        """Command to trigger the fine-tuning process."""
        self.ui_logger.system_message("Initiating fine-tuning process...")
        
        # Step 1: Prepare the data
        prepare_command = "python finetune/prepare_data.py"
        self.system_tools.execute_shell_command(prepare_command, "Preparing fine-tuning data...")

        # Step 2: Run the fine-tuning
        finetune_config = self.config.get('finetuning', {})
        base_model_path = self.config.get('model', {}).get('model_path', 'models/jenova.gguf') # Assuming model_path is in config
        training_file = finetune_config.get('training_file', 'finetune_train.jsonl')
        lora_output = finetune_config.get('lora_output_file', 'models/lora-jenova-adapter.bin')
        threads = self.config.get('hardware', {}).get('threads', 4)
        gpu_layers = self.config.get('hardware', {}).get('gpu_layers', 0)

        # This is a more realistic llama.cpp command
        # The user might need to adjust paths and parameters based on their setup
        finetune_command = f"""./llama.cpp/finetune --model-base {base_model_path} \
--train-data \"{training_file}\" \
--lora-out {lora_output} \
--threads {threads} --gpu-layers {gpu_layers} \
--batch-size 4 --epochs 3 --use-flash-attn"""

        self.ui_logger.system_message("Executing fine-tuning command. This may take a while...")
        self.system_tools.execute_shell_command(finetune_command, "Running fine-tuning...")
        self.ui_logger.system_message("Fine-tuning process completed. The new LoRA adapter is available at {lora_output}")
