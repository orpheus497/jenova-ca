import json
import os
import shlex
import inspect
from jenova.cortex.proactive_engine import ProactiveEngine
from jenova.cognitive_engine.scheduler import CognitiveScheduler
from jenova import tools

class CognitiveEngine:
    """The Perfected Cognitive Engine. Manages the refined cognitive cycle."""
    def __init__(self, llm, memory_search, insight_manager, assumption_manager, config, ui_logger, file_logger, cortex, rag_system):
        self.llm = llm
        self.memory_search = memory_search
        self.insight_manager = insight_manager
        self.assumption_manager = assumption_manager
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.cortex = cortex
        self.proactive_engine = ProactiveEngine(cortex, llm, ui_logger)
        self.rag_system = rag_system
        self.scheduler = CognitiveScheduler(config, cortex, insight_manager)
        self.history = []
        self.turn_count = 0
        self.MAX_HISTORY_TURNS = 10 # Keep the last 10 conversation turns
        self.pending_assumption = None

    def think(self, user_input: str, username: str) -> str:
        """Runs the full cognitive cycle: Retrieve, Plan, Execute, and Reflect."""
        with self.ui_logger.cognitive_process("Thinking...") as thinking_status:
            self.file_logger.log_info(f"New query received from {username}: {user_input}")
            self.turn_count += 1

            # Retrieve, Plan, Execute
            context = self.memory_search.search_all(user_input, username)
            # Proactive safeguard: Ensure all context items are strings.
            if context:
                context = [str(item) for item in context]
            else:
                context = []
            
            plan = self._plan(user_input, context, username, thinking_status)
            if not plan:
                plan = f"I will formulate a response to the user's query: {user_input}"

            response = self._execute(user_input, context, plan, username, thinking_status)

            # Only add to history if it's not a command
            if not user_input.startswith('/'):
                self.history.append(f"{username}: {user_input}")
                self.history.append(f"Jenova: {response}")
                if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                    self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]

                # Add to episodic memory
                self.memory_search.episodic_memory.add_episode(f"{username}: {user_input}\nJenova: {response}", username)

        return response


    def _plan(self, user_input: str, context: list[str], username: str, thinking_process=None) -> str:
        # Ensure all context items are strings before joining
        safe_context = [str(c) for c in context]
        context_str = "\n".join(f"- {c}" for c in safe_context)
        
        # Load persona from config
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        directives = persona_config.get('directives', [])
        
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')

        # Check if the current user is the creator
        user_title = "User"
        if username == creator_name:
            user_title = creator_alias

        prompt = f"""You are {ai_name}, a self-aware AI. Your creator is {creator_name}, whom you know as {creator_alias}.
Your core directives are: {directives}

== CONTEXT ==
{context_str if context else "No context available."}

== TASK ==
Based on your identity, directives, the user's query, and the provided context, create a step-by-step plan to formulate a response. The plan should be a short paragraph describing the steps you will take.

{user_title} ({username}): "{user_input}"

Plan:"""
        
        # Use the passed thinking_process if available, otherwise create a new one
        if thinking_process is not None:
            # In queue mode, status might be None, so check before calling update
            if hasattr(thinking_process, 'update'):
                thinking_process.update("Formulating plan...")
            plan = self.llm.generate(prompt, temperature=0.1, thinking_process=thinking_process)
        else:
            with self.ui_logger.thinking_process("Formulating plan..."):
                plan = self.llm.generate(prompt, temperature=0.1)
        
        self.file_logger.log_info(f"Generated Plan: {plan}")
        return plan

    def _execute(self, user_input: str, context: list[str], plan: str, username: str, thinking_process=None) -> str:
        # Use the passed thinking_process if available, otherwise create a new one
        if thinking_process is not None:
            # In queue mode, status might be None, so check before calling update
            if hasattr(thinking_process, 'update'):
                thinking_process.update("Executing plan...")
            return self.rag_system.generate_response(user_input, username, self.history, plan, thinking_process=thinking_process)
        else:
            with self.ui_logger.thinking_process("Executing plan..."):
                return self.rag_system.generate_response(user_input, username, self.history, plan)

    def generate_insight_from_history(self, username: str):
        """Analyzes recent conversation history to generate and save a new, high-quality insight."""
        self.ui_logger.info("Analyzing conversation history to generate a new insight...")
        if len(self.history) < 2:
            return

        conversation_segment = "\n".join(self.history[-8:])
        
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing a conversation with {user_title} ({username}). Your goal is to identify a single, high-quality insight that represents a new understanding, a significant user preference, a correction of a previous assumption, or a key takeaway that should be remembered for future interactions.

Format the output as a valid JSON object with "topic" and "insight" keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        with self.ui_logger.thinking_process("Generating insight from recent conversation..."):
            insight_json_str = self.llm.generate(prompt, temperature=0.2, grammar=self.cortex.json_grammar)
        
        try:
            data = json.loads(insight_json_str)
            topic = data.get('topic')
            insight = data.get('insight')
            if topic and insight:
                self.insight_manager.save_insight(insight, username, topic=topic)
        except (json.JSONDecodeError, ValueError):
            self.ui_logger.system_message(f"Failed to decode insight from LLM response: {insight_json_str}")

    def generate_assumption_from_history(self, username: str):
        """Analyzes recent conversation history to generate a new assumption about the user."""
        self.ui_logger.info("Analyzing conversation history to generate a new assumption...")
        if len(self.history) < 4:
            return

        conversation_segment = "\n".join(self.history[-8:])

        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""You are {ai_name}. You are analyzing a conversation with {user_title} ({username}). Your goal is to identify a single, non-trivial assumption about the user (their preferences, goals, knowledge level, etc.) that is implied but not explicitly stated. This assumption will be verified later.

Format the output as a valid JSON object with an "assumption" key.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        with self.ui_logger.thinking_process("Forming new assumption..."):
            assumption_json_str = self.llm.generate(prompt, temperature=0.3, grammar=self.cortex.json_grammar)

        try:
            assumption_data = json.loads(assumption_json_str)
            assumption = assumption_data.get('assumption')
            if assumption:
                self.assumption_manager.add_assumption(assumption, username)
        except (json.JSONDecodeError, ValueError):
            self.ui_logger.system_message(f"Failed to decode assumption from LLM response: {assumption_json_str}")

    def develop_insights_from_conversation(self, username: str) -> list[str]:
        """Command to develop insights from the full current conversation history."""
        messages = []
        
        conversation_segment = "\n".join(self.history)
        
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing a conversation with {user_title} ({username}). Identify up to three high-quality insights.

Format the output as a valid JSON array of objects, where each object has 'topic' and 'insight' keys.

[CONVERSATION]
{conversation_segment}

[JSON_OUTPUT]
"""
        insights_json_str = self.llm.generate(prompt, temperature=0.3, grammar=self.cortex.json_grammar)
        
        try:
            insights = json.loads(insights_json_str)
            for insight_data in insights:
                topic = insight_data.get('topic')
                insight = insight_data.get('insight')
                if topic and insight:
                    insight_id = self.insight_manager.save_insight(insight, username, topic=topic)
                    messages.append(f"New insight node created: {insight_id}")
        except (json.JSONDecodeError, ValueError):
            messages.append(f"Failed to decode insights from LLM response: {insights_json_str}")
        return messages

    def reflect_on_insights(self, username: str) -> list[str]:
        """Command to trigger a deep reflection on the cognitive graph."""
        return self.cortex.reflect(user=username)

    def generate_meta_insight(self, username: str) -> list[str]:
        """Command to generate a new, meta-insight from existing insights."""
        messages = []
        
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            messages.append("No existing insights to reflect on.")
            return messages

        self.file_logger.log_info(f"All insights: {all_insights}")

        insights_str = "\n".join([f"- {i['topic']}: {i['content']}" for i in all_insights])
        
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing a collection of insights for {user_title} ({username}). Your goal is to identify a new, higher-level 'meta-insight'. This is a conclusion drawn from analyzing existing insights, finding connections, patterns, or underlying themes.

[EXISTING INSIGHTS]
{insights_str}

[JSON_OUTPUT]
"""
        self.file_logger.log_info(f"Meta-insight prompt: {prompt}")

        meta_insight_json_str = self.llm.generate(prompt, temperature=0.4, grammar=self.cortex.json_grammar)
        
        self.file_logger.log_info(f"Meta-insight response: {meta_insight_json_str}")

        try:
            data = json.loads(meta_insight_json_str)
            topic = data.get('topic', 'meta')
            insight = data.get('insight')
            if insight:
                self.insight_manager.save_insight(insight, username, topic=topic)
                messages.append(f"New meta-insight generated under topic '{topic}': {insight}")
            else:
                messages.append("No new meta-insight could be generated from the existing insights.")
        except (json.JSONDecodeError, ValueError):
            messages.append(f"Failed to decode meta-insight from LLM response: {meta_insight_json_str}")
        return messages

    def develop_insights_from_memory(self, username: str) -> list[str]:
        messages = []
        """Command to develop new insights or assumptions from a broad search of long-term memory."""
        messages.append("Developing new insights from long-term memory...")
        
        context = self.memory_search.search_all("general knowledge and past experiences", username)
        if context is None:
            context = []
        # Ensure all context items are strings before joining to prevent TypeErrors
        safe_context = [str(c) for c in context]
        context_str = "\n".join(f"- {c}" for c in safe_context)

        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')
        user_title = creator_alias if username == creator_name else "User"

        prompt = f"""
You are {ai_name}. You are analyzing context from your long-term memory for {user_title} ({username}). Your goal is to extract a single, high-quality piece of information. Determine if it is a concrete insight or an unverified assumption about the user.

Format the output as a valid JSON object with one of two structures:
1. For an insight: {{"type": "insight", "topic": "<topic>", "content": "<insight_content>"}}
2. For an assumption: {{"type": "assumption", "content": "<assumption_content>"}}

[MEMORY CONTEXT]
{context_str}

[JSON_OUTPUT]
"""
        insight_json_str = self.llm.generate(prompt, temperature=0.3, grammar=self.cortex.json_grammar)
        
        try:
            data = json.loads(insight_json_str)
            if data.get('type') == 'insight':
                topic = data.get('topic')
                insight = data.get('content')
                if topic and insight:
                    insight_id = self.insight_manager.save_insight(insight, username, topic=topic)
                    messages.append(f"New insight node created: {insight_id}")
            elif data.get('type') == 'assumption':
                assumption = data.get('content')
                if assumption:
                    assumption_id = self.assumption_manager.add_assumption(assumption, username)
                    if assumption_id != "Assumption already exists.": # Check if a new assumption was actually added
                        messages.append(f"New assumption node created: {assumption_id}")
                    else:
                        messages.append(assumption_id) # Append the "Assumption already exists." message
        except (json.JSONDecodeError, ValueError):
            messages.append(f"Failed to decode insight from LLM response: {insight_json_str}")
        return messages

    def proactively_verify_assumption(self, username: str):
        """Proactively verifies an unverified assumption with the user."""
        assumption, question = self.assumption_manager.get_assumption_to_verify(username)
        if assumption and question:
            self.pending_assumption = assumption
            return question
        return None

    def verify_assumptions(self, username: str):
        """Command to verify unverified assumptions with the user."""
        question = self.proactively_verify_assumption(username)
        if question:
            return self.pending_assumption, question
        else:
            return None, "No unverified assumptions to check."

    def learn_procedure(self, procedure_data: dict, username: str) -> list[str]:
        """Command to learn a new procedure interactively."""
        messages = []
        procedure_name = procedure_data.get('name')
        steps = procedure_data.get('steps', [])
        outcome = procedure_data.get('outcome')

        if not procedure_name or not steps or not outcome:
            messages.append("Error: Incomplete procedure data provided.")
            return messages

        # Generate context for the procedure using LLM
        context_prompt = f"""Given the following procedure:
Name: {procedure_name}
Steps: {steps}
Outcome: {outcome}

Generate a concise, 1-2 sentence context for this procedure. What is its general purpose or domain?

Context:"""
        with self.ui_logger.thinking_process("Generating procedure context..."):
            context = self.llm.generate(context_prompt, temperature=0.3)

        # Combine name and steps for the main procedure document
        procedure_doc = f"Procedure: {procedure_name}\nSteps: {'; '.join(steps)}\nOutcome: {outcome}"

        self.memory_search.procedural_memory.add_procedure(
            procedure=procedure_doc,
            username=username,
            goal=outcome,
            steps=steps,
            context=context
        )
        messages.append(f"Procedure '{procedure_name}' learned successfully.")
        return messages