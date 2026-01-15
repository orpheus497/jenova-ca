##Script function and purpose: Cognitive Engine for The JENOVA Cognitive Architecture
##This module implements the core cognitive cycle: Retrieve, Plan, Execute, Reflect

import json
import os
import shlex
import inspect
from typing import List, Dict, Any, Optional, Tuple
from jenova.cortex.proactive_engine import ProactiveEngine
from jenova.cognitive_engine.scheduler import CognitiveScheduler
from jenova.cognitive_engine.query_analyzer import QueryAnalyzer
from jenova.utils.json_parser import extract_json
from jenova import tools

##Class purpose: Orchestrates the cognitive cycle and coordinates all cognitive functions
class CognitiveEngine:
    """The Perfected Cognitive Engine. Manages the refined cognitive cycle."""
    ##Function purpose: Initialize the cognitive engine with all required components
    def __init__(self, llm: Any, memory_search: Any, insight_manager: Any, assumption_manager: Any, config: Dict[str, Any], ui_logger: Any, file_logger: Any, cortex: Any, rag_system: Any) -> None:
        self.llm = llm
        self.memory_search = memory_search
        self.insight_manager = insight_manager
        self.assumption_manager = assumption_manager
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.cortex = cortex
        self.proactive_engine = ProactiveEngine(cortex, llm, ui_logger, config.get('cortex', {}))
        self.rag_system = rag_system
        self.scheduler = CognitiveScheduler(config, cortex, insight_manager)
        self.history = []
        self.turn_count = 0
        self.MAX_HISTORY_TURNS = 10 # Keep the last 10 conversation turns
        self.pending_assumption = None
        
        ##Block purpose: Initialize query analyzer for enhanced comprehension
        self.query_analyzer = QueryAnalyzer(llm, config, cortex.json_grammar if cortex else None)
        
        ##Block purpose: Set Cortex reference for entity linking (Phase C.2)
        # Note: We intentionally pass username=None here because CognitiveEngine may be reused
        # across different users/sessions. The per-request username context is set later in
        # the think() method via QueryAnalyzer.set_username(), after we know the actual user.
        if cortex:
            self.query_analyzer.set_cortex(cortex, None)
        
        ##Block purpose: Initialize integration layer reference (will be set by main.py after initialization)
        ##This enables Cortex-Memory feedback loops and unified knowledge representation
        self.integration_layer = None

    ##Function purpose: Execute the full cognitive cycle: Retrieve, Plan, Execute
    def think(self, user_input: str, username: str) -> str:
        """Runs the full cognitive cycle: Retrieve, Plan, Execute, and Reflect."""
        with self.ui_logger.cognitive_process("Thinking...") as thinking_status:
            self.file_logger.log_info(f"New query received from {username}: {user_input}")
            self.turn_count += 1
            
            ##Block purpose: Update QueryAnalyzer with current username for entity linking (Phase C.2)
            if self.query_analyzer.get_username() != username:
                self.query_analyzer.set_username(username)

            ##Block purpose: Analyze query for enhanced comprehension
            query_analysis = self.query_analyzer.analyze(user_input)
            self.file_logger.log_info(f"Query analysis: {query_analysis}")

            ##Block purpose: Retrieve context with query analysis for enhanced scoring
            context = self.memory_search.search_all(user_input, username, query_analysis)
            # Proactive safeguard: Ensure all context items are strings.
            if context:
                context = [str(item) for item in context]
            else:
                context = []
            
            plan = self._plan(user_input, context, username, query_analysis, thinking_status)
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
                episode_content = f"{username}: {user_input}\nJenova: {response}"
                self.memory_search.episodic_memory.add_episode(episode_content, username)
                
                ##Block purpose: Provide feedback from Memory to Cortex (if integration layer available)
                integration_config = self.config.get('cortex', {}).get('integration', {})
                if integration_config.get('memory_to_cortex_feedback', False) and hasattr(self, 'integration_layer') and self.integration_layer:
                    try:
                        self.integration_layer.feedback_memory_to_cortex(
                            episode_content, 'episodic', username
                        )
                    except Exception as e:
                        self.file_logger.log_error(f"Error providing Memory-to-Cortex feedback: {e}")

        return response


    ##Function purpose: Generate an internal plan for responding to the user query with enhanced multi-level planning
    def _plan(self, user_input: str, context: List[str], username: str, query_analysis: Optional[Dict[str, Any]] = None, thinking_process: Optional[Any] = None) -> str:
        """Generates plan with multi-level planning support based on query complexity."""
        
        ##Block purpose: Ensure query analysis exists (backward compatibility)
        if query_analysis is None:
            query_analysis = self.query_analyzer.analyze(user_input)
        
        ##Block purpose: Determine planning approach based on query complexity
        complexity = query_analysis.get('complexity', 'simple')
        comprehension_config = self.config.get('comprehension', {})
        planning_config = comprehension_config.get('planning', {})
        multi_level_enabled = planning_config.get('multi_level', True)
        
        ##Block purpose: Use simple planning for simple/moderate queries or if multi-level disabled
        if complexity in ['simple', 'moderate'] or not multi_level_enabled:
            return self._simple_plan(user_input, context, username, query_analysis, thinking_process)
        else:
            return self._complex_plan(user_input, context, username, query_analysis, thinking_process)
    
    ##Function purpose: Generate simple single-level plan (backward compatible)
    def _simple_plan(self, user_input: str, context: List[str], username: str, query_analysis: Dict[str, Any], thinking_process: Optional[Any]) -> str:
        """Generates a simple single-level plan."""
        
        ##Block purpose: Ensure all context items are strings before joining
        safe_context = [str(c) for c in context]
        context_str = "\n".join(f"- {c}" for c in safe_context)
        
        ##Block purpose: Load persona from config
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        directives = persona_config.get('directives', [])
        
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')

        ##Block purpose: Check if the current user is the creator
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
        
        ##Block purpose: Use the passed thinking_process if available, otherwise create a new one
        if thinking_process is not None:
            ##Block purpose: In queue mode, status might be None, so check before calling update
            if hasattr(thinking_process, 'update'):
                thinking_process.update("Formulating plan...")
            plan = self.llm.generate(prompt, temperature=0.1, thinking_process=thinking_process)
        else:
            with self.ui_logger.thinking_process("Formulating plan..."):
                plan = self.llm.generate(prompt, temperature=0.1)
        
        self.file_logger.log_info(f"Generated Plan: {plan}")
        return plan
    
    ##Function purpose: Generate complex multi-level plan with sub-goals and reasoning chain
    def _complex_plan(self, user_input: str, context: List[str], username: str, query_analysis: Dict[str, Any], thinking_process: Optional[Any]) -> str:
        """Generates a structured multi-level plan with sub-goals and reasoning chain."""
        
        ##Block purpose: Ensure all context items are strings before joining
        safe_context = [str(c) for c in context]
        context_str = "\n".join(f"- {c}" for c in safe_context)
        
        ##Block purpose: Load persona from config
        persona_config = self.config.get('persona', {})
        identity = persona_config.get('identity', {})
        directives = persona_config.get('directives', [])
        
        creator_name = identity.get('creator', 'orpheus497')
        creator_alias = identity.get('creator_alias', 'The Architect')
        ai_name = identity.get('name', 'Jenova')

        ##Block purpose: Check if the current user is the creator
        user_title = "User"
        if username == creator_name:
            user_title = creator_alias
        
        ##Block purpose: Extract query analysis information
        intent = query_analysis.get('intent', 'question')
        entities = query_analysis.get('entities', [])
        query_type = query_analysis.get('type', 'factual')
        max_sub_goals = self.config.get('comprehension', {}).get('planning', {}).get('max_sub_goals', 5)
        
        prompt = f"""You are {ai_name}, a self-aware AI. Your creator is {creator_name}, whom you know as {creator_alias}.
Your core directives are: {directives}

== CONTEXT ==
{context_str if context else "No context available."}

== QUERY ANALYSIS ==
Intent: {intent}
Query Type: {query_type}
Key Entities: {', '.join(entities) if entities else 'None'}

== TASK ==
Generate a structured plan with:
1. Main goal: The overall objective for responding to this query
2. Sub-goals: 3-{max_sub_goals} specific steps to achieve the main goal
3. Reasoning chain: Logical progression showing how sub-goals lead to the main goal

Respond with a valid JSON object:
{{
    "main_goal": "<overall objective>",
    "sub_goals": ["<step 1>", "<step 2>", ...],
    "reasoning_chain": ["<reasoning step 1>", "<reasoning step 2>", ...]
}}

{user_title} ({username}): "{user_input}"

JSON Plan:"""
        
        ##Block purpose: Use the passed thinking_process if available, otherwise create a new one
        if thinking_process is not None:
            ##Block purpose: In queue mode, status might be None, so check before calling update
            if hasattr(thinking_process, 'update'):
                thinking_process.update("Formulating multi-level plan...")
            plan_json_str = self.llm.generate(
                prompt, 
                temperature=0.3, 
                grammar=self.cortex.json_grammar if self.cortex else None,
                thinking_process=thinking_process
            )
        else:
            with self.ui_logger.thinking_process("Formulating multi-level plan..."):
                plan_json_str = self.llm.generate(
                    prompt, 
                    temperature=0.3, 
                    grammar=self.cortex.json_grammar if self.cortex else None
                )
        
        ##Block purpose: Parse structured plan JSON
        try:
            from jenova.utils.json_parser import extract_json
            plan_data = extract_json(plan_json_str)
            
            main_goal = plan_data.get('main_goal', '')
            sub_goals = plan_data.get('sub_goals', [])
            reasoning_chain = plan_data.get('reasoning_chain', [])
            
            ##Block purpose: Format structured plan as readable text
            plan_parts = [f"Main Goal: {main_goal}"]
            if sub_goals:
                plan_parts.append("\nSub-goals:")
                for i, sub_goal in enumerate(sub_goals, 1):
                    plan_parts.append(f"  {i}. {sub_goal}")
            if reasoning_chain:
                plan_parts.append("\nReasoning Chain:")
                for i, reasoning in enumerate(reasoning_chain, 1):
                    plan_parts.append(f"  {i}. {reasoning}")
            
            plan = "\n".join(plan_parts)
            self.file_logger.log_info(f"Generated Multi-level Plan: {plan}")
            return plan
            
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            ##Block purpose: Fallback to simple plan if JSON parsing fails
            self.file_logger.log_warning(f"Failed to parse structured plan JSON: {e}. Falling back to simple plan.")
            return self._simple_plan(user_input, context, username, query_analysis, thinking_process)

    ##Function purpose: Execute the plan and generate response using RAG system
    def _execute(self, user_input: str, context: List[str], plan: str, username: str, thinking_process: Optional[Any] = None) -> str:
        # Use the passed thinking_process if available, otherwise create a new one
        if thinking_process is not None:
            # In queue mode, status might be None, so check before calling update
            if hasattr(thinking_process, 'update'):
                thinking_process.update("Executing plan...")
            return self.rag_system.generate_response(user_input, username, self.history, plan, thinking_process=thinking_process)
        else:
            with self.ui_logger.thinking_process("Executing plan..."):
                return self.rag_system.generate_response(user_input, username, self.history, plan)

    ##Function purpose: Analyze conversation history to generate and save insights
    def generate_insight_from_history(self, username: str) -> None:
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
        
        ##Block purpose: Parse insight JSON using centralized utility
        data = extract_json(insight_json_str, default=None)
        if data is None:
            self.ui_logger.system_message(f"Failed to decode insight from LLM response: {insight_json_str}")
            return
        
        topic = data.get('topic')
        insight = data.get('insight')
        if topic and insight:
            self.insight_manager.save_insight(insight, username, topic=topic)

    ##Function purpose: Analyze conversation history to generate assumptions about the user
    def generate_assumption_from_history(self, username: str) -> None:
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

        ##Block purpose: Parse assumption JSON using centralized utility
        assumption_data = extract_json(assumption_json_str, default=None)
        if assumption_data is None:
            self.ui_logger.system_message(f"Failed to decode assumption from LLM response: {assumption_json_str}")
            return
        
        assumption = assumption_data.get('assumption')
        if assumption:
            self.assumption_manager.add_assumption(assumption, username)

    ##Function purpose: Develop multiple insights from the full conversation history
    def develop_insights_from_conversation(self, username: str) -> List[str]:
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
        
        ##Block purpose: Parse insights JSON using centralized utility
        insights = extract_json(insights_json_str, default=None)
        if insights is None:
            messages.append(f"Failed to decode insights from LLM response: {insights_json_str}")
            return messages
        
        ##Block purpose: Validate insights is a list, wrap dict in list if needed
        if isinstance(insights, dict):
            insights = [insights]  # Wrap single insight object in list
        elif not isinstance(insights, list):
            messages.append(f"Unexpected insights format (expected list): {type(insights)}")
            return messages
        
        for insight_data in insights:
            topic = insight_data.get('topic')
            insight = insight_data.get('insight')
            if topic and insight:
                insight_id = self.insight_manager.save_insight(insight, username, topic=topic)
                messages.append(f"New insight node created: {insight_id}")
        return messages

    ##Function purpose: Trigger deep reflection on the cognitive graph with unified knowledge integration
    def reflect_on_insights(self, username: str) -> List[str]:
        """Command to trigger a deep reflection on the cognitive graph."""
        messages = self.cortex.reflect(user=username)
        
        ##Block purpose: Integrate unified knowledge map into reflection (if integration layer available)
        integration_config = self.config.get('cortex', {}).get('integration', {})
        if integration_config.get('enabled', True) and hasattr(self, 'integration_layer') and self.integration_layer:
            try:
                ##Block purpose: Create unified knowledge map for comprehensive reflection
                knowledge_map = self.integration_layer.create_unified_knowledge_map(username)
                if knowledge_map:
                    cross_refs = knowledge_map.get('cross_references', [])
                    if cross_refs:
                        messages.append(f"Found {len(cross_refs)} cross-references between Memory and Cortex knowledge.")
                    self.file_logger.log_info(f"Unified knowledge map created: {len(knowledge_map.get('memory_items', []))} memory items, {len(knowledge_map.get('cortex_nodes', []))} Cortex nodes")
                
                ##Block purpose: Check knowledge consistency periodically during reflection
                consistency_report = self.integration_layer.check_knowledge_consistency(username)
                if not consistency_report.get('consistent', True):
                    gaps = consistency_report.get('gaps', [])
                    duplications = consistency_report.get('duplications', [])
                    if gaps:
                        messages.append(f"Knowledge consistency check: Found {len(gaps)} potential knowledge gaps.")
                    if duplications:
                        messages.append(f"Knowledge consistency check: Found {len(duplications)} potential duplications.")
                    recommendations = consistency_report.get('recommendations', [])
                    if recommendations:
                        for rec in recommendations[:2]:  # Limit to 2 recommendations
                            messages.append(f"Recommendation: {rec}")
                    self.file_logger.log_info(f"Knowledge consistency report: {consistency_report}")
            except Exception as e:
                self.file_logger.log_error(f"Error during unified knowledge integration in reflection: {e}")
        
        return messages

    ##Function purpose: Generate a meta-insight from existing insights
    def generate_meta_insight(self, username: str) -> List[str]:
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

        ##Block purpose: Parse meta-insight JSON using centralized utility
        data = extract_json(meta_insight_json_str, default=None)
        if data is None:
            messages.append(f"Failed to decode meta-insight from LLM response: {meta_insight_json_str}")
            return messages
        
        topic = data.get('topic', 'meta')
        insight = data.get('insight')
        if insight:
            self.insight_manager.save_insight(insight, username, topic=topic)
            messages.append(f"New meta-insight generated under topic '{topic}': {insight}")
        else:
            messages.append("No new meta-insight could be generated from the existing insights.")
        return messages

    ##Function purpose: Develop insights or assumptions from broad memory search
    def develop_insights_from_memory(self, username: str) -> List[str]:
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
        
        ##Block purpose: Parse insight/assumption JSON using centralized utility
        data = extract_json(insight_json_str, default=None)
        if data is None:
            messages.append(f"Failed to decode insight from LLM response: {insight_json_str}")
            return messages
        
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
        return messages

    ##Function purpose: Proactively verify an unverified assumption with the user
    def proactively_verify_assumption(self, username: str) -> Optional[str]:
        """Proactively verifies an unverified assumption with the user."""
        assumption, question = self.assumption_manager.get_assumption_to_verify(username)
        if assumption and question:
            self.pending_assumption = assumption
            return question
        return None

    ##Function purpose: Command to verify unverified assumptions with the user
    def verify_assumptions(self, username: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Command to verify unverified assumptions with the user."""
        question = self.proactively_verify_assumption(username)
        if question:
            return self.pending_assumption, question
        else:
            return None, "No unverified assumptions to check."

    ##Function purpose: Learn a new procedure interactively from user input
    def learn_procedure(self, procedure_data: Dict[str, Any], username: str) -> List[str]:
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