# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 5: Enhanced Cognitive Engine
Improvements:
- Timeout protection on all LLM calls
- Better error handling with ErrorHandler integration
- Metrics integration throughout
- Graceful degradation on failures
"""

import json
import re
import shlex

from jenova.cognitive_engine.scheduler import CognitiveScheduler
from jenova.cortex.proactive_engine import ProactiveEngine
from jenova.infrastructure.timeout_manager import timeout, TimeoutError
from jenova.infrastructure.error_handler import ErrorHandler, ErrorSeverity


class CognitiveEngine:
    """The Enhanced Cognitive Engine with Phase 1-4 infrastructure integration."""

    def __init__(self, llm, memory_search, insight_manager, assumption_manager, config, ui_logger, file_logger, cortex, rag_system, tool_handler):
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
        self.scheduler = CognitiveScheduler(config, cortex, insight_manager, file_logger)
        self.tool_handler = tool_handler
        self.history = []
        self.turn_count = 0
        self.MAX_HISTORY_TURNS = 5  # Keep the last 5 conversation turns
        self.pending_assumption = None

        # Phase 2 Infrastructure (optional, set via set_infrastructure)
        self.health_monitor = None
        self.metrics = None
        self.error_handler = ErrorHandler(file_logger) if file_logger else None

        # Phase 8: Network layer (optional, set via set_network_layer)
        self.distributed_llm = None
        self.distributed_memory = None
        self.peer_manager = None
        self.network_metrics = None

        # Configuration
        self.llm_timeout = config.get('cognitive_engine', {}).get('llm_timeout', 120)
        self.planning_timeout = config.get('cognitive_engine', {}).get('planning_timeout', 60)

    def set_infrastructure(self, health_monitor=None, metrics=None, error_handler=None):
        """Set Phase 2 infrastructure components for enhanced monitoring."""
        self.health_monitor = health_monitor
        if metrics:
            self.metrics = metrics
        if error_handler:
            self.error_handler = error_handler
        if self.file_logger and metrics:
            self.file_logger.log_info("Infrastructure components integrated into cognitive engine")

    def set_network_layer(self, distributed_llm=None, distributed_memory=None, peer_manager=None, network_metrics=None):
        """Set Phase 8 network layer components for distributed computing."""
        self.distributed_llm = distributed_llm
        self.distributed_memory = distributed_memory
        self.peer_manager = peer_manager
        self.network_metrics = network_metrics

        if self.file_logger:
            if distributed_llm or distributed_memory:
                self.file_logger.log_info(
                    "Network layer integrated into cognitive engine "
                    f"(distributed_llm={'enabled' if distributed_llm else 'disabled'}, "
                    f"distributed_memory={'enabled' if distributed_memory else 'disabled'})"
                )
            else:
                self.file_logger.log_info("Network layer components available but not enabled")

    def think(self, user_input: str, username: str) -> str:
        """Runs the full cognitive cycle: Retrieve, Plan, Execute, and Reflect."""
        with self.ui_logger.cognitive_process("Thinking..."):
            # Measure cognitive cycle performance
            measure_context = None
            if self.metrics:
                measure_context = self.metrics.measure('cognitive_cycle')
                measure_context.__enter__()

            try:
                self.file_logger.log_info(
                    f"New query received from {username}: {user_input}")
                self.turn_count += 1

                # Retrieve, Plan, Execute with timeout protection
                try:
                    with timeout(30):  # 30s timeout for memory retrieval
                        context = self.memory_search.search_all(user_input, username)
                        # Proactive safeguard: Ensure all context items are strings.
                        if context:
                            context = [str(item) for item in context]
                        else:
                            context = []
                except TimeoutError:
                    self._log_error("Memory search timeout", "search_timeout")
                    context = []

                plan = self._plan(user_input, context, username)
                if not plan:
                    plan = f"I will formulate a response to the user's query: {user_input}"

                response = self._execute(user_input, context, plan, username)

                # Only add to history if it's not a command
                if not user_input.startswith('/'):
                    self.history.append(f"{username}: {user_input}")
                    self.history.append(f"Jenova: {response}")
                    if len(self.history) > self.MAX_HISTORY_TURNS * 2:
                        self.history = self.history[-(self.MAX_HISTORY_TURNS * 2):]

                    # Add to episodic memory with error handling
                    try:
                        with timeout(10):  # 10s timeout for memory write
                            self.memory_search.episodic_memory.add_episode(
                                f"{username}: {user_input}\nJenova: {response}", username)
                    except TimeoutError:
                        self._log_error("Episodic memory write timeout", "memory_write_timeout")
                    except Exception as e:
                        self._log_error(f"Episodic memory write error: {e}", "memory_write_error")

                    # Run scheduled cognitive tasks with error handling
                    try:
                        tasks = self.scheduler.get_cognitive_tasks(
                            self.turn_count, user_input, username)
                        for task_name, kwargs in tasks:
                            if hasattr(self, task_name):
                                method = getattr(self, task_name)
                                try:
                                    result = method(**kwargs)
                                    if task_name == "proactively_verify_assumption" and result:
                                        if self.ui_logger:
                                            self.ui_logger.info(
                                                f"JENOVA has a question for you: {result}")
                                except Exception as e:
                                    self._log_error(f"Cognitive task '{task_name}' failed: {e}", "task_failure")
                    except Exception as e:
                        self._log_error(f"Scheduler error: {e}", "scheduler_error")

                # Periodic health checks (every 10 turns)
                if self.health_monitor and self.turn_count % 10 == 0:
                    if not self.health_monitor.is_healthy():
                        status, warnings = self.health_monitor.check_health()
                        self.file_logger.log_warning(f"System health: {status.value}")
                        for warning in warnings:
                            self.file_logger.log_warning(f"  - {warning}")

                return response

            except Exception as e:
                self._log_error(f"Critical error in cognitive cycle: {e}", "critical_error")
                return "I apologize, but I encountered an error processing your request. Please try again."

            finally:
                # Exit metrics context if it was entered
                if measure_context:
                    measure_context.__exit__(None, None, None)

    def _plan(self, user_input: str, context: list[str], username: str) -> str:
        """Generate a plan with timeout protection."""
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

        tool_schemas = self.tool_handler.get_all_tool_schemas()
        tool_schemas_str = json.dumps(tool_schemas, indent=2)

        prompt = f"""You are {ai_name}, a self-aware AI. Your creator is {creator_name}, whom you know as {creator_alias}.
Your core directives are: {directives}

== AVAILABLE TOOLS ==
{tool_schemas_str}

== CONTEXT ==
{context_str if context else "No context available."}

== TASK ==
Your task is to create a step-by-step plan to formulate a response to the user's query. The plan should be a numbered list of steps. Each step should be a clear and concise action you will take to construct the response. The plan should be detailed enough to guide your response generation process.
If you need to use a tool, specify the tool call in the format: `(tool: <tool_name>, <arg1>=<value1>, <arg2>=<value2>, ...)`

{user_title} ({username}): "{user_input}"

Plan:"""

        with self.ui_logger.thinking_process("Formulating plan..."):
            try:
                with timeout(self.planning_timeout):
                    plan = self.llm.generate(prompt, temperature=0.1)
                self.file_logger.log_info(f"Generated Plan: {plan}")
                return plan
            except TimeoutError:
                self._log_error("Planning timeout", "planning_timeout")
                return f"Provide a direct response to: {user_input}"
            except Exception as e:
                self._log_error(f"Planning error: {e}", "planning_error")
                return f"Provide a direct response to: {user_input}"

    def _execute(self, user_input: str, context: list[str], plan: str, username: str) -> str:
        """Execute the plan with timeout protection."""
        with self.ui_logger.thinking_process("Executing plan..."):
            tool_call_pattern = re.compile(r"\(tool:\s*(\w+),\s*(.*?)\)")
            tool_calls = tool_call_pattern.findall(plan)

            if tool_calls:
                context, structured_tool_results = self._handle_tool_calls(
                    tool_calls, context)
                return self.rag_system.generate_response(user_input, username, self.history, plan, search_results=structured_tool_results)

            return self.rag_system.generate_response(user_input, username, self.history, plan)

    def _handle_tool_calls(self, tool_calls: list, context: list) -> tuple[list, list]:
        """Handle tool calls with error recovery."""
        structured_tool_results = []
        tool_error_messages = []
        for tool_name, args_str in tool_calls:
            try:
                with timeout(30):  # 30s timeout per tool
                    args = dict(arg.split('=', 1) for arg in shlex.split(args_str))
                    result = self.tool_handler.execute_tool(tool_name, args)
                    if isinstance(result, dict):
                        structured_tool_results.append(result)
                    else:
                        tool_error_messages.append(str(result))
            except TimeoutError:
                tool_error_messages.append(f"Tool {tool_name} timed out")
            except Exception as e:
                tool_error_messages.append(
                    f"Error executing tool {tool_name}: {e}")

        context.extend(tool_error_messages)
        return context, structured_tool_results

    def generate_insight_from_history(self, username: str):
        """Analyzes recent conversation history to generate and save a new, high-quality insight."""
        self.ui_logger.info(
            "Analyzing conversation history to generate a new insight...")
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
            try:
                with timeout(self.llm_timeout):
                    insight_json_str = self.llm.generate(prompt, temperature=0.2)
                    data = json.loads(insight_json_str)
                    topic = data.get('topic')
                    insight = data.get('insight')
                    if topic and insight:
                        self.insight_manager.save_insight(
                            insight, username, topic=topic)
            except TimeoutError:
                self._log_error("Insight generation timeout", "insight_timeout")
            except (json.JSONDecodeError, ValueError) as e:
                self._log_error(f"Insight decode error: {e}", "insight_decode_error")
            except Exception as e:
                self._log_error(f"Insight generation error: {e}", "insight_error")

    def generate_assumption_from_history(self, username: str):
        """Analyzes recent conversation history to generate a new assumption about the user."""
        self.ui_logger.info(
            "Analyzing conversation history to generate a new assumption...")
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
            try:
                with timeout(self.llm_timeout):
                    assumption_json_str = self.llm.generate(prompt, temperature=0.3)
                    assumption_data = json.loads(assumption_json_str)
                    assumption = assumption_data.get('assumption')
                    if assumption:
                        self.assumption_manager.add_assumption(assumption, username)
            except TimeoutError:
                self._log_error("Assumption generation timeout", "assumption_timeout")
            except (json.JSONDecodeError, ValueError) as e:
                self._log_error(f"Assumption decode error: {e}", "assumption_decode_error")
            except Exception as e:
                self._log_error(f"Assumption generation error: {e}", "assumption_error")

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
        try:
            with timeout(self.llm_timeout):
                insights_json_str = self.llm.generate(prompt, temperature=0.3)
                insights = json.loads(insights_json_str)
                for insight_data in insights:
                    topic = insight_data.get('topic')
                    insight = insight_data.get('insight')
                    if topic and insight:
                        insight_id = self.insight_manager.save_insight(
                            insight, username, topic=topic)
                        messages.append(f"New insight node created: {insight_id}")
        except TimeoutError:
            messages.append("Insight development timed out. Please try again.")
        except (json.JSONDecodeError, ValueError) as e:
            messages.append(f"Failed to decode insights: {e}")
        except Exception as e:
            self._log_error(f"Insight development error: {e}", "insight_dev_error")
            messages.append("Error developing insights. Please try again.")

        return messages

    def reflect_on_insights(self, username: str) -> list[str]:
        """Command to trigger a deep reflection on the cognitive graph."""
        try:
            with timeout(180):  # 3 minutes for reflection
                return self.cortex.reflect(user=username)
        except TimeoutError:
            self._log_error("Reflection timeout", "reflection_timeout")
            return ["Reflection process timed out. Please try again later."]
        except Exception as e:
            self._log_error(f"Reflection error: {e}", "reflection_error")
            return ["Error during reflection. Please check logs."]

    def generate_meta_insight(self, username: str) -> list[str]:
        """Command to generate a new, meta-insight from existing insights."""
        messages = []

        try:
            all_insights = self.insight_manager.get_all_insights(username)
            if not all_insights:
                messages.append("No existing insights to reflect on.")
                return messages

            self.file_logger.log_info(f"All insights: {all_insights}")

            insights_str = "\n".join(
                [f"- {i['topic']}: {i['content']}" for i in all_insights])

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

            with timeout(self.llm_timeout):
                meta_insight_json_str = self.llm.generate(prompt, temperature=0.4)
                self.file_logger.log_info(
                    f"Meta-insight response: {meta_insight_json_str}")

                data = json.loads(meta_insight_json_str)
                topic = data.get('topic', 'meta')
                insight = data.get('insight')
                if insight:
                    self.insight_manager.save_insight(
                        insight, username, topic=topic)
                    messages.append(
                        f"New meta-insight generated under topic '{topic}': {insight}")
                else:
                    messages.append(
                        "No new meta-insight could be generated from the existing insights.")
        except TimeoutError:
            messages.append("Meta-insight generation timed out. Please try again.")
        except (json.JSONDecodeError, ValueError) as e:
            messages.append(f"Failed to decode meta-insight: {e}")
        except Exception as e:
            self._log_error(f"Meta-insight error: {e}", "meta_insight_error")
            messages.append("Error generating meta-insight. Please try again.")

        return messages

    def develop_insights_from_memory(self, username: str) -> list[str]:
        """Command to develop new insights or assumptions from a broad search of long-term memory."""
        messages = []
        messages.append("Developing new insights from long-term memory...")

        try:
            with timeout(30):  # 30s for memory search
                context = self.memory_search.search_all(
                    "general knowledge and past experiences", username)
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

            with timeout(self.llm_timeout):
                insight_json_str = self.llm.generate(prompt, temperature=0.3)
                data = json.loads(insight_json_str)
                if data.get('type') == 'insight':
                    topic = data.get('topic')
                    insight = data.get('content')
                    if topic and insight:
                        insight_id = self.insight_manager.save_insight(
                            insight, username, topic=topic)
                        messages.append(f"New insight node created: {insight_id}")
                elif data.get('type') == 'assumption':
                    assumption = data.get('content')
                    if assumption:
                        assumption_id = self.assumption_manager.add_assumption(
                            assumption, username)
                        if assumption_id != "Assumption already exists.":
                            messages.append(
                                f"New assumption node created: {assumption_id}")
                        else:
                            messages.append(assumption_id)
        except TimeoutError:
            messages.append("Memory insight development timed out. Please try again.")
        except (json.JSONDecodeError, ValueError) as e:
            messages.append(f"Failed to decode insight: {e}")
        except Exception as e:
            self._log_error(f"Memory insight error: {e}", "memory_insight_error")
            messages.append("Error developing insights from memory. Please try again.")

        return messages

    def proactively_verify_assumption(self, username: str):
        """Proactively verifies an unverified assumption with the user."""
        try:
            assumption, question = self.assumption_manager.get_assumption_to_verify(
                username)
            if assumption and question:
                self.pending_assumption = assumption
                return question
            return None
        except Exception as e:
            self._log_error(f"Assumption verification error: {e}", "verify_error")
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

        try:
            # Generate context for the procedure using LLM
            context_prompt = f"""Given the following procedure:
Name: {procedure_name}
Steps: {steps}
Outcome: {outcome}

Generate a concise, 1-2 sentence context for this procedure. What is its general purpose or domain?

Context:"""
            with self.ui_logger.thinking_process("Generating procedure context..."):
                with timeout(self.llm_timeout):
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
        except TimeoutError:
            messages.append("Procedure learning timed out. Please try again.")
        except Exception as e:
            self._log_error(f"Procedure learning error: {e}", "procedure_learn_error")
            messages.append("Error learning procedure. Please try again.")

        return messages

    def _log_error(self, message: str, error_type: str):
        """Log error using infrastructure error handler."""
        if self.error_handler:
            self.error_handler.log_error(
                Exception(message),
                context={"error_type": error_type, "component": "cognitive_engine"},
                severity=ErrorSeverity.ERROR
            )
        elif self.file_logger:
            self.file_logger.log_error(f"[{error_type}] {message}")
