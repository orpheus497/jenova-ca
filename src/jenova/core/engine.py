##Script function and purpose: CognitiveEngine - Central orchestrator for JENOVA's cognitive cycle.
##Dependency purpose: Coordinates KnowledgeStore, LLM, and ResponseGenerator to process user input and generate responses.
"""CognitiveEngine orchestrates the cognitive cycle for JENOVA.

This module provides the central engine that:
- Receives user input
- Retrieves relevant context from KnowledgeStore
- Generates responses via LLM
- Updates memory with new interactions
- Implements multi-level planning for complex queries
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import structlog

##Sec: Use JenovaMemoryError to avoid shadowing Python builtin MemoryError (P0-001)
from jenova.exceptions import (
    JenovaMemoryError,
    LLMError,
    LLMParseError,
    ProactiveError,
)
from jenova.llm.types import Prompt
from jenova.memory.types import MemoryType
from jenova.utils.json_safe import JSONSizeError, extract_json_from_response, safe_json_loads
from jenova.utils.sanitization import sanitize_user_query

##Sec: Import username validation for security (PATCH-001)
from jenova.utils.validation import validate_username

if TYPE_CHECKING:
    from jenova.assumptions.manager import AssumptionManager
    from jenova.config.models import JenovaConfig
    from jenova.core.integration import IntegrationHub
    from jenova.core.knowledge import KnowledgeStore
    from jenova.core.response import ResponseGenerator
    from jenova.core.scheduler import CognitiveScheduler
    from jenova.graph.proactive import ProactiveEngine, Suggestion
    from jenova.insights.manager import InsightManager
    from jenova.llm.interface import LLMInterface
else:
    ##Step purpose: Import managers for runtime use
    from jenova.assumptions.manager import AssumptionManager
    from jenova.graph.proactive import ProactiveEngine, Suggestion
    from jenova.insights.manager import InsightManager

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)

##Sec: Enhanced global regex-based redaction for PII protection (PATCH-004)
##Note: Matches emails while avoiding common URL contexts and trailing punctuation
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", flags=re.IGNORECASE
)
##Note: Global phone pattern capturing optional international prefixes and groupings
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4,6}"
)


##Class purpose: Enum defining planning complexity levels
class PlanComplexity(Enum):
    """Complexity level for query planning."""

    SIMPLE = "simple"
    """Simple query requiring basic single-step response."""

    MODERATE = "moderate"
    """Moderate query requiring some planning."""

    COMPLEX = "complex"
    """Complex query requiring multi-level planning."""

    VERY_COMPLEX = "very_complex"
    """Very complex query requiring extensive planning."""


##Class purpose: A single step in an execution plan
@dataclass(frozen=True)
class PlanStep:
    """A single step in an execution plan.

    Attributes:
        index: Step number (1-based).
        description: Description of what this step does.
        reasoning: Why this step is needed.
        status: Current status (pending, in_progress, completed).
    """

    index: int
    description: str
    reasoning: str = ""
    status: str = "pending"


##Class purpose: A structured multi-level plan for query response
@dataclass
class Plan:
    """A structured multi-level plan for query response.

    Attributes:
        main_goal: The overall objective for responding.
        sub_goals: List of sub-goals/steps to achieve main goal.
        reasoning_chain: Logical progression of reasoning.
        complexity: Complexity level of the plan.
        raw_text: Raw text representation for simple plans.
    """

    main_goal: str
    sub_goals: list[PlanStep] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)
    complexity: PlanComplexity = PlanComplexity.SIMPLE
    raw_text: str = ""

    ##Method purpose: Check if this is a structured multi-level plan
    def is_structured(self) -> bool:
        """Check if this is a structured multi-level plan."""
        return len(self.sub_goals) > 0

    ##Method purpose: Get plan as formatted text
    def as_text(self) -> str:
        """Get plan as formatted text for LLM prompt."""
        ##Condition purpose: Return raw text for simple plans
        if not self.is_structured():
            return self.raw_text or self.main_goal

        ##Step purpose: Build structured text
        parts = [f"Main Goal: {self.main_goal}"]

        ##Condition purpose: Add sub-goals if present
        if self.sub_goals:
            parts.append("\nSub-goals:")
            for step in self.sub_goals:
                parts.append(f"  {step.index}. {step.description}")

        ##Condition purpose: Add reasoning chain if present
        if self.reasoning_chain:
            parts.append("\nReasoning Chain:")
            for i, reasoning in enumerate(self.reasoning_chain, 1):
                parts.append(f"  {i}. {reasoning}")

        return "\n".join(parts)

    ##Method purpose: Create simple plan from text
    @classmethod
    def simple(cls, text: str) -> Plan:
        """Create a simple single-level plan from text."""
        return cls(
            main_goal=text,
            raw_text=text,
            complexity=PlanComplexity.SIMPLE,
        )

    ##Method purpose: Create structured plan from parsed data
    @classmethod
    def from_dict(cls, data: dict[str, object], complexity: PlanComplexity) -> Plan:
        """Create a structured plan from parsed dict data."""
        main_goal = str(data.get("main_goal", ""))

        ##Step purpose: Parse sub-goals into PlanStep objects
        raw_sub_goals = data.get("sub_goals", [])
        sub_goals: list[PlanStep] = []
        if isinstance(raw_sub_goals, list):
            for i, goal in enumerate(raw_sub_goals, 1):
                sub_goals.append(
                    PlanStep(
                        index=i,
                        description=str(goal),
                        status="pending",
                    )
                )

        ##Step purpose: Parse reasoning chain
        raw_reasoning = data.get("reasoning_chain", [])
        reasoning_chain: list[str] = []
        if isinstance(raw_reasoning, list):
            reasoning_chain = [str(r) for r in raw_reasoning]

        return cls(
            main_goal=main_goal,
            sub_goals=sub_goals,
            reasoning_chain=reasoning_chain,
            complexity=complexity,
        )


##Class purpose: Configuration for multi-level planning
@dataclass
class PlanningConfig:
    """Configuration for multi-level planning.

    Attributes:
        multi_level_enabled: Whether multi-level planning is enabled.
        max_sub_goals: Maximum number of sub-goals in a plan.
        complexity_threshold: Word count threshold for complex queries.
        plan_temperature: LLM temperature for plan generation.
    """

    multi_level_enabled: bool = True
    max_sub_goals: int = 5
    complexity_threshold: int = 20
    plan_temperature: float = 0.3


##Class purpose: Configuration for CognitiveEngine behavior
@dataclass
class EngineConfig:
    """Configuration for CognitiveEngine.

    Attributes:
        max_context_items: Maximum number of context items to retrieve.
        temperature: LLM temperature for response generation.
        enable_learning: Whether to store interactions in memory.
        max_history_turns: Maximum conversation history turns to maintain.
        planning: Planning configuration.
    """

    max_context_items: int = 10
    temperature: float = 0.7
    enable_learning: bool = True
    max_history_turns: int = 10
    planning: PlanningConfig = field(default_factory=PlanningConfig)


##Class purpose: Result of a cognitive cycle (think operation)
@dataclass
class ThinkResult:
    """Result from CognitiveEngine.think() operation.

    Attributes:
        content: The generated response content.
        context_used: Number of context items used.
        is_error: Whether an error occurred.
        error_message: Error message if is_error is True.
        plan_complexity: Complexity level of the plan used.
        plan_steps: Number of plan steps (for structured plans).
        suggestion: Optional proactive suggestion for the user.
    """

    content: str
    context_used: int = 0
    is_error: bool = False
    error_message: str | None = None
    plan_complexity: PlanComplexity = PlanComplexity.SIMPLE
    plan_steps: int = 0
    suggestion: Suggestion | None = None


##Class purpose: Central orchestrator for JENOVA's cognitive processing
class CognitiveEngine:
    """Orchestrates the cognitive cycle for JENOVA.

    The CognitiveEngine coordinates:
    - Context retrieval from KnowledgeStore
    - Response generation via LLM
    - Memory updates for learning
    - Multi-level planning for complex queries

    Attributes:
        config: Engine configuration.
        knowledge_store: Knowledge retrieval component.
        llm: Language model interface.
        response_generator: Response formatting component.
        integration_hub: Optional integration hub for unified knowledge.
    """

    ##Method purpose: Initialize engine with required components
    def __init__(
        self,
        config: JenovaConfig,
        knowledge_store: KnowledgeStore,
        llm: LLMInterface,
        response_generator: ResponseGenerator,
        engine_config: EngineConfig | None = None,
        integration_hub: IntegrationHub | None = None,
        insight_manager: InsightManager | None = None,
        assumption_manager: AssumptionManager | None = None,
        scheduler: CognitiveScheduler | None = None,
        proactive_engine: ProactiveEngine | None = None,
    ) -> None:
        """Initialize the CognitiveEngine.

        Args:
            config: JENOVA configuration.
            knowledge_store: Knowledge retrieval component.
            llm: Language model interface.
            response_generator: Response formatting component.
            engine_config: Optional engine-specific configuration.
            integration_hub: Optional integration hub for unified knowledge.
            insight_manager: Optional insight manager for insight generation.
            assumption_manager: Optional assumption manager for assumption verification.
            scheduler: Optional cognitive scheduler for turn-based background tasks.
            proactive_engine: Optional proactive engine for autonomous suggestions.
        """
        ##Step purpose: Store configuration and dependencies
        self.config = config
        self.knowledge_store = knowledge_store
        self.llm = llm
        self.response_generator = response_generator
        self.engine_config = engine_config or EngineConfig()
        self.integration_hub = integration_hub
        self._insight_manager = insight_manager
        self._assumption_manager = assumption_manager
        ##Update: WIRING-001 (2026-02-13T11:26:36Z) — Scheduler for turn-based cognitive tasks
        self._scheduler: CognitiveScheduler | None = scheduler
        ##Update: WIRING-003 (2026-02-14) — Proactive engine for suggestions
        self._proactive_engine: ProactiveEngine | None = proactive_engine

        ##Step purpose: Initialize conversation state
        self._history: list[tuple[str, str]] = []
        self._turn_count: int = 0
        self._current_username: str = "default"

        ##Action purpose: Log engine initialization
        logger.info(
            "cognitive_engine_initialized",
            max_context_items=self.engine_config.max_context_items,
            enable_learning=self.engine_config.enable_learning,
            multi_level_planning=self.engine_config.planning.multi_level_enabled,
            proactive_enabled=self._proactive_engine is not None,
        )

    ##Method purpose: Process user input and generate a response
    def think(self, user_input: str, username: str | None = None) -> ThinkResult:
        """Process user input through the cognitive cycle.

        Args:
            user_input: The user's input query or message.
            username: Optional username for personalization.

        Returns:
            ThinkResult containing the response and metadata.

        Raises:
            LLMError: If the language model fails to generate.
            JenovaMemoryError: If memory operations fail.
        """
        ##Step purpose: Increment turn counter and set username
        start_time = time.perf_counter()
        self._turn_count += 1
        if username:
            ##Sec: Validate username before database operations (PATCH-001)
            self._current_username = validate_username(username)
        logger.info("think_started", query=user_input[:100], turn=self._turn_count)

        ##Error purpose: Catch and handle errors during cognitive cycle
        try:
            ##Step purpose: Retrieve relevant context from knowledge store
            context = self._retrieve_context(user_input)
            context_count = len(context)

            ##Step purpose: Expand context with integration hub if available
            if self.integration_hub and self._current_username:
                context = self.integration_hub.expand_context_with_relationships(
                    context, self._current_username
                )
                context_count = len(context)

            ##Step purpose: Generate plan based on query complexity
            plan = self._plan(user_input, context)

            ##Step purpose: Build prompt with context, history, and plan
            prompt = self._build_prompt(user_input, context, plan)

            ##Action purpose: Generate response via LLM
            logger.debug(
                "generating_response",
                context_items=context_count,
                plan_complexity=plan.complexity.value,
            )
            completion = self.llm.generate(prompt)

            ##Step purpose: Format response through ResponseGenerator
            response = self.response_generator.generate(
                llm_output=completion.content,
                context=context,
                query=user_input,
            )

            ##Condition purpose: Store interaction if learning is enabled
            if self.engine_config.enable_learning:
                self._store_interaction(user_input, response.content)

                ##Condition purpose: Provide Memory → Cortex feedback if integration enabled
                if self.integration_hub and self._current_username:
                    episode_content = f"User: {user_input}\nJENOVA: {response.content}"
                    self.integration_hub.propagate_memory_to_cortex(
                        episode_content, "episodic", self._current_username
                    )

            ##Step purpose: Check for proactive suggestion
            suggestion = None
            if self._proactive_engine:
                try:
                    suggestion = self._proactive_engine.get_suggestion(self._current_username)
                except ProactiveError as e:
                    ##Fix: Narrow exception handling to ProactiveError
                    logger.warning("proactive_suggestion_failed", error=str(e))
                except Exception as e:
                    ##Fix: Log unexpected proactive errors without re-raising to preserve response (PATCH-005)
                    logger.exception(
                        "unexpected_error_in_proactive_suggestions",
                        error=str(e),
                        username=self._current_username,
                    )

            ##Action purpose: Log successful completion
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "think_completed",
                turn=self._turn_count,
                context_used=context_count,
                response_length=len(response.content),
                plan_complexity=plan.complexity.value,
                plan_steps=len(plan.sub_goals),
                duration_ms=round(duration_ms, 2),
                has_suggestion=suggestion is not None,
            )

            ##Update: WIRING-001 (2026-02-13T11:26:36Z) — Fire scheduler after successful turn
            if self._scheduler:
                try:
                    unverified = (
                        self._assumption_manager.unverified_count(self._current_username)
                        if self._assumption_manager
                        else 0
                    )
                    self._scheduler.on_turn_complete(self._current_username, unverified)
                except (RuntimeError, ValueError, OSError) as e:
                    ##Note: Scheduler errors must not break the think() response
                    logger.warning("scheduler_post_turn_error", error=str(e), exc_info=True)

            return ThinkResult(
                content=response.content,
                context_used=context_count,
                is_error=False,
                plan_complexity=plan.complexity,
                plan_steps=len(plan.sub_goals),
                suggestion=suggestion,
            )

        except LLMError as e:
            ##Step purpose: Handle LLM-specific errors
            logger.error("llm_error_in_think", error=str(e), query=user_input[:100])
            return ThinkResult(
                content="I apologize, but I encountered an issue generating a response.",
                is_error=True,
                error_message=str(e),
            )
        except JenovaMemoryError as e:
            ##Step purpose: Handle memory-specific errors
            logger.error("memory_error_in_think", error=str(e), query=user_input[:100])
            return ThinkResult(
                content="I encountered a memory access issue. Please try again.",
                is_error=True,
                error_message=str(e),
            )

    ##Method purpose: Retrieve relevant context for a query
    def _retrieve_context(self, query: str) -> list[str]:
        """Retrieve relevant context from knowledge store.

        Args:
            query: The query to find context for.

        Returns:
            List of relevant context strings.
        """
        ##Action purpose: Search knowledge store for relevant context
        knowledge_context = self.knowledge_store.search(
            query=query,
            n_results=self.engine_config.max_context_items,
            include_graph=True,
        )

        ##Step purpose: Build context list from memories
        context: list[str] = []

        ##Loop purpose: Add memory results to context
        for memory in knowledge_context.memories:
            context.append(f"[{memory.memory_type.value}] {memory.content}")

        ##Loop purpose: Add graph context to context
        for node in knowledge_context.graph_context:
            label = node.get("label", "Unknown")
            content = node.get("content", "")
            context.append(f"[knowledge] {label}: {content}")

        ##Step purpose: Add recent conversation history
        history_context = self._get_history_context()
        if history_context:
            context.append(f"[history] Recent conversation:\n{history_context}")

        return context

    ##Method purpose: Get formatted conversation history
    def _get_history_context(self) -> str:
        """Get recent conversation history as formatted string."""
        ##Condition purpose: Return empty if no history
        if not self._history:
            return ""

        ##Step purpose: Format recent turns
        recent = self._history[-self.engine_config.max_history_turns :]
        lines: list[str] = []
        ##Loop purpose: Format each turn
        for user_msg, ai_response in recent:
            lines.append(f"User: {user_msg}")
            lines.append(f"JENOVA: {ai_response[:200]}...")

        return "\n".join(lines)

    ##Method purpose: Build LLM prompt from query, context, and plan
    def _build_prompt(
        self,
        query: str,
        context: list[str],
        plan: Plan | None = None,
    ) -> Prompt:
        """Build a structured prompt for the LLM.

        Args:
            query: User's input query.
            context: List of context strings.
            plan: Optional execution plan.

        Returns:
            Structured Prompt for LLM.
        """
        ##Step purpose: Format context section
        context_section = ""
        if context:
            context_str = "\n".join(f"- {c}" for c in context)
            context_section = f"\n\n== CONTEXT ==\n{context_str}"

        ##Step purpose: Format plan section if present
        plan_section = ""
        if plan and plan.as_text():
            plan_section = f"\n\n== PLAN ==\n{plan.as_text()}"

        ##Step purpose: Build system prompt with persona
        persona = self.config.persona
        system_prompt = f"""You are {persona.name}, a self-aware AI cognitive assistant.

Your directives: {", ".join(persona.directives) if persona.directives else "Be helpful and informative."}

{persona.system_prompt}{context_section}{plan_section}

Respond thoughtfully to the user's message, using the provided context and following the plan when relevant."""

        ##Step purpose: Sanitize user query to prevent prompt injection
        safe_query = sanitize_user_query(query)

        return Prompt(
            system=system_prompt,
            user_message=safe_query,
        )

    ##Method purpose: Store interaction in memory for learning
    def _store_interaction(self, user_input: str, response: str) -> None:
        """Store the interaction in memory.

        Args:
            user_input: The user's input.
            response: The generated response.
        """
        ##Step purpose: Add to conversation history
        self._history.append((user_input, response))

        ##Condition purpose: Trim history if too long
        if len(self._history) > self.engine_config.max_history_turns:
            self._history = self._history[-self.engine_config.max_history_turns :]

        ##Step purpose: Store in episodic memory
        episode_content = f"User: {user_input}\nJENOVA: {response}"

        ##Error purpose: Log but don't fail on memory storage errors
        try:
            self.knowledge_store.add(
                content=episode_content,
                memory_type=MemoryType.EPISODIC,
                metadata={"turn": str(self._turn_count)},
            )
            logger.debug("interaction_stored", turn=self._turn_count)
        except JenovaMemoryError as e:
            logger.warning("failed_to_store_interaction", error=str(e))

    ##Method purpose: Generate execution plan based on query complexity
    def _plan(self, user_input: str, context: list[str]) -> Plan:
        """Generate an execution plan based on query complexity.

        Args:
            user_input: The user's input query.
            context: Retrieved context items.

        Returns:
            Plan object (simple or structured multi-level).
        """
        ##Step purpose: Assess query complexity
        complexity = self._assess_complexity(user_input)

        ##Step purpose: Determine planning approach
        planning_config = self.engine_config.planning

        ##Condition purpose: Use simple planning for simple/moderate or if disabled
        if complexity in (PlanComplexity.SIMPLE, PlanComplexity.MODERATE):
            return self._simple_plan(user_input, context, complexity)

        ##Condition purpose: Use simple planning if multi-level disabled
        if not planning_config.multi_level_enabled:
            return self._simple_plan(user_input, context, complexity)

        ##Step purpose: Use complex multi-level planning
        return self._complex_plan(user_input, context, complexity)

    ##Method purpose: Assess complexity of a query
    def _assess_complexity(self, user_input: str) -> PlanComplexity:
        """Assess the complexity level of a query.

        Args:
            user_input: The user's input query.

        Returns:
            PlanComplexity indicating query complexity.
        """
        ##Step purpose: Get word count and basic metrics
        words = user_input.split()
        word_count = len(words)
        threshold = self.engine_config.planning.complexity_threshold

        ##Step purpose: Check for complexity indicators
        complexity_indicators = [
            "explain",
            "compare",
            "analyze",
            "describe in detail",
            "step by step",
            "how does",
            "why does",
            "what are the",
            "difference between",
            "relationship between",
        ]

        has_indicators = any(indicator in user_input.lower() for indicator in complexity_indicators)

        ##Step purpose: Check for multiple questions
        question_count = user_input.count("?")

        ##Step purpose: Determine complexity level
        ##Condition purpose: Very complex - long with indicators and multiple questions
        if word_count > threshold * 2 and has_indicators and question_count > 1:
            return PlanComplexity.VERY_COMPLEX

        ##Condition purpose: Complex - long with indicators
        if word_count > threshold and has_indicators:
            return PlanComplexity.COMPLEX

        ##Condition purpose: Moderate - either long or has indicators
        if word_count > threshold or has_indicators:
            return PlanComplexity.MODERATE

        return PlanComplexity.SIMPLE

    ##Method purpose: Generate simple single-level plan
    def _simple_plan(
        self,
        user_input: str,
        context: list[str],
        complexity: PlanComplexity,
    ) -> Plan:
        """Generate a simple single-level plan.

        Args:
            user_input: The user's input query.
            context: Retrieved context items.
            complexity: Assessed complexity level.

        Returns:
            Simple Plan object.
        """
        ##Step purpose: Format context for prompt
        context_str = "\n".join(f"- {c}" for c in context) if context else "No context available."

        ##Step purpose: Build planning prompt
        persona = self.config.persona
        prompt = Prompt(
            system=f"""You are {persona.name}. Create a brief step-by-step plan to respond to the user's query.
The plan should be a short paragraph describing the approach.

== CONTEXT ==
{context_str}""",
            user_message=f'Query: "{user_input}"\n\nPlan:',
        )

        ##Error purpose: Handle LLM errors gracefully
        try:
            completion = self.llm.generate(prompt)
            plan_text = completion.content.strip()

            logger.debug("simple_plan_generated", plan_length=len(plan_text))

            return Plan.simple(plan_text)

        except LLMError as e:
            ##Step purpose: Fallback to basic plan on error
            logger.warning("simple_plan_generation_failed", error=str(e))
            return Plan.simple(f"Respond to the user's query: {user_input}")

    ##Method purpose: Generate complex multi-level plan with sub-goals
    def _complex_plan(
        self,
        user_input: str,
        context: list[str],
        complexity: PlanComplexity,
    ) -> Plan:
        """Generate a structured multi-level plan with sub-goals.

        Args:
            user_input: The user's input query.
            context: Retrieved context items.
            complexity: Assessed complexity level.

        Returns:
            Structured Plan object with sub-goals and reasoning.
        """
        ##Step purpose: Format context for prompt
        context_str = "\n".join(f"- {c}" for c in context) if context else "No context available."
        max_sub_goals = self.engine_config.planning.max_sub_goals

        ##Step purpose: Build structured planning prompt
        persona = self.config.persona
        prompt = Prompt(
            system=f"""You are {persona.name}. Generate a structured plan with:
1. Main goal: The overall objective
2. Sub-goals: 3-{max_sub_goals} specific steps
3. Reasoning chain: Logical progression

Respond with a valid JSON object:
{{
    "main_goal": "<overall objective>",
    "sub_goals": ["<step 1>", "<step 2>", ...],
    "reasoning_chain": ["<reasoning 1>", "<reasoning 2>", ...]
}}

== CONTEXT ==
{context_str}""",
            user_message=f'Query: "{user_input}"\n\nJSON Plan:',
        )

        ##Error purpose: Handle LLM and parsing errors
        try:
            completion = self.llm.generate(prompt)
            plan_json = completion.content.strip()

            ##Step purpose: Parse JSON response with size limits
            try:
                ##Step purpose: Extract JSON from response if needed
                try:
                    json_str = extract_json_from_response(plan_json)
                except ValueError:
                    json_str = plan_json

                ##Action purpose: Parse with size limits
                data = safe_json_loads(json_str)
            except (json.JSONDecodeError, JSONSizeError) as e:
                ##Step purpose: Try to extract JSON from response as fallback
                json_match = re.search(r"\{[^{}]*\}", plan_json, re.DOTALL)
                if json_match:
                    try:
                        data = safe_json_loads(json_match.group())
                    except (json.JSONDecodeError, JSONSizeError) as e:
                        raise LLMParseError(plan_json, f"Invalid JSON: {e}") from e
                else:
                    raise LLMParseError(plan_json, f"Invalid JSON: {e}") from e

            ##Step purpose: Create structured plan from parsed data
            plan = Plan.from_dict(data, complexity)

            logger.debug(
                "complex_plan_generated",
                main_goal=plan.main_goal[:50],
                sub_goals_count=len(plan.sub_goals),
                reasoning_count=len(plan.reasoning_chain),
            )

            return plan

        except (LLMError, LLMParseError) as e:
            ##Step purpose: Fallback to simple plan on error
            logger.warning("complex_plan_generation_failed", error=str(e))
            return self._simple_plan(user_input, context, complexity)

    ##Method purpose: Set integration hub for unified knowledge
    def set_integration_hub(self, hub: IntegrationHub) -> None:
        """Set the integration hub for unified knowledge operations.

        Args:
            hub: IntegrationHub instance.
        """
        self.integration_hub = hub
        logger.info("integration_hub_set")

    ##Update: WIRING-001 (2026-02-13T11:26:36Z) — Scheduler setter mirroring set_integration_hub
    ##Method purpose: Set cognitive scheduler for turn-based background tasks
    def set_scheduler(self, scheduler: CognitiveScheduler) -> None:
        """Set the cognitive scheduler for turn-based background tasks.

        Args:
            scheduler: CognitiveScheduler instance.
        """
        self._scheduler = scheduler
        logger.info("scheduler_set")

    ##Method purpose: Set the proactive engine for autonomous suggestions
    def set_proactive_engine(self, engine: ProactiveEngine) -> None:
        """Set the proactive engine for autonomous suggestions.

        Args:
            engine: ProactiveEngine instance.
        """
        self._proactive_engine = engine
        logger.info("proactive_engine_set")

    ##Update: WIRING-001 (2026-02-13T11:26:36Z) — Expose history for task executor
    ##Method purpose: Get recent conversation history for external consumers
    def get_recent_history(self, redact: bool = True) -> list[tuple[str, str]]:
        """Get recent conversation history.

        Args:
            redact: Whether to redact potential PII (default: True).

        Returns:
            List of (user_message, ai_response) tuples.
        """
        history = list(self._history)
        if redact:
            return [self._redact_pii(item) for item in history]
        return history

    ##Method purpose: Redact potential PII from a history item
    def _redact_pii(self, item: tuple[str, str]) -> tuple[str, str]:
        """Redact potential PII (Email, Phone) from a history item.

        Args:
            item: (user_message, ai_response) tuple.

        Returns:
            Sanitized tuple.
        """
        user_msg, ai_msg = item

        ##Refactor: Use pre-compiled regex for performance (D3-2026-02-14)
        def redact(text: str) -> str:
            ##Step purpose: Apply redaction patterns
            text = _EMAIL_RE.sub("[EMAIL]", text)
            text = _PHONE_RE.sub("[PHONE]", text)
            return text

        return (redact(user_msg), redact(ai_msg))

    ##Method purpose: Reset engine state for new conversation
    def reset(self) -> None:
        """Reset the engine state for a new conversation."""
        ##Step purpose: Clear conversation state
        self._history = []
        self._turn_count = 0

        ##Action purpose: Log reset
        logger.info("cognitive_engine_reset")

    ##Method purpose: Get insight manager instance
    @property
    def insight_manager(self) -> InsightManager | None:
        """Get the insight manager instance.

        Returns:
            InsightManager instance if available, None otherwise.
        """
        return self._insight_manager

    ##Method purpose: Get assumption manager instance
    @property
    def assumption_manager(self) -> AssumptionManager | None:
        """Get the assumption manager instance.

        Returns:
            AssumptionManager instance if available, None otherwise.
        """
        return self._assumption_manager
