##Script function and purpose: Planning module for CognitiveEngine
##Dependency purpose: Encapsulates planning logic, strategies, and complexity assessment
"""
Planning Module

Encapsulates all logic related to query planning, complexity assessment,
and multi-level plan generation using LLMs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from jenova.exceptions import LLMError, LLMParseError
from jenova.llm.types import Prompt
from jenova.utils.json_safe import JSONSizeError, extract_json_from_response, safe_json_loads

if TYPE_CHECKING:
    from jenova.config.models import PersonaConfig
    from jenova.llm.interface import LLMInterface

logger = structlog.get_logger(__name__)


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


##Class purpose: Handles plan generation and complexity assessment
class Planner:
    """Handles plan generation and complexity assessment."""

    def __init__(
        self,
        llm: LLMInterface,
        planning_config: PlanningConfig,
        persona_config: PersonaConfig,
    ) -> None:
        """Initialize the Planner.

        Args:
            llm: Language model interface.
            planning_config: Planning configuration.
            persona_config: Persona configuration.
        """
        self.llm = llm
        self.config = planning_config
        self.persona = persona_config

    ##Method purpose: Generate execution plan based on query complexity
    def plan(self, user_input: str, context: list[str]) -> Plan:
        """Generate an execution plan based on query complexity.

        Args:
            user_input: The user's input query.
            context: Retrieved context items.

        Returns:
            Plan object (simple or structured multi-level).
        """
        ##Step purpose: Assess query complexity
        complexity = self._assess_complexity(user_input)

        ##Condition purpose: Use simple planning for simple/moderate or if disabled
        if complexity in (PlanComplexity.SIMPLE, PlanComplexity.MODERATE):
            return self._simple_plan(user_input, context, complexity)

        ##Condition purpose: Use simple planning if multi-level disabled
        if not self.config.multi_level_enabled:
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
        threshold = self.config.complexity_threshold

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
        prompt = Prompt(
            system=f"""You are {self.persona.name}. Create a brief step-by-step plan to respond to the user's query.
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
        max_sub_goals = self.config.max_sub_goals

        ##Step purpose: Build structured planning prompt
        prompt = Prompt(
            system=f"""You are {self.persona.name}. Generate a structured plan with:
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
                    except (json.JSONDecodeError, JSONSizeError) as inner_e:
                        raise LLMParseError(plan_json, f"Invalid JSON: {inner_e}") from inner_e
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
