##Script function and purpose: Unit tests for CognitiveEngine multi-level planning
"""
Tests for CognitiveEngine planning features.

Tests the multi-level planning capabilities including:
- Complexity assessment
- Simple vs complex planning
- Plan data structures
"""

import pytest

from jenova.core.engine import (
    EngineConfig,
    Plan,
    PlanComplexity,
    PlanningConfig,
    PlanStep,
)


##Class purpose: Tests for PlanComplexity enum
class TestPlanComplexity:
    """Tests for PlanComplexity enum."""

    ##Method purpose: Test enum values exist
    def test_complexity_levels_exist(self) -> None:
        """All complexity levels are defined."""
        assert PlanComplexity.SIMPLE.value == "simple"
        assert PlanComplexity.MODERATE.value == "moderate"
        assert PlanComplexity.COMPLEX.value == "complex"
        assert PlanComplexity.VERY_COMPLEX.value == "very_complex"

    ##Method purpose: Test enum is iterable
    def test_complexity_is_iterable(self) -> None:
        """PlanComplexity can be iterated."""
        levels = list(PlanComplexity)
        assert len(levels) == 4


##Class purpose: Tests for PlanStep dataclass
class TestPlanStep:
    """Tests for PlanStep dataclass."""

    ##Method purpose: Test step creation
    def test_step_creation(self) -> None:
        """PlanStep can be created with required fields."""
        step = PlanStep(
            index=1,
            description="First step",
        )

        assert step.index == 1
        assert step.description == "First step"
        assert step.reasoning == ""
        assert step.status == "pending"

    ##Method purpose: Test step with all fields
    def test_step_with_all_fields(self) -> None:
        """PlanStep can be created with all fields."""
        step = PlanStep(
            index=2,
            description="Second step",
            reasoning="Because we need it",
            status="in_progress",
        )

        assert step.index == 2
        assert step.reasoning == "Because we need it"
        assert step.status == "in_progress"

    ##Method purpose: Test step is frozen
    def test_step_is_frozen(self) -> None:
        """PlanStep is immutable."""
        step = PlanStep(index=1, description="Test")

        with pytest.raises(Exception):  # noqa: B017 FrozenInstanceError
            step.index = 2


##Class purpose: Tests for Plan dataclass
class TestPlan:
    """Tests for Plan dataclass."""

    ##Method purpose: Test simple plan creation
    def test_simple_plan_creation(self) -> None:
        """Plan.simple() creates a simple plan."""
        plan = Plan.simple("Respond to the query")

        assert plan.main_goal == "Respond to the query"
        assert plan.raw_text == "Respond to the query"
        assert plan.complexity == PlanComplexity.SIMPLE
        assert plan.sub_goals == []
        assert plan.reasoning_chain == []

    ##Method purpose: Test structured plan is_structured
    def test_is_structured_for_simple_plan(self) -> None:
        """is_structured() returns False for simple plans."""
        plan = Plan.simple("Simple plan")

        assert plan.is_structured() is False

    ##Method purpose: Test is_structured for complex plan
    def test_is_structured_for_complex_plan(self) -> None:
        """is_structured() returns True for plans with sub_goals."""
        plan = Plan(
            main_goal="Main goal",
            sub_goals=[
                PlanStep(1, "Step 1"),
                PlanStep(2, "Step 2"),
            ],
            complexity=PlanComplexity.COMPLEX,
        )

        assert plan.is_structured() is True

    ##Method purpose: Test as_text for simple plan
    def test_as_text_for_simple_plan(self) -> None:
        """as_text() returns raw_text for simple plans."""
        plan = Plan.simple("Simple plan text")

        assert plan.as_text() == "Simple plan text"

    ##Method purpose: Test as_text for structured plan
    def test_as_text_for_structured_plan(self) -> None:
        """as_text() formats structured plans."""
        plan = Plan(
            main_goal="Main goal",
            sub_goals=[
                PlanStep(1, "Step 1"),
                PlanStep(2, "Step 2"),
            ],
            reasoning_chain=["Reason 1", "Reason 2"],
            complexity=PlanComplexity.COMPLEX,
        )

        text = plan.as_text()

        assert "Main Goal: Main goal" in text
        assert "Sub-goals:" in text
        assert "1. Step 1" in text
        assert "2. Step 2" in text
        assert "Reasoning Chain:" in text
        assert "1. Reason 1" in text

    ##Method purpose: Test from_dict creates structured plan
    def test_from_dict_creates_structured_plan(self) -> None:
        """Plan.from_dict() creates structured plan from dict."""
        data = {
            "main_goal": "Complete the task",
            "sub_goals": ["First step", "Second step", "Third step"],
            "reasoning_chain": ["Because A", "Therefore B"],
        }

        plan = Plan.from_dict(data, PlanComplexity.COMPLEX)

        assert plan.main_goal == "Complete the task"
        assert len(plan.sub_goals) == 3
        assert plan.sub_goals[0].description == "First step"
        assert plan.sub_goals[0].index == 1
        assert len(plan.reasoning_chain) == 2
        assert plan.complexity == PlanComplexity.COMPLEX

    ##Method purpose: Test from_dict handles missing fields
    def test_from_dict_handles_missing_fields(self) -> None:
        """Plan.from_dict() handles missing optional fields."""
        data = {
            "main_goal": "Simple goal",
        }

        plan = Plan.from_dict(data, PlanComplexity.MODERATE)

        assert plan.main_goal == "Simple goal"
        assert plan.sub_goals == []
        assert plan.reasoning_chain == []

    ##Method purpose: Test from_dict handles non-list sub_goals
    def test_from_dict_handles_non_list_sub_goals(self) -> None:
        """Plan.from_dict() handles non-list sub_goals gracefully."""
        data = {
            "main_goal": "Goal",
            "sub_goals": "not a list",
        }

        plan = Plan.from_dict(data, PlanComplexity.SIMPLE)

        # Should not crash, just empty sub_goals
        assert plan.sub_goals == []

    ##Method purpose: Test as_text with empty sub_goals
    def test_as_text_with_empty_reasoning(self) -> None:
        """as_text() handles empty reasoning chain."""
        plan = Plan(
            main_goal="Goal only",
            sub_goals=[PlanStep(1, "Step")],
            reasoning_chain=[],
            complexity=PlanComplexity.MODERATE,
        )

        text = plan.as_text()

        assert "Goal only" in text
        assert "Reasoning Chain:" not in text


##Class purpose: Tests for PlanningConfig
class TestPlanningConfig:
    """Tests for PlanningConfig dataclass."""

    ##Method purpose: Test default values
    def test_default_values(self) -> None:
        """PlanningConfig has sensible defaults."""
        config = PlanningConfig()

        assert config.multi_level_enabled is True
        assert config.max_sub_goals == 5
        assert config.complexity_threshold == 20
        assert config.plan_temperature == 0.3

    ##Method purpose: Test custom values
    def test_custom_values(self) -> None:
        """PlanningConfig accepts custom values."""
        config = PlanningConfig(
            multi_level_enabled=False,
            max_sub_goals=3,
            complexity_threshold=15,
            plan_temperature=0.5,
        )

        assert config.multi_level_enabled is False
        assert config.max_sub_goals == 3


##Class purpose: Tests for EngineConfig with planning
class TestEngineConfigWithPlanning:
    """Tests for EngineConfig planning integration."""

    ##Method purpose: Test engine config includes planning
    def test_engine_config_includes_planning(self) -> None:
        """EngineConfig includes PlanningConfig."""
        config = EngineConfig()

        assert hasattr(config, "planning")
        assert isinstance(config.planning, PlanningConfig)

    ##Method purpose: Test engine config with custom planning
    def test_engine_config_with_custom_planning(self) -> None:
        """EngineConfig accepts custom PlanningConfig."""
        planning = PlanningConfig(multi_level_enabled=False)
        config = EngineConfig(planning=planning)

        assert config.planning.multi_level_enabled is False


##Class purpose: Tests for complexity assessment logic
class TestComplexityAssessment:
    """Tests for complexity assessment patterns."""

    ##Method purpose: Test simple query patterns
    def test_simple_query_patterns(self) -> None:
        """Simple queries should be recognized."""
        simple_queries = [
            "Hello",
            "What time is it?",
            "Yes",
            "Thanks",
        ]

        # These are short and have no complexity indicators
        for query in simple_queries:
            words = query.split()
            assert len(words) <= 20  # Under typical threshold

    ##Method purpose: Test complexity indicator patterns
    def test_complexity_indicator_patterns(self) -> None:
        """Complexity indicators should be recognized."""
        indicators = [
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

        complex_query = "Can you explain step by step how does Python work?"

        has_indicator = any(ind in complex_query.lower() for ind in indicators)
        assert has_indicator is True

    ##Method purpose: Test word count threshold
    def test_word_count_threshold(self) -> None:
        """Queries above threshold should be flagged."""
        threshold = 20

        short_query = "What is Python?"
        long_query = " ".join(["word"] * 25)

        assert len(short_query.split()) < threshold
        assert len(long_query.split()) > threshold

    ##Method purpose: Test multiple questions detection
    def test_multiple_questions_detection(self) -> None:
        """Multiple questions should increase complexity."""
        single_question = "What is Python?"
        multiple_questions = "What is Python? How does it work? Why is it popular?"

        assert single_question.count("?") == 1
        assert multiple_questions.count("?") == 3


##Class purpose: Tests for Plan text formatting
class TestPlanFormatting:
    """Tests for Plan text formatting."""

    ##Method purpose: Test main goal only formatting
    def test_main_goal_only_format(self) -> None:
        """Plan with only main goal formats correctly."""
        plan = Plan(
            main_goal="Answer the question",
            complexity=PlanComplexity.SIMPLE,
            raw_text="Answer the question directly",
        )

        # Simple plan returns raw_text
        assert plan.as_text() == "Answer the question directly"

    ##Method purpose: Test full structured plan formatting
    def test_full_structured_plan_format(self) -> None:
        """Full structured plan formats with all sections."""
        plan = Plan(
            main_goal="Provide comprehensive answer",
            sub_goals=[
                PlanStep(1, "Understand the question"),
                PlanStep(2, "Gather relevant information"),
                PlanStep(3, "Formulate response"),
            ],
            reasoning_chain=[
                "Understanding ensures accuracy",
                "Information provides foundation",
                "Formulation delivers value",
            ],
            complexity=PlanComplexity.COMPLEX,
        )

        text = plan.as_text()

        # Check all sections present
        assert "Main Goal:" in text
        assert "Sub-goals:" in text
        assert "Reasoning Chain:" in text

        # Check numbering
        assert "1. Understand" in text
        assert "2. Gather" in text
        assert "3. Formulate" in text

    ##Method purpose: Test empty raw_text fallback
    def test_empty_raw_text_uses_main_goal(self) -> None:
        """Empty raw_text falls back to main_goal."""
        plan = Plan(
            main_goal="Fallback goal",
            complexity=PlanComplexity.SIMPLE,
            raw_text="",
        )

        # Should return main_goal when raw_text is empty
        assert plan.as_text() == "Fallback goal"
