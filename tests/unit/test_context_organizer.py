##Script function and purpose: Unit tests for ContextOrganizer - hierarchical context organization
"""
Tests for ContextOrganizer

Tests context organization including categorization, tier classification,
and prompt formatting.
"""

import pytest

from jenova.core.context_organizer import (
    ContextOrganizer,
    ContextOrganizerConfig,
    ContextTier,
    OrganizedContext,
)
from jenova.core.context_scorer import (
    ScoredContext,
    ScoringBreakdown,
)


##Class purpose: Test OrganizedContext dataclass
class TestOrganizedContext:
    """Tests for OrganizedContext dataclass."""
    
    ##Method purpose: Create organized context fixture
    @pytest.fixture
    def organized_context(self) -> OrganizedContext:
        """Create an OrganizedContext with test data."""
        return OrganizedContext(
            categorized={
                "Technical": ["Python code example", "API documentation"],
                "Procedural": ["Step by step guide"],
            },
            tiers={
                ContextTier.HIGH: ["Python code example"],
                ContextTier.MEDIUM: ["API documentation", "Step by step guide"],
                ContextTier.LOW: ["Unrelated content"],
            },
            query="How to use Python",
        )
    
    ##Method purpose: Test get_tier method
    def test_get_tier_returns_correct_items(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Get tier returns items for specified tier."""
        high = organized_context.get_tier(ContextTier.HIGH)
        
        assert len(high) == 1
        assert "Python code example" in high
    
    ##Method purpose: Test high_priority property
    def test_high_priority_property(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """High priority property returns high tier items."""
        high = organized_context.high_priority
        
        assert high == ["Python code example"]
    
    ##Method purpose: Test medium_priority property
    def test_medium_priority_property(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Medium priority property returns medium tier items."""
        medium = organized_context.medium_priority
        
        assert len(medium) == 2
    
    ##Method purpose: Test low_priority property
    def test_low_priority_property(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Low priority property returns low tier items."""
        low = organized_context.low_priority
        
        assert "Unrelated content" in low
    
    ##Method purpose: Test total_items property
    def test_total_items_counts_all(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Total items counts all items across tiers."""
        total = organized_context.total_items
        
        assert total == 4
    
    ##Method purpose: Test is_empty method
    def test_is_empty_returns_false_when_has_items(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Is empty returns False when items exist."""
        assert not organized_context.is_empty()
    
    ##Method purpose: Test is_empty with empty context
    def test_is_empty_returns_true_when_empty(self) -> None:
        """Is empty returns True when no items exist."""
        empty_context = OrganizedContext(
            categorized={},
            tiers={
                ContextTier.HIGH: [],
                ContextTier.MEDIUM: [],
                ContextTier.LOW: [],
            },
            query="test",
        )
        
        assert empty_context.is_empty()


##Class purpose: Test prompt formatting
class TestPromptFormatting:
    """Tests for prompt formatting functionality."""
    
    ##Method purpose: Create organized context fixture
    @pytest.fixture
    def organized_context(self) -> OrganizedContext:
        """Create an OrganizedContext with test data."""
        return OrganizedContext(
            categorized={},
            tiers={
                ContextTier.HIGH: ["Important context 1", "Important context 2"],
                ContextTier.MEDIUM: ["Related context"],
                ContextTier.LOW: ["Additional info"],
            },
            query="test query",
        )
    
    ##Method purpose: Test format_for_prompt returns string
    def test_format_for_prompt_returns_string(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Format for prompt returns string."""
        result = organized_context.format_for_prompt()
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    ##Method purpose: Test high priority is included
    def test_high_priority_included(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """High priority items are included in prompt."""
        result = organized_context.format_for_prompt()
        
        assert "Important context 1" in result
        assert "Highly Relevant" in result
    
    ##Method purpose: Test medium priority included
    def test_medium_priority_included(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Medium priority items are included in prompt."""
        result = organized_context.format_for_prompt()
        
        assert "Related context" in result
    
    ##Method purpose: Test max_tokens limits output
    def test_max_tokens_limits_output(
        self,
        organized_context: OrganizedContext,
    ) -> None:
        """Max tokens parameter limits output length."""
        short_result = organized_context.format_for_prompt(max_tokens=50)
        long_result = organized_context.format_for_prompt(max_tokens=500)
        
        # Short should be limited
        assert len(short_result) <= len(long_result)


##Class purpose: Test ContextOrganizer organization
class TestContextOrganization:
    """Tests for ContextOrganizer organization functionality."""
    
    ##Method purpose: Create organizer fixture
    @pytest.fixture
    def organizer(self) -> ContextOrganizer:
        """Create a ContextOrganizer instance."""
        return ContextOrganizer()
    
    ##Method purpose: Create scored context fixture
    @pytest.fixture
    def scored_context(self) -> ScoredContext:
        """Create a ScoredContext for testing."""
        items = [
            ScoringBreakdown(content="Python programming tutorial", total_score=0.9),
            ScoringBreakdown(content="API documentation guide", total_score=0.7),
            ScoringBreakdown(content="Random unrelated text", total_score=0.2),
        ]
        return ScoredContext(items=items, query="Learn Python")
    
    ##Method purpose: Test organize returns OrganizedContext
    def test_organize_returns_organized_context(
        self,
        organizer: ContextOrganizer,
        scored_context: ScoredContext,
    ) -> None:
        """Organize returns OrganizedContext."""
        result = organizer.organize(scored_context)
        
        assert isinstance(result, OrganizedContext)
    
    ##Method purpose: Test high scores go to high tier
    def test_high_scores_in_high_tier(
        self,
        organizer: ContextOrganizer,
        scored_context: ScoredContext,
    ) -> None:
        """High scoring items are placed in high tier."""
        result = organizer.organize(scored_context)
        
        # 0.9 score should be in high tier
        assert "Python programming tutorial" in result.high_priority
    
    ##Method purpose: Test low scores go to low tier
    def test_low_scores_in_low_tier(
        self,
        organizer: ContextOrganizer,
        scored_context: ScoredContext,
    ) -> None:
        """Low scoring items are placed in low tier."""
        result = organizer.organize(scored_context)
        
        # 0.2 score should be in low tier
        assert "Random unrelated text" in result.low_priority


##Class purpose: Test raw organization
class TestRawOrganization:
    """Tests for organizing raw context without scores."""
    
    ##Method purpose: Create organizer fixture
    @pytest.fixture
    def organizer(self) -> ContextOrganizer:
        """Create a ContextOrganizer instance."""
        return ContextOrganizer()
    
    ##Method purpose: Test organize_raw returns OrganizedContext
    def test_organize_raw_returns_organized_context(
        self,
        organizer: ContextOrganizer,
    ) -> None:
        """Organize raw returns OrganizedContext."""
        items = ["Item one", "Item two", "Item three"]
        
        result = organizer.organize_raw(items, "test query")
        
        assert isinstance(result, OrganizedContext)
        assert result.total_items == 3
    
    ##Method purpose: Test empty list handling
    def test_organize_raw_handles_empty_list(
        self,
        organizer: ContextOrganizer,
    ) -> None:
        """Organize raw handles empty list."""
        result = organizer.organize_raw([], "test query")
        
        assert result.is_empty() or result.total_items == 0


##Class purpose: Test categorization
class TestCategorization:
    """Tests for topic-based categorization."""
    
    ##Method purpose: Create organizer fixture
    @pytest.fixture
    def organizer(self) -> ContextOrganizer:
        """Create a ContextOrganizer instance."""
        config = ContextOrganizerConfig(
            categorization_enabled=True,
            tier_classification_enabled=True,
        )
        return ContextOrganizer(config=config)
    
    ##Method purpose: Test technical content categorized
    def test_technical_content_categorized(
        self,
        organizer: ContextOrganizer,
    ) -> None:
        """Technical content is categorized correctly."""
        scored_context = ScoredContext(
            items=[
                ScoringBreakdown(content="Python code example", total_score=0.8),
                ScoringBreakdown(content="API function documentation", total_score=0.7),
            ],
            query="programming",
        )
        
        result = organizer.organize(scored_context)
        
        # Should have categories
        assert len(result.categorized) > 0
    
    ##Method purpose: Test procedural content categorized
    def test_procedural_content_categorized(
        self,
        organizer: ContextOrganizer,
    ) -> None:
        """Procedural content is categorized correctly."""
        scored_context = ScoredContext(
            items=[
                ScoringBreakdown(content="Step by step guide to install", total_score=0.9),
                ScoringBreakdown(content="Process for setting up the system", total_score=0.8),
            ],
            query="how to",
        )
        
        result = organizer.organize(scored_context)
        
        # Check categorization happened
        assert len(result.categorized) > 0


##Class purpose: Test organizer configuration
class TestOrganizerConfiguration:
    """Tests for organizer configuration."""
    
    ##Method purpose: Test disabled organizer returns flat
    def test_disabled_organizer_returns_flat(self) -> None:
        """Disabled organizer returns flat organization."""
        config = ContextOrganizerConfig(enabled=False)
        organizer = ContextOrganizer(config=config)
        
        scored_context = ScoredContext(
            items=[
                ScoringBreakdown(content="Item 1", total_score=0.9),
                ScoringBreakdown(content="Item 2", total_score=0.5),
            ],
            query="test",
        )
        
        result = organizer.organize(scored_context)
        
        # All items should be in medium tier
        assert len(result.medium_priority) == 2
        assert len(result.high_priority) == 0
        assert len(result.low_priority) == 0
    
    ##Method purpose: Test tier thresholds are respected
    def test_tier_thresholds_respected(self) -> None:
        """Custom tier thresholds affect classification."""
        config = ContextOrganizerConfig(
            high_tier_threshold=0.9,  # Very high threshold
            medium_tier_threshold=0.5,
        )
        organizer = ContextOrganizer(config=config)
        
        scored_context = ScoredContext(
            items=[
                ScoringBreakdown(content="High score", total_score=0.95),
                ScoringBreakdown(content="Medium score", total_score=0.7),
                ScoringBreakdown(content="Low score", total_score=0.3),
            ],
            query="test",
        )
        
        result = organizer.organize(scored_context)
        
        # Only 0.95 should be high
        assert len(result.high_priority) == 1
        assert "High score" in result.high_priority
    
    ##Method purpose: Test max_items_per_tier is respected
    def test_max_items_per_tier_respected(self) -> None:
        """Max items per tier limits items."""
        config = ContextOrganizerConfig(max_items_per_tier=2)
        organizer = ContextOrganizer(config=config)
        
        scored_context = ScoredContext(
            items=[
                ScoringBreakdown(content="High 1", total_score=0.9),
                ScoringBreakdown(content="High 2", total_score=0.85),
                ScoringBreakdown(content="High 3", total_score=0.8),
                ScoringBreakdown(content="High 4", total_score=0.75),
            ],
            query="test",
        )
        
        result = organizer.organize(scored_context)
        
        # Should be limited to 2 per tier
        assert len(result.high_priority) <= 2


##Class purpose: Test tier summarization
class TestTierSummarization:
    """Tests for tier summarization functionality."""
    
    ##Method purpose: Create organizer fixture
    @pytest.fixture
    def organizer(self) -> ContextOrganizer:
        """Create a ContextOrganizer instance."""
        return ContextOrganizer()
    
    ##Method purpose: Test summarize_tier returns string
    def test_summarize_tier_returns_string(
        self,
        organizer: ContextOrganizer,
    ) -> None:
        """Summarize tier returns string."""
        organized = OrganizedContext(
            categorized={},
            tiers={
                ContextTier.HIGH: ["Item one", "Item two"],
                ContextTier.MEDIUM: [],
                ContextTier.LOW: [],
            },
            query="test",
        )
        
        summary = organizer.summarize_tier(organized, ContextTier.HIGH)
        
        assert isinstance(summary, str)
        assert "Item one" in summary
    
    ##Method purpose: Test summarize empty tier
    def test_summarize_empty_tier(
        self,
        organizer: ContextOrganizer,
    ) -> None:
        """Summarizing empty tier returns appropriate message."""
        organized = OrganizedContext(
            categorized={},
            tiers={
                ContextTier.HIGH: [],
                ContextTier.MEDIUM: [],
                ContextTier.LOW: [],
            },
            query="test",
        )
        
        summary = organizer.summarize_tier(organized, ContextTier.HIGH)
        
        assert "No" in summary or len(summary) > 0
    
    ##Method purpose: Test max_length truncation
    def test_summarize_respects_max_length(
        self,
        organizer: ContextOrganizer,
    ) -> None:
        """Summarize respects max_length parameter."""
        organized = OrganizedContext(
            categorized={},
            tiers={
                ContextTier.HIGH: [
                    "A very long item that contains lots of text " * 10,
                ],
                ContextTier.MEDIUM: [],
                ContextTier.LOW: [],
            },
            query="test",
        )
        
        summary = organizer.summarize_tier(organized, ContextTier.HIGH, max_length=50)
        
        assert len(summary) <= 53  # 50 + "..."
