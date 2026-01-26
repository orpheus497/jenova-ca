##Sec: Pydantic schemas for validating LLM JSON output (P1-001)
##Script function and purpose: Defines Pydantic models to validate LLM-generated JSON responses
"""
LLM Response Validation Schemas

Pydantic models for validating JSON responses from LLM calls in the
CognitiveGraph. Prevents crashes from malformed LLM output by providing
strict schema validation with sensible defaults and error recovery.

Security Note: These schemas are critical for preventing LLM output
from causing KeyError, TypeError, or other crashes. All LLM JSON
parsing should use these validators.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


##Class purpose: Validate emotion analysis LLM response
class EmotionAnalysisResponse(BaseModel):
    """Schema for emotion analysis LLM response.

    Expected format:
    {
        "primary_emotion": "joy",
        "confidence": 0.85,
        "emotion_scores": {"joy": 0.85, "neutral": 0.1, "curiosity": 0.05}
    }
    """

    primary_emotion: str = Field(default="neutral")
    """Primary detected emotion (validated against Emotion enum separately)."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    """Confidence score, clamped to 0.0-1.0."""

    emotion_scores: dict[str, float] = Field(default_factory=dict)
    """Optional breakdown of emotion scores."""

    ##Method purpose: Validate and clamp confidence value
    @field_validator("confidence", mode="before")
    @classmethod
    ##Refactor: Replace Any with object for Pydantic validator input - more explicit about intentional type acceptance (AP-001)
    def clamp_confidence(cls, v: object) -> float:
        """Clamp confidence to valid range, handling non-numeric input."""
        try:
            value = float(v)
            return max(0.0, min(1.0, value))
        except (TypeError, ValueError):
            return 0.5

    ##Method purpose: Validate emotion_scores dict values
    @field_validator("emotion_scores", mode="before")
    @classmethod
    ##Refactor: Replace Any with object for Pydantic validator input - more explicit about intentional type acceptance (AP-001)
    def validate_emotion_scores(cls, v: object) -> dict[str, float]:
        """Ensure emotion_scores is a dict with float values."""
        if not isinstance(v, dict):
            return {}

        result: dict[str, float] = {}
        for key, value in v.items():
            if isinstance(key, str):
                try:
                    result[key] = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    result[key] = 0.0
        return result


##Class purpose: Validate relationship analysis LLM response (link_orphans)
class RelationshipAnalysisResponse(BaseModel):
    """Schema for relationship analysis LLM response.

    Expected format:
    {
        "related_node_ids": ["abc12345", "def67890"],
        "relationship": "relates_to"
    }
    """

    related_node_ids: list[str] = Field(default_factory=list)
    """List of related node ID prefixes."""

    relationship: str = Field(default="relates_to")
    """Relationship type (validated against EdgeType separately)."""

    ##Method purpose: Ensure related_node_ids contains only strings
    @field_validator("related_node_ids", mode="before")
    @classmethod
    ##Refactor: Replace Any with object for Pydantic validator input - more explicit about intentional type acceptance (AP-001)
    def validate_related_ids(cls, v: object) -> list[str]:
        """Ensure related_node_ids is a list of strings."""
        if not isinstance(v, list):
            return []
        return [str(item) for item in v if item is not None]


##Class purpose: Validate contradiction detection LLM response
class ContradictionCheckResponse(BaseModel):
    """Schema for contradiction detection LLM response.

    Expected format:
    {
        "contradicts": true,
        "explanation": "These statements cannot both be true because..."
    }
    """

    contradicts: bool = Field(default=False)
    """Whether the two statements contradict each other."""

    explanation: str = Field(default="")
    """Optional explanation of the contradiction."""

    ##Method purpose: Handle non-boolean contradicts values
    @field_validator("contradicts", mode="before")
    @classmethod
    ##Refactor: Replace Any with object for Pydantic validator input - more explicit about intentional type acceptance (AP-001)
    def validate_contradicts(cls, v: object) -> bool:
        """Convert various truthy values to bool."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "yes", "1")
        return bool(v)


##Class purpose: Validate connection suggestion LLM response
class ConnectionSuggestionsResponse(BaseModel):
    """Schema for connection suggestion LLM response.

    Expected format:
    {
        "suggested_ids": ["abc12345", "def67890"]
    }
    """

    suggested_ids: list[str] = Field(default_factory=list)
    """List of suggested node ID prefixes."""

    ##Method purpose: Ensure suggested_ids contains only strings
    @field_validator("suggested_ids", mode="before")
    @classmethod
    ##Refactor: Replace Any with object for Pydantic validator input - more explicit about intentional type acceptance (AP-001)
    def validate_suggested_ids(cls, v: object) -> list[str]:
        """Ensure suggested_ids is a list of strings."""
        if not isinstance(v, list):
            return []
        return [str(item) for item in v if item is not None]


##Class purpose: Validate cluster label generation LLM response
class ClusterLabelResponse(BaseModel):
    """Schema for cluster label generation LLM response.

    Expected format:
    {
        "label": "Technology and Programming",
        "description": "Nodes related to software development..."
    }
    """

    label: str = Field(default="Unknown Cluster")
    """Generated label for the cluster."""

    description: str = Field(default="")
    """Optional description of the cluster theme."""


##Class purpose: Validate meta-insight generation LLM response
class MetaInsightResponse(BaseModel):
    """Schema for meta-insight generation LLM response.

    Expected format:
    {
        "insight": "The connection between X and Y suggests...",
        "confidence": 0.8
    }
    """

    insight: str = Field(default="")
    """The generated meta-insight."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    """Confidence in the insight."""

    ##Method purpose: Validate and clamp confidence value
    @field_validator("confidence", mode="before")
    @classmethod
    ##Refactor: Replace Any with object for Pydantic validator input - more explicit about intentional type acceptance (AP-001)
    def clamp_confidence(cls, v: object) -> float:
        """Clamp confidence to valid range."""
        try:
            value = float(v)
            return max(0.0, min(1.0, value))
        except (TypeError, ValueError):
            return 0.5
