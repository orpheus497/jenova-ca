# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Pydantic validators for all data structures.

Provides type-safe data models and validation for memory, insights, and other components.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, field_validator


class MemoryEntry(BaseModel):
    """Base model for memory entries."""

    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., min_length=1, description="Memory content")
    timestamp: float = Field(..., gt=0, description="Unix timestamp")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        """Validate embedding vector."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Embedding must be a list")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Embedding must contain only numbers")
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp is reasonable."""
        # Check it's not in the future (with 1 hour grace period)
        max_time = datetime.now().timestamp() + 3600
        if v > max_time:
            raise ValueError("Timestamp cannot be in the future")
        # Check it's not before year 2000
        min_time = 946684800  # 2000-01-01
        if v < min_time:
            raise ValueError("Timestamp must be after year 2000")
        return v


class EpisodicMemoryEntry(MemoryEntry):
    """Model for episodic memory entries."""

    context: Optional[str] = Field(None, description="Contextual information")
    emotional_valence: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Emotional value [-1, 1]"
    )
    importance: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Importance score [0, 1]"
    )


class SemanticMemoryEntry(MemoryEntry):
    """Model for semantic memory entries."""

    category: Optional[str] = Field(None, description="Knowledge category")
    relationships: List[str] = Field(
        default_factory=list, description="Related concept IDs"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score [0, 1]"
    )


class ProceduralMemoryEntry(MemoryEntry):
    """Model for procedural memory entries."""

    steps: List[str] = Field(..., min_length=1, description="Procedure steps")
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Success rate [0, 1]"
    )
    last_used: Optional[float] = Field(None, description="Last usage timestamp")


class InsightEntry(BaseModel):
    """Model for insight entries."""

    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., min_length=1, description="Insight content")
    timestamp: float = Field(..., gt=0, description="Creation timestamp")
    source: str = Field(..., description="Source of insight")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level [0, 1]"
    )
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    related_memories: List[str] = Field(
        default_factory=list, description="Related memory IDs"
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        """Validate source is non-empty."""
        if not v.strip():
            raise ValueError("Source cannot be empty")
        return v.strip()


class AssumptionEntry(BaseModel):
    """Model for assumption entries."""

    id: str = Field(..., description="Unique identifier")
    assumption: str = Field(..., min_length=1, description="Assumption statement")
    timestamp: float = Field(..., gt=0, description="Creation timestamp")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level [0, 1]"
    )
    context: Optional[str] = Field(None, description="Context where assumption applies")
    validated: bool = Field(False, description="Whether assumption has been validated")
    validation_result: Optional[bool] = Field(
        None, description="Validation outcome if tested"
    )


class ToolCall(BaseModel):
    """Model for tool execution calls."""

    tool_name: str = Field(..., description="Name of the tool")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    timestamp: float = Field(..., gt=0, description="Execution timestamp")
    result: Optional[str] = Field(None, description="Tool execution result")
    success: bool = Field(False, description="Whether execution was successful")
    error: Optional[str] = Field(None, description="Error message if failed")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v):
        """Validate tool name is non-empty and alphanumeric."""
        if not v.strip():
            raise ValueError("Tool name cannot be empty")
        # Allow alphanumeric and underscores
        if not all(c.isalnum() or c == "_" for c in v):
            raise ValueError("Tool name must be alphanumeric with underscores")
        return v.strip()


class ConversationTurn(BaseModel):
    """Model for conversation turns."""

    role: str = Field(..., description="Role (user/assistant/system)")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: float = Field(..., gt=0, description="Message timestamp")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        """Validate role is one of allowed values."""
        allowed_roles = {"user", "assistant", "system"}
        if v.lower() not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v.lower()


class SearchQuery(BaseModel):
    """Model for search queries."""

    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Similarity threshold"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional filters"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query is non-empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    """Model for search results."""

    id: str = Field(..., description="Result ID")
    content: str = Field(..., description="Result content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ConfigUpdate(BaseModel):
    """Model for configuration updates."""

    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="New value")
    timestamp: float = Field(..., gt=0, description="Update timestamp")
    reason: Optional[str] = Field(None, description="Reason for update")

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        """Validate key format."""
        if not v.strip():
            raise ValueError("Key cannot be empty")
        # Keys should be dot-separated paths
        parts = v.split(".")
        if not all(part.isidentifier() for part in parts):
            raise ValueError(
                "Key must be valid Python identifier path (e.g., 'model.gpu_layers')"
            )
        return v


class PerformanceMetric(BaseModel):
    """Model for performance metrics."""

    operation: str = Field(..., description="Operation name")
    duration_seconds: float = Field(..., ge=0.0, description="Operation duration")
    success: bool = Field(..., description="Operation success")
    timestamp: float = Field(..., gt=0, description="Measurement timestamp")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metrics"
    )

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v):
        """Validate operation name."""
        if not v.strip():
            raise ValueError("Operation name cannot be empty")
        return v.strip()


class DataValidator:
    """Helper class for validating data structures."""

    @staticmethod
    def validate_memory_entry(
        data: Dict[str, Any], memory_type: str = "episodic"
    ) -> MemoryEntry:
        """
        Validate and create a memory entry.

        Args:
            data: Raw memory data
            memory_type: Type of memory (episodic/semantic/procedural)

        Returns:
            Validated memory entry

        Raises:
            ValueError: If validation fails
        """
        if memory_type == "episodic":
            return EpisodicMemoryEntry(**data)
        elif memory_type == "semantic":
            return SemanticMemoryEntry(**data)
        elif memory_type == "procedural":
            return ProceduralMemoryEntry(**data)
        else:
            return MemoryEntry(**data)

    @staticmethod
    def validate_insight(data: Dict[str, Any]) -> InsightEntry:
        """
        Validate and create an insight entry.

        Args:
            data: Raw insight data

        Returns:
            Validated insight entry

        Raises:
            ValueError: If validation fails
        """
        return InsightEntry(**data)

    @staticmethod
    def validate_assumption(data: Dict[str, Any]) -> AssumptionEntry:
        """
        Validate and create an assumption entry.

        Args:
            data: Raw assumption data

        Returns:
            Validated assumption entry

        Raises:
            ValueError: If validation fails
        """
        return AssumptionEntry(**data)

    @staticmethod
    def validate_tool_call(data: Dict[str, Any]) -> ToolCall:
        """
        Validate and create a tool call entry.

        Args:
            data: Raw tool call data

        Returns:
            Validated tool call

        Raises:
            ValueError: If validation fails
        """
        return ToolCall(**data)

    @staticmethod
    def validate_search_query(data: Dict[str, Any]) -> SearchQuery:
        """
        Validate and create a search query.

        Args:
            data: Raw search query data

        Returns:
            Validated search query

        Raises:
            ValueError: If validation fails
        """
        return SearchQuery(**data)

    @staticmethod
    def sanitize_dict(
        data: Dict[str, Any], allowed_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Sanitize dictionary by removing invalid keys.

        Args:
            data: Dictionary to sanitize
            allowed_keys: List of allowed keys (None = allow all)

        Returns:
            Sanitized dictionary
        """
        if allowed_keys is None:
            return data.copy()

        return {k: v for k, v in data.items() if k in allowed_keys}

    @staticmethod
    def validate_timestamp(timestamp: float) -> float:
        """
        Validate a timestamp value.

        Args:
            timestamp: Unix timestamp to validate

        Returns:
            Validated timestamp

        Raises:
            ValueError: If timestamp is invalid
        """
        max_time = datetime.now().timestamp() + 3600
        min_time = 946684800  # 2000-01-01

        if timestamp > max_time:
            raise ValueError("Timestamp cannot be in the future")
        if timestamp < min_time:
            raise ValueError("Timestamp must be after year 2000")

        return timestamp

    @staticmethod
    def validate_score(
        score: float, min_val: float = 0.0, max_val: float = 1.0
    ) -> float:
        """
        Validate a score/confidence value.

        Args:
            score: Score to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated score

        Raises:
            ValueError: If score is out of range
        """
        if not isinstance(score, (int, float)):
            raise ValueError(f"Score must be a number, got {type(score)}")

        if score < min_val or score > max_val:
            raise ValueError(
                f"Score must be between {min_val} and {max_val}, got {score}"
            )

        return float(score)
