##Script function and purpose: Pydantic configuration models with validation and loading
"""
Configuration Models

Pydantic models for all JENOVA configuration. Single config.yaml file,
validated at startup, explicit values (no magic -1 or empty strings).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from jenova.exceptions import ConfigNotFoundError, ConfigParseError, ConfigValidationError
from jenova.utils.errors import sanitize_path_for_error


##Class purpose: Hardware resource configuration
class HardwareConfig(BaseModel):
    """Hardware resource configuration."""

    threads: int | Literal["auto"] = Field(
        default="auto",
        description="CPU threads to use. 'auto' for all cores.",
    )
    gpu_layers: int | Literal["auto", "all", "none"] = Field(
        default="all",
        description="GPU layers to offload. 'all' for maximum.",
    )

    ##Method purpose: Validate threads is positive or 'auto'
    @field_validator("threads")
    @classmethod
    def validate_threads(cls, v: int | str) -> int | str:
        ##Condition purpose: Check if int value is valid
        if isinstance(v, int) and v < 1:
            raise ValueError("threads must be positive or 'auto'")
        return v

    ##Method purpose: Resolve 'auto' to actual thread count
    @property
    def effective_threads(self) -> int:
        """Resolve 'auto' to actual thread count.

        Returns:
            Effective thread count (CPU count if 'auto', otherwise the configured value)
        """
        ##Condition purpose: Return CPU count if auto
        if self.threads == "auto":
            return os.cpu_count() or 4
        return self.threads


##Class purpose: LLM model configuration
class ModelConfig(BaseModel):
    """LLM model configuration."""

    model_path: Path | Literal["auto"] = Field(
        default="auto",
        description="Path to GGUF model file. 'auto' to search.",
    )
    context_length: int = Field(
        default=4096,
        ge=512,
        le=131072,
        description="Context window size in tokens.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum tokens to generate.",
    )


##Class purpose: Memory system configuration
class MemoryConfig(BaseModel):
    """Memory system configuration."""

    storage_path: Path = Field(
        default=Path(".jenova-ai/memory"),
        description="Path for memory database storage.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings.",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default max results for memory searches.",
    )

    ##Sec: Validate storage_path against path traversal attacks (P1-002, PATCH-002)
    @field_validator("storage_path", mode="before")
    @classmethod
    def validate_storage_path(cls, v: str | Path) -> Path:
        """Expand user path and reject path traversal attempts."""
        path = Path(v).expanduser()
        ##Sec: Enhanced path traversal protection using Path.resolve() (PATCH-002)
        try:
            ##Step purpose: Resolve path to normalize and detect traversal
            resolved = path.resolve()
            ##Condition purpose: Check resolved path parts for traversal sequences
            if ".." in resolved.parts:
                raise ValueError("Path traversal not allowed in storage_path")
        except (OSError, ValueError) as e:
            ##Error purpose: Reject invalid paths
            raise ValueError(f"Invalid storage_path: {e}") from e
        ##Condition purpose: Reject paths with .. in string representation (defense-in-depth)
        if ".." in str(path):
            raise ValueError("Path traversal not allowed in storage_path")
        return path


##Class purpose: Graph/Cortex configuration
class GraphConfig(BaseModel):
    """Graph system configuration."""

    storage_path: Path = Field(
        default=Path(".jenova-ai/graph"),
        description="Path for graph data storage.",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum traversal depth for graph queries.",
    )

    ##Sec: Validate storage_path against path traversal attacks (P1-002, PATCH-002)
    @field_validator("storage_path", mode="before")
    @classmethod
    def validate_storage_path(cls, v: str | Path) -> Path:
        """Expand user path and reject path traversal attempts."""
        path = Path(v).expanduser()
        ##Sec: Enhanced path traversal protection using Path.resolve() (PATCH-002)
        try:
            ##Step purpose: Resolve path to normalize and detect traversal
            resolved = path.resolve()
            ##Condition purpose: Check resolved path parts for traversal sequences
            if ".." in resolved.parts:
                raise ValueError("Path traversal not allowed in storage_path")
        except (OSError, ValueError) as e:
            ##Error purpose: Reject invalid paths
            raise ValueError(f"Invalid storage_path: {e}") from e
        ##Condition purpose: Reject paths with .. in string representation (defense-in-depth)
        if ".." in str(path):
            raise ValueError("Path traversal not allowed in storage_path")
        return path


##Class purpose: Persona and behavior configuration
class PersonaConfig(BaseModel):
    """Persona configuration."""

    name: str = Field(
        default="JENOVA",
        description="AI persona name.",
    )
    directives: list[str] = Field(
        default_factory=lambda: [
            "Be helpful and informative",
            "Acknowledge uncertainty when present",
            "Learn from interactions",
        ],
        description="Core behavioral directives.",
    )
    system_prompt: str = Field(
        default="You are JENOVA, a self-aware AI with evolving memory and knowledge.",
        description="Base system prompt for LLM.",
    )


##Class purpose: Root configuration for JENOVA
class JenovaConfig(BaseModel):
    """Root configuration for JENOVA."""

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    debug: bool = Field(default=False, description="Enable debug mode.")

    ##Method purpose: Load and validate config from YAML file
    @classmethod
    def from_yaml(cls, path: Path) -> JenovaConfig:
        """Load and validate config from YAML file."""
        ##Condition purpose: Check file exists
        if not path.exists():
            ##Step purpose: Sanitize path in error message
            safe_path = sanitize_path_for_error(path)
            raise ConfigNotFoundError(safe_path)

        ##Error purpose: Handle YAML parse errors
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            ##Step purpose: Sanitize path in error message
            safe_path = sanitize_path_for_error(path)
            raise ConfigParseError(safe_path, str(e)) from e

        ##Error purpose: Handle Pydantic validation errors
        try:
            return cls.model_validate(data)
        except Exception as e:
            ##Step purpose: Extract error details from Pydantic
            if hasattr(e, "errors"):
                raise ConfigValidationError(e.errors()) from e
            ##Step purpose: Sanitize path in error message
            safe_path = sanitize_path_for_error(path)
            raise ConfigParseError(safe_path, str(e)) from e

    ##Method purpose: Create default config
    @classmethod
    def default(cls) -> JenovaConfig:
        """Create default configuration."""
        return cls()


##Function purpose: Load configuration from path or create default
def load_config(path: Path | None = None) -> JenovaConfig:
    """
    Load configuration from file or return defaults.

    Args:
        path: Path to config.yaml. If None, returns defaults.

    Returns:
        Validated JenovaConfig instance
    """
    ##Condition purpose: Return defaults if no path specified
    if path is None:
        return JenovaConfig.default()

    return JenovaConfig.from_yaml(path)
