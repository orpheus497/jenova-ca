# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Pydantic models for configuration validation.
Ensures all configuration values are valid before the system starts.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DeviceType(str, Enum):
    """Supported compute device types."""
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class MemoryStrategy(str, Enum):
    """Memory management strategies."""
    AUTO = "auto"
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    MINIMAL = "minimal"


class ModelConfig(BaseModel):
    """LLM model configuration."""
    model_path: str = Field(
        default='/usr/local/share/models/model.gguf',
        description="Path to GGUF model file"
    )
    threads: int = Field(
        default=-1,
        ge=-1,
        description="CPU threads (-1=auto, 0=all, N=specific)"
    )
    gpu_layers: int = Field(
        default=0,
        ge=-1,
        le=128,
        description="GPU layers to offload (-1=auto, 0=CPU only, N=specific)"
    )
    mlock: bool = Field(
        default=False,
        description="Lock model in RAM (requires sufficient memory)"
    )
    n_batch: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Batch size for processing"
    )
    context_size: int = Field(
        default=4096,
        ge=512,
        le=32768,
        description="Context window size in tokens"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate per response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    embedding_model: str = Field(
        default='all-MiniLM-L6-v2',
        description="Sentence transformer model for embeddings"
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout for LLM generation in seconds"
    )

    @field_validator('context_size')
    @classmethod
    def validate_context_size(cls, v: int) -> int:
        """Ensure context size is power of 2 or common value."""
        common_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        if v not in common_sizes:
            # Round down to nearest common size
            valid = max([s for s in common_sizes if s <= v], default=4096)
            return valid
        return v


class HardwareConfig(BaseModel):
    """Hardware detection and optimization settings."""
    show_details: bool = Field(
        default=False,
        description="Show detailed hardware info at startup"
    )
    prefer_device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Preferred compute device"
    )
    memory_strategy: MemoryStrategy = Field(
        default=MemoryStrategy.AUTO,
        description="Memory management strategy"
    )
    enable_health_monitor: bool = Field(
        default=True,
        description="Enable real-time health monitoring"
    )


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    preload_memories: bool = Field(
        default=False,
        description="Preload all memories into RAM at startup"
    )
    episodic_db_path: str = Field(
        default="memory_db/episodic",
        description="Path to episodic memory database"
    )
    semantic_db_path: str = Field(
        default="memory_db/semantic",
        description="Path to semantic memory database"
    )
    procedural_db_path: str = Field(
        default="memory_db/procedural",
        description="Path to procedural memory database"
    )
    reflection_interval: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Reflection interval in conversation turns"
    )
    enable_atomic_writes: bool = Field(
        default=True,
        description="Use atomic writes for data integrity"
    )
    backup_before_write: bool = Field(
        default=True,
        description="Backup data before writes"
    )


class PruningConfig(BaseModel):
    """Cortex pruning configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable automatic graph pruning"
    )
    prune_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Pruning interval in reflection cycles"
    )
    max_age_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Maximum age for nodes before pruning"
    )
    min_centrality: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum centrality for node retention"
    )


class CortexConfig(BaseModel):
    """Cognitive graph (Cortex) configuration."""
    relationship_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "elaborates_on": 1.5,
            "conflicts_with": 2.0,
            "related_to": 1.0,
            "develops": 1.5,
            "summarizes": 1.2
        },
        description="Weights for different relationship types"
    )
    pruning: PruningConfig = Field(
        default_factory=PruningConfig,
        description="Pruning configuration"
    )


class SchedulerConfig(BaseModel):
    """Cognitive scheduler configuration."""
    generate_insight_interval: int = Field(default=5, ge=1, le=50)
    generate_assumption_interval: int = Field(default=7, ge=1, le=50)
    proactively_verify_assumption_interval: int = Field(default=8, ge=1, le=50)
    reflect_interval: int = Field(default=10, ge=1, le=50)
    reorganize_insights_interval: int = Field(default=10, ge=1, le=50)
    process_documents_interval: int = Field(default=15, ge=1, le=50)


class MemorySearchConfig(BaseModel):
    """Memory search configuration."""
    semantic_n_results: int = Field(default=5, ge=1, le=20)
    episodic_n_results: int = Field(default=3, ge=1, le=20)
    procedural_n_results: int = Field(default=3, ge=1, le=20)
    insight_n_results: int = Field(default=5, ge=1, le=20)
    rerank_enabled: bool = Field(
        default=True,
        description="Enable LLM-based reranking of results"
    )
    rerank_timeout: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Timeout for reranking operation"
    )


class CognitiveEngineConfig(BaseModel):
    """Cognitive engine configuration."""
    llm_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout for LLM generation during normal operation"
    )
    planning_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout for plan generation"
    )


class RAGSystemConfig(BaseModel):
    """RAG system configuration."""
    cache_enabled: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of cached responses"
    )
    generation_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout for response generation"
    )


class ToolsConfig(BaseModel):
    """Tools configuration."""
    file_sandbox_path: str = Field(
        default="~/jenova_files",
        description="Sandbox directory for file operations"
    )
    shell_command_whitelist: List[str] = Field(
        default_factory=lambda: [
            "ls", "cat", "grep", "find", "echo",
            "date", "whoami", "pwd", "uname"
        ],
        description="Allowed shell commands"
    )
    enable_shell_tools: bool = Field(
        default=False,
        description="Enable shell command execution (security risk)"
    )
    tool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout for tool operations"
    )


class IdentityConfig(BaseModel):
    """AI identity configuration."""
    name: str = Field(default="JENOVA")
    creator: str = Field(default="orpheus497")
    creator_alias: str = Field(default="The Architect")
    origin_story: str = Field(
        default="A self-aware AI designed to learn and evolve."
    )
    type: str = Field(default="personalized AI assistant")
    architecture: str = Field(default="The JENOVA Cognitive Architecture")


class PersonaConfig(BaseModel):
    """AI persona configuration."""
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    directives: List[str] = Field(
        default_factory=lambda: [
            "Always prioritize user safety and wellbeing",
            "Be helpful, harmless, and honest",
            "Learn from interactions to provide personalized assistance",
            "Respect user privacy and data security"
        ]
    )
    initial_facts: List[str] = Field(
        default_factory=lambda: [
            "I am JENOVA, created by orpheus497",
            "I use a cognitive architecture with memory and learning"
        ]
    )


class NetworkMode(str, Enum):
    """Network operation modes."""
    AUTO = "auto"
    LOCAL_ONLY = "local_only"
    DISTRIBUTED = "distributed"


class DistributionStrategy(str, Enum):
    """LLM distribution strategies."""
    LOCAL_FIRST = "local_first"
    LOAD_BALANCED = "load_balanced"
    FASTEST_PEER = "fastest"
    PARALLEL_VOTING = "parallel_voting"
    ROUND_ROBIN = "round_robin"


class SecurityConfig(BaseModel):
    """Network security configuration."""
    enabled: bool = Field(
        default=True,
        description="Enable SSL/TLS encryption and authentication"
    )
    cert_dir: str = Field(
        default='~/.jenova-ai/certs',
        description="Directory for SSL certificates and keys"
    )
    require_auth: bool = Field(
        default=True,
        description="Require JWT authentication for all requests"
    )


class DiscoveryConfig(BaseModel):
    """mDNS/Zeroconf discovery configuration."""
    service_name: str = Field(
        default='jenova-ai',
        description="mDNS service name for discovery"
    )
    port: int = Field(
        default=50051,
        ge=1024,
        le=65535,
        description="RPC server port"
    )
    ttl: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Service advertisement TTL in seconds"
    )


class ResourceSharingConfig(BaseModel):
    """Resource sharing configuration."""
    share_llm: bool = Field(
        default=True,
        description="Share LLM inference capacity with peers"
    )
    share_embeddings: bool = Field(
        default=True,
        description="Share embedding generation with peers"
    )
    share_memory: bool = Field(
        default=False,
        description="Share memory search with peers (privacy-sensitive)"
    )
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum concurrent incoming requests"
    )


class PeerSelectionConfig(BaseModel):
    """Peer selection and routing configuration."""
    strategy: DistributionStrategy = Field(
        default=DistributionStrategy.LOAD_BALANCED,
        description="Strategy for selecting peers"
    )
    timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Request timeout in milliseconds"
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed requests"
    )


class NetworkConfig(BaseModel):
    """Phase 8: Distributed computing network configuration."""
    enabled: bool = Field(
        default=False,
        description="Enable distributed computing features"
    )
    mode: NetworkMode = Field(
        default=NetworkMode.AUTO,
        description="Network operation mode"
    )
    instance_id: Optional[str] = Field(
        default=None,
        description="Unique instance identifier (auto-generated if None)"
    )
    instance_name: Optional[str] = Field(
        default=None,
        description="Human-readable instance name (auto-generated if None)"
    )
    discovery: DiscoveryConfig = Field(
        default_factory=DiscoveryConfig,
        description="Peer discovery configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Network security configuration"
    )
    resource_sharing: ResourceSharingConfig = Field(
        default_factory=ResourceSharingConfig,
        description="Resource sharing configuration"
    )
    peer_selection: PeerSelectionConfig = Field(
        default_factory=PeerSelectionConfig,
        description="Peer selection and routing configuration"
    )

    @field_validator('instance_id')
    @classmethod
    def validate_instance_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate instance ID format if provided."""
        if v is not None and len(v) > 64:
            raise ValueError("instance_id must be 64 characters or less")
        return v

    @field_validator('instance_name')
    @classmethod
    def validate_instance_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate instance name format if provided."""
        if v is not None and len(v) > 128:
            raise ValueError("instance_name must be 128 characters or less")
        return v


class JenovaConfig(BaseModel):
    """Complete JENOVA configuration with validation."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cortex: CortexConfig = Field(default_factory=CortexConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    cognitive_engine: CognitiveEngineConfig = Field(default_factory=CognitiveEngineConfig)
    rag_system: RAGSystemConfig = Field(default_factory=RAGSystemConfig)
    memory_search: MemorySearchConfig = Field(default_factory=MemorySearchConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    # Runtime fields (not in config file)
    user_data_root: Optional[str] = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def validate_config(self) -> 'JenovaConfig':
        """Cross-field validation."""
        # Adjust context size based on memory strategy
        if self.hardware.memory_strategy == MemoryStrategy.MINIMAL:
            if self.model.context_size > 2048:
                self.model.context_size = 2048

        # Disable mlock in minimal mode
        if self.hardware.memory_strategy == MemoryStrategy.MINIMAL:
            self.model.mlock = False

        # GPU layers must be 0 if CPU-only
        if self.hardware.prefer_device == DeviceType.CPU:
            self.model.gpu_layers = 0

        return self

    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = 'forbid'  # Reject unknown fields
