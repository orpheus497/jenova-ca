##Script function and purpose: Defines typed exception hierarchy for all JENOVA errors
"""
JENOVA Exception Hierarchy

All custom exceptions for the JENOVA cognitive architecture.
Typed exceptions enable explicit error handling and clear error paths.
"""


##Class purpose: Base exception for all JENOVA errors
class JenovaError(Exception):
    """Base exception for all JENOVA errors."""
    pass


##Class purpose: Configuration is invalid or missing
class ConfigError(JenovaError):
    """Configuration is invalid or missing."""
    pass


##Class purpose: Configuration file not found
class ConfigNotFoundError(ConfigError):
    """Configuration file does not exist at specified path."""
    
    ##Method purpose: Initialize with the missing path
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Config file not found: {path}")


##Class purpose: Configuration file has invalid YAML syntax
class ConfigParseError(ConfigError):
    """Configuration file has invalid YAML syntax."""
    
    ##Method purpose: Initialize with path and parse error details
    def __init__(self, path: str, error: str) -> None:
        self.path = path
        self.error = error
        super().__init__(f"Invalid YAML in {path}: {error}")


##Class purpose: Configuration values fail validation
class ConfigValidationError(ConfigError):
    """Configuration values fail Pydantic validation."""
    
    ##Method purpose: Initialize with validation error details
    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        error_msgs = "; ".join(f"{e.get('loc', '?')}: {e.get('msg', '?')}" for e in errors)
        super().__init__(f"Config validation failed: {error_msgs}")


##Class purpose: Base exception for LLM operations
class LLMError(JenovaError):
    """LLM operation failed."""
    pass


##Class purpose: Failed to load LLM model file
class LLMLoadError(LLMError):
    """Failed to load LLM model."""
    
    ##Method purpose: Initialize with model path and error details
    def __init__(self, model_path: str, error: str) -> None:
        self.model_path = model_path
        self.error = error
        super().__init__(f"Failed to load model {model_path}: {error}")


##Class purpose: LLM generation timed out or failed
class LLMGenerationError(LLMError):
    """LLM generation failed."""
    
    ##Method purpose: Initialize with prompt context and error
    def __init__(self, message: str, prompt_preview: str | None = None) -> None:
        self.prompt_preview = prompt_preview
        super().__init__(message)


##Class purpose: Failed to parse LLM output to expected format
class LLMParseError(LLMError):
    """Failed to parse LLM output."""
    
    ##Method purpose: Initialize with raw output and parse error
    def __init__(self, raw_output: str, parse_error: str) -> None:
        self.raw_output = raw_output
        self.parse_error = parse_error
        super().__init__(f"Parse failed: {parse_error}")


##Sec: Renamed from MemoryError to avoid shadowing Python builtin MemoryError (P0-001)
##Class purpose: Base exception for memory operations
class JenovaMemoryError(JenovaError):
    """Memory system operation failed."""
    pass


##Class purpose: Memory storage operation failed
class MemoryStoreError(JenovaMemoryError):
    """Failed to store content in memory."""
    
    ##Method purpose: Initialize with content preview and error
    def __init__(self, content_preview: str, error: str) -> None:
        self.content_preview = content_preview[:100]
        self.error = error
        super().__init__(f"Failed to store: {error}")


##Class purpose: Memory search operation failed
class MemorySearchError(JenovaMemoryError):
    """Failed to search memory."""
    
    ##Method purpose: Initialize with query and error
    def __init__(self, query: str, error: str) -> None:
        self.query = query
        self.error = error
        super().__init__(f"Search failed for '{query}': {error}")


##Class purpose: Base exception for graph operations
class GraphError(JenovaError):
    """Graph operation failed."""
    pass


##Class purpose: Referenced node does not exist
class NodeNotFoundError(GraphError):
    """Node not found in graph."""
    
    ##Method purpose: Initialize with missing node ID
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


##Class purpose: Edge operation failed
class EdgeError(GraphError):
    """Edge operation failed."""
    pass


##Class purpose: Graph pruning operation failed
class GraphPruneError(GraphError):
    """Graph pruning operation failed."""
    
    ##Method purpose: Initialize with pruning context and error
    def __init__(self, nodes_attempted: int, error: str) -> None:
        self.nodes_attempted = nodes_attempted
        self.error = error
        super().__init__(f"Pruning failed ({nodes_attempted} nodes): {error}")


##Class purpose: Graph clustering operation failed
class GraphClusterError(GraphError):
    """Graph clustering operation failed."""
    
    ##Method purpose: Initialize with cluster context and error
    def __init__(self, node_count: int, error: str) -> None:
        self.node_count = node_count
        self.error = error
        super().__init__(f"Clustering failed ({node_count} nodes): {error}")


##Class purpose: Graph analysis operation failed
class GraphAnalysisError(GraphError):
    """Graph analysis operation (emotion, contradiction, etc.) failed."""
    
    ##Method purpose: Initialize with analysis type and error
    def __init__(self, analysis_type: str, error: str) -> None:
        self.analysis_type = analysis_type
        self.error = error
        super().__init__(f"{analysis_type} analysis failed: {error}")


##Class purpose: Base exception for data migration
class MigrationError(JenovaError):
    """Data migration failed."""
    pass


##Class purpose: Data schema version is newer than supported
class SchemaVersionError(MigrationError):
    """Schema version is unsupported."""
    
    ##Method purpose: Initialize with found and supported versions
    def __init__(self, found: int, supported: int) -> None:
        self.found = found
        self.supported = supported
        super().__init__(
            f"Schema version {found} is newer than supported version {supported}. "
            "Please update JENOVA."
        )


##Class purpose: Migration from one version to another failed
class MigrationFailedError(MigrationError):
    """Migration between versions failed."""
    
    ##Method purpose: Initialize with version info and error
    def __init__(self, from_version: int, to_version: int, error: str) -> None:
        self.from_version = from_version
        self.to_version = to_version
        self.error = error
        super().__init__(
            f"Migration from v{from_version} to v{to_version} failed: {error}"
        )


##Class purpose: Base exception for embedding operations
class EmbeddingError(JenovaError):
    """Embedding operation failed."""
    pass


##Class purpose: Failed to load embedding model
class EmbeddingLoadError(EmbeddingError):
    """Failed to load embedding model."""
    
    ##Method purpose: Initialize with model info and error
    def __init__(self, model_name: str, error: str) -> None:
        self.model_name = model_name
        self.error = error
        super().__init__(f"Failed to load embedding model {model_name}: {error}")


##Class purpose: Base exception for UI operations
class UIError(JenovaError):
    """UI operation failed."""
    pass


##Class purpose: Base exception for assumption operations
class AssumptionError(JenovaError):
    """Assumption operation failed."""
    pass


##Class purpose: Assumption with duplicate content already exists
class AssumptionDuplicateError(AssumptionError):
    """Assumption with same content already exists."""
    
    ##Method purpose: Initialize with content and existing status
    def __init__(self, content: str, existing_status: str) -> None:
        self.content = content
        self.existing_status = existing_status
        super().__init__(
            f"Assumption already exists with status '{existing_status}': {content[:50]}..."
        )


##Class purpose: Assumption not found for update or resolution
class AssumptionNotFoundError(AssumptionError):
    """Assumption not found."""
    
    ##Method purpose: Initialize with content that was not found
    def __init__(self, content: str) -> None:
        self.content = content
        super().__init__(f"Assumption not found: {content[:50]}...")


##Class purpose: Base exception for insight operations
class InsightError(JenovaError):
    """Insight operation failed."""
    pass


##Class purpose: Failed to save insight to storage
class InsightSaveError(InsightError):
    """Failed to save insight."""
    
    ##Method purpose: Initialize with content and error
    def __init__(self, content: str, error: str) -> None:
        self.content = content[:100]
        self.error = error
        super().__init__(f"Failed to save insight: {error}")


##Class purpose: Base exception for concern operations
class ConcernError(JenovaError):
    """Concern operation failed."""
    pass


##Class purpose: Base exception for integration operations
class IntegrationError(JenovaError):
    """Integration operation failed."""
    pass


##Class purpose: Knowledge consistency check failed
class ConsistencyError(IntegrationError):
    """Knowledge consistency check failed."""
    
    ##Method purpose: Initialize with details about inconsistency
    def __init__(self, message: str, gaps: int = 0, duplications: int = 0) -> None:
        self.gaps = gaps
        self.duplications = duplications
        super().__init__(message)


##Class purpose: Base exception for scheduler operations
class SchedulerError(JenovaError):
    """Scheduler operation failed."""
    pass


##Class purpose: Base exception for grammar operations
class GrammarError(JenovaError):
    """Grammar loading or parsing failed."""
    pass


##Class purpose: Base exception for tool operations
class ToolError(JenovaError):
    """Tool execution failed."""
    pass


##Class purpose: Base exception for proactive engine operations
class ProactiveError(JenovaError):
    """Proactive suggestion generation failed."""
    pass
