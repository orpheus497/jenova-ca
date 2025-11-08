# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.

"""Infrastructure layer for The JENOVA Cognitive Architecture."""

from jenova.infrastructure.error_handler import ErrorHandler, ErrorSeverity
from jenova.infrastructure.timeout_manager import (
    timeout,
    with_timeout,
    with_short_timeout,
    with_medium_timeout,
    with_long_timeout,
    TimeoutError,
)
from jenova.infrastructure.health_monitor import (
    HealthMonitor,
    HealthStatus,
    SystemHealth,
)
from jenova.infrastructure.data_validator import (
    DataValidator,
    MemoryEntry,
    EpisodicMemoryEntry,
    SemanticMemoryEntry,
    ProceduralMemoryEntry,
    InsightEntry,
    AssumptionEntry,
    ToolCall,
    ConversationTurn,
    SearchQuery,
    SearchResult,
    PerformanceMetric,
)
from jenova.infrastructure.file_manager import FileManager
from jenova.infrastructure.metrics_collector import MetricsCollector, MetricStats
from jenova.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
    get_registry,
)

__all__ = [
    # Error handling
    "ErrorHandler",
    "ErrorSeverity",
    # Timeouts
    "timeout",
    "with_timeout",
    "with_short_timeout",
    "with_medium_timeout",
    "with_long_timeout",
    "TimeoutError",
    # Health monitoring
    "HealthMonitor",
    "HealthStatus",
    "SystemHealth",
    # Data validation
    "DataValidator",
    "MemoryEntry",
    "EpisodicMemoryEntry",
    "SemanticMemoryEntry",
    "ProceduralMemoryEntry",
    "InsightEntry",
    "AssumptionEntry",
    "ToolCall",
    "ConversationTurn",
    "SearchQuery",
    "SearchResult",
    "PerformanceMetric",
    # File operations
    "FileManager",
    # Metrics
    "MetricsCollector",
    "MetricStats",
    # Circuit Breaker (Phase 20)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerMetrics",
    "CircuitBreakerRegistry",
    "CircuitState",
    "circuit_breaker",
    "get_registry",
]
