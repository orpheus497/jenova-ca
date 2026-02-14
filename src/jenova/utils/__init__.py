##Script function and purpose: Utility package initialization - exposes logging, migration, validation, sanitization, and error utilities
"""Utility modules for JENOVA."""

from jenova.utils.cache import (
    CacheEntry,
    CacheManager,
    CacheStats,
    TTLCache,
)
from jenova.utils.errors import (
    safe_error_with_path,
    sanitize_error_message,
    sanitize_path_for_error,
)
from jenova.utils.grammar import (
    BuiltinGrammars,
    GrammarLoader,
)
from jenova.utils.json_safe import (
    MAX_JSON_DEPTH,
    MAX_JSON_SIZE,
    JSONSizeError,
    extract_json_from_response,
    safe_json_loads,
)
from jenova.utils.migrations import (
    SCHEMA_VERSION,
    load_json_with_migration,
    save_json_atomic,
)
from jenova.utils.performance import (
    PerformanceMonitor,
    TimingStats,
    log_slow,
    timed,
)
from jenova.utils.sanitization import (
    SanitizationConfig,
    SanitizationResult,
    sanitize_for_context,
    sanitize_for_prompt,
    sanitize_user_query,
)
from jenova.utils.validation import (
    validate_path_component,
    validate_path_within_base,
    validate_topic,
    validate_username,
    validate_username_and_topic,
)

__all__ = [
    # Migration utilities
    "SCHEMA_VERSION",
    "load_json_with_migration",
    "save_json_atomic",
    # Validation utilities
    "validate_username",
    "validate_topic",
    "validate_path_component",
    "validate_path_within_base",
    "validate_username_and_topic",
    # Sanitization utilities
    "sanitize_for_prompt",
    "sanitize_user_query",
    "sanitize_for_context",
    "SanitizationConfig",
    "SanitizationResult",
    # JSON utilities
    "safe_json_loads",
    "extract_json_from_response",
    "JSONSizeError",
    "MAX_JSON_SIZE",
    "MAX_JSON_DEPTH",
    # Error utilities
    "sanitize_path_for_error",
    "sanitize_error_message",
    "safe_error_with_path",
    # Cache utilities
    "TTLCache",
    "CacheEntry",
    "CacheStats",
    "CacheManager",
    # Performance utilities
    "PerformanceMonitor",
    "TimingStats",
    "timed",
    "log_slow",
    # Grammar utilities
    "GrammarLoader",
    "BuiltinGrammars",
]
