##Script function and purpose: Utility package initialization - exposes logging, migration, validation, sanitization, and error utilities
"""Utility modules for JENOVA."""

from jenova.utils.migrations import (
    SCHEMA_VERSION,
    load_json_with_migration,
    save_json_atomic,
)
from jenova.utils.validation import (
    validate_username,
    validate_topic,
    validate_path_component,
    validate_path_within_base,
    validate_username_and_topic,
)
from jenova.utils.sanitization import (
    sanitize_for_prompt,
    sanitize_user_query,
    sanitize_for_context,
    SanitizationConfig,
)
from jenova.utils.json_safe import (
    safe_json_loads,
    extract_json_from_response,
    JSONSizeError,
    MAX_JSON_SIZE,
    MAX_JSON_DEPTH,
)
from jenova.utils.errors import (
    sanitize_path_for_error,
    sanitize_error_message,
    safe_error_with_path,
)
from jenova.utils.cache import (
    TTLCache,
    CacheEntry,
    CacheStats,
    CacheManager,
)
from jenova.utils.performance import (
    PerformanceMonitor,
    TimingStats,
    timed,
    log_slow,
)
from jenova.utils.grammar import (
    GrammarLoader,
    BuiltinGrammars,
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
