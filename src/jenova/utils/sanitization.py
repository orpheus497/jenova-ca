##Script function and purpose: Input sanitization utilities for LLM prompts and user content
"""
Input Sanitization Utilities

Provides sanitization functions to prevent prompt injection attacks
and ensure safe content for LLM processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

##Step purpose: Define size limits
MAX_USER_INPUT_LENGTH = 10000
"""Maximum length for user input queries."""

MAX_CONTENT_LENGTH = 500
"""Maximum length for content snippets in prompts."""


##Class purpose: Configuration for input sanitization
@dataclass
class SanitizationConfig:
    """Configuration for input sanitization."""

    max_user_input_length: int = MAX_USER_INPUT_LENGTH
    """Maximum length for user input."""

    max_content_length: int = MAX_CONTENT_LENGTH
    """Maximum length for content snippets."""

    enable_strict_mode: bool = True
    """Whether to enable strict sanitization mode."""


##Step purpose: Define injection patterns (case-insensitive)
INJECTION_PATTERNS = [
    r"(?i)(ignore|forget|disregard)\s+(previous|above|all|instructions)",
    r"(?i)(new\s+)?(instructions?|rules?|directives?)\s*:",
    r"(?i)system\s*:\s*",
    r"(?i)you\s+are\s+now",
    r"(?i)override\s+",
    r"(?i)disregard\s+the\s+above",
    r"(?i)forget\s+everything",
    r"(?i)pretend\s+you\s+are",
    r"(?i)act\s+as\s+if",
    r"(?i)roleplay\s+as",
]
"""List of regex patterns to detect prompt injection attempts."""


##Function purpose: Sanitize user content for LLM prompts
def sanitize_for_prompt(
    content: str,
    config: SanitizationConfig | None = None,
) -> str:
    """Sanitize user content for LLM prompts.

    Removes injection patterns, limits length, and ensures content
    is safe for inclusion in LLM prompts.

    Args:
        content: Raw user content to sanitize
        config: Sanitization configuration (uses defaults if None)

    Returns:
        Sanitized content safe for prompts
    """
    ##Step purpose: Use default config if not provided
    if config is None:
        config = SanitizationConfig()

    ##Condition purpose: Handle empty content
    if not content:
        return ""

    ##Sec: Validate input length BEFORE regex matching to prevent ReDoS
    ##     attacks (P1-001 Daedelus audit)
    ##Step purpose: Reject inputs exceeding maximum length before any regex operations
    if len(content) > config.max_user_input_length:
        raise ValueError(
            f"Input too long: {len(content)} characters > {config.max_user_input_length} maximum"
        )

    ##Step purpose: Limit length first to prevent processing large inputs
    sanitized = content[: config.max_content_length]

    ##Step purpose: Replace code block markers that could break JSON/prompts
    sanitized = sanitized.replace("```", "'''")
    sanitized = sanitized.replace("`", "'")

    ##Sec: Apply regex patterns after length validation to prevent ReDoS (P1-001 Daedelus audit)
    ##Step purpose: Remove injection patterns (case-insensitive)
    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized)

    ##Step purpose: Remove control characters (except newline and tab)
    sanitized = "".join(char for char in sanitized if ord(char) >= 32 or char in "\n\t")

    return sanitized


##Function purpose: Sanitize user query for processing
def sanitize_user_query(query: str) -> str:
    """Sanitize user query for processing.

    Convenience function that uses user input length limit
    instead of content length limit.

    Args:
        query: User query to sanitize

    Returns:
        Sanitized query safe for processing
    """
    ##Step purpose: Create config with user input length
    config = SanitizationConfig()
    config.max_content_length = config.max_user_input_length

    return sanitize_for_prompt(query, config)


##Function purpose: Sanitize content for context inclusion
def sanitize_for_context(content: str) -> str:
    """Sanitize content for inclusion in context.

    Uses standard content length limit for context snippets.

    Args:
        content: Content to sanitize for context

    Returns:
        Sanitized content safe for context
    """
    return sanitize_for_prompt(content)
