##Script function and purpose: Input sanitization utilities for LLM prompts and user content
"""
Input Sanitization Utilities

Provides sanitization functions to prevent prompt injection attacks
and ensure safe content for LLM processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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


##Class purpose: Result of sanitization with metadata
@dataclass
class SanitizationResult:
    """Result of sanitization including metadata about matched patterns."""

    text: str
    """Sanitized text."""

    flagged_injection: bool = False
    """Whether any injection pattern was matched."""

    matched_patterns: list[str] = field(default_factory=list)
    """Names/indices of patterns that matched."""


##Update: Enhanced injection patterns to capture modern prompt injection vectors (ISSUE-005)
INJECTION_PATTERNS = [
    r"(?i)(ignore|forget|disregard|override|skip)\s+(previous|above|all|existing|following)\s+(instructions|rules|directives|guidelines|prompts|context)",
    r"(?i)(new\s+)?(instructions?|rules?|directives?)\s*[:\-]",
    r"(?i)(new|special|override|custom)\s+mode\s*[:\-]",
    r"(?i)system\s+(prompt|message|instruction)\s*[:\-]",
    r"(?i)you\s+are\s+now(?:\s+(?:a|an))?",
    r"(?i)disregard\s+the\s+above",
    r"(?i)forget\s+everything",
    r"(?i)(?:ignore|override)\s+.{0,50}?(pretend|act|roleplay|simulate)\s+(to\s+be|as\s+if|as)\b",
    r"(?i)(Do\s+Anything\s+Now|\bDAN\s+mode\b|\benable\s+DAN\b|\bactivate\s+DAN\b)",
    r"(?i)(enable|activate|enter|switch\s+to)\s+Developer\s+Mode",
    r"(?i)(enable|activate|enter|switch\s+to)\s+Debug\s+Mode",
    r"(?i)\bJailbreak\b\s*(mode|prompt|this|the|system|assistant)",
    r"(?i)\b(unfiltered|uncensored)\b\s*(?:mode|response|output|filter|settings)\b",
    r"(?i)Stay\s+in\s+character",
    r"(?i)Assistant\s+Settings\b",
    r"(?i)Override\s+Safety",
    r"(?i)Kernel\s+Prompt",
    r"(?i)\[(SYSTEM|ADMIN|USER)\](?=\s*[:\-])",
    ##Refactor: Catch bare role-tag prefixes without brackets (D3-2026-02-14T10:24:30Z)
    r"(?im)(^|\n)\s*(SYSTEM|ADMIN|USER)\s*:",
]
"""List of regex patterns to detect prompt injection attempts."""


##Function purpose: Sanitize user content for LLM prompts and return metadata
def sanitize_for_prompt(
    content: str,
    config: SanitizationConfig | None = None,
) -> SanitizationResult:
    """Sanitize user content for LLM prompts.

    Detects injection patterns, limits length, and ensures content
    is safe for inclusion in LLM prompts. Returns metadata about
    matched patterns so callers can decide how to handle borderline cases.

    Args:
        content: Raw user content to sanitize
        config: Sanitization configuration (uses defaults if None)

    Returns:
        SanitizationResult with sanitized text and injection metadata
    """
    ##Step purpose: Use default config if not provided
    if config is None:
        config = SanitizationConfig()

    ##Condition purpose: Handle empty content
    if not content:
        return SanitizationResult(text="")

    ##Sec: Validate input length BEFORE regex matching to prevent ReDoS attacks (P1-001 Daedelus audit)
    ##Step purpose: Reject inputs exceeding maximum length before any regex operations
    if len(content) > config.max_user_input_length:
        raise ValueError(
            f"Input too long: {len(content)} characters > {config.max_user_input_length} maximum"
        )

    ##Step purpose: Limit length first to prevent processing large inputs
    sanitized = content[: config.max_content_length]

    ##Step purpose: Remove control characters (except newline and tab) before pattern matching
    sanitized = "".join(char for char in sanitized if ord(char) >= 32 or char in "\n\t")

    ##Step purpose: Replace code block markers that could break JSON/prompts
    sanitized = sanitized.replace("```", "'''")
    sanitized = sanitized.replace("`", "'")

    ##Sec: Apply regex patterns after length validation to prevent ReDoS (P1-001 Daedelus audit)
    ##Step purpose: Detect and replace injection patterns (case-insensitive)
    flagged = False
    matched: list[str] = []
    for i, pattern in enumerate(INJECTION_PATTERNS):
        sanitized, count = re.subn(pattern, "[REDACTED]", sanitized)
        if count > 0:
            flagged = True
            matched.append(f"INJECTION_PATTERN_{i}")

    return SanitizationResult(text=sanitized, flagged_injection=flagged, matched_patterns=matched)


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

    return sanitize_for_prompt(query, config).text


##Function purpose: Sanitize content for context inclusion
def sanitize_for_context(content: str) -> str:
    """Sanitize content for inclusion in context.

    Uses standard content length limit for context snippets.

    Args:
        content: Content to sanitize for context

    Returns:
        Sanitized content safe for context
    """
    return sanitize_for_prompt(content).text
