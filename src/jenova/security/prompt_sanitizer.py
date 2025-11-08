"""
JENOVA Prompt Sanitizer - Defense against LLM prompt injection attacks.

This module implements comprehensive protection against prompt injection, ensuring
that user input cannot manipulate the AI's behavior or access unauthorized information.

Fixes: VULN-H2 (High Severity) - LLM Prompt Injection vulnerability

Copyright (c) 2024-2025, orpheus497. All rights reserved.
Licensed under the MIT License.
"""

import re
from typing import Dict, Any, List, Optional
from string import Template
import logging

logger = logging.getLogger(__name__)


class PromptSanitizer:
    """
    Sanitizes user input and LLM prompts to prevent injection attacks.

    Features:
    - Detects common injection patterns
    - Validates and escapes special characters
    - Uses template-based prompts to prevent manipulation
    - Validates LLM output for safety
    - Configurable injection pattern detection
    """

    # Common prompt injection patterns (case-insensitive)
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above|prior)\s+(instructions?|directives?|rules?)",
        r"disregard\s+(previous|all|above|prior)\s+(instructions?|directives?|rules?)",
        r"forget\s+(previous|all|above|prior)\s+(instructions?|directives?|rules?)",
        r"new\s+(instructions?|directives?|rules?)\s*:",
        r"system\s*:\s*",
        r"assistant\s*:\s*",
        r"you\s+are\s+now\s+(a|an)\s+\w+",
        r"act\s+as\s+(a|an)\s+\w+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"roleplay\s+as",
        r"simulate\s+(being|a|an)",
        r"reveal\s+(your|the)\s+(prompt|instructions?|system|directives?)",
        r"show\s+(me\s+)?(your|the)\s+(prompt|instructions?|system)",
        r"what\s+(are|is)\s+your\s+(instructions?|system\s+prompt|directives?)",
        r"<\|system\|>",
        r"<\|assistant\|>",
        r"<\|user\|>",
    ]

    # Characters that need escaping in prompts
    DANGEROUS_CHARS = ['<', '>', '{', '}', '$', '`']

    # Maximum input length to prevent resource exhaustion
    MAX_INPUT_LENGTH = 50000  # ~50KB

    # Template for safe prompt construction
    SAFE_PROMPT_TEMPLATES = {
        "plan": Template(
            "You are JENOVA, a cognitive AI assistant.\n\n"
            "Retrieved Context:\n$context\n\n"
            "User Query: $query\n\n"
            "Task: Create a structured plan to answer the user's query based on the retrieved context."
        ),
        "generate": Template(
            "You are JENOVA, a cognitive AI assistant.\n\n"
            "Context:\n$context\n\n"
            "Plan:\n$plan\n\n"
            "User Query: $query\n\n"
            "Task: Generate a helpful, accurate response based on the context and plan."
        ),
        "analyze": Template(
            "Analyze the following content:\n\n$content\n\n"
            "Task: $task"
        ),
    }

    def __init__(
        self,
        enable_injection_detection: bool = True,
        enable_output_validation: bool = True,
        max_input_length: int = MAX_INPUT_LENGTH,
        custom_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the PromptSanitizer.

        Args:
            enable_injection_detection: Enable detection of injection patterns
            enable_output_validation: Validate LLM output for safety
            max_input_length: Maximum allowed input length
            custom_patterns: Additional regex patterns to detect
        """
        self.enable_injection_detection = enable_injection_detection
        self.enable_output_validation = enable_output_validation
        self.max_input_length = max_input_length

        # Compile injection patterns for efficiency
        self.injection_regexes = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.INJECTION_PATTERNS
        ]

        # Add custom patterns if provided
        if custom_patterns:
            for pattern in custom_patterns:
                try:
                    self.injection_regexes.append(
                        re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    )
                except re.error as e:
                    logger.warning(f"Invalid custom pattern '{pattern}': {e}")

    def sanitize_input(self, user_input: str) -> str:
        """
        Sanitize user input to prevent injection attacks.

        Args:
            user_input: Raw user input string

        Returns:
            Sanitized input string

        Raises:
            ValueError: If input exceeds max length or contains injection patterns
        """
        if not isinstance(user_input, str):
            raise TypeError(f"Expected string input, got {type(user_input).__name__}")

        # Check length limit
        if len(user_input) > self.max_input_length:
            raise ValueError(
                f"Input exceeds maximum length of {self.max_input_length} characters"
            )

        # Detect injection patterns
        if self.enable_injection_detection:
            for regex in self.injection_regexes:
                if regex.search(user_input):
                    match = regex.search(user_input).group(0)
                    logger.warning(
                        f"Potential prompt injection detected: '{match[:50]}...'"
                    )
                    raise ValueError(
                        "Input contains potentially dangerous patterns. "
                        "Please rephrase your query."
                    )

        # Escape dangerous characters (but preserve basic formatting)
        sanitized = user_input
        for char in self.DANGEROUS_CHARS:
            if char in sanitized:
                # Log but don't reject - just escape
                logger.debug(f"Escaping dangerous character '{char}' in input")
                sanitized = sanitized.replace(char, f"\\{char}")

        # Remove null bytes (can cause issues with C-based LLM backends)
        sanitized = sanitized.replace('\x00', '')

        # Normalize whitespace (prevent whitespace-based obfuscation)
        sanitized = ' '.join(sanitized.split())

        return sanitized

    def build_safe_prompt(
        self,
        template_name: str,
        **kwargs: Any
    ) -> str:
        """
        Build a safe prompt using templates to prevent manipulation.

        Args:
            template_name: Name of the template to use ('plan', 'generate', 'analyze')
            **kwargs: Variables to substitute into the template

        Returns:
            Constructed prompt string

        Raises:
            ValueError: If template not found or required variables missing
        """
        if template_name not in self.SAFE_PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(self.SAFE_PROMPT_TEMPLATES.keys())}"
            )

        template = self.SAFE_PROMPT_TEMPLATES[template_name]

        # Sanitize all template variables
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                # Don't reject context/plan - they're from trusted sources
                # But still remove null bytes
                sanitized_kwargs[key] = value.replace('\x00', '')
            else:
                sanitized_kwargs[key] = str(value).replace('\x00', '')

        try:
            return template.substitute(**sanitized_kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing required template variable: {e}. "
                f"Required: {template.template}"
            )

    def validate_output(self, llm_output: str) -> str:
        """
        Validate LLM output for safety (prevent jailbreak responses).

        Args:
            llm_output: Raw LLM output string

        Returns:
            Validated output string

        Raises:
            ValueError: If output contains unsafe content
        """
        if not self.enable_output_validation:
            return llm_output

        if not isinstance(llm_output, str):
            raise TypeError(
                f"Expected string output, got {type(llm_output).__name__}"
            )

        # Check for signs of successful jailbreak
        jailbreak_indicators = [
            r"I('m|\s+am)\s+(sorry|unable|cannot|can't).+?(follow|comply|assist)",
            r"(OpenAI|Anthropic|Google|Meta)\s+policy",
            r"I\s+don'?t\s+have\s+access\s+to",
            r"I\s+(cannot|can't|won't)\s+(assist|help)\s+with",
        ]

        # These are actually GOOD patterns (model refusing unsafe requests)
        # We log them but don't reject - this is desired behavior
        for pattern in jailbreak_indicators:
            if re.search(pattern, llm_output, re.IGNORECASE):
                logger.debug("LLM output contains safety refusal - this is expected")
                break

        # Check for leaked system information
        leaked_info_patterns = [
            r"<\|system\|>",
            r"<\|assistant\|>",
            r"<\|user\|>",
            r"System\s+prompt:",
            r"My\s+instructions\s+are:",
        ]

        for pattern in leaked_info_patterns:
            if re.search(pattern, llm_output, re.IGNORECASE):
                logger.error(
                    f"LLM output may contain leaked system information: {pattern}"
                )
                raise ValueError(
                    "Generated response contains unsafe content. "
                    "Please try rephrasing your query."
                )

        return llm_output

    def check_injection_risk(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for injection risk without rejecting it.

        Useful for logging/monitoring without blocking legitimate queries.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with risk analysis:
            {
                'has_risk': bool,
                'risk_score': float (0.0 to 1.0),
                'matched_patterns': List[str],
                'dangerous_chars': List[str]
            }
        """
        matched_patterns = []
        for regex in self.injection_regexes:
            matches = regex.findall(text)
            if matches:
                matched_patterns.extend(matches)

        dangerous_chars = [char for char in self.DANGEROUS_CHARS if char in text]

        # Calculate risk score
        risk_score = 0.0
        if matched_patterns:
            risk_score += 0.5 * min(len(matched_patterns) / 3.0, 1.0)
        if dangerous_chars:
            risk_score += 0.3 * min(len(dangerous_chars) / 4.0, 1.0)
        if len(text) > self.max_input_length * 0.8:
            risk_score += 0.2

        risk_score = min(risk_score, 1.0)

        return {
            'has_risk': risk_score > 0.3,
            'risk_score': risk_score,
            'matched_patterns': matched_patterns[:5],  # First 5 only
            'dangerous_chars': dangerous_chars,
        }


# Singleton instance for convenience
_default_sanitizer: Optional[PromptSanitizer] = None


def get_default_sanitizer() -> PromptSanitizer:
    """Get the default singleton PromptSanitizer instance."""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = PromptSanitizer()
    return _default_sanitizer
