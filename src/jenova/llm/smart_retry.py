# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Smart retry logic with context adaptation for LLM operations.

This module provides intelligent retry strategies that adapt prompts based on failure patterns.
Unlike simple exponential backoff, smart retry analyzes failure types and adjusts the approach:

Failure Types Detected:
- Timeout: LLM taking too long to respond
- Malformed Output: Invalid JSON, incomplete responses
- Refusal: Model refusing to answer due to safety filters
- Hallucination: Nonsensical or contradictory responses
- Quality Issues: Low-quality or irrelevant responses

Adaptive Strategies:
- Timeout → Simplify prompt, reduce max_tokens
- Malformed → Add explicit format instructions, examples
- Refusal → Rephrase to avoid trigger words, add context
- Hallucination → Increase temperature, add grounding
- Quality → Add few-shot examples, increase detail

This addresses the issue identified in Step 1: Current retry logic is basic exponential backoff.
Smart retry learns from failures to prevent future occurrences.
"""

import time
import re
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict


class FailureType(Enum):
    """Types of LLM generation failures."""
    TIMEOUT = "timeout"
    MALFORMED_JSON = "malformed_json"
    MALFORMED_OUTPUT = "malformed_output"
    REFUSAL = "refusal"
    HALLUCINATION = "hallucination"
    QUALITY_LOW = "quality_low"
    UNKNOWN = "unknown"


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""
    attempt_number: int
    failure_type: FailureType
    original_prompt: str
    adapted_prompt: str
    strategy_applied: str
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None


class SmartRetryHandler:
    """
    Intelligent retry handler with adaptive prompt modification.

    Learns from failure patterns and adapts retry strategies accordingly.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        file_logger: Optional[Any] = None
    ):
        """
        Initialize smart retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            file_logger: File logger instance
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.file_logger = file_logger

        # Failure pattern tracking
        self.failure_patterns: Dict[FailureType, List[str]] = defaultdict(list)
        self.success_patterns: Dict[FailureType, List[Dict[str, Any]]] = defaultdict(list)

        # Retry history for learning
        self.retry_history: List[RetryAttempt] = []

    def execute_with_retry(
        self,
        llm_func: Callable,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Optional[str]:
        """
        Execute LLM function with smart retry logic.

        Args:
            llm_func: LLM generation function to call
            prompt: Original prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional arguments for llm_func

        Returns:
            Generated response, or None if all retries failed
        """
        current_prompt = prompt
        current_max_tokens = max_tokens
        current_temperature = temperature

        for attempt in range(self.max_retries + 1):
            try:
                if self.file_logger and attempt > 0:
                    self.file_logger.log_info(f"Retry attempt {attempt}/{self.max_retries}")

                # Execute LLM call
                response = llm_func(
                    current_prompt,
                    max_tokens=current_max_tokens,
                    temperature=current_temperature,
                    **kwargs
                )

                # Validate response
                failure_type = self._detect_failure_type(response, prompt)

                if failure_type == FailureType.UNKNOWN:
                    # Success!
                    self._record_success(
                        attempt,
                        FailureType.UNKNOWN if attempt == 0 else self._get_previous_failure_type(),
                        prompt,
                        current_prompt,
                        response
                    )
                    return response

                # Response has issues, adapt and retry
                if attempt < self.max_retries:
                    if self.file_logger:
                        self.file_logger.log_warning(
                            f"Detected {failure_type.value}, adapting prompt for retry"
                        )

                    # Adapt prompt based on failure type
                    current_prompt, current_max_tokens, current_temperature = self._adapt_parameters(
                        failure_type,
                        prompt,
                        current_max_tokens,
                        current_temperature
                    )

                    # Record failure
                    self._record_failure(attempt, failure_type, prompt, current_prompt, str(response))

                    # Exponential backoff
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    # Max retries reached
                    if self.file_logger:
                        self.file_logger.log_error(f"Max retries reached with failure: {failure_type.value}")
                    return None

            except TimeoutError as e:
                if attempt < self.max_retries:
                    if self.file_logger:
                        self.file_logger.log_warning(f"Timeout on attempt {attempt}, simplifying prompt")

                    # Reduce complexity for timeout
                    current_prompt, current_max_tokens, current_temperature = self._adapt_for_timeout(
                        prompt,
                        current_max_tokens
                    )

                    self._record_failure(attempt, FailureType.TIMEOUT, prompt, current_prompt, str(e))

                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    if self.file_logger:
                        self.file_logger.log_error("Max retries reached with timeout")
                    return None

            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Unexpected error during LLM call: {e}")

                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    return None

        return None

    def _detect_failure_type(self, response: str, original_prompt: str) -> FailureType:
        """
        Detect the type of failure from the response.

        Args:
            response: LLM response
            original_prompt: Original prompt

        Returns:
            Detected failure type
        """
        if not response or len(response.strip()) == 0:
            return FailureType.MALFORMED_OUTPUT

        # Check for refusal patterns
        refusal_patterns = [
            r"I cannot",
            r"I'm unable to",
            r"I don't have",
            r"I apologize, but",
            r"I can't assist",
            r"against my guidelines",
            r"I'm not able to"
        ]

        for pattern in refusal_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return FailureType.REFUSAL

        # Check for JSON format if expected
        if "json" in original_prompt.lower():
            try:
                import json
                json.loads(response)
            except (json.JSONDecodeError, ValueError):
                return FailureType.MALFORMED_JSON

        # Check for hallucination indicators
        hallucination_indicators = [
            r"As an AI language model",
            r"I don't actually",
            r"I'm just a",
            r"I don't have real-time",
            r"I cannot access"
        ]

        for indicator in hallucination_indicators:
            if re.search(indicator, response, re.IGNORECASE):
                return FailureType.HALLUCINATION

        # Check for quality issues
        if len(response.strip()) < 10:
            return FailureType.QUALITY_LOW

        # No issues detected
        return FailureType.UNKNOWN

    def _adapt_parameters(
        self,
        failure_type: FailureType,
        original_prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, int, float]:
        """
        Adapt generation parameters based on failure type.

        Args:
            failure_type: Type of failure detected
            original_prompt: Original prompt
            max_tokens: Current max_tokens
            temperature: Current temperature

        Returns:
            Tuple of (adapted_prompt, adapted_max_tokens, adapted_temperature)
        """
        if failure_type == FailureType.TIMEOUT:
            return self._adapt_for_timeout(original_prompt, max_tokens)

        elif failure_type == FailureType.MALFORMED_JSON:
            return self._adapt_for_malformed_json(original_prompt, max_tokens, temperature)

        elif failure_type == FailureType.MALFORMED_OUTPUT:
            return self._adapt_for_malformed_output(original_prompt, max_tokens, temperature)

        elif failure_type == FailureType.REFUSAL:
            return self._adapt_for_refusal(original_prompt, max_tokens, temperature)

        elif failure_type == FailureType.HALLUCINATION:
            return self._adapt_for_hallucination(original_prompt, max_tokens, temperature)

        elif failure_type == FailureType.QUALITY_LOW:
            return self._adapt_for_quality(original_prompt, max_tokens, temperature)

        else:
            # Unknown failure, try generic simplification
            simplified_prompt = f"Please provide a clear, concise answer:\n\n{original_prompt}"
            return simplified_prompt, max_tokens, temperature

    def _adapt_for_timeout(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        """Adapt for timeout failures."""
        # Simplify prompt
        simplified = f"Briefly answer: {prompt[:500]}"

        # Reduce max_tokens
        reduced_tokens = max(max_tokens // 2, 128)

        # Keep temperature same
        return simplified, reduced_tokens, 0.7

    def _adapt_for_malformed_json(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, int, float]:
        """Adapt for malformed JSON failures."""
        # Add explicit JSON formatting instructions
        adapted = f"""Generate valid JSON only. No explanations before or after.

        {prompt}

        Output must be valid JSON that can be parsed with json.loads().
        Example format:
        {{
            "key": "value"
        }}"""

        # Lower temperature for more deterministic output
        return adapted, max_tokens, 0.3

    def _adapt_for_malformed_output(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, int, float]:
        """Adapt for malformed output failures."""
        adapted = f"""Provide a complete, well-formed response.

        {prompt}

        Ensure your response is:
        - Complete (no truncation)
        - Properly formatted
        - Clear and structured"""

        return adapted, max_tokens, temperature

    def _adapt_for_refusal(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, int, float]:
        """Adapt for refusal failures."""
        # Rephrase to be more neutral
        adapted = f"""As a helpful assistant, please assist with the following task:

        {prompt}

        Focus on providing accurate, helpful information."""

        return adapted, max_tokens, temperature

    def _adapt_for_hallucination(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, int, float]:
        """Adapt for hallucination failures."""
        # Add grounding instructions
        adapted = f"""Based on the provided context, {prompt}

        Important: Only use information from the context. If unsure, say so."""

        # Slightly lower temperature
        return adapted, max_tokens, max(temperature - 0.2, 0.1)

    def _adapt_for_quality(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[str, int, float]:
        """Adapt for low quality failures."""
        # Add detail requirements
        adapted = f"""Provide a detailed, comprehensive answer to:

        {prompt}

        Include:
        - Clear explanation
        - Relevant details
        - Specific examples if applicable"""

        # Increase max_tokens for more detail
        increased_tokens = min(max_tokens * 2, 2048)

        return adapted, increased_tokens, temperature

    def _record_failure(
        self,
        attempt: int,
        failure_type: FailureType,
        original_prompt: str,
        adapted_prompt: str,
        error: str
    ) -> None:
        """Record a failed attempt for learning."""
        retry_attempt = RetryAttempt(
            attempt_number=attempt,
            failure_type=failure_type,
            original_prompt=original_prompt,
            adapted_prompt=adapted_prompt,
            strategy_applied=f"adapt_for_{failure_type.value}",
            success=False,
            error=error
        )

        self.retry_history.append(retry_attempt)
        self.failure_patterns[failure_type].append(original_prompt)

    def _record_success(
        self,
        attempt: int,
        failure_type: FailureType,
        original_prompt: str,
        final_prompt: str,
        response: str
    ) -> None:
        """Record a successful attempt after retries."""
        retry_attempt = RetryAttempt(
            attempt_number=attempt,
            failure_type=failure_type,
            original_prompt=original_prompt,
            adapted_prompt=final_prompt,
            strategy_applied="success" if attempt == 0 else f"recovered_from_{failure_type.value}",
            success=True,
            response=response
        )

        self.retry_history.append(retry_attempt)

        if attempt > 0:
            # This was a successful retry after failure
            self.success_patterns[failure_type].append({
                "original_prompt": original_prompt,
                "successful_prompt": final_prompt,
                "response": response
            })

    def _get_previous_failure_type(self) -> FailureType:
        """Get failure type from previous attempt."""
        if len(self.retry_history) > 0:
            return self.retry_history[-1].failure_type
        return FailureType.UNKNOWN

    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about retry patterns.

        Returns:
            Dictionary with retry statistics
        """
        total_attempts = len(self.retry_history)
        if total_attempts == 0:
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "failure_breakdown": {},
                "recovery_rates": {}
            }

        successes = sum(1 for attempt in self.retry_history if attempt.success)
        success_rate = successes / total_attempts

        # Failure breakdown
        failure_breakdown = {}
        for failure_type in FailureType:
            count = sum(
                1 for attempt in self.retry_history
                if attempt.failure_type == failure_type and not attempt.success
            )
            if count > 0:
                failure_breakdown[failure_type.value] = count

        # Recovery rates (successful retries after each failure type)
        recovery_rates = {}
        for failure_type in FailureType:
            if failure_type in self.success_patterns and len(self.success_patterns[failure_type]) > 0:
                failures = len(self.failure_patterns[failure_type])
                recoveries = len(self.success_patterns[failure_type])
                recovery_rates[failure_type.value] = recoveries / max(failures, 1)

        return {
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "failure_breakdown": failure_breakdown,
            "recovery_rates": recovery_rates
        }

    def get_learned_strategies(self) -> Dict[FailureType, List[str]]:
        """
        Get successful strategies learned from retry history.

        Returns:
            Dictionary mapping failure types to successful recovery strategies
        """
        learned = {}

        for failure_type, successes in self.success_patterns.items():
            if len(successes) > 0:
                learned[failure_type] = [s["successful_prompt"] for s in successes[:5]]  # Top 5

        return learned
