##Script function and purpose: Safe JSON parsing utilities with size and depth limits
"""
Safe JSON Parsing Utilities

Provides safe JSON parsing functions with size and depth limits
to prevent denial of service attacks via resource exhaustion.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

##Step purpose: Define size and depth limits
MAX_JSON_SIZE = 1024 * 1024  # 1MB
"""Maximum JSON size in bytes."""

MAX_JSON_DEPTH = 100
"""Maximum JSON nesting depth."""

##Sec: Define timeout for JSON parsing to prevent DoS attacks (P1-003 Daedelus audit)
JSON_PARSE_TIMEOUT = 5.0  # 5 seconds
"""Maximum time allowed for JSON parsing in seconds."""


##Class purpose: Exception for JSON size limit violations
class JSONSizeError(Exception):
    """Raised when JSON exceeds size or depth limits."""

    pass


##Class purpose: Exception for JSON parsing timeout
class JSONTimeoutError(Exception):
    """Raised when JSON parsing exceeds timeout limit."""

    pass


##Function purpose: Check JSON nesting depth recursively
def _check_depth(obj: Any, depth: int = 0, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Check JSON object nesting depth.

    Args:
        obj: Object to check depth for
        depth: Current depth level
        max_depth: Maximum allowed depth

    Raises:
        JSONSizeError: If depth exceeds maximum
    """
    ##Condition purpose: Check depth limit
    if depth > max_depth:
        raise JSONSizeError(f"JSON nesting too deep: {depth} > {max_depth}")

    ##Condition purpose: Recursively check dict values
    if isinstance(obj, dict):
        for value in obj.values():
            _check_depth(value, depth + 1, max_depth)
    ##Condition purpose: Recursively check list items
    elif isinstance(obj, list):
        for item in obj:
            _check_depth(item, depth + 1, max_depth)


##Function purpose: Safely parse JSON with size and depth limits
def safe_json_loads(
    json_str: str,
    max_size: int = MAX_JSON_SIZE,
    max_depth: int = MAX_JSON_DEPTH,
    timeout: float = JSON_PARSE_TIMEOUT,
) -> Any:
    """Safely parse JSON with size and depth limits.

    Prevents denial of service attacks by limiting the size and
    nesting depth of JSON payloads. Also includes timeout protection
    to prevent hanging on malicious JSON.

    Args:
        json_str: JSON string to parse
        max_size: Maximum size in bytes
        max_depth: Maximum nesting depth
        timeout: Maximum time allowed for parsing in seconds

    Returns:
        Parsed JSON value

    Raises:
        JSONSizeError: If size or depth exceeds limits
        JSONTimeoutError: If parsing exceeds timeout
        json.JSONDecodeError: If JSON is invalid
    """
    ##Condition purpose: Check size limit
    json_bytes = json_str.encode("utf-8")
    if len(json_bytes) > max_size:
        raise JSONSizeError(f"JSON response too large: {len(json_bytes)} bytes > {max_size} bytes")

    ##Sec: Parse JSON with timeout protection to prevent DoS attacks (P1-003 Daedelus audit)
    ##Step purpose: Define parsing function for timeout wrapper
    def _parse_json() -> Any:
        """Parse JSON and check depth."""
        data = json.loads(json_str)
        _check_depth(data, max_depth=max_depth)
        return data

    ##Error purpose: Parse JSON with timeout protection
    try:
        ##Step purpose: Use ThreadPoolExecutor for cross-platform timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_parse_json)
            data = future.result(timeout=timeout)
    except FutureTimeoutError:
        raise JSONTimeoutError(f"JSON parsing exceeded timeout of {timeout} seconds") from None
    except json.JSONDecodeError as e:
        ##Step purpose: Re-raise with context
        raise json.JSONDecodeError(
            f"Invalid JSON: {e.msg}",
            e.doc,
            e.pos,
        ) from e

    return data


##Function purpose: Extract JSON from text response
def extract_json_from_response(response: str) -> str:
    """Extract JSON object from text response.

    Many LLMs return JSON wrapped in text. This function extracts
    the JSON portion for parsing.

    Args:
        response: Text response that may contain JSON

    Returns:
        Extracted JSON string

    Raises:
        ValueError: If no JSON found
    """
    ##Step purpose: Find JSON object boundaries
    json_start = response.find("{")
    json_end = response.rfind("}") + 1

    ##Condition purpose: Check if JSON found
    if json_start >= 0 and json_end > json_start:
        return response[json_start:json_end]

    ##Step purpose: Try to find JSON array
    json_start = response.find("[")
    json_end = response.rfind("]") + 1

    ##Condition purpose: Check if JSON array found
    if json_start >= 0 and json_end > json_start:
        return response[json_start:json_end]

    ##Step purpose: No JSON found
    raise ValueError("No JSON object or array found in response")
