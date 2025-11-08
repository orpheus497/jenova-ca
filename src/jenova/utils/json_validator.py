# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
JSON validation utilities for LLM-generated responses.

This module provides robust JSON parsing and validation for LLM-generated content,
with schema validation, error correction, and retry logic.

Addresses issue identified in Step 1 diagnostic: LLM responses may contain malformed JSON.
Provides defense-in-depth with multiple fallback strategies.
"""

import json
import re
from typing import Dict, Any, Optional, List
from jsonschema import validate, ValidationError, Draft7Validator


# JSON schemas for common LLM response types
SCHEMAS = {
    "links": {
        "type": "object",
        "properties": {
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "type": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["target", "type"]
                }
            }
        },
        "required": ["links"]
    },
    "meta_insight": {
        "type": "object",
        "properties": {
            "meta_insight": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["meta_insight"]
    },
    "plan": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"}
            },
            "reasoning": {"type": "string"}
        },
        "required": ["steps"]
    },
    "insight": {
        "type": "object",
        "properties": {
            "insight": {"type": "string"},
            "concern": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["insight"]
    },
    "assumption": {
        "type": "object",
        "properties": {
            "assumption": {"type": "string"},
            "category": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["assumption"]
    }
}


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON content from text that may contain markdown, explanations, or other formatting.

    Handles common LLM response patterns:
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON with surrounding explanatory text
    - Multiple JSON objects (returns first valid one)

    Args:
        text: Raw text from LLM that should contain JSON

    Returns:
        Extracted JSON string, or None if no valid JSON found
    """
    # Try to extract from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Try to find JSON object boundaries
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Try to find JSON array boundaries
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        return array_match.group(0)

    return None


def fix_common_json_errors(text: str) -> str:
    """
    Attempt to fix common JSON formatting errors from LLM responses.

    Common issues:
    - Single quotes instead of double quotes
    - Trailing commas
    - Unescaped newlines in strings
    - Missing quotes around keys

    Args:
        text: Potentially malformed JSON string

    Returns:
        Corrected JSON string
    """
    # Replace single quotes with double quotes (but not in strings)
    # This is a heuristic and may not work for all cases
    text = text.replace("'", '"')

    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Escape unescaped newlines in strings
    # This is complex and may not catch all cases
    text = text.replace('\n', '\\n')

    # Remove comments (not valid JSON but LLMs sometimes include them)
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    return text


def parse_llm_json(
    text: str,
    schema_name: Optional[str] = None,
    max_retries: int = 3,
    file_logger: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Robust JSON parsing with error correction and validation.

    Attempts multiple strategies to extract and parse JSON from LLM responses:
    1. Direct JSON parsing
    2. Extract from markdown/text and parse
    3. Apply error corrections and parse
    4. Schema validation if schema_name provided

    Args:
        text: Raw text from LLM
        schema_name: Name of schema to validate against (from SCHEMAS dict)
        max_retries: Maximum parsing attempts with different strategies
        file_logger: Optional file logger for debugging

    Returns:
        Parsed JSON as dictionary, or None if parsing failed
    """
    if not text or not isinstance(text, str):
        if file_logger:
            file_logger.log_error(f"Invalid input to parse_llm_json: {type(text)}")
        return None

    strategies = [
        ("direct", lambda t: json.loads(t)),
        ("extract", lambda t: json.loads(extract_json_from_text(t) or "")),
        ("fix_errors", lambda t: json.loads(fix_common_json_errors(t))),
        ("extract_and_fix", lambda t: json.loads(fix_common_json_errors(extract_json_from_text(t) or "")))
    ]

    for strategy_name, strategy_func in strategies:
        try:
            result = strategy_func(text)

            # Validate against schema if provided
            if schema_name and schema_name in SCHEMAS:
                try:
                    validate(instance=result, schema=SCHEMAS[schema_name])
                except ValidationError as e:
                    if file_logger:
                        file_logger.log_warning(
                            f"JSON schema validation failed for '{schema_name}': {e.message}"
                        )
                    # Continue - return result even if validation fails (partial success)

            if file_logger:
                file_logger.log_debug(f"Successfully parsed JSON using strategy: {strategy_name}")

            return result

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            if file_logger:
                file_logger.log_debug(f"JSON parsing strategy '{strategy_name}' failed: {e}")
            continue

    # All strategies failed
    if file_logger:
        file_logger.log_error(f"All JSON parsing strategies failed for text: {text[:200]}...")

    return None


def validate_json_schema(data: Dict[str, Any], schema_name: str) -> bool:
    """
    Validate JSON data against a named schema.

    Args:
        data: Parsed JSON data
        schema_name: Name of schema from SCHEMAS dict

    Returns:
        True if valid, False otherwise
    """
    if schema_name not in SCHEMAS:
        return False

    try:
        validate(instance=data, schema=SCHEMAS[schema_name])
        return True
    except ValidationError:
        return False


def get_validation_errors(data: Dict[str, Any], schema_name: str) -> List[str]:
    """
    Get detailed validation errors for debugging.

    Args:
        data: Parsed JSON data
        schema_name: Name of schema from SCHEMAS dict

    Returns:
        List of error messages
    """
    if schema_name not in SCHEMAS:
        return [f"Unknown schema: {schema_name}"]

    validator = Draft7Validator(SCHEMAS[schema_name])
    errors = []

    for error in validator.iter_errors(data):
        errors.append(f"{'.'.join(str(p) for p in error.path)}: {error.message}")

    return errors


def register_custom_schema(name: str, schema: Dict[str, Any]) -> None:
    """
    Register a custom JSON schema for validation.

    Args:
        name: Schema name
        schema: JSON schema dict (following JSON Schema Draft 7 format)
    """
    SCHEMAS[name] = schema
