##Script function and purpose: Input validation utilities for path components and user data
"""
Input Validation Utilities

Provides validation functions for usernames, topics, and path components
to prevent path traversal attacks and ensure filesystem safety.
"""

from __future__ import annotations

import re
from pathlib import Path

##Step purpose: Define validation patterns
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
"""Pattern for valid usernames: 1-64 alphanumeric, underscore, or hyphen."""

TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9_\s-]{1,128}$")
"""Pattern for valid topics: 1-128 alphanumeric, space, underscore, or hyphen."""

MAX_USERNAME_LENGTH = 64
"""Maximum username length."""

MAX_TOPIC_LENGTH = 128
"""Maximum topic length."""


##Function purpose: Validate and sanitize username
def validate_username(username: str) -> str:
    """Validate and sanitize username.

    Args:
        username: Username to validate

    Returns:
        Validated and sanitized username

    Raises:
        ValueError: If username is invalid
    """
    ##Fix: Reject None/non-str so callers get ValueError instead of AttributeError
    if username is None or not isinstance(username, str):
        raise ValueError("Username cannot be empty")

    ##Condition purpose: Check username is not empty
    if not username:
        raise ValueError("Username cannot be empty")

    ##Step purpose: Strip whitespace
    username = username.strip()

    ##Condition purpose: Check username is not empty after stripping
    if not username:
        raise ValueError("Username cannot be empty or whitespace only")

    ##Condition purpose: Validate against pattern
    if not USERNAME_PATTERN.match(username):
        raise ValueError("Username must be 1-64 alphanumeric characters, underscores, or hyphens")

    return username


##Function purpose: Validate and sanitize topic name
def validate_topic(topic: str) -> str:
    """Validate and sanitize topic name.

    Args:
        topic: Topic name to validate

    Returns:
        Validated and sanitized topic name

    Raises:
        ValueError: If topic is invalid
    """
    ##Fix: Reject None/non-str so callers get ValueError instead of AttributeError
    if topic is None or not isinstance(topic, str):
        raise ValueError("Topic cannot be empty")

    ##Condition purpose: Check topic is not empty
    if not topic:
        raise ValueError("Topic cannot be empty")

    ##Step purpose: Strip whitespace
    topic = topic.strip()

    ##Condition purpose: Check topic is not empty after stripping
    if not topic:
        raise ValueError("Topic cannot be empty or whitespace only")

    ##Condition purpose: Validate against pattern
    if not TOPIC_PATTERN.match(topic):
        raise ValueError(
            "Topic must be 1-128 alphanumeric characters, spaces, underscores, or hyphens"
        )

    return topic


##Function purpose: Validate path component for filesystem safety
def validate_path_component(component: str) -> str:
    """Validate and sanitize a path component.

    Prevents path traversal attacks by rejecting components containing
    directory traversal sequences or path separators.

    Args:
        component: Path component to validate

    Returns:
        Validated path component

    Raises:
        ValueError: If component contains invalid characters
    """
    ##Condition purpose: Check component is not empty
    if not component:
        raise ValueError("Path component cannot be empty")

    ##Step purpose: Strip whitespace
    component = component.strip()

    ##Condition purpose: Check for path traversal sequences
    if ".." in component:
        raise ValueError("Path component cannot contain '..'")

    ##Condition purpose: Check for path separators
    if "/" in component or "\\" in component:
        raise ValueError("Path component cannot contain path separators")

    ##Condition purpose: Check for null bytes
    if "\x00" in component:
        raise ValueError("Path component cannot contain null bytes")

    ##Condition purpose: Check length
    if len(component) > 255:  # Filesystem limit
        raise ValueError(f"Path component too long: {len(component)} > 255")

    return component


##Function purpose: Verify path is within base directory
def validate_path_within_base(path: Path, base: Path) -> Path:
    """Verify that a path is within a base directory.

    Prevents path traversal attacks by ensuring the resolved path
    is contained within the base directory.

    Args:
        path: Path to validate
        base: Base directory that path must be within

    Returns:
        Resolved path if valid

    Raises:
        ValueError: If path traversal is detected
    """
    ##Step purpose: Resolve both paths to absolute
    resolved_path = path.resolve()
    resolved_base = base.resolve()

    ##Error purpose: Check if path is within base
    try:
        ##Action purpose: Attempt to get relative path
        resolved_path.relative_to(resolved_base)
    except ValueError as e:
        ##Step purpose: Path traversal detected
        raise ValueError(f"Path traversal detected: {path} resolves outside base {base}") from e

    return resolved_path


##Function purpose: Validate username and topic together
def validate_username_and_topic(username: str, topic: str) -> tuple[str, str]:
    """Validate username and topic together.

    Convenience function to validate both username and topic
    in a single call.

    Args:
        username: Username to validate
        topic: Topic to validate

    Returns:
        Tuple of (validated_username, validated_topic)

    Raises:
        ValueError: If either is invalid
    """
    safe_username = validate_username(username)
    safe_topic = validate_topic(topic)
    return safe_username, safe_topic
