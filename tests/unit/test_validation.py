##Script function and purpose: Unit tests for validation utilities
"""
Validation Unit Tests

Tests for input validation, path safety, and username/topic constraints.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jenova.utils.validation import (
    validate_path_component,
    validate_path_within_base,
    validate_topic,
    validate_username,
    validate_username_and_topic,
)


##Class purpose: Test username validation logic
class TestUsernameValidation:
    """Tests for validate_username function."""

    ##Method purpose: Test valid usernames
    @pytest.mark.parametrize("username", [
        "alice",
        "bob_123",
        "user-name",
        "A",
        "x" * 64,
    ])
    def test_valid_usernames(self, username: str) -> None:
        """Valid usernames should be returned as-is (stripped)."""
        assert validate_username(f"  {username}  ") == username

    ##Method purpose: Test invalid usernames
    @pytest.mark.parametrize("username", [
        "",
        " ",
        "user!",
        "user name",
        "x" * 65,
        "@",
        "\x00",
    ])
    def test_invalid_usernames(self, username: str) -> None:
        """Invalid usernames should raise ValueError."""
        with pytest.raises(ValueError):
            validate_username(username)

    ##Method purpose: Test None and non-string inputs
    def test_none_and_non_string(self) -> None:
        """None and non-string inputs should raise ValueError."""
        with pytest.raises(ValueError, match="Username cannot be empty"):
            validate_username(None)  # type: ignore
        with pytest.raises(ValueError, match="Username must be a string"):
            validate_username(123)  # type: ignore


##Class purpose: Test topic validation logic
class TestTopicValidation:
    """Tests for validate_topic function."""

    ##Method purpose: Test valid topics
    @pytest.mark.parametrize("topic", [
        "General",
        "My Topic 123",
        "topic_name",
        "topic-name",
        "x" * 128,
    ])
    def test_valid_topics(self, topic: str) -> None:
        """Valid topics should be returned as-is (stripped)."""
        assert validate_topic(f"  {topic}  ") == topic

    ##Method purpose: Test invalid topics
    @pytest.mark.parametrize("topic", [
        "",
        " ",
        "topic!",
        "x" * 129,
        "@",
    ])
    def test_invalid_topics(self, topic: str) -> None:
        """Invalid topics should raise ValueError."""
        with pytest.raises(ValueError):
            validate_topic(topic)

    ##Method purpose: Test None and non-string inputs for topic
    def test_none_and_non_string_topic(self) -> None:
        """None and non-string inputs should raise ValueError."""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            validate_topic(None)  # type: ignore
        with pytest.raises(ValueError, match="Topic must be a string"):
            validate_topic(123)  # type: ignore


##Class purpose: Test path component validation
class TestPathComponentValidation:
    """Tests for validate_path_component function."""

    ##Method purpose: Test valid components
    @pytest.mark.parametrize("component", [
        "file.txt",
        "sub_dir",
        "my-file",
        "x" * 255,
    ])
    def test_valid_components(self, component: str) -> None:
        """Valid path components should be returned as-is (stripped)."""
        assert validate_path_component(f"  {component}  ") == component

    ##Method purpose: Test invalid components
    @pytest.mark.parametrize("component", [
        "",
        "..",
        "dir/file",
        r"dir\file",
        "\x00",
        "x" * 256,
    ])
    def test_invalid_components(self, component: str) -> None:
        """Invalid path components should raise ValueError."""
        with pytest.raises(ValueError):
            validate_path_component(component)


##Class purpose: Test path traversal protection
class TestPathWithinBase:
    """Tests for validate_path_within_base function."""

    ##Method purpose: Test valid path within base
    def test_valid_path_within_base(self, tmp_path: Path) -> None:
        """Path within base should be returned resolved."""
        base = tmp_path / "base"
        base.mkdir()
        file_path = base / "file.txt"
        file_path.touch()

        assert validate_path_within_base(file_path, base) == file_path.resolve()

    ##Method purpose: Test path traversal attempt
    def test_path_traversal_trapped(self, tmp_path: Path) -> None:
        """Path outside base should raise ValueError."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.touch()

        traversal_path = base / ".." / "outside.txt"

        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path_within_base(traversal_path, base)


##Class purpose: Test combined validation
class TestCombinedValidation:
    """Tests for validate_username_and_topic function."""

    ##Method purpose: Test valid pair
    def test_valid_pair(self) -> None:
        """Valid username and topic should be returned as tuple."""
        assert validate_username_and_topic("user", "topic") == ("user", "topic")

    ##Method purpose: Test invalid pair
    def test_invalid_pair(self) -> None:
        """Invalid components should raise ValueError."""
        with pytest.raises(ValueError):
            validate_username_and_topic("user!", "topic")
        with pytest.raises(ValueError):
            validate_username_and_topic("user", "topic!")
