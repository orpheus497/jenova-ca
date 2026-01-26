##Script function and purpose: Unit tests for exceptions module
"""
Exception Unit Tests

Tests for the typed exception hierarchy.
"""

from __future__ import annotations

from jenova.exceptions import (
    ConfigError,
    ConfigNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    GraphError,
    JenovaError,
    JenovaMemoryError,
    LLMError,
    LLMParseError,
    MigrationError,
    SchemaVersionError,
)


##Class purpose: Test exception hierarchy
class TestExceptionHierarchy:
    """Tests for exception inheritance."""

    ##Method purpose: Test all exceptions inherit from JenovaError
    def test_all_inherit_from_jenova_error(self) -> None:
        """All custom exceptions should inherit from JenovaError."""
        exceptions = [
            ConfigError("test"),
            ConfigNotFoundError("/path"),
            LLMError("test"),
            JenovaMemoryError("test"),
            GraphError("test"),
            MigrationError("test"),
        ]

        ##Loop purpose: Verify inheritance
        for exc in exceptions:
            assert isinstance(exc, JenovaError)

    ##Method purpose: Test config exceptions inherit from ConfigError
    def test_config_exceptions_inherit_from_config_error(self) -> None:
        """Config-related exceptions should inherit from ConfigError."""
        exceptions = [
            ConfigNotFoundError("/path"),
            ConfigParseError("/path", "error"),
            ConfigValidationError([{"loc": "test", "msg": "error"}]),
        ]

        ##Loop purpose: Verify inheritance
        for exc in exceptions:
            assert isinstance(exc, ConfigError)
            assert isinstance(exc, JenovaError)


##Class purpose: Test exception messages
class TestExceptionMessages:
    """Tests for exception message formatting."""

    ##Method purpose: Test ConfigNotFoundError message
    def test_config_not_found_error_message(self) -> None:
        """ConfigNotFoundError should include path in message."""
        exc = ConfigNotFoundError("/path/to/config.yaml")
        assert "/path/to/config.yaml" in str(exc)
        assert exc.path == "/path/to/config.yaml"

    ##Method purpose: Test LLMParseError preserves raw output
    def test_llm_parse_error_preserves_raw_output(self) -> None:
        """LLMParseError should preserve raw output for debugging."""
        raw = '{"invalid": json}'
        exc = LLMParseError(raw, "JSON decode error")
        assert exc.raw_output == raw
        assert exc.parse_error == "JSON decode error"

    ##Method purpose: Test SchemaVersionError message
    def test_schema_version_error_message(self) -> None:
        """SchemaVersionError should show found and supported versions."""
        exc = SchemaVersionError(found=5, supported=3)
        assert "5" in str(exc)
        assert "3" in str(exc)
        assert exc.found == 5
        assert exc.supported == 3
