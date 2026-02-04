##Script function and purpose: Unit tests for grammar loader utilities
"""
Test suite for GrammarLoader - Loading and caching llama.cpp grammars.

Tests cover:
- Built-in grammar definitions
- Grammar loading from strings
- Grammar loading from files
- Caching behavior
- Error handling
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from jenova.exceptions import GrammarError
from jenova.utils.grammar import (
    BuiltinGrammars,
    GrammarLoader,
)


##Class purpose: Fixture providing temporary grammar directory
@pytest.fixture
def grammar_dir(tmp_path: Path) -> Path:
    """##Test case: Temporary directory for grammar files."""
    return tmp_path


##Class purpose: Fixture providing grammar loader
@pytest.fixture
def loader(grammar_dir: Path) -> GrammarLoader:
    """##Test case: GrammarLoader with temp directory."""
    return GrammarLoader(grammar_dir=grammar_dir)


##Class purpose: Fixture providing loader without directory
@pytest.fixture
def loader_no_dir() -> GrammarLoader:
    """##Test case: GrammarLoader without directory."""
    return GrammarLoader(grammar_dir=None)


##Function purpose: Test builtin grammars exist
def test_builtin_grammars_json() -> None:
    """##Test case: JSON grammar is defined."""
    ##Assertion purpose: Verify JSON grammar exists
    assert BuiltinGrammars.JSON is not None
    assert isinstance(BuiltinGrammars.JSON, str)
    assert "value" in BuiltinGrammars.JSON
    assert "object" in BuiltinGrammars.JSON


##Function purpose: Test builtin simple json
def test_builtin_grammars_simple_json() -> None:
    """##Test case: Simple JSON grammar is defined."""
    ##Assertion purpose: Verify simple JSON grammar
    assert BuiltinGrammars.SIMPLE_JSON is not None
    assert isinstance(BuiltinGrammars.SIMPLE_JSON, str)
    assert "members" in BuiltinGrammars.SIMPLE_JSON


##Function purpose: Test builtin boolean grammar
def test_builtin_grammars_boolean() -> None:
    """##Test case: Boolean grammar is defined."""
    ##Assertion purpose: Verify boolean grammar
    assert BuiltinGrammars.BOOLEAN is not None
    assert "true" in BuiltinGrammars.BOOLEAN
    assert "false" in BuiltinGrammars.BOOLEAN


##Function purpose: Test builtin integer grammar
def test_builtin_grammars_integer() -> None:
    """##Test case: Integer grammar is defined."""
    ##Assertion purpose: Verify integer grammar
    assert BuiltinGrammars.INTEGER is not None
    assert "[0-9]" in BuiltinGrammars.INTEGER


##Function purpose: Test builtin confidence grammar
def test_builtin_grammars_confidence() -> None:
    """##Test case: Confidence grammar is defined."""
    ##Assertion purpose: Verify confidence grammar
    assert BuiltinGrammars.CONFIDENCE is not None
    assert "0" in BuiltinGrammars.CONFIDENCE
    assert "1" in BuiltinGrammars.CONFIDENCE


##Function purpose: Test loader initialization
def test_loader_initialization(loader: GrammarLoader, grammar_dir: Path) -> None:
    """##Test case: Loader initializes correctly."""
    ##Assertion purpose: Verify state
    assert loader._grammar_dir == grammar_dir
    assert len(loader._cache) == 0


##Function purpose: Test loader check llama available
def test_loader_check_llama_available(loader: GrammarLoader) -> None:
    """##Test case: Can check llama-cpp-python availability."""
    ##Assertion purpose: Verify check works
    is_available = loader._check_llama_available()
    assert isinstance(is_available, bool)


##Function purpose: Test loader load from string without llama
def test_loader_load_from_string_no_llama(loader_no_dir: GrammarLoader) -> None:
    """##Test case: load_from_string returns None if llama not available."""
    ##Step purpose: Patch llama import to fail
    with patch.object(loader_no_dir, "_llama_available", False):
        ##Action purpose: Try to load
        result = loader_no_dir.load_from_string("test grammar", name="test")

        ##Assertion purpose: Verify None returned
        assert result is None


##Function purpose: Test loader load from string with llama
def test_loader_load_from_string_with_llama(loader: GrammarLoader) -> None:
    """##Test case: load_from_string creates grammar if llama available."""
    ##Step purpose: Patch llama import
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        ##Step purpose: Set llama available
        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load grammar
            result = loader.load_from_string(BuiltinGrammars.JSON, name="json")

            ##Assertion purpose: Verify returned
            assert result is mock_grammar


##Function purpose: Test loader caches grammars
def test_loader_caches_from_string(loader: GrammarLoader) -> None:
    """##Test case: load_from_string caches result."""
    ##Step purpose: Create mock grammar
    mock_grammar = Mock()

    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load same grammar twice
            result1 = loader.load_from_string("test", name="test")
            result2 = loader.load_from_string("test", name="test")

            ##Assertion purpose: Verify cached (only one call to llama)
            assert mock_llama_class.from_string.call_count == 1
            assert result1 is result2


##Function purpose: Test loader grammar parsing error
def test_loader_load_from_string_parse_error(loader: GrammarLoader) -> None:
    """##Test case: Grammar parsing error raises GrammarError."""
    ##Step purpose: Set up error
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(side_effect=ValueError("Invalid grammar"))

        with patch.object(loader, "_llama_available", True), pytest.raises(GrammarError):  # noqa: SIM117
            ##Action purpose: Try to load
            loader.load_from_string("invalid", name="bad")


##Function purpose: Test loader load from file
def test_loader_load_from_file(loader: GrammarLoader, grammar_dir: Path) -> None:
    """##Test case: load_from_file reads and loads grammar."""
    ##Step purpose: Create grammar file
    grammar_file = grammar_dir / "test.gbnf"
    grammar_file.write_text(BuiltinGrammars.JSON)

    ##Step purpose: Mock llama
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load from file
            result = loader.load_from_file("test.gbnf")

            ##Assertion purpose: Verify loaded
            assert result is mock_grammar


##Function purpose: Test loader load from file not found
def test_loader_load_from_file_not_found(loader: GrammarLoader) -> None:
    """##Test case: load_from_file raises if file not found."""
    ##Step purpose: Try to load nonexistent file
    with pytest.raises(GrammarError) as exc_info:
        loader.load_from_file("nonexistent.gbnf")

    ##Assertion purpose: Verify error
    assert "not found" in str(exc_info.value)


##Function purpose: Test loader load from file no directory
def test_loader_load_from_file_no_directory(loader_no_dir: GrammarLoader) -> None:
    """##Test case: load_from_file raises if no directory configured."""
    ##Step purpose: Try to load
    with pytest.raises(GrammarError) as exc_info:
        loader_no_dir.load_from_file("test.gbnf")

    ##Assertion purpose: Verify error
    assert "not configured" in str(exc_info.value)


##Function purpose: Test loader load from file read error
def test_loader_load_from_file_read_error(loader: GrammarLoader, grammar_dir: Path) -> None:
    """##Test case: load_from_file handles OSError."""
    ##Step purpose: Create file and simulate read error
    grammar_file = grammar_dir / "test.gbnf"
    grammar_file.write_text("test")

    ##Action purpose: Patch read to fail
    with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
        ##Action purpose: Try to load
        with pytest.raises(GrammarError) as exc_info:
            loader.load_from_file("test.gbnf")

        ##Assertion purpose: Verify error
        assert "read" in str(exc_info.value).lower()


##Function purpose: Test loader load json grammar
def test_loader_load_json_grammar(loader: GrammarLoader) -> None:
    """##Test case: load_json_grammar loads built-in JSON."""
    ##Step purpose: Mock llama
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load JSON
            result = loader.load_json_grammar()

            ##Assertion purpose: Verify loaded and cached as "json"
            assert result is mock_grammar
            assert "json" in loader._cache


##Function purpose: Test loader load simple json grammar
def test_loader_load_simple_json_grammar(loader: GrammarLoader) -> None:
    """##Test case: load_simple_json_grammar loads built-in simple JSON."""
    ##Step purpose: Mock llama
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load simple JSON
            result = loader.load_simple_json_grammar()

            ##Assertion purpose: Verify loaded and cached
            assert result is mock_grammar
            assert "simple_json" in loader._cache


##Function purpose: Test loader load boolean grammar
def test_loader_load_boolean_grammar(loader: GrammarLoader) -> None:
    """##Test case: load_boolean_grammar loads built-in boolean."""
    ##Step purpose: Mock llama
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load boolean
            result = loader.load_boolean_grammar()

            ##Assertion purpose: Verify loaded and cached
            assert result is mock_grammar
            assert "boolean" in loader._cache


##Function purpose: Test loader load integer grammar
def test_loader_load_integer_grammar(loader: GrammarLoader) -> None:
    """##Test case: load_integer_grammar loads built-in integer."""
    ##Step purpose: Mock llama
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load integer
            result = loader.load_integer_grammar()

            ##Assertion purpose: Verify loaded and cached
            assert result is mock_grammar
            assert "integer" in loader._cache


##Function purpose: Test loader load confidence grammar
def test_loader_load_confidence_grammar(loader: GrammarLoader) -> None:
    """##Test case: load_confidence_grammar loads built-in confidence."""
    ##Step purpose: Mock llama
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load confidence
            result = loader.load_confidence_grammar()

            ##Assertion purpose: Verify loaded and cached
            assert result is mock_grammar
            assert "confidence" in loader._cache


##Function purpose: Test loader clear cache
def test_loader_clear_cache(loader: GrammarLoader) -> None:
    """##Test case: clear_cache empties cache."""
    ##Step purpose: Add to cache
    loader._cache["test"] = Mock()
    loader._cache["test2"] = Mock()

    ##Action purpose: Clear
    loader.clear_cache()

    ##Assertion purpose: Verify cleared
    assert len(loader._cache) == 0


##Function purpose: Test loader is available property
def test_loader_is_available_property(loader: GrammarLoader) -> None:
    """##Test case: is_available returns llama availability."""
    ##Assertion purpose: Verify property
    result = loader.is_available
    assert isinstance(result, bool)


##Function purpose: Test different grammars are different
def test_builtin_grammars_different() -> None:
    """##Test case: Built-in grammars are distinct."""
    ##Assertion purpose: Verify differences
    assert BuiltinGrammars.JSON != BuiltinGrammars.SIMPLE_JSON
    assert BuiltinGrammars.BOOLEAN != BuiltinGrammars.INTEGER
    assert BuiltinGrammars.JSON != BuiltinGrammars.CONFIDENCE


##Function purpose: Test loader cache key format
def test_loader_cache_key_format(loader: GrammarLoader) -> None:
    """##Test case: Cache keys are correct."""
    ##Step purpose: Mock llama and load several
    mock_grammar = Mock()
    with patch("llama_cpp.LlamaGrammar") as mock_llama_class:
        mock_llama_class.from_string = Mock(return_value=mock_grammar)

        with patch.object(loader, "_llama_available", True):
            ##Action purpose: Load grammars
            loader.load_from_string(BuiltinGrammars.JSON, name="json")
            loader.load_from_string(BuiltinGrammars.BOOLEAN, name="bool")
            loader.load_from_string("custom", name="custom_name")

            ##Assertion purpose: Verify cache keys
            assert "json" in loader._cache
            assert "bool" in loader._cache
            assert "custom_name" in loader._cache
