##Script function and purpose: Grammar loading utilities for llama.cpp structured output
"""
Grammar Loader - Loading and caching llama.cpp grammars.

This module provides utilities for loading GBNF grammars used with
llama-cpp-python for structured JSON output generation.

Reference: .devdocs/resources/src/jenova/utils/grammar_loader.py
"""

from pathlib import Path
from typing import Protocol

import structlog

from jenova.exceptions import GrammarError

##Class purpose: Define logger for grammar operations
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for llama grammar objects
class LlamaGrammarProtocol(Protocol):
    """Protocol for LlamaGrammar objects."""
    pass


##Class purpose: Built-in GBNF grammar definitions
class BuiltinGrammars:
    """Built-in GBNF grammar definitions for common use cases."""
    
    ##Step purpose: JSON grammar for structured output
    JSON = r'''
root   ::= value
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n] ws)?
'''
    
    ##Step purpose: Simple key-value JSON
    SIMPLE_JSON = r'''
root   ::= "{" ws members "}" ws
members ::= pair ("," ws pair)*
pair   ::= string ":" ws value
value  ::= string | number | "true" | "false" | "null"
string ::= "\"" [a-zA-Z0-9_ -]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
'''
    
    ##Step purpose: Boolean response
    BOOLEAN = r'''
root ::= "true" | "false"
'''
    
    ##Step purpose: Integer response
    INTEGER = r'''
root ::= "-"? [0-9]+
'''
    
    ##Step purpose: Float response with confidence (0.0-1.0)
    CONFIDENCE = r'''
root ::= "0" ("." [0-9]+)? | "1" ("." "0"+)?
'''


##Class purpose: Grammar loader with caching
class GrammarLoader:
    """Loader for llama.cpp GBNF grammars with caching.
    
    Provides loading of grammars from files or built-in definitions,
    with caching to avoid repeated parsing.
    
    Example:
        >>> loader = GrammarLoader()
        >>> grammar = loader.load_json_grammar()
        >>> # Use grammar with llama-cpp-python
    """
    
    ##Method purpose: Initialize the grammar loader
    def __init__(self, grammar_dir: Path | None = None) -> None:
        """Initialize the grammar loader.
        
        Args:
            grammar_dir: Directory containing grammar files (optional)
        """
        ##Step purpose: Store configuration
        self._grammar_dir = grammar_dir
        
        ##Step purpose: Initialize cache
        self._cache: dict[str, object] = {}
        
        ##Step purpose: Track if llama-cpp-python is available
        self._llama_available = self._check_llama_available()
        
        logger.debug(
            "grammar_loader_initialized",
            grammar_dir=str(grammar_dir) if grammar_dir else None,
            llama_available=self._llama_available,
        )
    
    ##Method purpose: Check if llama-cpp-python is available
    def _check_llama_available(self) -> bool:
        """Check if llama-cpp-python is available.
        
        Returns:
            True if llama-cpp-python is importable, False otherwise
        """
        try:
            from llama_cpp import LlamaGrammar
            return True
        except ImportError:
            return False
    
    ##Method purpose: Load a grammar from string
    def load_from_string(self, grammar_str: str, name: str = "custom") -> object:
        """Load a grammar from a GBNF string.
        
        Args:
            grammar_str: GBNF grammar string
            name: Name for caching
            
        Returns:
            LlamaGrammar object or None if not available
            
        Raises:
            GrammarError: If grammar parsing fails
        """
        ##Condition purpose: Check cache first
        if name in self._cache:
            return self._cache[name]
        
        ##Condition purpose: Return None if llama not available
        if not self._llama_available:
            logger.warning("grammar_llama_not_available")
            return None
        
        ##Error purpose: Handle grammar parsing errors
        try:
            from llama_cpp import LlamaGrammar
            
            grammar = LlamaGrammar.from_string(grammar_str)
            self._cache[name] = grammar
            
            logger.debug("grammar_loaded", name=name)
            return grammar
            
        except Exception as e:
            raise GrammarError(f"Failed to parse grammar '{name}': {e}") from e
    
    ##Method purpose: Load grammar from file
    def load_from_file(self, filename: str) -> object:
        """Load a grammar from a GBNF file.
        
        Args:
            filename: Name of grammar file
            
        Returns:
            LlamaGrammar object or None if not available
            
        Raises:
            GrammarError: If file not found or parsing fails
        """
        ##Condition purpose: Check cache first
        if filename in self._cache:
            return self._cache[filename]
        
        ##Condition purpose: Determine file path
        if self._grammar_dir is None:
            raise GrammarError("Grammar directory not configured")
        
        file_path = self._grammar_dir / filename
        
        ##Condition purpose: Check file exists
        if not file_path.exists():
            raise GrammarError(f"Grammar file not found: {file_path}")
        
        ##Error purpose: Handle file reading errors
        try:
            grammar_str = file_path.read_text(encoding="utf-8")
            return self.load_from_string(grammar_str, name=filename)
            
        except OSError as e:
            raise GrammarError(f"Failed to read grammar file: {e}") from e
    
    ##Method purpose: Load the built-in JSON grammar
    def load_json_grammar(self) -> object:
        """Load the built-in JSON grammar.
        
        Returns:
            LlamaGrammar object or None if not available
        """
        return self.load_from_string(BuiltinGrammars.JSON, name="json")
    
    ##Method purpose: Load the simple JSON grammar
    def load_simple_json_grammar(self) -> object:
        """Load the simple JSON grammar.
        
        Returns:
            LlamaGrammar object or None if not available
        """
        return self.load_from_string(BuiltinGrammars.SIMPLE_JSON, name="simple_json")
    
    ##Method purpose: Load the boolean grammar
    def load_boolean_grammar(self) -> object:
        """Load the boolean grammar.
        
        Returns:
            LlamaGrammar object or None if not available
        """
        return self.load_from_string(BuiltinGrammars.BOOLEAN, name="boolean")
    
    ##Method purpose: Load the integer grammar
    def load_integer_grammar(self) -> object:
        """Load the integer grammar.
        
        Returns:
            LlamaGrammar object or None if not available
        """
        return self.load_from_string(BuiltinGrammars.INTEGER, name="integer")
    
    ##Method purpose: Load the confidence grammar
    def load_confidence_grammar(self) -> object:
        """Load the confidence (0.0-1.0) grammar.
        
        Returns:
            LlamaGrammar object or None if not available
        """
        return self.load_from_string(BuiltinGrammars.CONFIDENCE, name="confidence")
    
    ##Method purpose: Clear the grammar cache
    def clear_cache(self) -> None:
        """Clear the grammar cache."""
        self._cache.clear()
        logger.debug("grammar_cache_cleared")
    
    ##Method purpose: Check if llama grammars are available
    @property
    def is_available(self) -> bool:
        """Check if llama grammar support is available."""
        return self._llama_available
