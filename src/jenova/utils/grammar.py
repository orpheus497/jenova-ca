##Script function and purpose: Grammar Loader - Centralized JSON grammar loading utility for llama.cpp
##Dependency purpose: Provides a single point of grammar loading to avoid code duplication across modules
"""Grammar Loader for JENOVA.

This module provides centralized JSON grammar loading for llama.cpp to ensure
structured LLM responses. Loads grammar files from the llama.cpp submodule.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from jenova.config.models import JenovaConfig

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for UI logger operations
class UILoggerProtocol(Protocol):
    """Protocol for UI logger operations."""
    
    ##Method purpose: Log system message
    def system_message(self, message: str) -> None:
        """Log system message."""
        ...
    
    ##Method purpose: Log error
    def log_error(self, message: str) -> None:
        """Log error message."""
        ...


##Class purpose: Protocol for file logger operations
class FileLoggerProtocol(Protocol):
    """Protocol for file logger operations."""
    
    ##Method purpose: Log warning
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        ...
    
    ##Method purpose: Log error
    def log_error(self, message: str) -> None:
        """Log error message."""
        ...


##Function purpose: Load JSON grammar from llama.cpp for structured LLM responses
def load_json_grammar(
    grammar_path: Path | None = None,
    ui_logger: UILoggerProtocol | None = None,
    file_logger: FileLoggerProtocol | None = None,
) -> object | None:
    """Load JSON grammar from llama.cpp submodule for structured LLM output.
    
    Args:
        grammar_path: Optional custom path to grammar file. If None, uses
            module-relative default.
        ui_logger: Optional UI logger instance for user-facing messages.
        file_logger: Optional file logger instance for error logging.
        
    Returns:
        LlamaGrammar instance if successful, None otherwise.
    """
    ##Step purpose: Construct path to grammar file
    if grammar_path is None:
        ##Step purpose: Get module directory
        module_dir = Path(__file__).parent
        
        ##Step purpose: Walk up to project root (utils/ -> jenova/ -> src/ -> project root)
        project_root = module_dir.parent.parent.parent
        
        ##Step purpose: Construct grammar path
        grammar_path = project_root / "llama.cpp" / "grammars" / "json.gbnf"
    
    ##Condition purpose: Check if grammar file exists
    if not grammar_path.exists():
        error_msg = f"JSON grammar file not found at {grammar_path}"
        if ui_logger:
            ui_logger.system_message(error_msg)
        if file_logger:
            file_logger.log_warning(error_msg)
        logger.warning("grammar_file_not_found", path=str(grammar_path))
        return None
    
    ##Error purpose: Handle grammar loading errors
    try:
        ##Step purpose: Import LlamaGrammar
        try:
            from llama_cpp import LlamaGrammar
        except ImportError as e:
            error_msg = f"Could not import LlamaGrammar: {e}"
            if file_logger:
                file_logger.log_error(error_msg)
            if ui_logger:
                ui_logger.log_error(error_msg)
            logger.error("grammar_import_failed", error=str(e))
            return None
        
        ##Step purpose: Read grammar file
        with open(grammar_path, "r", encoding="utf-8") as f:
            grammar_text = f.read()
        
        ##Step purpose: Create LlamaGrammar instance
        grammar = LlamaGrammar.from_string(grammar_text)
        
        logger.info("grammar_loaded", path=str(grammar_path))
        return grammar
        
    except Exception as e:
        error_msg = f"Could not load JSON grammar: {e}"
        if file_logger:
            file_logger.log_error(error_msg)
        if ui_logger:
            ui_logger.log_error(error_msg)
        logger.error("grammar_load_failed", error=str(e), path=str(grammar_path))
        return None
