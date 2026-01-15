##Script function and purpose: Centralized JSON grammar loading utility for The JENOVA Cognitive Architecture
##This module provides a single point of grammar loading to avoid code duplication across modules

import os
from typing import Any, Optional

##Function purpose: Load JSON grammar from llama.cpp for structured LLM responses
def load_json_grammar(
    ui_logger: Optional[Any] = None, 
    file_logger: Optional[Any] = None,
    grammar_path: Optional[str] = None
) -> Optional[Any]:
    """
    Load JSON grammar from llama.cpp submodule for structured LLM output.
    
    Args:
        ui_logger: Optional UILogger instance for user-facing messages
        file_logger: Optional FileLogger instance for error logging
        grammar_path: Optional custom path to grammar file. If None, uses module-relative default.
        
    Returns:
        LlamaGrammar instance if successful, None otherwise
    """
    ##Block purpose: Construct path to grammar file using module-relative path (not cwd)
    if grammar_path is None:
        # Walk up from this file to project root, then to llama.cpp/grammars/json.gbnf
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # utils/ -> jenova/ -> src/ -> project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(module_dir)))
        grammar_path = os.path.join(project_root, "llama.cpp", "grammars", "json.gbnf")
    
    ##Block purpose: Check if grammar file exists
    if not os.path.exists(grammar_path):
        if ui_logger:
            ui_logger.system_message(f"JSON grammar file not found at {grammar_path}")
        if file_logger:
            file_logger.log_warning(f"JSON grammar file not found at {grammar_path}")
        return None
    
    ##Block purpose: Load grammar file and create LlamaGrammar instance
    try:
        with open(grammar_path, 'r', encoding='utf-8') as f:
            grammar_text = f.read()
        
        from llama_cpp.llama_grammar import LlamaGrammar
        return LlamaGrammar.from_string(grammar_text)
        
    except ImportError as e:
        error_msg = f"Could not import LlamaGrammar: {e}"
        if file_logger:
            file_logger.log_error(error_msg)
        if ui_logger:
            ui_logger.log_error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Could not load JSON grammar: {e}"
        if file_logger:
            file_logger.log_error(error_msg)
        if ui_logger:
            ui_logger.log_error(error_msg)
        return None
