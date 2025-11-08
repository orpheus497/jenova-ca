# The JENOVA Cognitive Architecture - Code Tools Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Enhanced code operations module for JENOVA.

Provides advanced file editing, code parsing, refactoring, syntax highlighting,
codebase mapping, and interactive terminal support.

Phases 13-17: CLI enhancement to match capabilities of Gemini CLI, Copilot CLI, and Claude Code.
"""

from jenova.code_tools.file_editor import FileEditor
from jenova.code_tools.code_parser import CodeParser
from jenova.code_tools.refactoring_engine import RefactoringEngine
from jenova.code_tools.syntax_highlighter import SyntaxHighlighter
from jenova.code_tools.codebase_mapper import CodebaseMapper
from jenova.code_tools.interactive_terminal import InteractiveTerminal

__all__ = [
    'FileEditor',
    'CodeParser',
    'RefactoringEngine',
    'SyntaxHighlighter',
    'CodebaseMapper',
    'InteractiveTerminal',
]

__version__ = '5.2.0'
