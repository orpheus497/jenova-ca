# The JENOVA Cognitive Architecture - Syntax Highlighter
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Syntax highlighting using Pygments.

Provides terminal-friendly syntax highlighting for code display.
"""

from typing import Optional


class SyntaxHighlighter:
    """
    Syntax highlighter using Pygments.

    Capabilities:
    - Multi-language syntax highlighting
    - Terminal-friendly color schemes
    - Line numbering
    - Diff highlighting
    """

    def __init__(self, style: str = "monokai"):
        """
        Initialize syntax highlighter.

        Args:
            style: Pygments style name
        """
        self.style = style
        self._pygments_available = False

        try:
            from pygments import highlight
            from pygments.lexers import get_lexer_by_name, guess_lexer
            from pygments.formatters import TerminalFormatter, Terminal256Formatter
            from pygments.styles import get_style_by_name

            self.highlight = highlight
            self.get_lexer_by_name = get_lexer_by_name
            self.guess_lexer = guess_lexer
            self.TerminalFormatter = TerminalFormatter
            self.Terminal256Formatter = Terminal256Formatter
            self.get_style_by_name = get_style_by_name
            self._pygments_available = True

        except ImportError:
            pass

    def highlight_code(
        self, code: str, language: Optional[str] = None, line_numbers: bool = False
    ) -> str:
        """
        Highlight code with syntax coloring.

        Args:
            code: Source code to highlight
            language: Language name (auto-detect if None)
            line_numbers: Whether to show line numbers

        Returns:
            Highlighted code string
        """
        if not self._pygments_available:
            # Fallback: return code with line numbers if requested
            if line_numbers:
                lines = code.split("\n")
                width = len(str(len(lines)))
                return "\n".join(
                    f"{i+1:>{width}} | {line}" for i, line in enumerate(lines)
                )
            return code

        try:
            # Get lexer
            if language:
                lexer = self.get_lexer_by_name(language)
            else:
                lexer = self.guess_lexer(code)

            # Get formatter
            formatter = self.Terminal256Formatter(
                style=self.style, linenos=line_numbers
            )

            # Highlight
            return self.highlight(code, lexer, formatter)

        except Exception:
            # Fallback to plain code
            if line_numbers:
                lines = code.split("\n")
                width = len(str(len(lines)))
                return "\n".join(
                    f"{i+1:>{width}} | {line}" for i, line in enumerate(lines)
                )
            return code

    def highlight_file(self, file_path: str, line_numbers: bool = True) -> str:
        """
        Highlight file with syntax coloring.

        Args:
            file_path: Path to file
            line_numbers: Whether to show line numbers

        Returns:
            Highlighted file content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            # Detect language from extension
            language = self._detect_language(file_path)

            return self.highlight_code(code, language, line_numbers)

        except Exception as e:
            return f"Error highlighting file: {e}"

    def highlight_diff(self, diff_text: str) -> str:
        """
        Highlight diff output.

        Args:
            diff_text: Diff text

        Returns:
            Highlighted diff
        """
        if not self._pygments_available:
            return diff_text

        try:
            lexer = self.get_lexer_by_name("diff")
            formatter = self.Terminal256Formatter(style=self.style)
            return self.highlight(diff_text, lexer, formatter)

        except Exception:
            return diff_text

    def _detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Language name or None
        """
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rs": "rust",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
            ".sql": "sql",
        }

        import os

        _, ext = os.path.splitext(file_path)
        return ext_map.get(ext.lower())
