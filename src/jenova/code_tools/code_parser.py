# The JENOVA Cognitive Architecture - Code Parser
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
AST-based code parsing and symbol extraction.

Provides intelligent code understanding for Python files.
"""

import ast
import os
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Symbol:
    """Represents a code symbol (class, function, variable)."""

    name: str
    type: str  # 'class', 'function', 'method', 'variable', 'import'
    line: int
    col: int
    end_line: int
    end_col: int
    docstring: Optional[str] = None
    parent: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


@dataclass
class CodeStructure:
    """Represents the structure of a code file."""

    file_path: str
    symbols: List[Symbol]
    imports: List[str]
    classes: List[str]
    functions: List[str]
    complexity: int = 0


class CodeParser:
    """
    AST-based Python code parser.

    Capabilities:
    - Symbol extraction (classes, functions, methods, variables)
    - Import detection
    - Docstring extraction
    - Function signature parsing
    - Decorator detection
    - Code structure analysis
    """

    def __init__(self, ui_logger=None, file_logger=None):
        """
        Initialize code parser.

        Args:
            ui_logger: UI logger for user feedback
            file_logger: File logger for operation logging
        """
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def parse_file(self, file_path: str) -> Optional[CodeStructure]:
        """
        Parse Python file and extract structure.

        Args:
            file_path: Path to Python file

        Returns:
            CodeStructure or None if error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)

            symbols = self._extract_symbols(tree)
            imports = self._extract_imports(tree)
            classes = [s.name for s in symbols if s.type == "class"]
            functions = [s.name for s in symbols if s.type == "function"]

            return CodeStructure(
                file_path=file_path,
                symbols=symbols,
                imports=imports,
                classes=classes,
                functions=functions,
            )

        except SyntaxError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Syntax error parsing {file_path}: {e}")
            return None

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error parsing {file_path}: {e}")
            return None

    def _extract_symbols(self, tree: ast.AST) -> List[Symbol]:
        """
        Extract all symbols from AST.

        Args:
            tree: AST node

        Returns:
            List of Symbol objects
        """
        symbols = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(self._parse_class(node))

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        symbols.append(self._parse_function(item, parent=node.name))

            elif isinstance(node, ast.FunctionDef):
                # Top-level functions only (methods handled above)
                if not any(
                    isinstance(p, ast.ClassDef)
                    for p in ast.walk(tree)
                    if hasattr(p, "body") and node in getattr(p, "body", [])
                ):
                    symbols.append(self._parse_function(node))

        return symbols

    def _parse_class(self, node: ast.ClassDef) -> Symbol:
        """
        Parse class definition.

        Args:
            node: ClassDef AST node

        Returns:
            Symbol object
        """
        docstring = ast.get_docstring(node)
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        return Symbol(
            name=node.name,
            type="class",
            line=node.lineno,
            col=node.col_offset,
            end_line=node.end_lineno or node.lineno,
            end_col=node.end_col_offset or node.col_offset,
            docstring=docstring,
            decorators=decorators,
        )

    def _parse_function(
        self, node: ast.FunctionDef, parent: Optional[str] = None
    ) -> Symbol:
        """
        Parse function/method definition.

        Args:
            node: FunctionDef AST node
            parent: Parent class name if method

        Returns:
            Symbol object
        """
        docstring = ast.get_docstring(node)
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {ast.unparse(arg.annotation)}"
            parameters.append(param_str)

        # Extract return annotation
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns)

        return Symbol(
            name=node.name,
            type="method" if parent else "function",
            line=node.lineno,
            col=node.col_offset,
            end_line=node.end_lineno or node.lineno,
            end_col=node.end_col_offset or node.col_offset,
            docstring=docstring,
            parent=parent,
            parameters=parameters,
            return_annotation=return_annotation,
            decorators=decorators,
        )

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """
        Extract decorator name from AST node.

        Args:
            decorator: Decorator AST node

        Returns:
            Decorator name as string
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return ast.unparse(decorator)

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """
        Extract all imports from AST.

        Args:
            tree: AST node

        Returns:
            List of import statements
        """
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return imports

    def find_symbol(
        self, structure: CodeStructure, symbol_name: str
    ) -> Optional[Symbol]:
        """
        Find symbol by name in code structure.

        Args:
            structure: CodeStructure to search
            symbol_name: Name of symbol to find

        Returns:
            Symbol or None if not found
        """
        for symbol in structure.symbols:
            if symbol.name == symbol_name:
                return symbol
        return None

    def find_definition(
        self, file_path: str, symbol_name: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find definition location of symbol.

        Args:
            file_path: Path to file
            symbol_name: Name of symbol

        Returns:
            Tuple of (line, column) or None if not found
        """
        structure = self.parse_file(file_path)
        if not structure:
            return None

        symbol = self.find_symbol(structure, symbol_name)
        if symbol:
            return (symbol.line, symbol.col)

        return None

    def get_symbol_at_line(self, file_path: str, line_num: int) -> Optional[Symbol]:
        """
        Get symbol at specific line number.

        Args:
            file_path: Path to file
            line_num: Line number

        Returns:
            Symbol or None if not found
        """
        structure = self.parse_file(file_path)
        if not structure:
            return None

        for symbol in structure.symbols:
            if symbol.line <= line_num <= symbol.end_line:
                return symbol

        return None

    def extract_function_calls(self, file_path: str) -> Dict[str, List[str]]:
        """
        Extract function calls from file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary mapping function names to list of called functions
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            calls = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_calls = []

                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                func_calls.append(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                func_calls.append(child.func.attr)

                    calls[node.name] = func_calls

            return calls

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error extracting calls from {file_path}: {e}"
                )
            return {}

    def get_dependencies(self, file_path: str) -> Set[str]:
        """
        Get all module dependencies.

        Args:
            file_path: Path to file

        Returns:
            Set of module names
        """
        structure = self.parse_file(file_path)
        if not structure:
            return set()

        dependencies = set()
        for imp in structure.imports:
            # Get top-level module name
            module = imp.split(".")[0]
            dependencies.add(module)

        return dependencies

    def format_symbol(self, symbol: Symbol) -> str:
        """
        Format symbol as readable string.

        Args:
            symbol: Symbol to format

        Returns:
            Formatted string
        """
        parts = []

        # Decorators
        if symbol.decorators:
            parts.append("Decorators: " + ", ".join(f"@{d}" for d in symbol.decorators))

        # Type and name
        if symbol.parent:
            parts.append(f"{symbol.type.capitalize()}: {symbol.parent}.{symbol.name}")
        else:
            parts.append(f"{symbol.type.capitalize()}: {symbol.name}")

        # Parameters
        if symbol.parameters:
            params = ", ".join(symbol.parameters)
            parts.append(f"Parameters: ({params})")

        # Return type
        if symbol.return_annotation:
            parts.append(f"Returns: {symbol.return_annotation}")

        # Location
        parts.append(f"Location: Line {symbol.line}, Col {symbol.col}")

        # Docstring
        if symbol.docstring:
            parts.append(f"Docstring: {symbol.docstring[:100]}...")

        return "\n".join(parts)
