# The JENOVA Cognitive Architecture - Refactoring Engine
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Code refactoring operations using rope library.

Provides safe, AST-aware code refactoring capabilities.
"""

import os
from typing import Optional, List, Dict
from pathlib import Path


class RefactoringEngine:
    """
    Code refactoring engine using rope.

    Capabilities:
    - Rename symbol (function, class, variable)
    - Extract method
    - Inline variable
    - Move class/function
    - Organize imports
    - Format code
    """

    def __init__(self, project_root: str, ui_logger=None, file_logger=None):
        """
        Initialize refactoring engine.

        Args:
            project_root: Root directory of project
            ui_logger: UI logger for user feedback
            file_logger: File logger for operation logging
        """
        self.project_root = os.path.abspath(project_root)
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.rope_project = None

    def _init_rope_project(self):
        """Initialize rope project if not already initialized."""
        if self.rope_project is None:
            try:
                from rope.base.project import Project
                self.rope_project = Project(self.project_root)
            except ImportError:
                if self.file_logger:
                    self.file_logger.log_error("rope library not installed")
                return False
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Error initializing rope project: {e}")
                return False
        return True

    def rename_symbol(self, file_path: str, offset: int, new_name: str) -> Optional[str]:
        """
        Rename symbol at given offset.

        Args:
            file_path: Path to file
            offset: Character offset of symbol
            new_name: New name for symbol

        Returns:
            Success message or None if error
        """
        if not self._init_rope_project():
            return "Error: rope library not available"

        try:
            from rope.refactor.rename import Rename

            resource = self.rope_project.get_file(file_path)
            changes = Rename(self.rope_project, resource, offset).get_changes(new_name)

            if changes:
                self.rope_project.do(changes)
                return f"Renamed symbol to '{new_name}' in {len(changes.changes)} locations"

            return "No changes needed"

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error renaming symbol: {e}")
            return f"Error: {e}"

    def extract_method(self, file_path: str, start_offset: int, end_offset: int,
                      method_name: str) -> Optional[str]:
        """
        Extract selected code into new method.

        Args:
            file_path: Path to file
            start_offset: Start character offset
            end_offset: End character offset
            method_name: Name for extracted method

        Returns:
            Success message or None if error
        """
        if not self._init_rope_project():
            return "Error: rope library not available"

        try:
            from rope.refactor.extract import ExtractMethod

            resource = self.rope_project.get_file(file_path)
            changes = ExtractMethod(
                self.rope_project, resource, start_offset, end_offset
            ).get_changes(method_name)

            if changes:
                self.rope_project.do(changes)
                return f"Extracted method '{method_name}'"

            return "No changes needed"

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error extracting method: {e}")
            return f"Error: {e}"

    def inline_variable(self, file_path: str, offset: int) -> Optional[str]:
        """
        Inline variable at given offset.

        Args:
            file_path: Path to file
            offset: Character offset of variable

        Returns:
            Success message or None if error
        """
        if not self._init_rope_project():
            return "Error: rope library not available"

        try:
            from rope.refactor.inline import InlineVariable

            resource = self.rope_project.get_file(file_path)
            changes = InlineVariable(self.rope_project, resource, offset).get_changes()

            if changes:
                self.rope_project.do(changes)
                return "Inlined variable"

            return "No changes needed"

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error inlining variable: {e}")
            return f"Error: {e}"

    def organize_imports(self, file_path: str) -> Optional[str]:
        """
        Organize and sort imports in file.

        Args:
            file_path: Path to file

        Returns:
            Success message or None if error
        """
        try:
            # Use isort for import organization
            try:
                import isort
                result = isort.file(file_path)
                if result:
                    return f"Organized imports in {file_path}"
                return "No changes needed"

            except ImportError:
                # Fallback to basic sorting
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                import_lines = []
                other_lines = []
                in_imports = False

                for line in lines:
                    if line.startswith('import ') or line.startswith('from '):
                        import_lines.append(line)
                        in_imports = True
                    elif in_imports and line.strip() == '':
                        other_lines.append(line)
                        in_imports = False
                    elif not in_imports:
                        other_lines.append(line)

                import_lines.sort()

                with open(file_path, 'w') as f:
                    f.writelines(import_lines)
                    f.writelines(other_lines)

                return f"Organized imports in {file_path}"

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error organizing imports: {e}")
            return f"Error: {e}"

    def format_file(self, file_path: str, style: str = 'pep8') -> Optional[str]:
        """
        Format file according to style guide.

        Args:
            file_path: Path to file
            style: Style guide ('pep8', 'black')

        Returns:
            Success message or None if error
        """
        try:
            if style == 'black':
                try:
                    import black
                    mode = black.Mode()
                    with open(file_path, 'r') as f:
                        source = f.read()

                    formatted = black.format_file_contents(source, fast=False, mode=mode)

                    with open(file_path, 'w') as f:
                        f.write(formatted)

                    return f"Formatted {file_path} with black"

                except ImportError:
                    return "Error: black not installed"

            else:  # pep8/autopep8
                try:
                    import autopep8
                    options = {'aggressive': 1}

                    with open(file_path, 'r') as f:
                        source = f.read()

                    formatted = autopep8.fix_code(source, options=options)

                    with open(file_path, 'w') as f:
                        f.write(formatted)

                    return f"Formatted {file_path} with autopep8"

                except ImportError:
                    return "Error: autopep8 not installed"

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error formatting file: {e}")
            return f"Error: {e}"

    def find_usages(self, file_path: str, symbol_name: str) -> List[Dict]:
        """
        Find all usages of symbol.

        Args:
            file_path: Path to file
            symbol_name: Name of symbol

        Returns:
            List of usage locations
        """
        usages = []

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if symbol_name in line:
                    usages.append({
                        'file': file_path,
                        'line': i,
                        'content': line.strip()
                    })

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error finding usages: {e}")

        return usages

    def close(self):
        """Close rope project and clean up resources."""
        if self.rope_project:
            try:
                self.rope_project.close()
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Error closing rope project: {e}")
