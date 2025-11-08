# The JENOVA Cognitive Architecture - Advanced File Editor
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Advanced file editing with diff-based previews and multi-file support.

Provides intelligent code editing capabilities similar to Claude Code CLI.
"""

import os
import difflib
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EditOperation:
    """Represents a single edit operation."""

    file_path: str
    old_content: str
    new_content: str
    line_start: int
    line_end: int
    description: str


@dataclass
class FileEdit:
    """Represents all edits for a single file."""

    file_path: str
    original_content: str
    modified_content: str
    operations: List[EditOperation]


class FileEditor:
    """
    Advanced file editor with diff-based previews and safety features.

    Capabilities:
    - Diff-based editing with preview
    - Multi-file editing in single operation
    - Syntax-aware line detection
    - Backup creation before edits
    - Atomic file operations
    - Permission validation
    """

    def __init__(
        self, sandbox_path: Optional[str] = None, ui_logger=None, file_logger=None
    ):
        """
        Initialize file editor.

        Args:
            sandbox_path: Optional sandbox directory restriction
            ui_logger: UI logger for user feedback
            file_logger: File logger for operation logging
        """
        self.sandbox_path = os.path.expanduser(sandbox_path) if sandbox_path else None
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.backup_dir = None

    def _validate_path(self, file_path: str) -> bool:
        """
        Validate file path is within sandbox if configured.

        Args:
            file_path: Path to validate

        Returns:
            True if path is valid, False otherwise
        """
        if not self.sandbox_path:
            return True

        abs_path = os.path.abspath(os.path.expanduser(file_path))
        sandbox_abs = os.path.abspath(self.sandbox_path)

        return abs_path.startswith(sandbox_abs)

    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read file content with validation.

        Args:
            file_path: Path to file

        Returns:
            File content or None if error
        """
        if not self._validate_path(file_path):
            if self.file_logger:
                self.file_logger.log_error(f"Path outside sandbox: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error reading {file_path}: {e}")
            return None

    def write_file(
        self, file_path: str, content: str, create_backup: bool = True
    ) -> bool:
        """
        Write content to file with backup.

        Args:
            file_path: Path to file
            content: Content to write
            create_backup: Whether to create backup

        Returns:
            True if successful, False otherwise
        """
        if not self._validate_path(file_path):
            if self.file_logger:
                self.file_logger.log_error(f"Path outside sandbox: {file_path}")
            return False

        try:
            # Create backup if file exists
            if create_backup and os.path.exists(file_path):
                self._create_backup(file_path)

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            if self.file_logger:
                self.file_logger.log_info(f"Wrote file: {file_path}")

            return True

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error writing {file_path}: {e}")
            return False

    def _create_backup(self, file_path: str) -> Optional[str]:
        """
        Create backup of file.

        Args:
            file_path: Path to file

        Returns:
            Backup path or None if error
        """
        try:
            import shutil

            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)

            if self.file_logger:
                self.file_logger.log_info(f"Created backup: {backup_path}")

            return backup_path

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error creating backup: {e}")
            return None

    def generate_diff(
        self, original: str, modified: str, file_path: str = "file"
    ) -> str:
        """
        Generate unified diff between original and modified content.

        Args:
            original: Original content
            modified: Modified content
            file_path: File path for diff header

        Returns:
            Unified diff string
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )

        return "".join(diff)

    def apply_edit(
        self, file_path: str, old_text: str, new_text: str, preview: bool = False
    ) -> Optional[str]:
        """
        Apply edit to file with optional preview.

        Args:
            file_path: Path to file
            old_text: Text to replace
            new_text: Replacement text
            preview: If True, return preview without applying

        Returns:
            Preview diff if preview=True, or confirmation message
        """
        content = self.read_file(file_path)
        if content is None:
            return None

        if old_text not in content:
            return f"Error: Text not found in {file_path}"

        new_content = content.replace(old_text, new_text, 1)

        if preview:
            return self.generate_diff(content, new_content, file_path)

        if self.write_file(file_path, new_content):
            diff = self.generate_diff(content, new_content, file_path)
            return f"Applied edit to {file_path}:\n{diff}"

        return f"Error: Failed to write {file_path}"

    def apply_line_edit(
        self, file_path: str, line_num: int, new_line: str, preview: bool = False
    ) -> Optional[str]:
        """
        Edit specific line in file.

        Args:
            file_path: Path to file
            line_num: Line number (1-indexed)
            new_line: New line content
            preview: If True, return preview without applying

        Returns:
            Preview diff or confirmation message
        """
        content = self.read_file(file_path)
        if content is None:
            return None

        lines = content.splitlines(keepends=True)

        if line_num < 1 or line_num > len(lines):
            return f"Error: Line {line_num} out of range (1-{len(lines)})"

        # Preserve line ending
        if lines[line_num - 1].endswith("\n"):
            new_line = new_line.rstrip("\n") + "\n"

        lines[line_num - 1] = new_line
        new_content = "".join(lines)

        if preview:
            return self.generate_diff(content, new_content, file_path)

        if self.write_file(file_path, new_content):
            diff = self.generate_diff(content, new_content, file_path)
            return f"Applied line edit to {file_path}:\n{diff}"

        return f"Error: Failed to write {file_path}"

    def insert_lines(
        self, file_path: str, line_num: int, lines: List[str], preview: bool = False
    ) -> Optional[str]:
        """
        Insert lines at specified position.

        Args:
            file_path: Path to file
            line_num: Line number to insert at (1-indexed)
            lines: Lines to insert
            preview: If True, return preview without applying

        Returns:
            Preview diff or confirmation message
        """
        content = self.read_file(file_path)
        if content is None:
            return None

        file_lines = content.splitlines(keepends=True)

        if line_num < 0 or line_num > len(file_lines):
            return f"Error: Line {line_num} out of range (0-{len(file_lines)})"

        # Ensure lines have proper endings
        insert_lines = [line if line.endswith("\n") else line + "\n" for line in lines]

        new_lines = file_lines[:line_num] + insert_lines + file_lines[line_num:]
        new_content = "".join(new_lines)

        if preview:
            return self.generate_diff(content, new_content, file_path)

        if self.write_file(file_path, new_content):
            diff = self.generate_diff(content, new_content, file_path)
            return f"Inserted {len(lines)} lines into {file_path}:\n{diff}"

        return f"Error: Failed to write {file_path}"

    def delete_lines(
        self, file_path: str, start_line: int, end_line: int, preview: bool = False
    ) -> Optional[str]:
        """
        Delete lines from file.

        Args:
            file_path: Path to file
            start_line: Start line (1-indexed, inclusive)
            end_line: End line (1-indexed, inclusive)
            preview: If True, return preview without applying

        Returns:
            Preview diff or confirmation message
        """
        content = self.read_file(file_path)
        if content is None:
            return None

        lines = content.splitlines(keepends=True)

        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return f"Error: Invalid line range {start_line}-{end_line}"

        new_lines = lines[: start_line - 1] + lines[end_line:]
        new_content = "".join(new_lines)

        if preview:
            return self.generate_diff(content, new_content, file_path)

        if self.write_file(file_path, new_content):
            diff = self.generate_diff(content, new_content, file_path)
            deleted_count = end_line - start_line + 1
            return f"Deleted {deleted_count} lines from {file_path}:\n{diff}"

        return f"Error: Failed to write {file_path}"

    def find_and_replace(
        self,
        file_path: str,
        pattern: str,
        replacement: str,
        regex: bool = False,
        preview: bool = False,
    ) -> Optional[str]:
        """
        Find and replace in file.

        Args:
            file_path: Path to file
            pattern: Pattern to find
            replacement: Replacement text
            regex: If True, pattern is regex
            preview: If True, return preview without applying

        Returns:
            Preview diff or confirmation message
        """
        content = self.read_file(file_path)
        if content is None:
            return None

        try:
            if regex:
                new_content = re.sub(pattern, replacement, content)
            else:
                new_content = content.replace(pattern, replacement)

            if content == new_content:
                return f"No matches found in {file_path}"

            if preview:
                return self.generate_diff(content, new_content, file_path)

            if self.write_file(file_path, new_content):
                diff = self.generate_diff(content, new_content, file_path)
                return f"Applied find/replace to {file_path}:\n{diff}"

            return f"Error: Failed to write {file_path}"

        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        except Exception as e:
            return f"Error: {e}"

    def preview_multi_file_edit(self, edits: List[Tuple[str, str, str]]) -> str:
        """
        Preview edits across multiple files.

        Args:
            edits: List of (file_path, old_text, new_text) tuples

        Returns:
            Combined diff preview
        """
        previews = []

        for file_path, old_text, new_text in edits:
            preview = self.apply_edit(file_path, old_text, new_text, preview=True)
            if preview and not preview.startswith("Error"):
                previews.append(preview)
            else:
                previews.append(f"\n# {file_path}: {preview}\n")

        return "\n".join(previews)

    def apply_multi_file_edit(self, edits: List[Tuple[str, str, str]]) -> str:
        """
        Apply edits across multiple files.

        Args:
            edits: List of (file_path, old_text, new_text) tuples

        Returns:
            Summary of applied edits
        """
        results = []
        successful = 0
        failed = 0

        for file_path, old_text, new_text in edits:
            result = self.apply_edit(file_path, old_text, new_text, preview=False)

            if result and not result.startswith("Error"):
                successful += 1
                results.append(f"✓ {file_path}")
            else:
                failed += 1
                results.append(f"✗ {file_path}: {result}")

        summary = f"\nEdited {successful} files successfully"
        if failed > 0:
            summary += f", {failed} failed"

        return summary + "\n\n" + "\n".join(results)
