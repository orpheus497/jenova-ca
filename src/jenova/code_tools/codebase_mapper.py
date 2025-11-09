# The JENOVA Cognitive Architecture - Codebase Mapper
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Codebase structure analysis and mapping.

Provides project-wide code understanding and dependency analysis.
"""

import os
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field


@dataclass
class FileInfo:
    """Information about a file in the codebase."""

    path: str
    language: str
    size: int
    lines: int
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)


@dataclass
class ProjectStructure:
    """Represents the complete project structure."""

    root_path: str
    files: List[FileInfo]
    total_files: int
    total_lines: int
    languages: Dict[str, int]  # language -> file count
    dependencies: Set[str]


class CodebaseMapper:
    """
    Codebase structure analyzer and mapper.

    Capabilities:
    - Project structure discovery
    - File classification
    - Dependency extraction
    - Code statistics
    - Directory tree visualization
    """

    def __init__(self, ui_logger=None, file_logger=None):
        """
        Initialize codebase mapper.

        Args:
            ui_logger: UI logger for user feedback
            file_logger: File logger for operation logging
        """
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def map_project(
        self, root_path: str, exclude_dirs: Optional[List[str]] = None
    ) -> ProjectStructure:
        """
        Map entire project structure.

        Args:
            root_path: Root directory of project
            exclude_dirs: Directories to exclude

        Returns:
            ProjectStructure object
        """
        if exclude_dirs is None:
            exclude_dirs = [
                "__pycache__",
                ".git",
                "node_modules",
                "venv",
                ".venv",
                "build",
                "dist",
            ]

        files = []
        total_lines = 0
        languages = {}
        all_dependencies = set()

        for root, dirs, filenames in os.walk(root_path):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for filename in filenames:
                file_path = os.path.join(root, filename)
                language = self._detect_language(filename)

                if not language:
                    continue

                file_info = self._analyze_file(file_path, language)
                if file_info:
                    files.append(file_info)
                    total_lines += file_info.lines

                    # Update language count
                    languages[language] = languages.get(language, 0) + 1

                    # Collect dependencies
                    all_dependencies.update(file_info.imports)

        return ProjectStructure(
            root_path=root_path,
            files=files,
            total_files=len(files),
            total_lines=total_lines,
            languages=languages,
            dependencies=all_dependencies,
        )

    def _analyze_file(self, file_path: str, language: str) -> Optional[FileInfo]:
        """
        Analyze single file.

        Args:
            file_path: Path to file
            language: Detected language

        Returns:
            FileInfo or None if error
        """
        try:
            size = os.path.getsize(file_path)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = len(content.splitlines())

            classes = []
            functions = []
            imports = set()

            # Python-specific analysis
            if language == "python":
                try:
                    from jenova.code_tools.code_parser import CodeParser

                    parser = CodeParser()
                    structure = parser.parse_file(file_path)

                    if structure:
                        classes = structure.classes
                        functions = structure.functions
                        imports = set(structure.imports)

                except (SyntaxError, ValueError, UnicodeDecodeError, ImportError) as e:
                    # File has syntax errors, invalid structure, or encoding issues
                    # Continue with basic file info without detailed structure
                    logger.debug(f"Could not parse {file_path}: {type(e).__name__}")

            return FileInfo(
                path=file_path,
                language=language,
                size=size,
                lines=lines,
                classes=classes,
                functions=functions,
                imports=imports,
            )

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error analyzing {file_path}: {e}")
            return None

    def _detect_language(self, filename: str) -> Optional[str]:
        """
        Detect programming language from filename.

        Args:
            filename: Filename

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
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".sh": "bash",
        }

        _, ext = os.path.splitext(filename)
        return ext_map.get(ext.lower())

    def generate_tree(
        self,
        root_path: str,
        max_depth: int = 3,
        exclude_dirs: Optional[List[str]] = None,
    ) -> str:
        """
        Generate directory tree visualization.

        Args:
            root_path: Root directory
            max_depth: Maximum depth to display
            exclude_dirs: Directories to exclude

        Returns:
            Tree visualization string
        """
        if exclude_dirs is None:
            exclude_dirs = ["__pycache__", ".git", "node_modules", "venv", ".venv"]

        lines = [os.path.basename(root_path) + "/"]

        def walk_dir(path: str, prefix: str = "", depth: int = 0):
            if depth >= max_depth:
                return

            try:
                entries = sorted(os.listdir(path))
            except PermissionError:
                return

            # Separate dirs and files
            dirs = [
                e
                for e in entries
                if os.path.isdir(os.path.join(path, e)) and e not in exclude_dirs
            ]
            files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

            # Process directories
            for i, dir_name in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and not files
                connector = "└── " if is_last_dir else "├── "
                lines.append(f"{prefix}{connector}{dir_name}/")

                extension = "    " if is_last_dir else "│   "
                walk_dir(os.path.join(path, dir_name), prefix + extension, depth + 1)

            # Process files
            for i, file_name in enumerate(files):
                is_last = i == len(files) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{file_name}")

        walk_dir(root_path)
        return "\n".join(lines)

    def find_files(self, root_path: str, pattern: str) -> List[str]:
        """
        Find files matching pattern.

        Args:
            root_path: Root directory
            pattern: Filename pattern (supports wildcards)

        Returns:
            List of matching file paths
        """
        import fnmatch

        matches = []

        for root, dirs, files in os.walk(root_path):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    matches.append(os.path.join(root, filename))

        return matches

    def get_statistics(self, structure: ProjectStructure) -> str:
        """
        Generate statistics summary.

        Args:
            structure: ProjectStructure object

        Returns:
            Formatted statistics string
        """
        lines = [
            "Project Statistics",
            "=" * 50,
            f"Root: {structure.root_path}",
            f"Total Files: {structure.total_files}",
            f"Total Lines: {structure.total_lines:,}",
            "",
            "Languages:",
        ]

        for lang, count in sorted(
            structure.languages.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {lang}: {count} files")

        if structure.dependencies:
            lines.append("")
            lines.append(f"External Dependencies: {len(structure.dependencies)}")

        return "\n".join(lines)

    def find_symbol_references(self, root_path: str, symbol_name: str) -> List[Dict]:
        """
        Find all references to a symbol across the codebase.

        Args:
            root_path: Root directory
            symbol_name: Name of symbol to find

        Returns:
            List of reference locations
        """
        references = []

        for root, dirs, files in os.walk(root_path):
            # Exclude common build/cache directories
            dirs[:] = [
                d
                for d in dirs
                if d not in ["__pycache__", ".git", "node_modules", "venv"]
            ]

            for filename in files:
                if not filename.endswith(".py"):
                    continue

                file_path = os.path.join(root, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines, 1):
                        if symbol_name in line:
                            references.append(
                                {"file": file_path, "line": i, "content": line.strip()}
                            )

                except (FileNotFoundError, PermissionError, UnicodeDecodeError, OSError) as e:
                    # File unreadable or encoding issues - skip this file
                    logger.debug(f"Cannot read {file_path} for symbol search: {type(e).__name__}")
                    continue

        return references
