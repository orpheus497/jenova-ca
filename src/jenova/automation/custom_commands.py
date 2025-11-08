"""
JENOVA Cognitive Architecture - Custom Commands Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides custom command management with Markdown template support.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class CustomCommand:
    """Represents a custom command."""

    name: str
    content: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "content_length": len(self.content)
        }


class CustomCommandManager:
    """
    Manages custom commands with Markdown template support.

    Features:
    - Load commands from Markdown files
    - Variable extraction from templates
    - Command discovery and indexing
    - Metadata parsing from frontmatter
    - Command validation
    """

    def __init__(self, commands_dir: str = ".jenova/commands"):
        """
        Initialize the custom command manager.

        Args:
            commands_dir: Directory containing command templates
        """
        self.commands_dir = Path(commands_dir)
        self.commands: Dict[str, CustomCommand] = {}

        # Create commands directory if it doesn't exist
        self.commands_dir.mkdir(parents=True, exist_ok=True)

        # Load existing commands
        self.reload_commands()

    def load_command(self, name: str) -> Optional[str]:
        """
        Load a custom command template by name.

        Args:
            name: Command name

        Returns:
            Command content or None if not found
        """
        command = self.commands.get(name)
        if command:
            return command.content

        # Try loading from file
        command_path = self.commands_dir / f"{name}.md"
        if command_path.exists():
            try:
                with open(command_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse and cache
                command = self._parse_command_file(name, content)
                self.commands[name] = command

                logger.info(f"Loaded command: {name}")
                return command.content
            except Exception as e:
                logger.error(f"Error loading command {name}: {e}")
                return None

        return None

    def _parse_command_file(self, name: str, content: str) -> CustomCommand:
        """
        Parse a command file and extract metadata.

        Args:
            name: Command name
            content: File content

        Returns:
            CustomCommand object
        """
        # Extract frontmatter if present
        metadata = {}
        description = ""
        actual_content = content

        # Check for YAML frontmatter (---...---)
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            frontmatter_text = match.group(1)
            actual_content = match.group(2)

            # Parse simple YAML-like frontmatter
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key.lower() == 'description':
                        description = value
                    else:
                        metadata[key] = value

        # Extract variables from content ({{variable}} pattern)
        variables = self._extract_variables(actual_content)

        return CustomCommand(
            name=name,
            content=actual_content,
            description=description,
            variables=variables,
            metadata=metadata
        )

    def _extract_variables(self, content: str) -> List[str]:
        """
        Extract variable names from template content.

        Args:
            content: Template content

        Returns:
            List of variable names
        """
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, content)
        return list(set(matches))  # Unique variables

    def reload_commands(self) -> int:
        """
        Reload all commands from the commands directory.

        Returns:
            Number of commands loaded
        """
        self.commands.clear()
        count = 0

        for command_file in self.commands_dir.glob("*.md"):
            name = command_file.stem
            try:
                with open(command_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                command = self._parse_command_file(name, content)
                self.commands[name] = command
                count += 1
            except Exception as e:
                logger.error(f"Error loading command {name}: {e}")

        logger.info(f"Loaded {count} custom commands")
        return count

    def list_commands(self) -> List[Dict[str, Any]]:
        """
        List all available commands.

        Returns:
            List of command metadata dictionaries
        """
        return [cmd.to_dict() for cmd in self.commands.values()]

    def get_command_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a command.

        Args:
            name: Command name

        Returns:
            Command metadata or None if not found
        """
        command = self.commands.get(name)
        return command.to_dict() if command else None

    def create_command(
        self,
        name: str,
        content: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new custom command.

        Args:
            name: Command name
            content: Command template content
            description: Command description
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare frontmatter
            frontmatter_lines = ["---"]
            if description:
                frontmatter_lines.append(f"description: {description}")
            if metadata:
                for key, value in metadata.items():
                    frontmatter_lines.append(f"{key}: {value}")
            frontmatter_lines.append("---")
            frontmatter_lines.append("")

            # Combine frontmatter and content
            full_content = "\n".join(frontmatter_lines) + content

            # Write to file
            command_path = self.commands_dir / f"{name}.md"
            with open(command_path, 'w', encoding='utf-8') as f:
                f.write(full_content)

            # Parse and cache
            command = self._parse_command_file(name, full_content)
            self.commands[name] = command

            logger.info(f"Created command: {name}")
            return True

        except Exception as e:
            logger.error(f"Error creating command {name}: {e}")
            return False

    def update_command(
        self,
        name: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing command.

        Args:
            name: Command name
            content: New content (optional)
            description: New description (optional)
            metadata: New metadata (optional)

        Returns:
            True if successful, False otherwise
        """
        command = self.commands.get(name)
        if not command:
            logger.error(f"Command not found: {name}")
            return False

        try:
            # Update fields
            if content is not None:
                command.content = content
                command.variables = self._extract_variables(content)
            if description is not None:
                command.description = description
            if metadata is not None:
                command.metadata.update(metadata)

            command.modified_at = datetime.now()

            # Write to file
            return self._write_command_file(command)

        except Exception as e:
            logger.error(f"Error updating command {name}: {e}")
            return False

    def _write_command_file(self, command: CustomCommand) -> bool:
        """Write a command to file."""
        try:
            # Prepare frontmatter
            frontmatter_lines = ["---"]
            if command.description:
                frontmatter_lines.append(f"description: {command.description}")
            for key, value in command.metadata.items():
                frontmatter_lines.append(f"{key}: {value}")
            frontmatter_lines.append("---")
            frontmatter_lines.append("")

            full_content = "\n".join(frontmatter_lines) + command.content

            command_path = self.commands_dir / f"{command.name}.md"
            with open(command_path, 'w', encoding='utf-8') as f:
                f.write(full_content)

            logger.info(f"Wrote command file: {command.name}")
            return True

        except Exception as e:
            logger.error(f"Error writing command {command.name}: {e}")
            return False

    def delete_command(self, name: str) -> bool:
        """
        Delete a command.

        Args:
            name: Command name

        Returns:
            True if successful, False otherwise
        """
        try:
            command_path = self.commands_dir / f"{name}.md"
            if command_path.exists():
                command_path.unlink()

            if name in self.commands:
                del self.commands[name]

            logger.info(f"Deleted command: {name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting command {name}: {e}")
            return False

    def search_commands(self, query: str) -> List[Dict[str, Any]]:
        """
        Search commands by name or description.

        Args:
            query: Search query

        Returns:
            List of matching command metadata
        """
        query_lower = query.lower()
        results = []

        for command in self.commands.values():
            if (query_lower in command.name.lower() or
                query_lower in command.description.lower()):
                results.append(command.to_dict())

        return results

    def validate_command(self, name: str) -> Dict[str, Any]:
        """
        Validate a command template.

        Args:
            name: Command name

        Returns:
            Dictionary with validation results
        """
        command = self.commands.get(name)
        if not command:
            return {
                "valid": False,
                "errors": [f"Command not found: {name}"]
            }

        errors = []
        warnings = []

        # Check for empty content
        if not command.content.strip():
            errors.append("Command content is empty")

        # Check for unbalanced variables
        open_braces = command.content.count("{{")
        close_braces = command.content.count("}}")
        if open_braces != close_braces:
            warnings.append("Unbalanced variable braces")

        # Check for invalid variable names
        for var in command.variables:
            if not re.match(r'^\w+$', var):
                warnings.append(f"Invalid variable name: {var}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "variables": command.variables
        }
