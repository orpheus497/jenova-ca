# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Base Tool Classes

"""
Base classes for tool implementation.

This module provides the foundational abstractions for all tools in the JENOVA
system, ensuring consistent interfaces, error handling, and result formatting.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.

    Attributes:
        success: Whether the tool executed successfully
        data: Result data (type varies by tool)
        error: Error message if execution failed
        metadata: Additional metadata about execution
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Returns:
            Dictionary representation of result
        """
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata or {}
        }


class ToolError(Exception):
    """
    Exception raised when tool execution fails.

    Attributes:
        tool_name: Name of the tool that failed
        message: Error message
        context: Additional context about the failure
    """

    def __init__(self, tool_name: str, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize tool error.

        Args:
            tool_name: Name of the tool
            message: Error message
            context: Additional error context
        """
        self.tool_name = tool_name
        self.message = message
        self.context = context or {}
        super().__init__(f"{tool_name}: {message}")


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    All tools in the JENOVA system should inherit from this class and implement
    the execute() method. This ensures consistent interfaces and error handling
    across all tools.

    Attributes:
        name: Tool identifier
        description: Human-readable tool description
        config: Configuration dictionary
        ui_logger: UI logger instance
        file_logger: File logger instance
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        ui_logger: Any,
        file_logger: Any
    ):
        """
        Initialize base tool.

        Args:
            name: Tool identifier
            description: Tool description
            config: Configuration dictionary
            ui_logger: UI logger instance
            file_logger: File logger instance
        """
        self.name = name
        self.description = description
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult containing execution results

        Raises:
            ToolError: If execution fails
        """
        pass

    def _create_success_result(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Create successful tool result.

        Args:
            data: Result data
            metadata: Additional metadata

        Returns:
            ToolResult indicating success
        """
        return ToolResult(success=True, data=data, metadata=metadata)

    def _create_error_result(
        self,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Create error tool result.

        Args:
            error: Error message
            metadata: Additional metadata

        Returns:
            ToolResult indicating failure
        """
        return ToolResult(success=False, error=error, metadata=metadata)

    def _log_execution(self, args: Dict[str, Any], result: ToolResult) -> None:
        """
        Log tool execution for audit trail.

        Args:
            args: Tool arguments
            result: Execution result
        """
        if self.file_logger:
            if result.success:
                self.file_logger.log_info(
                    f"Tool '{self.name}' executed successfully"
                )
            else:
                self.file_logger.log_error(
                    f"Tool '{self.name}' failed: {result.error}"
                )
