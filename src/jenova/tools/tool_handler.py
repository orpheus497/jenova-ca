# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Tool Handler

"""
Centralized tool handler and router.

This module provides unified tool management and execution routing for all
tools in the JENOVA cognitive architecture.
"""

from typing import Any, Dict, List, Optional, Type
from jenova.tools.base import BaseTool, ToolResult, ToolError
from jenova.tools.time_tools import TimeTools
from jenova.tools.shell_tools import ShellTools
from jenova.tools.web_tools import WebTools
from jenova.tools.file_tools import FileTools


class ToolHandler:
    """
    Centralized tool handler and router.

    Manages tool registration, initialization, and execution routing for all
    tools in the JENOVA system. Provides a unified interface for tool operations.

    Features:
        - Automatic tool registration and initialization
        - Unified tool execution interface
        - Tool discovery and metadata
        - Configuration management per tool
        - Error handling and logging

    Example:
        >>> handler = ToolHandler(config, ui_logger, file_logger)
        >>> result = handler.execute_tool('get_current_datetime', {})
        >>> print(result.data)  # "2025-11-08 14:30:00"
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ui_logger: Any,
        file_logger: Any
    ):
        """
        Initialize tool handler.

        Args:
            config: Configuration dictionary
            ui_logger: UI logger instance
            file_logger: File logger instance
        """
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger

        # Tool registry
        self.tools: Dict[str, BaseTool] = {}

        # Initialize all tools
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """
        Initialize and register all available tools.

        Instantiates all tool classes with configuration and registers them
        in the tool registry for execution routing.
        """
        try:
            # Initialize time tools
            time_tools = TimeTools(
                config=self.config,
                ui_logger=self.ui_logger,
                file_logger=self.file_logger
            )
            self.register_tool('time_tools', time_tools)

            # Initialize shell tools
            shell_tools = ShellTools(
                config=self.config,
                ui_logger=self.ui_logger,
                file_logger=self.file_logger
            )
            self.register_tool('shell_tools', shell_tools)

            # Initialize web tools
            web_tools = WebTools(
                config=self.config,
                ui_logger=self.ui_logger,
                file_logger=self.file_logger
            )
            self.register_tool('web_tools', web_tools)

            # Initialize file tools
            file_tools = FileTools(
                config=self.config,
                ui_logger=self.ui_logger,
                file_logger=self.file_logger
            )
            self.register_tool('file_tools', file_tools)

            if self.file_logger:
                self.file_logger.log_info(
                    f"Initialized {len(self.tools)} tools: "
                    f"{', '.join(self.tools.keys())}"
                )

        except Exception as e:
            error_msg = f"Failed to initialize tools: {str(e)}"
            if self.file_logger:
                self.file_logger.log_error(error_msg)
            raise ToolError("ToolHandler", error_msg)

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """
        Register a tool in the handler.

        Args:
            name: Tool identifier
            tool: Tool instance

        Example:
            >>> custom_tool = CustomTool(config, ui_logger, file_logger)
            >>> handler.register_tool('custom_tool', custom_tool)
        """
        self.tools[name] = tool

        if self.file_logger:
            self.file_logger.log_info(
                f"Registered tool: {name} ({tool.description})"
            )

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the handler.

        Args:
            name: Tool identifier

        Returns:
            True if tool was unregistered, False if not found

        Example:
            >>> handler.unregister_tool('custom_tool')
            True
        """
        if name in self.tools:
            del self.tools[name]
            if self.file_logger:
                self.file_logger.log_info(f"Unregistered tool: {name}")
            return True
        return False

    def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult from tool execution

        Raises:
            ToolError: If tool not found or execution fails

        Example:
            >>> # Execute time tool
            >>> result = handler.execute_tool(
            ...     'time_tools',
            ...     format_str='%Y-%m-%d',
            ...     timezone='America/New_York'
            ... )
            >>> print(result.data)  # "2025-11-08"

            >>> # Execute shell command
            >>> result = handler.execute_tool(
            ...     'shell_tools',
            ...     command='ls -la'
            ... )
            >>> print(result.data)  # Directory listing
        """
        try:
            # Validate tool exists
            if tool_name not in self.tools:
                available = ', '.join(self.tools.keys())
                raise ToolError(
                    "ToolHandler",
                    f"Tool '{tool_name}' not found. Available: {available}"
                )

            # Get tool instance
            tool = self.tools[tool_name]

            # Execute tool
            result = tool.execute(**kwargs)

            return result

        except ToolError:
            raise
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            if self.file_logger:
                self.file_logger.log_error(
                    f"Tool '{tool_name}' failed: {error_msg}"
                )
            raise ToolError(tool_name, error_msg)

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get tool instance by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance or None if not found

        Example:
            >>> time_tool = handler.get_tool('time_tools')
            >>> timestamp = time_tool.get_timestamp()
        """
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names.

        Returns:
            List of tool identifiers

        Example:
            >>> tools = handler.list_tools()
            >>> print(tools)  # ['time_tools', 'shell_tools', 'web_tools', 'file_tools']
        """
        return list(self.tools.keys())

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool metadata.

        Args:
            tool_name: Name of tool

        Returns:
            Dictionary with tool information or None if not found

        Example:
            >>> info = handler.get_tool_info('time_tools')
            >>> print(info['description'])  # "Provides current datetime..."
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return None

        return {
            'name': tool.name,
            'description': tool.description,
            'type': type(tool).__name__
        }

    def get_all_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered tools.

        Returns:
            Dictionary mapping tool names to their metadata

        Example:
            >>> all_info = handler.get_all_tools_info()
            >>> for name, info in all_info.items():
            ...     print(f"{name}: {info['description']}")
        """
        return {
            name: self.get_tool_info(name)
            for name in self.tools.keys()
        }

    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if tool is available.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is registered, False otherwise

        Example:
            >>> handler.is_tool_available('time_tools')
            True
        """
        return tool_name in self.tools
