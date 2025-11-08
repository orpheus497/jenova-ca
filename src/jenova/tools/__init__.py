# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Tools Module Refactoring

"""
Specialized tools module for The JENOVA Cognitive Architecture.

This module provides a clean, modular architecture for tool implementations,
replacing the monolithic default_api.py (970 lines) with specialized tool modules.

The tools system is organized into specialized categories:
    - TimeTools: Datetime and temporal operations
    - ShellTools: Shell command execution with security controls
    - WebTools: Web search and content retrieval
    - FileTools: File system operations with sandboxing
    - CodeTools: Code analysis and manipulation (via CLI modules)
    - GitTools: Git operations and version control (via CLI modules)
    - AnalysisTools: Code quality and security analysis (via CLI modules)
    - OrchestrationTools: Task planning and execution (via CLI modules)
    - AutomationTools: Custom commands and workflows (via CLI modules)

Example:
    >>> from jenova.tools import ToolHandler
    >>> handler = ToolHandler(config, ui_logger, file_logger)
    >>> result = handler.execute_tool('get_current_datetime', {})
"""

from jenova.tools.base import BaseTool, ToolResult, ToolError
from jenova.tools.time_tools import TimeTools
from jenova.tools.shell_tools import ShellTools
from jenova.tools.web_tools import WebTools
from jenova.tools.file_tools import FileTools
from jenova.tools.tool_handler import ToolHandler

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolError",
    "TimeTools",
    "ShellTools",
    "WebTools",
    "FileTools",
    "ToolHandler",
]
