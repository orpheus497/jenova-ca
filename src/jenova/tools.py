# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for handling the tools for the JENOVA Cognitive Architecture.
"""

import inspect

from jenova.default_api import get_current_datetime, execute_shell_command, web_search, FileTools


class ToolHandler:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.tools = {}
        self._register_tools()

    def _register_tools(self):
        """
        Registers all the tools available in the default_api module.
        """
        self.tools["get_current_datetime"] = get_current_datetime
        self.tools["execute_shell_command"] = execute_shell_command
        self.tools["web_search"] = web_search

        # Register FileTools class methods
        file_tools = FileTools(self.config['tools']['file_sandbox_path'])
        self.tools["read_file"] = file_tools.read_file
        self.tools["write_file"] = file_tools.write_file
        self.tools["list_directory"] = file_tools.list_directory

    def execute_tool(self, tool_name: str, kwargs: dict) -> any:
        """
        Executes a tool with the given arguments.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        # Security check for shell commands
        if tool_name == 'execute_shell_command':
            command_to_run = kwargs.get('command', '').split()[0]
            whitelist = self.config.get('tools', {}).get(
                'shell_command_whitelist', [])
            if command_to_run not in whitelist:
                if self.file_logger:
                    self.file_logger.log_warning(
                        f"Blocked shell command: {kwargs.get('command')}")
                return f"Error: Command '{command_to_run}' is not in the allowed list."

        try:
            return self.tools[tool_name](**kwargs)
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error executing tool '{tool_name}': {e}")
            return f"Error executing tool '{tool_name}': {e}"

    def get_tool_schema(self, tool_name: str) -> dict | None:
        """
        Returns the schema of a tool.
        """
        if tool_name not in self.tools:
            return None

        tool = self.tools[tool_name]
        argspec = inspect.getfullargspec(tool)

        # Handle methods of FileTools class
        if inspect.ismethod(tool):
            # Exclude 'self' from the arguments
            args = [arg for arg in argspec.args if arg != 'self']
        else:
            args = argspec.args

        return {
            "name": tool_name,
            "description": inspect.getdoc(tool),
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string"} for arg in args},
                "required": args,
            },
        }

    def get_all_tool_schemas(self) -> list[dict]:
        """
        Returns the schemas of all available tools.
        """
        return [self.get_tool_schema(tool_name) for tool_name in self.tools]


# Global llm_interface to be set by main.py
llm_interface = None
