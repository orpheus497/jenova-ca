import inspect
import json
from jenova.default_api import *

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
        # Register functions
        for name, func in inspect.getmembers(inspect.getmodule(inspect.currentframe()), inspect.isfunction):
            if name != 'ToolHandler':
                self.tools[name] = func

        # Register FileTools class methods
        file_tools = FileTools(self.config['tools']['file_sandbox_path'])
        for name, func in inspect.getmembers(file_tools, inspect.ismethod):
            if not name.startswith('_'):
                self.tools[name] = func

    def execute_tool(self, tool_name: str, kwargs: dict) -> any:
        """
        Executes a tool with the given arguments.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        try:
            return self.tools[tool_name](**kwargs)
        except Exception as e:
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
