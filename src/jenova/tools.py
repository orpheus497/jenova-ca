# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for handling the tools for the JENOVA Cognitive Architecture.
"""

import inspect

from jenova.default_api import (
    get_current_datetime, execute_shell_command, web_search, FileTools,
    CodeToolsAPI, GitToolsAPI, AnalysisToolsAPI, OrchestrationToolsAPI, AutomationToolsAPI
)


class ToolHandler:
    def __init__(self, config, ui_logger, file_logger, **cli_modules):
        """
        Initialize ToolHandler with core config and optional CLI enhancement modules.

        Args:
            config: Configuration dictionary
            ui_logger: UI logger instance
            file_logger: File logger instance
            **cli_modules: Optional Phase 13-17 CLI enhancement modules
        """
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.tools = {}

        # Phase 13-17: Store CLI enhancement module instances
        self.file_editor = cli_modules.get('file_editor')
        self.code_parser = cli_modules.get('code_parser')
        self.refactoring_engine = cli_modules.get('refactoring_engine')
        self.syntax_highlighter = cli_modules.get('syntax_highlighter')
        self.codebase_mapper = cli_modules.get('codebase_mapper')
        self.interactive_terminal = cli_modules.get('interactive_terminal')

        self.git_interface = cli_modules.get('git_interface')
        self.commit_assistant = cli_modules.get('commit_assistant')
        self.diff_analyzer = cli_modules.get('diff_analyzer')
        self.hooks_manager = cli_modules.get('hooks_manager')
        self.branch_manager = cli_modules.get('branch_manager')

        self.context_optimizer = cli_modules.get('context_optimizer')
        self.code_metrics = cli_modules.get('code_metrics')
        self.security_scanner = cli_modules.get('security_scanner')
        self.intent_classifier = cli_modules.get('intent_classifier')
        self.command_disambiguator = cli_modules.get('command_disambiguator')

        self.task_planner = cli_modules.get('task_planner')
        self.subagent_manager = cli_modules.get('subagent_manager')
        self.execution_engine = cli_modules.get('execution_engine')
        self.checkpoint_manager = cli_modules.get('checkpoint_manager')
        self.background_tasks = cli_modules.get('background_tasks')

        self.custom_commands = cli_modules.get('custom_commands')
        self.hooks_system = cli_modules.get('hooks_system')
        self.template_engine = cli_modules.get('template_engine')
        self.workflow_library = cli_modules.get('workflow_library')

        # Initialize Phase 13-17 API wrappers if modules are available
        self.code_tools_api = None
        self.git_tools_api = None
        self.analysis_tools_api = None
        self.orchestration_tools_api = None
        self.automation_tools_api = None

        if all([self.file_editor, self.code_parser, self.refactoring_engine,
                self.syntax_highlighter, self.codebase_mapper, self.interactive_terminal]):
            self.code_tools_api = CodeToolsAPI(
                self.file_editor, self.code_parser, self.refactoring_engine,
                self.syntax_highlighter, self.codebase_mapper, self.interactive_terminal
            )

        if all([self.git_interface, self.commit_assistant, self.diff_analyzer,
                self.hooks_manager, self.branch_manager]):
            self.git_tools_api = GitToolsAPI(
                self.git_interface, self.commit_assistant, self.diff_analyzer,
                self.hooks_manager, self.branch_manager
            )

        if all([self.context_optimizer, self.code_metrics, self.security_scanner,
                self.intent_classifier, self.command_disambiguator]):
            self.analysis_tools_api = AnalysisToolsAPI(
                self.context_optimizer, self.code_metrics, self.security_scanner,
                self.intent_classifier, self.command_disambiguator
            )

        if all([self.task_planner, self.subagent_manager, self.execution_engine,
                self.checkpoint_manager, self.background_tasks]):
            self.orchestration_tools_api = OrchestrationToolsAPI(
                self.task_planner, self.subagent_manager, self.execution_engine,
                self.checkpoint_manager, self.background_tasks
            )

        if all([self.custom_commands, self.hooks_system, self.template_engine,
                self.workflow_library]):
            self.automation_tools_api = AutomationToolsAPI(
                self.custom_commands, self.hooks_system, self.template_engine,
                self.workflow_library
            )

        self._register_tools()

    def _register_tools(self):
        """
        Registers all the tools available in the default_api module.
        Includes Phase 13-17 CLI enhancement tools if modules are available.
        """
        # Core tools (always available)
        self.tools["get_current_datetime"] = get_current_datetime
        self.tools["execute_shell_command"] = execute_shell_command
        self.tools["web_search"] = web_search

        # Register FileTools class methods
        file_tools = FileTools(self.config['tools']['file_sandbox_path'])
        self.tools["read_file"] = file_tools.read_file
        self.tools["write_file"] = file_tools.write_file
        self.tools["list_directory"] = file_tools.list_directory

        # Phase 13-17: Code Tools (if available)
        if self.code_tools_api:
            self.tools["edit_file"] = self.code_tools_api.edit_file
            self.tools["parse_code"] = self.code_tools_api.parse_code
            self.tools["refactor_rename"] = self.code_tools_api.refactor_rename
            self.tools["highlight_syntax"] = self.code_tools_api.highlight_syntax
            self.tools["map_codebase"] = self.code_tools_api.map_codebase

        # Phase 13-17: Git Tools (if available)
        if self.git_tools_api:
            self.tools["git_status"] = self.git_tools_api.git_status
            self.tools["git_diff"] = self.git_tools_api.git_diff
            self.tools["git_commit"] = self.git_tools_api.git_commit
            self.tools["git_branch"] = self.git_tools_api.git_branch

        # Phase 13-17: Analysis Tools (if available)
        if self.analysis_tools_api:
            self.tools["optimize_context"] = self.analysis_tools_api.optimize_context
            self.tools["analyze_code_metrics"] = self.analysis_tools_api.analyze_code_metrics
            self.tools["scan_security"] = self.analysis_tools_api.scan_security
            self.tools["classify_intent"] = self.analysis_tools_api.classify_intent
            self.tools["disambiguate_command"] = self.analysis_tools_api.disambiguate_command

        # Phase 13-17: Orchestration Tools (if available)
        if self.orchestration_tools_api:
            self.tools["create_task_plan"] = self.orchestration_tools_api.create_task_plan
            self.tools["execute_task_plan"] = self.orchestration_tools_api.execute_task_plan
            self.tools["spawn_subagent"] = self.orchestration_tools_api.spawn_subagent
            self.tools["save_checkpoint"] = self.orchestration_tools_api.save_checkpoint
            self.tools["run_background_task"] = self.orchestration_tools_api.run_background_task

        # Phase 13-17: Automation Tools (if available)
        if self.automation_tools_api:
            self.tools["create_custom_command"] = self.automation_tools_api.create_custom_command
            self.tools["execute_workflow"] = self.automation_tools_api.execute_workflow
            self.tools["register_hook"] = self.automation_tools_api.register_hook
            self.tools["render_template"] = self.automation_tools_api.render_template

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
