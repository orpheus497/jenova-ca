# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module provides the default API for the JENOVA Cognitive Architecture.
"""

import datetime
import os
import shlex
import subprocess

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options


def get_current_datetime() -> str:
    """
    Returns the current date and time in ISO 8601 format.
    """
    return datetime.datetime.now().isoformat()


def execute_shell_command(command: str) -> dict:
    """
    Executes a shell command and returns the result.
    """
    try:
        # Use shlex.split to safely parse the command string
        command_args = shlex.split(command)
        result = subprocess.run(
            command_args, capture_output=True, text=True, check=False, timeout=30)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": result.returncode != 0
        }
    except FileNotFoundError:
        return {
            "stdout": "",
            "stderr": f"Command not found: {command.split()[0]}",
            "returncode": -1,
            "error": True
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Command timed out after 30 seconds.",
            "returncode": -1,
            "error": True
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": True
        }


def web_search(query: str) -> list[dict] | str:
    """
    Performs a web search using DuckDuckGo and returns the results.
    """
    # Preserve CUDA environment for subprocess (geckodriver/Firefox)
    # This prevents CUDA context conflicts when spawning the browser
    import os
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    options = Options()
    options.headless = True
    driver = None
    try:
        # Temporarily hide CUDA from the browser subprocess to prevent conflicts
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        driver = webdriver.Firefox(options=options)
        driver.get(f"https://duckduckgo.com/html/?q={query}")
        results = []
        # Limit to top 5 results
        for result in driver.find_elements(By.CLASS_NAME, "result")[:5]:
            title = result.find_element(By.CLASS_NAME, "result__title").text
            link = result.find_element(
                By.CLASS_NAME, "result__url").get_attribute("href")
            snippet = result.find_element(
                By.CLASS_NAME, "result__snippet").text
            results.append({"title": title, "link": link, "summary": snippet})
        return results
    except WebDriverException as e:
        return f"Error: Web search failed. Could not initialize browser. Please ensure Firefox is installed. Details: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred during web search: {e}"
    finally:
        # Always restore original CUDA visibility setting
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        if driver:
            try:
                driver.quit()
            except Exception:
                # Suppress cleanup errors
                pass


class FileTools:
    def __init__(self, sandbox_path: str):
        self.sandbox_path = os.path.realpath(os.path.expanduser(sandbox_path))
        if not os.path.exists(self.sandbox_path):
            os.makedirs(self.sandbox_path)

    def _get_safe_path(self, path: str) -> str | None:
        """
        Resolves a path to a real, absolute path within the sandbox.
        Returns None if the path is outside the sandbox or is a symlink.
        """
        # Normalize the user-provided path by removing any relative path components
        normalized_path = os.path.normpath(path)
        # Prevent absolute paths from being treated as relative
        if os.path.isabs(normalized_path):
            return None

        # Join with the sandbox root
        prospective_path = os.path.join(self.sandbox_path, normalized_path)

        # Get the real, absolute path, resolving any symlinks
        real_path = os.path.realpath(prospective_path)

        # Check if the resolved path is within the sandbox directory
        if os.path.commonprefix([self.sandbox_path, real_path]) != self.sandbox_path:
            return None

        return real_path

    def read_file(self, path: str) -> str:
        """
        Reads the content of a file within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox or is invalid."
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found at '{path}'."
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, path: str, content: str) -> str:
        """
        Writes content to a file within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox or is invalid."
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(safe_path)
            os.makedirs(parent_dir, exist_ok=True)

            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File written successfully to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def list_directory(self, path: str) -> list[str] | str:
        """
        Lists the contents of a directory within the sandbox.
        """
        safe_path = self._get_safe_path(path)
        if not safe_path:
            return "Error: Path is outside the sandbox or is invalid."
        try:
            return os.listdir(safe_path)
        except FileNotFoundError:
            return f"Error: Directory not found at '{path}'."
        except Exception as e:
            return f"Error listing directory: {e}"


# Phase 13-17: Enhanced CLI Capabilities - Tool API Classes
# These classes provide LLM-callable wrappers for all Phase 13-17 modules

class CodeToolsAPI:
    """
    API wrapper for code tools (file editing, parsing, refactoring, etc.).

    Provides LLM-callable functions for advanced code operations.
    """

    def __init__(self, file_editor, code_parser, refactoring_engine,
                 syntax_highlighter, codebase_mapper, interactive_terminal):
        """Initialize CodeToolsAPI with all code tool instances."""
        self.file_editor = file_editor
        self.code_parser = code_parser
        self.refactoring_engine = refactoring_engine
        self.syntax_highlighter = syntax_highlighter
        self.codebase_mapper = codebase_mapper
        self.interactive_terminal = interactive_terminal

    def edit_file(self, file_path: str, old_text: str, new_text: str) -> dict:
        """
        Edit a file by replacing old_text with new_text.

        Args:
            file_path: Path to file to edit
            old_text: Text to find and replace
            new_text: Replacement text

        Returns:
            dict with status, message, and optional diff
        """
        try:
            if not self.file_editor:
                return {"status": "error", "message": "File editor not available"}

            result = self.file_editor.apply_edit(file_path, old_text, new_text)
            return {
                "status": "success" if result else "error",
                "message": "File edited successfully" if result else "Edit failed",
                "file_path": file_path
            }
        except Exception as e:
            return {"status": "error", "message": f"Error editing file: {e}"}

    def parse_code(self, file_path: str) -> dict:
        """
        Parse code file and extract structure (classes, functions, etc.).

        Args:
            file_path: Path to source code file

        Returns:
            dict with parsed structure and symbols
        """
        try:
            if not self.code_parser:
                return {"status": "error", "message": "Code parser not available"}

            structure = self.code_parser.parse_file(file_path)
            return {
                "status": "success",
                "file_path": file_path,
                "structure": structure
            }
        except Exception as e:
            return {"status": "error", "message": f"Error parsing code: {e}"}

    def refactor_rename(self, file_path: str, old_name: str, new_name: str) -> dict:
        """
        Rename a symbol (variable, function, class) in code.

        Args:
            file_path: Path to source file
            old_name: Current symbol name
            new_name: New symbol name

        Returns:
            dict with refactoring status
        """
        try:
            if not self.refactoring_engine:
                return {"status": "error", "message": "Refactoring engine not available"}

            result = self.refactoring_engine.rename(file_path, old_name, new_name)
            return {
                "status": "success" if result else "error",
                "message": f"Renamed {old_name} to {new_name}" if result else "Rename failed",
                "file_path": file_path
            }
        except Exception as e:
            return {"status": "error", "message": f"Error during refactoring: {e}"}

    def highlight_syntax(self, code: str, language: str = "python") -> dict:
        """
        Apply syntax highlighting to code for terminal display.

        Args:
            code: Source code to highlight
            language: Programming language (default: python)

        Returns:
            dict with highlighted code
        """
        try:
            if not self.syntax_highlighter:
                return {"status": "error", "message": "Syntax highlighter not available"}

            highlighted = self.syntax_highlighter.highlight(code, language)
            return {
                "status": "success",
                "language": language,
                "highlighted_code": highlighted
            }
        except Exception as e:
            return {"status": "error", "message": f"Error highlighting syntax: {e}"}

    def map_codebase(self, directory: str, max_depth: int = 3) -> dict:
        """
        Generate codebase structure map and dependency graph.

        Args:
            directory: Root directory to analyze
            max_depth: Maximum depth to traverse

        Returns:
            dict with codebase structure and dependencies
        """
        try:
            if not self.codebase_mapper:
                return {"status": "error", "message": "Codebase mapper not available"}

            structure = self.codebase_mapper.map_directory(directory, max_depth)
            return {
                "status": "success",
                "directory": directory,
                "structure": structure
            }
        except Exception as e:
            return {"status": "error", "message": f"Error mapping codebase: {e}"}


class GitToolsAPI:
    """
    API wrapper for Git operations (status, commit, branch, etc.).

    Provides LLM-callable functions for Git workflow automation.
    """

    def __init__(self, git_interface, commit_assistant, diff_analyzer,
                 hooks_manager, branch_manager):
        """Initialize GitToolsAPI with all git tool instances."""
        self.git_interface = git_interface
        self.commit_assistant = commit_assistant
        self.diff_analyzer = diff_analyzer
        self.hooks_manager = hooks_manager
        self.branch_manager = branch_manager

    def git_status(self) -> dict:
        """
        Get repository status (modified files, staged changes, etc.).

        Returns:
            dict with repository status information
        """
        try:
            if not self.git_interface:
                return {"status": "error", "message": "Git interface not available"}

            status = self.git_interface.status()
            return {
                "status": "success",
                "repository_status": status
            }
        except Exception as e:
            return {"status": "error", "message": f"Error getting git status: {e}"}

    def git_diff(self, staged: bool = False) -> dict:
        """
        View changes in working directory or staging area.

        Args:
            staged: Show staged changes if True, unstaged if False

        Returns:
            dict with diff output and analysis
        """
        try:
            if not self.git_interface:
                return {"status": "error", "message": "Git interface not available"}

            diff = self.git_interface.diff(staged=staged)

            # Analyze diff if diff_analyzer available
            analysis = None
            if self.diff_analyzer and diff:
                analysis = self.diff_analyzer.analyze(diff)

            return {
                "status": "success",
                "diff": diff,
                "analysis": analysis,
                "staged": staged
            }
        except Exception as e:
            return {"status": "error", "message": f"Error getting diff: {e}"}

    def git_commit(self, message: str = None, auto_generate: bool = False) -> dict:
        """
        Commit staged changes with optional auto-generated message.

        Args:
            message: Commit message (if None and auto_generate=True, generates message)
            auto_generate: Auto-generate commit message from diff

        Returns:
            dict with commit status
        """
        try:
            if not self.git_interface:
                return {"status": "error", "message": "Git interface not available"}

            # Auto-generate message if requested
            if auto_generate and not message and self.commit_assistant:
                diff = self.git_interface.diff(staged=True)
                message = self.commit_assistant.generate_message(diff)

            if not message:
                return {"status": "error", "message": "No commit message provided"}

            result = self.git_interface.commit(message)
            return {
                "status": "success" if result else "error",
                "message": message,
                "committed": result
            }
        except Exception as e:
            return {"status": "error", "message": f"Error committing: {e}"}

    def git_branch(self, operation: str, branch_name: str = None) -> dict:
        """
        Perform branch operations (list, create, delete, switch).

        Args:
            operation: Operation to perform (list, create, delete, switch)
            branch_name: Name of branch (required for create, delete, switch)

        Returns:
            dict with operation result
        """
        try:
            if not self.branch_manager:
                return {"status": "error", "message": "Branch manager not available"}

            if operation == "list":
                branches = self.branch_manager.list_branches()
                return {"status": "success", "branches": branches}
            elif operation == "create" and branch_name:
                result = self.branch_manager.create_branch(branch_name)
                return {"status": "success" if result else "error", "branch": branch_name}
            elif operation == "delete" and branch_name:
                result = self.branch_manager.delete_branch(branch_name)
                return {"status": "success" if result else "error", "branch": branch_name}
            elif operation == "switch" and branch_name:
                result = self.branch_manager.switch_branch(branch_name)
                return {"status": "success" if result else "error", "branch": branch_name}
            else:
                return {"status": "error", "message": f"Invalid operation or missing branch_name"}
        except Exception as e:
            return {"status": "error", "message": f"Error with branch operation: {e}"}


class AnalysisToolsAPI:
    """
    API wrapper for analysis tools (context optimization, code metrics, security, etc.).

    Provides LLM-callable functions for code analysis and intent classification.
    """

    def __init__(self, context_optimizer, code_metrics, security_scanner,
                 intent_classifier, command_disambiguator):
        """Initialize AnalysisToolsAPI with all analysis tool instances."""
        self.context_optimizer = context_optimizer
        self.code_metrics = code_metrics
        self.security_scanner = security_scanner
        self.intent_classifier = intent_classifier
        self.command_disambiguator = command_disambiguator

    def optimize_context(self, text: str, max_tokens: int = 2000) -> dict:
        """
        Optimize text for context window with relevance scoring.

        Args:
            text: Text to optimize
            max_tokens: Maximum tokens to keep

        Returns:
            dict with optimized text and metadata
        """
        try:
            if not self.context_optimizer:
                return {"status": "error", "message": "Context optimizer not available"}

            optimized = self.context_optimizer.optimize(text, max_tokens)
            return {
                "status": "success",
                "optimized_text": optimized,
                "original_tokens": len(text.split()),
                "optimized_tokens": len(optimized.split())
            }
        except Exception as e:
            return {"status": "error", "message": f"Error optimizing context: {e}"}

    def analyze_code_metrics(self, file_path: str) -> dict:
        """
        Analyze code complexity and maintainability metrics.

        Args:
            file_path: Path to source code file

        Returns:
            dict with complexity metrics and maintainability index
        """
        try:
            if not self.code_metrics:
                return {"status": "error", "message": "Code metrics analyzer not available"}

            metrics = self.code_metrics.analyze_file(file_path)
            return {
                "status": "success",
                "file_path": file_path,
                "metrics": metrics
            }
        except Exception as e:
            return {"status": "error", "message": f"Error analyzing code metrics: {e}"}

    def scan_security(self, path: str, output_format: str = "text") -> dict:
        """
        Scan code for security vulnerabilities using Bandit.

        Args:
            path: File or directory to scan
            output_format: Output format (text, json, html)

        Returns:
            dict with security scan results
        """
        try:
            if not self.security_scanner:
                return {"status": "error", "message": "Security scanner not available"}

            results = self.security_scanner.scan(path, output_format)
            return {
                "status": "success",
                "path": path,
                "format": output_format,
                "results": results
            }
        except Exception as e:
            return {"status": "error", "message": f"Error scanning security: {e}"}

    def classify_intent(self, text: str) -> dict:
        """
        Classify user intent from natural language input.

        Args:
            text: User input text

        Returns:
            dict with classified intent and confidence score
        """
        try:
            if not self.intent_classifier:
                return {"status": "error", "message": "Intent classifier not available"}

            intent = self.intent_classifier.classify(text)
            return {
                "status": "success",
                "text": text,
                "intent": intent
            }
        except Exception as e:
            return {"status": "error", "message": f"Error classifying intent: {e}"}

    def disambiguate_command(self, partial: str) -> dict:
        """
        Find best command match using fuzzy matching.

        Args:
            partial: Partial command string

        Returns:
            dict with best matches and confidence scores
        """
        try:
            if not self.command_disambiguator:
                return {"status": "error", "message": "Command disambiguator not available"}

            matches = self.command_disambiguator.find_matches(partial)
            return {
                "status": "success",
                "partial": partial,
                "matches": matches
            }
        except Exception as e:
            return {"status": "error", "message": f"Error disambiguating command: {e}"}


class OrchestrationToolsAPI:
    """
    API wrapper for orchestration tools (task planning, subagents, execution, etc.).

    Provides LLM-callable functions for complex multi-step task management.
    """

    def __init__(self, task_planner, subagent_manager, execution_engine,
                 checkpoint_manager, background_tasks):
        """Initialize OrchestrationToolsAPI with all orchestration tool instances."""
        self.task_planner = task_planner
        self.subagent_manager = subagent_manager
        self.execution_engine = execution_engine
        self.checkpoint_manager = checkpoint_manager
        self.background_tasks = background_tasks

    def create_task_plan(self, description: str) -> dict:
        """
        Create multi-step task plan with dependency graph.

        Args:
            description: Natural language task description

        Returns:
            dict with task plan ID and steps
        """
        try:
            if not self.task_planner:
                return {"status": "error", "message": "Task planner not available"}

            plan = self.task_planner.create_plan(description)
            return {
                "status": "success",
                "plan_id": plan.get("id"),
                "description": description,
                "steps": plan.get("steps", [])
            }
        except Exception as e:
            return {"status": "error", "message": f"Error creating task plan: {e}"}

    def execute_task_plan(self, plan_id: str) -> dict:
        """
        Execute a task plan with retry logic and error handling.

        Args:
            plan_id: ID of task plan to execute

        Returns:
            dict with execution results
        """
        try:
            if not self.execution_engine:
                return {"status": "error", "message": "Execution engine not available"}

            result = self.execution_engine.execute(plan_id)
            return {
                "status": "success" if result.get("completed") else "error",
                "plan_id": plan_id,
                "result": result
            }
        except Exception as e:
            return {"status": "error", "message": f"Error executing task plan: {e}"}

    def spawn_subagent(self, task_description: str, priority: int = 5) -> dict:
        """
        Spawn background subagent to handle task asynchronously.

        Args:
            task_description: Description of task for subagent
            priority: Task priority (1-10, higher is more important)

        Returns:
            dict with subagent ID and status
        """
        try:
            if not self.subagent_manager:
                return {"status": "error", "message": "Subagent manager not available"}

            subagent_id = self.subagent_manager.create_subagent(task_description, priority)
            return {
                "status": "success",
                "subagent_id": subagent_id,
                "task": task_description,
                "priority": priority
            }
        except Exception as e:
            return {"status": "error", "message": f"Error spawning subagent: {e}"}

    def save_checkpoint(self, checkpoint_id: str, state: dict) -> dict:
        """
        Save execution checkpoint for resume capability.

        Args:
            checkpoint_id: Unique checkpoint identifier
            state: State dictionary to save

        Returns:
            dict with save status
        """
        try:
            if not self.checkpoint_manager:
                return {"status": "error", "message": "Checkpoint manager not available"}

            result = self.checkpoint_manager.save(checkpoint_id, state)
            return {
                "status": "success" if result else "error",
                "checkpoint_id": checkpoint_id,
                "saved": result
            }
        except Exception as e:
            return {"status": "error", "message": f"Error saving checkpoint: {e}"}

    def run_background_task(self, command: str, timeout: int = 3600) -> dict:
        """
        Run command in background with output capture and monitoring.

        Args:
            command: Shell command to run
            timeout: Maximum execution time in seconds

        Returns:
            dict with task ID and initial status
        """
        try:
            if not self.background_tasks:
                return {"status": "error", "message": "Background task manager not available"}

            task_id = self.background_tasks.start_task(command, timeout)
            return {
                "status": "success",
                "task_id": task_id,
                "command": command,
                "timeout": timeout
            }
        except Exception as e:
            return {"status": "error", "message": f"Error starting background task: {e}"}


class AutomationToolsAPI:
    """
    API wrapper for automation tools (custom commands, workflows, hooks, templates).

    Provides LLM-callable functions for workflow automation and custom commands.
    """

    def __init__(self, custom_commands, hooks_system, template_engine,
                 workflow_library):
        """Initialize AutomationToolsAPI with all automation tool instances."""
        self.custom_commands = custom_commands
        self.hooks_system = hooks_system
        self.template_engine = template_engine
        self.workflow_library = workflow_library

    def create_custom_command(self, name: str, template: str, description: str = "") -> dict:
        """
        Create custom command from template.

        Args:
            name: Command name
            template: Command template with variables
            description: Command description

        Returns:
            dict with creation status
        """
        try:
            if not self.custom_commands:
                return {"status": "error", "message": "Custom command manager not available"}

            result = self.custom_commands.create(name, template, description)
            return {
                "status": "success" if result else "error",
                "command_name": name,
                "created": result
            }
        except Exception as e:
            return {"status": "error", "message": f"Error creating custom command: {e}"}

    def execute_workflow(self, workflow_name: str, context: dict = None) -> dict:
        """
        Execute predefined workflow (code review, testing, deployment, etc.).

        Args:
            workflow_name: Name of workflow to execute
            context: Context variables for workflow

        Returns:
            dict with workflow execution results
        """
        try:
            if not self.workflow_library:
                return {"status": "error", "message": "Workflow library not available"}

            result = self.workflow_library.execute(workflow_name, context or {})
            return {
                "status": "success" if result.get("completed") else "error",
                "workflow_name": workflow_name,
                "result": result
            }
        except Exception as e:
            return {"status": "error", "message": f"Error executing workflow: {e}"}

    def register_hook(self, event: str, script: str, priority: int = 5) -> dict:
        """
        Register event-driven hook for automation.

        Args:
            event: Event to hook (pre_commit, post_execute, on_error, etc.)
            script: Script to execute on event
            priority: Hook priority (1-10, higher executes first)

        Returns:
            dict with registration status
        """
        try:
            if not self.hooks_system:
                return {"status": "error", "message": "Hooks system not available"}

            hook_id = self.hooks_system.register(event, script, priority)
            return {
                "status": "success",
                "hook_id": hook_id,
                "event": event,
                "priority": priority
            }
        except Exception as e:
            return {"status": "error", "message": f"Error registering hook: {e}"}

    def render_template(self, template: str, variables: dict) -> dict:
        """
        Render template with variables and filters.

        Args:
            template: Template string with {{variable}} syntax
            variables: Dictionary of variable values

        Returns:
            dict with rendered output
        """
        try:
            if not self.template_engine:
                return {"status": "error", "message": "Template engine not available"}

            rendered = self.template_engine.render(template, variables)
            return {
                "status": "success",
                "rendered": rendered,
                "variables": list(variables.keys())
            }
        except Exception as e:
            return {"status": "error", "message": f"Error rendering template: {e}"}
