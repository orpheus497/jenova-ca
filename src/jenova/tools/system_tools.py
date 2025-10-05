import subprocess
import platform
import datetime
import re

class SystemTools:
    """A collection of tools for interacting with the underlying system."""
    def __init__(self, ui_logger, file_logger):
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def handle_tool_request(self, response: str) -> str:
        system_info_pattern = re.compile(r'<TOOL:GET_SYSTEM_INFO\(\)>')
        datetime_pattern = re.compile(r'<TOOL:GET_CURRENT_DATETIME\(\)>')
        shell_pattern = re.compile(r'<TOOL:EXECUTE_SHELL\(command="([^"]+)"\)>')

        system_info_match = system_info_pattern.search(response)
        datetime_match = datetime_pattern.search(response)
        shell_match = shell_pattern.search(response)

        if system_info_match:
            self.ui_logger.info("Tool request detected: GET_SYSTEM_INFO")
            self.file_logger.log_info("Tool request detected: GET_SYSTEM_INFO")
            return system_info_pattern.sub(self.get_system_info(), response)
        elif datetime_match:
            self.ui_logger.info("Tool request detected: GET_CURRENT_DATETIME")
            self.file_logger.log_info("Tool request detected: GET_CURRENT_DATETIME")
            return datetime_pattern.sub(self.get_current_datetime(), response)
        elif shell_match:
            self.ui_logger.info("Tool request detected: EXECUTE_SHELL")
            self.file_logger.log_info("Tool request detected: EXECUTE_SHELL")
            command = shell_match.group(1)
            return shell_pattern.sub(self.execute_shell_command(command, "Executing user-requested shell command."), response)
        else:
            return response

    def execute_shell_command(self, command: str, description: str) -> str:
        """Executes a shell command and returns a dictionary with the output."""
        self.ui_logger.system_message(f"Executing command: {description}")
        self.file_logger.log_info(f"Executing command: {command}")
        try:
            process = subprocess.run(command, shell=True, capture_output=True, text=True, executable="/bin/bash")
            if process.returncode == 0:
                self.ui_logger.system_message("Command executed successfully.")
                self.file_logger.log_info(f"Command output:\n{process.stdout}")
                return process.stdout
            else:
                error_message = f"Command failed with return code {process.returncode}."
                self.ui_logger.system_message(f"{error_message}\n{process.stderr}")
                self.file_logger.log_error(f"{error_message}\n{process.stderr}")
                return f"{error_message}\n{process.stderr}"
        except Exception as e:
            error_message = f"An exception occurred while executing the command: {e}"
            self.ui_logger.system_message(error_message)
            self.file_logger.log_error(error_message)
            return error_message

    def get_system_info(self) -> str:
        """Returns a string containing system information."""
        info = f"System: {platform.system()}\n"
        info += f"Node Name: {platform.node()}\n"
        info += f"Release: {platform.release()}\n"
        info += f"Version: {platform.version()}\n"
        info += f"Machine: {platform.machine()}\n"
        info += f"Processor: {platform.processor()}\n"
        return info

    def get_current_datetime(self) -> str:
        """Returns the current date and time as a string."""
        return datetime.datetime.now().isoformat()