import subprocess

class SystemTools:
    """A collection of tools for interacting with the underlying system."""
    def __init__(self, ui_logger, file_logger):
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def execute_shell_command(self, command: str, description: str):
        """Executes a shell command and logs the output."""
        self.ui_logger.system_message(f"Executing command: {description}")
        self.file_logger.log_info(f"Executing command: {command}")
        try:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                self.ui_logger.system_message("Command executed successfully.")
                self.file_logger.log_info(f"Command output:\n{stdout}")
                return stdout
            else:
                self.ui_logger.system_message(f"Command failed with error:\n{stderr}")
                self.file_logger.log_error(f"Command failed with error:\n{stderr}")
                return None
        except Exception as e:
            self.ui_logger.system_message(f"An error occurred while executing the command: {e}")
            self.file_logger.log_error(f"An error occurred while executing the command: {e}")
            return None
