import os
import re

class FileTools:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        
        # Set up the secure file sandbox
        sandbox_path = self.config.get('tools', {}).get('file_sandbox_path', '~/jenova_files')
        self.sandbox_dir = os.path.realpath(os.path.expanduser(sandbox_path))
        
        os.makedirs(self.sandbox_dir, exist_ok=True)
        self.ui_logger.info(f"File sandbox initialized at: {self.sandbox_dir}")

    def _is_in_sandbox(self, path: str) -> bool:
        """Checks if a given path is securely within the sandbox directory."""
        full_path = os.path.realpath(path)
        return os.path.commonprefix((full_path, self.sandbox_dir)) == self.sandbox_dir

    def handle_tool_request(self, response: str) -> str:
        write_pattern = re.compile(r'<TOOL:WRITE_FILE(path="([^"]+)", content="(.+?)")>', re.DOTALL)
        read_pattern = re.compile(r'<TOOL:READ_FILE(path="([^"]+)")>', re.DOTALL)
        list_pattern = re.compile(r'<TOOL:LIST_DIRECTORY(path="([^"]+)")>', re.DOTALL)

        write_match = write_pattern.search(response)
        read_match = read_pattern.search(response)
        list_match = list_pattern.search(response)

        if write_match:
            self.ui_logger.info("Tool request detected: WRITE_FILE")
            file_path_str = write_match.group(1)
            content = write_match.group(2)
            return write_pattern.sub(self.write_file(file_path_str, content), response)
        elif read_match:
            self.ui_logger.info("Tool request detected: READ_FILE")
            file_path_str = read_match.group(1)
            return read_pattern.sub(self.read_file(file_path_str), response)
        elif list_match:
            self.ui_logger.info("Tool request detected: LIST_DIRECTORY")
            path_str = list_match.group(1)
            return list_pattern.sub(self.list_directory(path_str), response)
        else:
            return response

    def write_file(self, file_path: str, content: str) -> str:
        # Prevent writing to hidden files
        if os.path.basename(file_path).startswith('.'):
            error_message = "Error: Writing to hidden files is not permitted."
            self.ui_logger.system_message(error_message)
            return error_message

        full_path = os.path.join(self.sandbox_dir, file_path)

        if not self._is_in_sandbox(full_path):
            error_message = f"Error: Attempted to write to a file outside of the allowed sandbox directory ({self.sandbox_dir})."
            self.ui_logger.system_message(error_message)
            return error_message

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            tool_success_message = f"I have successfully written the content to the file: {file_path}"
            self.ui_logger.info(f"Successfully wrote file to {full_path}")
            return tool_success_message
        except Exception as e:
            error_message = f"I encountered an error while trying to write the file: {e}"
            self.ui_logger.system_message(f"Failed to write file: {e}")
            return error_message

    def read_file(self, file_path: str) -> str:
        if os.path.basename(file_path).startswith('.'):
            error_message = "Error: Reading hidden files is not permitted."
            self.ui_logger.system_message(error_message)
            return error_message
            
        full_path = os.path.join(self.sandbox_dir, file_path)

        if not self._is_in_sandbox(full_path):
            error_message = f"Error: Attempted to read a file outside of the allowed sandbox directory ({self.sandbox_dir})."
            self.ui_logger.system_message(error_message)
            return error_message

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"Content of {file_path}:\n{content}"
        except FileNotFoundError:
            return f"Error: The file '{file_path}' was not found in the sandbox."
        except Exception as e:
            error_message = f"I encountered an error while trying to read the file: {e}"
            self.ui_logger.system_message(f"Failed to read file: {e}")
            return error_message

    def list_directory(self, path: str) -> str:
        if path.strip() in ['.', '..', '/']:
            full_path = self.sandbox_dir
        else:
            full_path = os.path.join(self.sandbox_dir, path)

        if not self._is_in_sandbox(full_path):
            error_message = f"Error: Attempted to list a directory outside of the allowed sandbox directory ({self.sandbox_dir})."
            self.ui_logger.system_message(error_message)
            return error_message

        try:
            files = os.listdir(full_path)
            # Filter out hidden files
            visible_files = [f for f in files if not f.startswith('.')]
            return f"Files in '{path}':\n" + "\n".join(visible_files)
        except FileNotFoundError:
            return f"Error: The directory '{path}' was not found in the sandbox."
        except Exception as e:
            error_message = f"I encountered an error while trying to list the directory: {e}"
            self.ui_logger.system_message(f"Failed to list directory: {e}")
            return error_message
