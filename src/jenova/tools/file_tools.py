import os
import re

class FileTools:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.output_dir = os.path.join(config['user_data_root'], "generated_files")
        os.makedirs(self.output_dir, exist_ok=True)

    def handle_tool_request(self, response: str) -> str:
        tool_pattern = re.compile(r'<TOOL:WRITE_FILE\(path="([^"]+)", content="(.+?)"\)>', re.DOTALL)
        match = tool_pattern.search(response)
        if not match:
            return response

        self.ui_logger.info("Tool request detected: WRITE_FILE")
        self.file_logger.log_info("Tool request detected: WRITE_FILE")
        file_path_str = match.group(1)
        content = match.group(2)
        
        filename = os.path.basename(file_path_str)
        full_path = os.path.join(self.output_dir, filename)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            tool_success_message = f"I have successfully written the content to the file: {filename}"
            self.ui_logger.info(f"Successfully wrote file to {full_path}")
            self.file_logger.log_info(f"Successfully wrote file to {full_path}")
            return tool_pattern.sub(tool_success_message, response)
        except Exception as e:
            error_message = f"I encountered an error while trying to write the file: {e}"
            self.ui_logger.system_message(f"Failed to write file: {e}")
            self.file_logger.log_error(f"Failed to write file: {e}")
            return tool_pattern.sub(error_message, response)