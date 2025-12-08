import os
import logging
from logging.handlers import RotatingFileHandler

class FileLogger:
    def __init__(self, user_data_root: str, log_file_name: str = "jenova.log"):
        log_dir = os.path.join(user_data_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, log_file_name)

        self.logger = logging.getLogger('JenovaFileLogger')
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding handlers if they already exist
        if not self.logger.handlers:
            # Use a rotating file handler to keep log sizes manageable (5MB per file, 2 backups)
            handler = RotatingFileHandler(self.log_file_path, maxBytes=5*1024*1024, backupCount=2)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_warning(self, message: str):
        self.logger.warning(message)

    def log_error(self, message: str):
        self.logger.error(message)