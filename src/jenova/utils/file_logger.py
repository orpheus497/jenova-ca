##Script function and purpose: File Logger for The JENOVA Cognitive Architecture
##This module provides rotating file-based logging for persistent debug and error tracking

import os
import logging
from logging.handlers import RotatingFileHandler

##Class purpose: Manages file-based logging with automatic rotation
class FileLogger:
    ##Function purpose: Initialize file logger with rotating file handler
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

    ##Function purpose: Log informational messages to file
    def log_info(self, message: str):
        self.logger.info(message)

    ##Function purpose: Log warning messages to file
    def log_warning(self, message: str):
        self.logger.warning(message)

    ##Function purpose: Log error messages to file
    def log_error(self, message: str):
        self.logger.error(message)