# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for the file logging of the JENOVA Cognitive Architecture."""

import logging
import os
from logging.handlers import RotatingFileHandler


class FileLogger:
    """
    File-based logging system for JENOVA.

    Provides persistent logging to rotating log files with automatic size
    management and backup rotation.

    Attributes:
        log_file_path: Path to the current log file
        logger: Python logging.Logger instance

    Example:
        >>> logger = FileLogger("/home/user/.jenova", "app.log")
        >>> logger.log_info("Application started")
        >>> logger.log_error("Connection failed")
    """

    def __init__(self, user_data_root: str, log_file_name: str = "jenova.log"):
        """
        Initialize file logger with rotating file handler.

        Creates a log directory within user_data_root and sets up a rotating
        file handler that keeps log files at 5MB with 2 backup files.

        Args:
            user_data_root: Root directory for user data
            log_file_name: Name of the log file (default: "jenova.log")
        """
        log_dir = os.path.join(user_data_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, log_file_name)

        self.logger = logging.getLogger("JenovaFileLogger")
        self.logger.setLevel(logging.INFO)

        # Avoid adding handlers if they already exist
        if not self.logger.handlers:
            # Use a rotating file handler to keep log sizes manageable (5MB per file, 2 backups)
            handler = RotatingFileHandler(
                self.log_file_path, maxBytes=5 * 1024 * 1024, backupCount=2
            )
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_info(self, message: str):
        """
        Log an informational message.

        Args:
            message: Message to log

        Example:
            >>> logger.log_info("Configuration loaded successfully")
        """
        self.logger.info(message)

    def log_warning(self, message: str):
        """
        Log a warning message.

        Args:
            message: Warning message to log

        Example:
            >>> logger.log_warning("Fallback to default configuration")
        """
        self.logger.warning(message)

    def log_error(self, message: str):
        """
        Log an error message.

        Args:
            message: Error message to log

        Example:
            >>> logger.log_error("Failed to connect to database")
        """
        self.logger.error(message)
