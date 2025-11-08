# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Time Tools

"""
Temporal operations tool module.

This module provides datetime and time-related operations for the JENOVA
cognitive architecture, enabling time-aware responses and temporal context.
"""

from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

from jenova.tools.base import BaseTool, ToolResult


class TimeTools(BaseTool):
    """
    Time and datetime operations tool.

    Provides current datetime, timezone conversions, and temporal calculations
    to enable time-aware cognitive responses.

    Methods:
        execute: Get current datetime in specified format and timezone
        get_current_datetime: Get current datetime string
        get_timestamp: Get Unix timestamp
        format_datetime: Format datetime object
    """

    def __init__(self, config: Dict[str, Any], ui_logger: Any, file_logger: Any):
        """
        Initialize time tools.

        Args:
            config: Configuration dictionary
            ui_logger: UI logger instance
            file_logger: File logger instance
        """
        super().__init__(
            name="time_tools",
            description="Provides current datetime and temporal operations",
            config=config,
            ui_logger=ui_logger,
            file_logger=file_logger
        )

    def execute(
        self,
        format_str: str = "%Y-%m-%d %H:%M:%S",
        timezone: str = "UTC"
    ) -> ToolResult:
        """
        Get current datetime with specified format and timezone.

        Args:
            format_str: strftime format string (default: "%Y-%m-%d %H:%M:%S")
            timezone: Timezone identifier (default: "UTC")

        Returns:
            ToolResult with formatted datetime string

        Example:
            >>> tools = TimeTools(config, ui_logger, file_logger)
            >>> result = tools.execute(format_str="%Y-%m-%d", timezone="America/New_York")
            >>> print(result.data)  # "2025-11-08"
        """
        try:
            # Get current datetime in specified timezone
            try:
                tz = ZoneInfo(timezone)
            except Exception:
                # Fallback to UTC if timezone invalid
                tz = ZoneInfo("UTC")
                if self.file_logger:
                    self.file_logger.log_warning(
                        f"Invalid timezone '{timezone}', using UTC"
                    )

            now = datetime.now(tz)

            # Format datetime
            formatted = now.strftime(format_str)

            result = self._create_success_result(
                data=formatted,
                metadata={
                    'timezone': str(tz),
                    'format': format_str,
                    'iso_format': now.isoformat()
                }
            )

            self._log_execution({'format_str': format_str, 'timezone': timezone}, result)
            return result

        except Exception as e:
            error_msg = f"Failed to get current datetime: {str(e)}"
            result = self._create_error_result(error_msg)
            self._log_execution({'format_str': format_str, 'timezone': timezone}, result)
            return result

    def get_current_datetime(self) -> str:
        """
        Get current datetime as formatted string (UTC).

        Returns:
            Formatted datetime string

        Example:
            >>> tools = TimeTools(config, ui_logger, file_logger)
            >>> dt = tools.get_current_datetime()
            >>> print(dt)  # "2025-11-08 14:30:00"
        """
        result = self.execute()
        return result.data if result.success else "Unable to retrieve datetime"

    def get_timestamp(self) -> int:
        """
        Get current Unix timestamp.

        Returns:
            Unix timestamp (seconds since epoch)

        Example:
            >>> tools = TimeTools(config, ui_logger, file_logger)
            >>> ts = tools.get_timestamp()
            >>> print(ts)  # 1731075000
        """
        try:
            return int(datetime.now(ZoneInfo("UTC")).timestamp())
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to get timestamp: {e}")
            return 0

    def format_datetime(
        self,
        dt: datetime,
        format_str: str = "%Y-%m-%d %H:%M:%S"
    ) -> str:
        """
        Format datetime object to string.

        Args:
            dt: Datetime object to format
            format_str: strftime format string

        Returns:
            Formatted datetime string

        Example:
            >>> tools = TimeTools(config, ui_logger, file_logger)
            >>> dt = datetime.now()
            >>> formatted = tools.format_datetime(dt, "%B %d, %Y")
            >>> print(formatted)  # "November 08, 2025"
        """
        try:
            return dt.strftime(format_str)
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to format datetime: {e}")
            return str(dt)
