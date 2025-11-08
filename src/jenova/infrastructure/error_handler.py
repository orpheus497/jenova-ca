# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.

"""
Centralized error handling and recovery system.
"""

import traceback
from enum import Enum
from typing import Any, Callable, Optional, Type


class ErrorSeverity(Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorHandler:
    """Centralized error handler with recovery strategies."""

    def __init__(self, ui_logger=None, file_logger=None):
        """
        Initialize error handler.

        Args:
            ui_logger: Optional UI logger
            file_logger: Optional file logger
        """
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.error_count = 0
        self.cuda_error_count = 0

    def log_error(
        self,
        error: Exception,
        context: Any = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> None:
        """
        Log an error (convenience method that wraps handle_error).

        Args:
            error: The exception that occurred
            context: Context about where/when the error occurred (dict or str)
            severity: Severity level
        """
        # Convert context to string if it's a dict
        if isinstance(context, dict):
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        elif context is None:
            context_str = ""
        else:
            context_str = str(context)

        self.handle_error(error, context_str, severity)

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recovery_action: Optional[Callable] = None,
    ) -> bool:
        """
        Handle an error with optional recovery.

        Args:
            error: The exception that occurred
            context: Context about where/when the error occurred
            severity: Severity level
            recovery_action: Optional function to attempt recovery

        Returns:
            True if recovered, False otherwise
        """
        self.error_count += 1

        # Check for CUDA-specific errors
        is_cuda_error = self._is_cuda_error(error)
        if is_cuda_error:
            self.cuda_error_count += 1

        # Format error message
        error_msg = self._format_error(error, context, is_cuda_error)

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            if self.ui_logger:
                self.ui_logger.error(f"CRITICAL: {error_msg}")
            if self.file_logger:
                self.file_logger.log_error(f"CRITICAL ERROR: {error_msg}")
                self.file_logger.log_error(f"Traceback: {traceback.format_exc()}")
        elif severity == ErrorSeverity.ERROR:
            if self.ui_logger:
                self.ui_logger.error(error_msg)
            if self.file_logger:
                self.file_logger.log_error(error_msg)
                self.file_logger.log_error(f"Traceback: {traceback.format_exc()}")
        elif severity == ErrorSeverity.WARNING:
            if self.ui_logger:
                self.ui_logger.system_message(f"Warning: {error_msg}")
            if self.file_logger:
                self.file_logger.log_warning(error_msg)
        else:  # INFO
            if self.file_logger:
                self.file_logger.log_info(error_msg)

        # Attempt recovery if provided
        if recovery_action:
            try:
                recovery_action()
                if self.file_logger:
                    self.file_logger.log_info(
                        f"Recovery action succeeded for: {context}"
                    )
                return True
            except Exception as recovery_error:
                if self.file_logger:
                    self.file_logger.log_error(
                        f"Recovery failed for {context}: {recovery_error}"
                    )
                return False

        return False

    def _is_cuda_error(self, error: Exception) -> bool:
        """Check if error is CUDA-related."""
        error_str = str(error).lower()
        cuda_keywords = [
            "cuda",
            "gpu",
            "vram",
            "out of memory",
            "device-side",
            "cublas",
            "cudnn",
            "ggml-cuda",
        ]
        return any(keyword in error_str for keyword in cuda_keywords)

    def _format_error(self, error: Exception, context: str, is_cuda: bool) -> str:
        """Format error message with context."""
        error_type = type(error).__name__
        error_msg = str(error)

        formatted = f"{error_type}: {error_msg}"
        if context:
            formatted = f"[{context}] {formatted}"

        if is_cuda:
            formatted += " (CUDA-related)"
            if self.cuda_error_count > 3:
                formatted += (
                    f" [CUDA errors: {self.cuda_error_count}, consider CPU-only mode]"
                )

        return formatted

    def get_cuda_recommendation(self) -> Optional[str]:
        """Get recommendation based on CUDA error count."""
        if self.cuda_error_count >= 3:
            return (
                "Multiple CUDA errors detected. Recommendations:\n"
                "  1. Reduce gpu_layers in config (try 16, 8, or 0)\n"
                "  2. Reduce context_size to 2048\n"
                "  3. Set prefer_device to 'cpu' for CPU-only mode\n"
                "  4. Check nvidia-smi for GPU memory usage"
            )
        return None


def safe_execute(
    func: Callable,
    error_handler: ErrorHandler,
    context: str = "",
    default_return: Any = None,
    raise_on_error: bool = False,
) -> Any:
    """
    Execute a function with error handling.

    Args:
        func: Function to execute
        error_handler: Error handler instance
        context: Context for error messages
        default_return: Value to return on error
        raise_on_error: Whether to re-raise the error after handling

    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        error_handler.handle_error(e, context=context)
        if raise_on_error:
            raise
        return default_return
