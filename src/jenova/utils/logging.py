##Script function and purpose: Provides structured logging configuration using structlog
"""
Structured Logging Configuration

Configures structlog for consistent, parseable logging across JENOVA.
All logs use key-value pairs for easy grep/analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


##Refactor: Custom BoundLogger subclass for Textual compatibility (D3-2026-02-11T08:22:24Z)
##Class purpose: Extend structlog BoundLogger with Textual framework methods
class JenovaBoundLogger(structlog.stdlib.BoundLogger):
    """Custom BoundLogger with Textual framework compatibility methods."""

    def system(self, event: str | None = None, **kw: object) -> object:
        """System-level logging for Textual framework."""
        return self.debug(event, **kw)

    def __call__(self, event: str | None = None, **kw: object) -> object:
        """Make logger callable for Textual framework."""
        return self.debug(event, **kw)


##Refactor: Extracted to module scope for testability (D3-2026-02-11T08:22:24Z)
##Function purpose: Patch structlog lazy proxy for Textual framework compatibility
def _patch_structlog_lazy_proxy() -> None:
    """Patch structlog lazy proxy for Textual framework compatibility."""
    # Patch the lazy proxy - check class dict and wrap in try/except for safety
    try:
        if "__call__" not in structlog._config.BoundLoggerLazyProxy.__dict__:

            def proxy_call(self: Any, event: str | None = None, **kw: object) -> object:
                """Make logger proxy callable."""
                return self.bind().debug(event, **kw)

            structlog._config.BoundLoggerLazyProxy.__call__ = proxy_call
    except (AttributeError, ImportError) as e:
        ##Refactor: Added diagnostic logging instead of silent pass (D3-2026-02-11T08:22:24Z)
        logger = logging.getLogger(__name__)
        logger.debug(
            "Skipped structlog BoundLoggerLazyProxy patching (API may have changed): %s",
            str(e),
            exc_info=True,
        )


##Function purpose: Configure application-wide logging settings
def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: If True, output JSON formatted logs
    """
    ##Step purpose: Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    ##Fix: Patch structlog lazy proxy BEFORE configuration (see PR BH-2026-02-11)
    ##Note: Must patch BEFORE structlog.configure() to affect logger instances
    ##Refactor: Replaced monkey-patching loop with custom subclass (D3-2026-02-11T08:22:24Z)
    _patch_structlog_lazy_proxy()

    ##Step purpose: Build processor chain for structlog
    processors: list[structlog.types.Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    ##Condition purpose: Choose renderer based on format preference
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    ##Action purpose: Apply structlog configuration with custom wrapper class
    ##Refactor: Use JenovaBoundLogger subclass instead of monkey-patching (D3-2026-02-11T08:22:24Z)
    structlog.configure(
        processors=processors,
        wrapper_class=JenovaBoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    ##Condition purpose: Set up file handler if log file specified
    if log_file is not None:
        ##Action purpose: Add file handler to root logger
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)


##Function purpose: Get a logger instance bound to the given name
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


##Function purpose: Create a logger with pre-bound context
def get_logger_with_context(
    name: str,
    **context: str | int | float | bool,
) -> structlog.stdlib.BoundLogger:
    """
    Get a logger with pre-bound context values.

    Args:
        name: Logger name
        **context: Key-value pairs to bind to all log entries

    Returns:
        Logger with bound context
    """
    return structlog.get_logger(name).bind(**context)
