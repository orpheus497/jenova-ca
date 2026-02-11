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

import structlog


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

    ##Fix: Patch structlog BoundLogger classes BEFORE configuration (2026-02-11T06:32:20Z)
    ##Note: Must patch BEFORE structlog.configure() to affect logger instances
    def _patch_bound_logger_for_textual():
        """Patch all structlog BoundLogger classes for Textual framework compatibility."""
        def system(self, event=None, **kw):
            """System-level logging for Textual framework."""
            return self.debug(event, **kw)
        
        def call_method(self, event=None, **kw):
            """Make logger callable for Textual framework."""
            return self.debug(event, **kw)
        
        # Patch all BoundLogger variant classes
        for cls_name in dir(structlog._native):
            cls = getattr(structlog._native, cls_name)
            if isinstance(cls, type) and "BoundLogger" in cls_name:
                if not hasattr(cls, "system"):
                    cls.system = system
                if not hasattr(cls, "__call__"):
                    cls.__call__ = call_method
        
        # Also patch the lazy proxy - ALWAYS set it (hasattr checks don't work reliably on proxies)
        def proxy_call(self, event=None, **kw):
            """Make logger proxy callable."""
            return self.bind().debug(event, **kw)
        
        structlog._config.BoundLoggerLazyProxy.__call__ = proxy_call
    
    _patch_bound_logger_for_textual()

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

    ##Action purpose: Apply structlog configuration
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
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
