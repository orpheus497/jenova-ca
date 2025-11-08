"""
JENOVA Security Audit Logging - Structured security event logging.

Implements comprehensive audit logging for security events including:
- Authentication and authorization events
- Input validation failures
- File access attempts
- Configuration changes
- Security errors

Implements: FEATURE-C2 - Comprehensive audit logging

Copyright (c) 2024-2025, orpheus497. All rights reserved.
Licensed under the MIT License.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class SecurityEventType(Enum):
    """Types of security events."""
    # Authentication
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_TOKEN_GENERATED = "auth_token_generated"
    AUTH_TOKEN_REVOKED = "auth_token_revoked"

    # Authorization
    AUTHZ_GRANTED = "authz_granted"
    AUTHZ_DENIED = "authz_denied"

    # Input Validation
    VALIDATION_FAILURE = "validation_failure"
    INJECTION_ATTEMPT = "injection_attempt"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"

    # File Access
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_ACCESS_DENIED = "file_access_denied"

    # Configuration
    CONFIG_CHANGE = "config_change"
    SECRET_ROTATION = "secret_rotation"

    # Errors
    SECURITY_ERROR = "security_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class SecurityEvent:
    """Represents a security event."""

    def __init__(
        self,
        event_type: SecurityEventType,
        username: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
    ):
        """
        Initialize a security event.

        Args:
            event_type: Type of security event
            username: User associated with event
            details: Additional event details
            severity: Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.event_type = event_type
        self.username = username
        self.details = details or {}
        self.severity = severity
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "username": self.username,
            "severity": self.severity,
            "details": self.details,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)


class AuditLogger:
    """
    Security audit logger with structured logging support.

    Logs security events in JSON format for SIEM integration.
    Privacy-aware: does not log PII beyond username.
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        enable_structured_logging: bool = True,
    ):
        """
        Initialize the AuditLogger.

        Args:
            log_file: Path to audit log file (None = use default logger)
            enable_structured_logging: Use structlog if available
        """
        self.log_file = log_file
        self.use_structlog = enable_structured_logging and HAS_STRUCTLOG

        if self.use_structlog:
            self.logger = structlog.get_logger("jenova.security.audit")
        else:
            self.logger = logging.getLogger("jenova.security.audit")

        # Set up file handler if specified
        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(message)s')
            )
            if not self.use_structlog:
                self.logger.addHandler(handler)

    def log_event(self, event: SecurityEvent) -> None:
        """
        Log a security event.

        Args:
            event: SecurityEvent to log
        """
        if self.use_structlog:
            # Structured logging with all fields
            self.logger.log(
                logging.getLevelName(event.severity),
                "security_event",
                event_type=event.event_type.value,
                username=event.username,
                **event.details
            )
        else:
            # JSON logging for standard logger
            log_msg = event.to_json()
            level = logging.getLevelName(event.severity)
            self.logger.log(level, log_msg)

    def log_auth_success(self, username: str, method: str = "password") -> None:
        """Log successful authentication."""
        event = SecurityEvent(
            SecurityEventType.AUTH_SUCCESS,
            username=username,
            details={"method": method},
            severity="INFO",
        )
        self.log_event(event)

    def log_auth_failure(
        self,
        username: str,
        reason: str,
        ip_address: Optional[str] = None
    ) -> None:
        """Log failed authentication attempt."""
        event = SecurityEvent(
            SecurityEventType.AUTH_FAILURE,
            username=username,
            details={
                "reason": reason,
                "ip_address": ip_address,
            },
            severity="WARNING",
        )
        self.log_event(event)

    def log_injection_attempt(
        self,
        username: Optional[str],
        input_type: str,
        pattern: str,
    ) -> None:
        """Log detected injection attempt."""
        event = SecurityEvent(
            SecurityEventType.INJECTION_ATTEMPT,
            username=username,
            details={
                "input_type": input_type,
                "pattern": pattern[:100],  # Truncate for privacy
            },
            severity="ERROR",
        )
        self.log_event(event)

    def log_path_traversal_attempt(
        self,
        username: Optional[str],
        path: str,
    ) -> None:
        """Log detected path traversal attempt."""
        event = SecurityEvent(
            SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            username=username,
            details={"path": path[:200]},  # Truncate for privacy
            severity="ERROR",
        )
        self.log_event(event)

    def log_file_access(
        self,
        username: str,
        action: str,
        file_path: str,
        success: bool,
    ) -> None:
        """Log file access attempt."""
        event_type = {
            "read": SecurityEventType.FILE_READ,
            "write": SecurityEventType.FILE_WRITE,
            "delete": SecurityEventType.FILE_DELETE,
        }.get(action, SecurityEventType.FILE_READ)

        if not success:
            event_type = SecurityEventType.FILE_ACCESS_DENIED

        event = SecurityEvent(
            event_type,
            username=username,
            details={
                "action": action,
                "file_path": file_path[:200],
                "success": success,
            },
            severity="WARNING" if not success else "INFO",
        )
        self.log_event(event)

    def log_rate_limit_exceeded(
        self,
        username: str,
        operation: str,
        limit: int,
    ) -> None:
        """Log rate limit exceedance."""
        event = SecurityEvent(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            username=username,
            details={
                "operation": operation,
                "limit": limit,
            },
            severity="WARNING",
        )
        self.log_event(event)

    def log_config_change(
        self,
        username: str,
        setting: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Log configuration change."""
        # Sanitize sensitive values
        def sanitize(value):
            if isinstance(value, str) and any(
                k in setting.lower() for k in ['password', 'secret', 'token', 'key']
            ):
                return "***REDACTED***"
            return value

        event = SecurityEvent(
            SecurityEventType.CONFIG_CHANGE,
            username=username,
            details={
                "setting": setting,
                "old_value": sanitize(old_value),
                "new_value": sanitize(new_value),
            },
            severity="INFO",
        )
        self.log_event(event)


# Singleton instance for convenience
_default_audit_logger: Optional[AuditLogger] = None


def get_default_audit_logger() -> AuditLogger:
    """Get the default singleton AuditLogger instance."""
    global _default_audit_logger
    if _default_audit_logger is None:
        _default_audit_logger = AuditLogger()
    return _default_audit_logger
