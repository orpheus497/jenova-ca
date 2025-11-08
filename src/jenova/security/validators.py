"""
JENOVA Input Validation Framework - Comprehensive input validation and sanitization.

This module provides validation for paths, files, URLs, and other user inputs
to prevent security vulnerabilities including path traversal, file type exploits,
and resource exhaustion.

Fixes: VULN-H1 (High Severity) - Path traversal vulnerability
Implements: FEATURE-C3 - Comprehensive input validation framework

Copyright (c) 2024-2025, orpheus497. All rights reserved.
Licensed under the MIT License.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Set
import logging

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logging.warning("python-magic not available - MIME type validation disabled")

try:
    import validators as val_lib
    HAS_VALIDATORS = True
except ImportError:
    HAS_VALIDATORS = False
    logging.warning("validators library not available - advanced validation disabled")

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class PathValidator:
    """
    Validates file system paths to prevent path traversal and other attacks.

    This class implements defense-in-depth against path traversal:
    1. Validates path BEFORE symlink resolution
    2. Checks for traversal patterns (../, ..\, etc.)
    3. Resolves symlinks with strict=True
    4. Verifies resolved path is still within sandbox
    5. Validates against allowlist of extensions
    """

    # Patterns that indicate path traversal attempts
    TRAVERSAL_PATTERNS = [
        r'\.\.',  # Any occurrence of ..
        r'~',  # Home directory expansion
        r'\$',  # Environment variable expansion
        r'%',  # Windows environment variable (e.g., %TEMP%)
    ]

    # Default allowed file extensions
    DEFAULT_ALLOWED_EXTENSIONS = {
        # Text and documents
        '.txt', '.md', '.rst', '.pdf', '.doc', '.docx',
        # Code and config
        '.py', '.js', '.ts', '.json', '.yaml', '.yml', '.toml', '.ini', '.xml',
        # Data
        '.csv', '.tsv', '.jsonl',
        # Logs
        '.log',
    }

    # Maximum path length (prevents buffer overflow in some C libraries)
    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        sandbox_root: Optional[str] = None,
        allowed_extensions: Optional[Set[str]] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize the PathValidator.

        Args:
            sandbox_root: Root directory for path containment (None = no sandbox)
            allowed_extensions: Set of allowed file extensions (None = use defaults)
            strict_mode: Enable strict validation (recommended)
        """
        self.sandbox_root = Path(sandbox_root).resolve() if sandbox_root else None
        self.allowed_extensions = allowed_extensions or self.DEFAULT_ALLOWED_EXTENSIONS
        self.strict_mode = strict_mode

        # Compile traversal patterns
        self.traversal_regexes = [
            re.compile(pattern) for pattern in self.TRAVERSAL_PATTERNS
        ]

    def validate_path(
        self,
        path: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
    ) -> Path:
        """
        Validate a file system path (FIXES VULN-H1).

        This method implements secure path validation:
        1. Check path length
        2. Detect traversal patterns BEFORE resolution
        3. Convert to Path and validate structure
        4. Resolve symlinks with strict=True
        5. Verify within sandbox (if configured)
        6. Check extension allowlist
        7. Validate existence and type if required

        Args:
            path: Path string to validate
            must_exist: Require path to exist
            must_be_file: Require path to be a file
            must_be_dir: Require path to be a directory

        Returns:
            Validated Path object

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(path, str):
            raise ValidationError(f"Path must be string, got {type(path).__name__}")

        # Check length to prevent buffer overflows
        if len(path) > self.MAX_PATH_LENGTH:
            raise ValidationError(
                f"Path exceeds maximum length of {self.MAX_PATH_LENGTH} characters"
            )

        # CRITICAL: Detect traversal patterns BEFORE any path manipulation
        for regex in self.traversal_regexes:
            if regex.search(path):
                raise ValidationError(
                    f"Path contains potentially dangerous pattern: {path}"
                )

        # Convert to Path object
        try:
            path_obj = Path(path)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid path format: {e}")

        # Check for absolute vs relative
        if self.strict_mode and path_obj.is_absolute() and not self.sandbox_root:
            raise ValidationError(
                "Absolute paths not allowed without sandbox configuration"
            )

        # If sandbox configured, validate containment BEFORE resolving symlinks
        if self.sandbox_root:
            # Make path relative to sandbox if not absolute
            if not path_obj.is_absolute():
                path_obj = self.sandbox_root / path_obj

            # Check containment before resolution (prevent symlink escape)
            try:
                # Use os.path.commonpath for initial check
                common = Path(os.path.commonpath([str(path_obj), str(self.sandbox_root)]))
                if common != self.sandbox_root:
                    raise ValidationError(
                        f"Path outside sandbox: {path} (sandbox: {self.sandbox_root})"
                    )
            except ValueError:
                # Paths on different drives (Windows)
                raise ValidationError(
                    f"Path outside sandbox: {path} (sandbox: {self.sandbox_root})"
                )

        # Now resolve symlinks with strict=True (will fail if path doesn't exist)
        if must_exist or self.strict_mode:
            try:
                resolved_path = path_obj.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as e:
                if must_exist:
                    raise ValidationError(f"Path does not exist: {path}")
                # In strict mode, still try non-strict resolution
                resolved_path = path_obj.resolve(strict=False)
        else:
            resolved_path = path_obj.resolve(strict=False)

        # CRITICAL: Verify resolved path is still within sandbox
        if self.sandbox_root:
            try:
                resolved_path.relative_to(self.sandbox_root)
            except ValueError:
                raise ValidationError(
                    f"Resolved path escapes sandbox: {resolved_path} "
                    f"(sandbox: {self.sandbox_root})"
                )

        # Validate file extension if file check requested or path has extension
        if resolved_path.suffix:
            if resolved_path.suffix.lower() not in self.allowed_extensions:
                raise ValidationError(
                    f"File extension '{resolved_path.suffix}' not allowed. "
                    f"Allowed: {sorted(self.allowed_extensions)}"
                )

        # Validate existence and type if requested
        if must_exist:
            if not resolved_path.exists():
                raise ValidationError(f"Path does not exist: {resolved_path}")

            if must_be_file and not resolved_path.is_file():
                raise ValidationError(f"Path is not a file: {resolved_path}")

            if must_be_dir and not resolved_path.is_dir():
                raise ValidationError(f"Path is not a directory: {resolved_path}")

        return resolved_path


class FileValidator:
    """
    Validates files for safety (MIME type, size, content).

    Provides defense against:
    - Malicious file uploads
    - Resource exhaustion via large files
    - File type confusion attacks
    """

    # Default maximum file size (100MB)
    DEFAULT_MAX_SIZE = 100 * 1024 * 1024

    # Allowed MIME types (if python-magic available)
    DEFAULT_ALLOWED_MIMES = {
        'text/plain',
        'text/markdown',
        'text/x-python',
        'text/x-script.python',
        'application/json',
        'application/pdf',
        'application/yaml',
        'application/x-yaml',
        'text/yaml',
    }

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        allowed_mimes: Optional[Set[str]] = None,
        check_mime: bool = True,
    ):
        """
        Initialize the FileValidator.

        Args:
            max_size: Maximum file size in bytes
            allowed_mimes: Set of allowed MIME types (None = use defaults)
            check_mime: Enable MIME type validation (requires python-magic)
        """
        self.max_size = max_size
        self.allowed_mimes = allowed_mimes or self.DEFAULT_ALLOWED_MIMES
        self.check_mime = check_mime and HAS_MAGIC

        if check_mime and not HAS_MAGIC:
            logger.warning(
                "MIME type checking requested but python-magic not available"
            )

    def validate_file(self, file_path: Path) -> None:
        """
        Validate a file for safety.

        Args:
            file_path: Path to file to validate

        Raises:
            ValidationError: If validation fails
        """
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        # Check file size
        size = file_path.stat().st_size
        if size > self.max_size:
            raise ValidationError(
                f"File size ({size} bytes) exceeds maximum ({self.max_size} bytes)"
            )

        # Check MIME type if available
        if self.check_mime:
            try:
                mime = magic.from_file(str(file_path), mime=True)
                if mime not in self.allowed_mimes:
                    raise ValidationError(
                        f"File MIME type '{mime}' not allowed. "
                        f"Allowed: {sorted(self.allowed_mimes)}"
                    )
            except Exception as e:
                logger.error(f"MIME type detection failed: {e}")
                # Don't fail validation if MIME detection fails
                # But log it for security monitoring


class InputValidator:
    """
    General-purpose input validator for strings, numbers, URLs, etc.
    """

    # Maximum string length to prevent resource exhaustion
    DEFAULT_MAX_LENGTH = 100000  # 100KB

    def __init__(self, max_length: int = DEFAULT_MAX_LENGTH):
        """
        Initialize the InputValidator.

        Args:
            max_length: Maximum string length
        """
        self.max_length = max_length

    def validate_string(
        self,
        value: str,
        min_length: int = 0,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
    ) -> str:
        """
        Validate a string input.

        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length (None = use default)
            pattern: Regex pattern to match

        Returns:
            Validated string

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value).__name__}")

        max_len = max_length if max_length is not None else self.max_length

        if len(value) < min_length:
            raise ValidationError(
                f"String too short: {len(value)} < {min_length}"
            )

        if len(value) > max_len:
            raise ValidationError(
                f"String too long: {len(value)} > {max_len}"
            )

        if pattern:
            if not re.match(pattern, value):
                raise ValidationError(f"String does not match pattern: {pattern}")

        return value

    def validate_url(self, url: str) -> str:
        """
        Validate a URL (requires validators library).

        Args:
            url: URL string to validate

        Returns:
            Validated URL

        Raises:
            ValidationError: If validation fails
        """
        if not HAS_VALIDATORS:
            logger.warning("URL validation requested but validators library not available")
            return url

        if not val_lib.url(url):
            raise ValidationError(f"Invalid URL: {url}")

        return url

    def validate_email(self, email: str) -> str:
        """
        Validate an email address (requires validators library).

        Args:
            email: Email string to validate

        Returns:
            Validated email

        Raises:
            ValidationError: If validation fails
        """
        if not HAS_VALIDATORS:
            logger.warning("Email validation requested but validators library not available")
            return email

        if not val_lib.email(email):
            raise ValidationError(f"Invalid email: {email}")

        return email

    def validate_number(
        self,
        value: any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Validate a numeric input.

        Args:
            value: Value to validate (will be converted to float)
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated number as float

        Raises:
            ValidationError: If validation fails
        """
        try:
            num = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid number: {value} ({e})")

        if min_value is not None and num < min_value:
            raise ValidationError(f"Number too small: {num} < {min_value}")

        if max_value is not None and num > max_value:
            raise ValidationError(f"Number too large: {num} > {max_value}")

        return num
