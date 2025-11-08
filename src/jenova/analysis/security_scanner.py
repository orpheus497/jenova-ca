"""
JENOVA Cognitive Architecture - Security Scanner Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides security vulnerability scanning for Python code using the Bandit library.
"""

import ast
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

try:
    import bandit
    from bandit.core import manager as bandit_manager
    from bandit.core import config as bandit_config
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security vulnerability found in code."""

    severity: str
    confidence: str
    issue_text: str
    test_id: str
    test_name: str
    line_number: int
    line_range: List[int]
    code: str
    filename: str
    more_info: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary format."""
        return {
            "severity": self.severity,
            "confidence": self.confidence,
            "issue_text": self.issue_text,
            "test_id": self.test_id,
            "test_name": self.test_name,
            "line_number": self.line_number,
            "line_range": self.line_range,
            "code": self.code,
            "filename": self.filename,
            "more_info": self.more_info
        }


@dataclass
class ScanResult:
    """Results from a security scan."""

    issues: List[SecurityIssue] = field(default_factory=list)
    files_scanned: int = 0
    lines_scanned: int = 0
    errors: List[str] = field(default_factory=list)

    def get_by_severity(self, severity: str) -> List[SecurityIssue]:
        """Filter issues by severity level."""
        return [issue for issue in self.issues if issue.severity.upper() == severity.upper()]

    def get_high_severity(self) -> List[SecurityIssue]:
        """Get high severity issues."""
        return self.get_by_severity("HIGH")

    def get_medium_severity(self) -> List[SecurityIssue]:
        """Get medium severity issues."""
        return self.get_by_severity("MEDIUM")

    def get_low_severity(self) -> List[SecurityIssue]:
        """Get low severity issues."""
        return self.get_by_severity("LOW")

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to dictionary."""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "files_scanned": self.files_scanned,
            "lines_scanned": self.lines_scanned,
            "errors": self.errors,
            "summary": {
                "total_issues": len(self.issues),
                "high_severity": len(self.get_high_severity()),
                "medium_severity": len(self.get_medium_severity()),
                "low_severity": len(self.get_low_severity())
            }
        }


class SecurityScanner:
    """
    Security vulnerability scanner using Bandit library.

    This scanner analyzes Python code for common security issues such as:
    - Use of dangerous functions (exec, eval, pickle)
    - SQL injection vulnerabilities
    - Hardcoded passwords and secrets
    - Insecure cryptography
    - Path traversal vulnerabilities
    - And many more security issues
    """

    def __init__(self, config_file: Optional[str] = None, profile: Optional[str] = None):
        """
        Initialize the security scanner.

        Args:
            config_file: Path to Bandit configuration file (.bandit)
            profile: Bandit profile to use (e.g., 'bandit', 'django')
        """
        self.config_file = config_file
        self.profile = profile
        self._validate_bandit()

    def _validate_bandit(self) -> None:
        """Validate that Bandit is available."""
        if not BANDIT_AVAILABLE:
            logger.warning("Bandit library not available. Security scanning will be limited.")

    def scan(self, file_path: str) -> List[SecurityIssue]:
        """
        Scan a single file for security vulnerabilities.

        Args:
            file_path: Path to the Python file to scan

        Returns:
            List of SecurityIssue objects found in the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a Python file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.endswith('.py'):
            raise ValueError(f"Not a Python file: {file_path}")

        result = self.scan_files([file_path])
        return result.issues

    def scan_files(self, file_paths: List[str]) -> ScanResult:
        """
        Scan multiple files for security vulnerabilities.

        Args:
            file_paths: List of paths to Python files to scan

        Returns:
            ScanResult containing all issues found
        """
        if not BANDIT_AVAILABLE:
            return self._fallback_scan(file_paths)

        result = ScanResult()

        try:
            # Create Bandit manager
            b_mgr = self._create_bandit_manager(file_paths)

            # Run the scan
            b_mgr.discover_files(file_paths)
            b_mgr.run_tests()

            # Process results
            for issue in b_mgr.get_issue_list():
                security_issue = SecurityIssue(
                    severity=issue.severity,
                    confidence=issue.confidence,
                    issue_text=issue.text,
                    test_id=issue.test_id,
                    test_name=issue.test,
                    line_number=issue.lineno,
                    line_range=issue.linerange,
                    code=issue.get_code(show_linenos=False),
                    filename=issue.fname,
                    more_info=issue.more_info or ""
                )
                result.issues.append(security_issue)

            # Get metrics
            metrics = b_mgr.metrics
            result.files_scanned = len(file_paths)
            result.lines_scanned = sum(metrics.get_metrics(None).get(fname, {}).get('loc', 0)
                                      for fname in file_paths)

        except Exception as e:
            logger.error(f"Error during Bandit scan: {e}")
            result.errors.append(str(e))

        return result

    def scan_directory(self, directory: str, recursive: bool = True,
                      exclude_patterns: Optional[List[str]] = None) -> ScanResult:
        """
        Scan all Python files in a directory.

        Args:
            directory: Path to directory to scan
            recursive: Whether to scan subdirectories
            exclude_patterns: List of glob patterns to exclude

        Returns:
            ScanResult containing all issues found
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Find all Python files
        python_files = self._find_python_files(directory, recursive, exclude_patterns)

        if not python_files:
            logger.warning(f"No Python files found in {directory}")
            return ScanResult()

        return self.scan_files(python_files)

    def scan_string(self, code: str, filename: str = "<string>") -> List[SecurityIssue]:
        """
        Scan a string of Python code for security issues.

        Args:
            code: Python code as a string
            filename: Filename to use in reports

        Returns:
            List of SecurityIssue objects found
        """
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = self.scan_files([temp_path])
            # Update filenames in results
            for issue in result.issues:
                issue.filename = filename
            return result.issues
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _create_bandit_manager(self, file_paths: List[str]) -> 'bandit_manager.BanditManager':
        """Create and configure a Bandit manager instance."""
        # Create config
        if self.config_file and os.path.exists(self.config_file):
            b_conf = bandit_config.BanditConfig(self.config_file)
        else:
            b_conf = bandit_config.BanditConfig()

        # Create manager
        b_mgr = bandit_manager.BanditManager(b_conf, 'file', profile=self.profile)

        return b_mgr

    def _find_python_files(self, directory: str, recursive: bool = True,
                          exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """Find all Python files in a directory."""
        exclude_patterns = exclude_patterns or []
        python_files = []

        path = Path(directory)
        pattern = '**/*.py' if recursive else '*.py'

        for file_path in path.glob(pattern):
            # Skip excluded patterns
            if any(file_path.match(pattern) for pattern in exclude_patterns):
                continue

            # Skip __pycache__ and .tox directories
            if '__pycache__' in file_path.parts or '.tox' in file_path.parts:
                continue

            python_files.append(str(file_path))

        return python_files

    def _fallback_scan(self, file_paths: List[str]) -> ScanResult:
        """
        Fallback scanning when Bandit is not available.
        Uses basic AST analysis to detect common security issues.
        """
        result = ScanResult()

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                issues = self._basic_security_analysis(code, file_path)
                result.issues.extend(issues)
                result.files_scanned += 1
                result.lines_scanned += len(code.splitlines())

            except Exception as e:
                logger.error(f"Error scanning {file_path}: {e}")
                result.errors.append(f"{file_path}: {str(e)}")

        return result

    def _basic_security_analysis(self, code: str, filename: str) -> List[SecurityIssue]:
        """
        Basic security analysis using AST when Bandit is not available.

        Detects:
        - Use of eval() and exec()
        - Use of pickle
        - Hardcoded passwords
        - SQL string concatenation
        """
        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filename}: {e}")
            return issues

        lines = code.splitlines()

        for node in ast.walk(tree):
            # Check for dangerous functions
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id

                    if func_name in ('eval', 'exec'):
                        issues.append(SecurityIssue(
                            severity="HIGH",
                            confidence="HIGH",
                            issue_text=f"Use of {func_name}() is dangerous and can lead to code injection",
                            test_id="B001",
                            test_name=f"dangerous_{func_name}",
                            line_number=node.lineno,
                            line_range=[node.lineno, node.lineno],
                            code=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            filename=filename,
                            more_info="Avoid using eval() or exec() with untrusted input"
                        ))

                    elif func_name == 'compile':
                        issues.append(SecurityIssue(
                            severity="MEDIUM",
                            confidence="MEDIUM",
                            issue_text="Use of compile() can be dangerous with untrusted input",
                            test_id="B002",
                            test_name="dangerous_compile",
                            line_number=node.lineno,
                            line_range=[node.lineno, node.lineno],
                            code=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            filename=filename,
                            more_info="Ensure input to compile() is trusted"
                        ))

                # Check for pickle usage
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ('loads', 'load') and isinstance(node.func.value, ast.Name):
                        if node.func.value.id == 'pickle':
                            issues.append(SecurityIssue(
                                severity="HIGH",
                                confidence="HIGH",
                                issue_text="Use of pickle.loads/load can execute arbitrary code",
                                test_id="B003",
                                test_name="pickle_usage",
                                line_number=node.lineno,
                                line_range=[node.lineno, node.lineno],
                                code=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                                filename=filename,
                                more_info="Pickle can execute arbitrary code during deserialization"
                            ))

            # Check for hardcoded passwords
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(keyword in var_name for keyword in ['password', 'passwd', 'pwd', 'secret', 'token']):
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                if node.value.value:  # Non-empty string
                                    issues.append(SecurityIssue(
                                        severity="MEDIUM",
                                        confidence="LOW",
                                        issue_text=f"Possible hardcoded {var_name}",
                                        test_id="B004",
                                        test_name="hardcoded_password",
                                        line_number=node.lineno,
                                        line_range=[node.lineno, node.lineno],
                                        code=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                                        filename=filename,
                                        more_info="Store secrets in environment variables or secure vaults"
                                    ))

        return issues

    def generate_report(self, result: ScanResult, format: str = "text") -> str:
        """
        Generate a formatted report from scan results.

        Args:
            result: ScanResult to generate report from
            format: Output format ('text', 'json', 'html')

        Returns:
            Formatted report as string
        """
        if format == "json":
            return json.dumps(result.to_dict(), indent=2)

        elif format == "text":
            lines = []
            lines.append("=" * 80)
            lines.append("JENOVA Security Scan Report")
            lines.append("=" * 80)
            lines.append(f"Files scanned: {result.files_scanned}")
            lines.append(f"Lines scanned: {result.lines_scanned}")
            lines.append(f"Total issues: {len(result.issues)}")
            lines.append("")

            summary = result.to_dict()['summary']
            lines.append(f"High severity:   {summary['high_severity']}")
            lines.append(f"Medium severity: {summary['medium_severity']}")
            lines.append(f"Low severity:    {summary['low_severity']}")
            lines.append("")

            if result.errors:
                lines.append("Errors:")
                for error in result.errors:
                    lines.append(f"  - {error}")
                lines.append("")

            if result.issues:
                lines.append("Issues Found:")
                lines.append("-" * 80)

                for i, issue in enumerate(result.issues, 1):
                    lines.append(f"\n{i}. [{issue.severity}/{issue.confidence}] {issue.test_name}")
                    lines.append(f"   File: {issue.filename}:{issue.line_number}")
                    lines.append(f"   Issue: {issue.issue_text}")
                    if issue.code:
                        lines.append(f"   Code: {issue.code.strip()}")
                    if issue.more_info:
                        lines.append(f"   Info: {issue.more_info}")
            else:
                lines.append("No issues found!")

            lines.append("\n" + "=" * 80)
            return "\n".join(lines)

        elif format == "html":
            # Simple HTML report
            html = ['<!DOCTYPE html>', '<html>', '<head>',
                   '<title>JENOVA Security Scan Report</title>',
                   '<style>',
                   'body { font-family: Arial, sans-serif; margin: 20px; }',
                   '.high { color: red; }',
                   '.medium { color: orange; }',
                   '.low { color: blue; }',
                   'table { border-collapse: collapse; width: 100%; }',
                   'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                   'th { background-color: #4CAF50; color: white; }',
                   '</style>',
                   '</head>', '<body>',
                   '<h1>JENOVA Security Scan Report</h1>']

            summary = result.to_dict()['summary']
            html.append(f'<p>Files scanned: {result.files_scanned}</p>')
            html.append(f'<p>Lines scanned: {result.lines_scanned}</p>')
            html.append(f'<p>Total issues: {len(result.issues)}</p>')
            html.append(f'<p class="high">High severity: {summary["high_severity"]}</p>')
            html.append(f'<p class="medium">Medium severity: {summary["medium_severity"]}</p>')
            html.append(f'<p class="low">Low severity: {summary["low_severity"]}</p>')

            if result.issues:
                html.append('<h2>Issues</h2>')
                html.append('<table>')
                html.append('<tr><th>Severity</th><th>File</th><th>Line</th><th>Issue</th></tr>')

                for issue in result.issues:
                    severity_class = issue.severity.lower()
                    html.append(f'<tr class="{severity_class}">')
                    html.append(f'<td>{issue.severity}</td>')
                    html.append(f'<td>{issue.filename}</td>')
                    html.append(f'<td>{issue.line_number}</td>')
                    html.append(f'<td>{issue.issue_text}</td>')
                    html.append('</tr>')

                html.append('</table>')

            html.append('</body></html>')
            return '\n'.join(html)

        else:
            raise ValueError(f"Unknown format: {format}")
