# The JENOVA Cognitive Architecture - Code Metrics Analyzer
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Comprehensive code quality and complexity analysis.

Uses radon library for cyclomatic complexity, Halstead metrics,
and maintainability index calculations.
"""

import os
import ast
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from radon.complexity import cc_visit, cc_rank
    from radon.metrics import h_visit, mi_visit
    from radon.raw import analyze

    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False


@dataclass
class ComplexityResult:
    """Complexity analysis for a single function/class."""

    name: str
    type: str  # 'function', 'method', 'class'
    complexity: int
    rank: str  # A-F ranking
    lineno: int
    col_offset: int
    endline: int


@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics."""

    h1: int = 0  # Number of distinct operators
    h2: int = 0  # Number of distinct operands
    N1: int = 0  # Total number of operators
    N2: int = 0  # Total number of operands
    vocabulary: int = 0
    length: int = 0
    calculated_length: float = 0.0
    volume: float = 0.0
    difficulty: float = 0.0
    effort: float = 0.0
    time: float = 0.0
    bugs: float = 0.0


@dataclass
class CodeMetricsResult:
    """Complete code quality analysis results."""

    file_path: str
    loc: int = 0  # Lines of code (total)
    lloc: int = 0  # Logical lines of code
    sloc: int = 0  # Source lines of code
    comments: int = 0
    multi: int = 0  # Multi-line strings
    blank: int = 0
    single_comments: int = 0
    complexity_blocks: List[ComplexityResult] = field(default_factory=list)
    avg_complexity: float = 0.0
    max_complexity: int = 0
    maintainability_index: float = 0.0
    halstead: Optional[HalsteadMetrics] = None
    issues: List[str] = field(default_factory=list)


class CodeMetrics:
    """
    Comprehensive code quality and complexity analyzer.

    Features:
    - Cyclomatic complexity analysis (McCabe)
    - Halstead complexity metrics
    - Maintainability index calculation
    - Lines of code analysis
    - Function/class size metrics
    - Code quality issues detection
    """

    def __init__(self):
        """Initialize code metrics analyzer."""
        self.radon_available = RADON_AVAILABLE

        # Thresholds for quality issues
        self.complexity_threshold = 10
        self.function_length_threshold = 50
        self.maintainability_threshold = 20.0

    def calculate(self, code: str, file_path: str = "inline") -> Dict:
        """
        Calculate comprehensive code metrics.

        Args:
            code: Python source code
            file_path: Optional file path for context

        Returns:
            Dictionary with all metrics
        """
        result = CodeMetricsResult(file_path=file_path)

        if not code or not code.strip():
            return self._result_to_dict(result)

        # Basic metrics
        result.loc = len(code.split("\n"))
        result.blank = code.count("\n\n")
        result.comments = self._count_comments(code)

        if not self.radon_available:
            # Fallback: simple analysis without radon
            return self._simple_analysis(code, result)

        try:
            # Raw metrics (LOC, LLOC, SLOC, comments)
            raw_metrics = analyze(code)
            result.lloc = raw_metrics.lloc
            result.sloc = raw_metrics.sloc
            result.comments = raw_metrics.comments
            result.multi = raw_metrics.multi
            result.blank = raw_metrics.blank
            result.single_comments = raw_metrics.single_comments

            # Cyclomatic complexity
            complexity_blocks = cc_visit(code)
            result.complexity_blocks = [
                ComplexityResult(
                    name=block.name,
                    type=block.classname if hasattr(block, "classname") else "function",
                    complexity=block.complexity,
                    rank=cc_rank(block.complexity),
                    lineno=block.lineno,
                    col_offset=block.col_offset,
                    endline=block.endline,
                )
                for block in complexity_blocks
            ]

            if result.complexity_blocks:
                complexities = [b.complexity for b in result.complexity_blocks]
                result.avg_complexity = sum(complexities) / len(complexities)
                result.max_complexity = max(complexities)

            # Halstead metrics
            halstead_report = h_visit(code)
            if halstead_report:
                result.halstead = HalsteadMetrics(
                    h1=halstead_report.h1,
                    h2=halstead_report.h2,
                    N1=halstead_report.N1,
                    N2=halstead_report.N2,
                    vocabulary=halstead_report.vocabulary,
                    length=halstead_report.length,
                    calculated_length=halstead_report.calculated_length,
                    volume=halstead_report.volume,
                    difficulty=halstead_report.difficulty,
                    effort=halstead_report.effort,
                    time=halstead_report.time,
                    bugs=halstead_report.bugs,
                )

            # Maintainability index
            mi_score = mi_visit(code, multi=True)
            result.maintainability_index = mi_score

            # Identify quality issues
            result.issues = self._identify_issues(result, code)

        except Exception as e:
            result.issues.append(f"Error during analysis: {str(e)}")

        return self._result_to_dict(result)

    def _count_comments(self, code: str) -> int:
        """Count comment lines in code."""
        lines = code.split("\n")
        comments = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                comments += 1

        return comments

    def _simple_analysis(self, code: str, result: CodeMetricsResult) -> Dict:
        """
        Perform simple analysis without radon.

        Args:
            code: Source code
            result: CodeMetricsResult object

        Returns:
            Metrics dictionary
        """
        try:
            # Parse AST for basic complexity
            tree = ast.parse(code)

            # Count functions and classes
            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Simple complexity: count decision points
                    complexity = 1  # Base complexity
                    for child in ast.walk(node):
                        if isinstance(
                            child, (ast.If, ast.While, ast.For, ast.ExceptHandler)
                        ):
                            complexity += 1
                        elif isinstance(child, ast.BoolOp):
                            complexity += len(child.values) - 1

                    functions.append((node.name, complexity, node.lineno))

                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)

            # Create complexity results
            for name, complexity, lineno in functions:
                rank = "A" if complexity <= 5 else "B" if complexity <= 10 else "C"
                result.complexity_blocks.append(
                    ComplexityResult(
                        name=name,
                        type="function",
                        complexity=complexity,
                        rank=rank,
                        lineno=lineno,
                        col_offset=0,
                        endline=lineno,
                    )
                )

            if result.complexity_blocks:
                complexities = [b.complexity for b in result.complexity_blocks]
                result.avg_complexity = sum(complexities) / len(complexities)
                result.max_complexity = max(complexities)

        except SyntaxError as e:
            result.issues.append(f"Syntax error: {str(e)}")

        return self._result_to_dict(result)

    def _identify_issues(self, result: CodeMetricsResult, code: str) -> List[str]:
        """
        Identify code quality issues.

        Args:
            result: Metrics result
            code: Source code

        Returns:
            List of issue descriptions
        """
        issues = []

        # High complexity
        for block in result.complexity_blocks:
            if block.complexity > self.complexity_threshold:
                issues.append(
                    f"High complexity in {block.type} '{block.name}' "
                    f"(complexity: {block.complexity}, rank: {block.rank}) "
                    f"at line {block.lineno}"
                )

        # Low maintainability
        if result.maintainability_index < self.maintainability_threshold:
            issues.append(
                f"Low maintainability index: {result.maintainability_index:.1f} "
                f"(threshold: {self.maintainability_threshold})"
            )

        # Long functions
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_length = self._get_function_length(node)
                    if func_length > self.function_length_threshold:
                        issues.append(
                            f"Long function '{node.name}' ({func_length} lines) "
                            f"at line {node.lineno}"
                        )
        except Exception as e:
            # Unable to parse code for function length analysis
            # This is not critical, other metrics can still be computed
            pass

        # Too many comments might indicate unclear code
        if result.loc > 0:
            comment_ratio = result.comments / result.loc
            if comment_ratio > 0.3:
                issues.append(
                    f"High comment ratio ({comment_ratio:.1%}). "
                    f"Consider refactoring for clarity."
                )

        # Very few comments might indicate poor documentation
        if result.sloc > 50 and result.comments == 0:
            issues.append("No comments found. Consider adding documentation.")

        return issues

    def _get_function_length(self, node: ast.FunctionDef) -> int:
        """Get length of function in lines."""
        if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
            return node.end_lineno - node.lineno + 1
        return 0

    def _result_to_dict(self, result: CodeMetricsResult) -> Dict:
        """Convert result object to dictionary."""
        return {
            "file_path": result.file_path,
            "lines_of_code": {
                "total": result.loc,
                "logical": result.lloc,
                "source": result.sloc,
                "comments": result.comments,
                "blank": result.blank,
                "multi_line_strings": result.multi,
                "single_comments": result.single_comments,
            },
            "complexity": {
                "average": round(result.avg_complexity, 2),
                "max": result.max_complexity,
                "blocks": [
                    {
                        "name": b.name,
                        "type": b.type,
                        "complexity": b.complexity,
                        "rank": b.rank,
                        "lineno": b.lineno,
                    }
                    for b in result.complexity_blocks
                ],
            },
            "maintainability_index": round(result.maintainability_index, 2),
            "halstead": (
                {
                    "distinct_operators": result.halstead.h1,
                    "distinct_operands": result.halstead.h2,
                    "total_operators": result.halstead.N1,
                    "total_operands": result.halstead.N2,
                    "vocabulary": result.halstead.vocabulary,
                    "length": result.halstead.length,
                    "volume": round(result.halstead.volume, 2),
                    "difficulty": round(result.halstead.difficulty, 2),
                    "effort": round(result.halstead.effort, 2),
                    "time": round(result.halstead.time, 2),
                    "bugs": round(result.halstead.bugs, 3),
                }
                if result.halstead
                else {}
            ),
            "issues": result.issues,
            "quality_grade": self._calculate_grade(result),
        }

    def _calculate_grade(self, result: CodeMetricsResult) -> str:
        """
        Calculate overall quality grade.

        Args:
            result: Metrics result

        Returns:
            Grade (A-F)
        """
        score = 100

        # Deduct for high complexity
        if result.max_complexity > 20:
            score -= 30
        elif result.max_complexity > 10:
            score -= 15

        # Deduct for low maintainability
        if result.maintainability_index < 20:
            score -= 30
        elif result.maintainability_index < 40:
            score -= 15

        # Deduct for issues
        score -= min(len(result.issues) * 5, 25)

        # Convert to grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Metrics dictionary
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            return self.calculate(code, file_path)
        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "issues": [f"Failed to analyze file: {str(e)}"],
            }

    def analyze_directory(self, directory: str, recursive: bool = True) -> List[Dict]:
        """
        Analyze all Python files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively

        Returns:
            List of metrics dictionaries
        """
        results = []

        if recursive:
            for root, dirs, files in os.walk(directory):
                # Skip hidden and __pycache__ directories
                dirs[:] = [
                    d for d in dirs if not d.startswith(".") and d != "__pycache__"
                ]

                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        results.append(self.analyze_file(file_path))
        else:
            for file in os.listdir(directory):
                if file.endswith(".py"):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        results.append(self.analyze_file(file_path))

        return results
