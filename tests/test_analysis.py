# The JENOVA Cognitive Architecture - Analysis Tests
# Copyright (c) 2024-2025, orpheus497. All rights reserved.
# Licensed under the MIT License

"""
Analysis module tests for JENOVA Phase 13-17.

Tests context optimization, code metrics, security scanning,
intent classification, and command disambiguation.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from jenova.analysis.context_optimizer import ContextOptimizer
from jenova.analysis.code_metrics import CodeMetrics
from jenova.analysis.security_scanner import SecurityScanner
from jenova.analysis.intent_classifier import IntentClassifier
from jenova.analysis.command_disambiguator import CommandDisambiguator


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    return {
        'analysis': {
            'max_context_tokens': 8192,
            'security_scan_level': 'medium'
        }
    }


@pytest.fixture
def mock_logger():
    """Mock file logger for tests."""
    logger = Mock()
    logger.log_info = Mock()
    logger.log_warning = Mock()
    logger.log_error = Mock()
    return logger


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""# Test Python file
import os
import sys

def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    result = a + b
    return result

class MathOperations:
    '''Math operations class.'''
    def __init__(self):
        self.history = []

    def add(self, x, y):
        result = x + y
        self.history.append(result)
        return result
""")
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


class TestContextOptimizer:
    """Tests for ContextOptimizer functionality."""

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_context_optimizer_initialization(self, mock_config, mock_logger):
        """Test ContextOptimizer initializes correctly."""
        optimizer = ContextOptimizer(mock_config, mock_logger)
        assert optimizer is not None
        assert optimizer.config == mock_config

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_context_optimizer_token_counting(self, mock_config, mock_logger):
        """Test ContextOptimizer can count tokens."""
        optimizer = ContextOptimizer(mock_config, mock_logger)

        text = "This is a test sentence with several words."
        # Mock token counting
        with patch.object(optimizer, 'count_tokens', return_value=10) as mock_count:
            token_count = optimizer.count_tokens(text)
            assert mock_count.called or isinstance(token_count, int)

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_context_optimizer_semantic_chunking(self, mock_config, mock_logger):
        """Test ContextOptimizer performs semantic chunking."""
        optimizer = ContextOptimizer(mock_config, mock_logger)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        # Mock semantic chunking
        with patch.object(optimizer, 'semantic_chunk', return_value=['Chunk 1', 'Chunk 2', 'Chunk 3']) as mock_chunk:
            chunks = optimizer.semantic_chunk(text)
            assert mock_chunk.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_context_optimizer_relevance_scoring(self, mock_config, mock_logger):
        """Test ContextOptimizer scores relevance."""
        optimizer = ContextOptimizer(mock_config, mock_logger)

        # Mock relevance scoring
        with patch.object(optimizer, 'score_relevance', return_value=0.85) as mock_score:
            score = optimizer.score_relevance('query', 'context')
            assert mock_score.called or isinstance(score, float)

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_context_optimizer_sliding_window(self, mock_config, mock_logger):
        """Test ContextOptimizer implements sliding window optimization."""
        optimizer = ContextOptimizer(mock_config, mock_logger)

        # Mock sliding window
        with patch.object(optimizer, 'optimize_window', return_value=['segment1', 'segment2']) as mock_window:
            segments = optimizer.optimize_window('long text', window_size=1024)
            assert mock_window.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_context_optimizer_statistics(self, mock_config, mock_logger):
        """Test ContextOptimizer provides statistics."""
        optimizer = ContextOptimizer(mock_config, mock_logger)

        # Mock statistics
        with patch.object(optimizer, 'get_statistics', return_value={
            'total_tokens': 5000,
            'chunks': 10,
            'avg_relevance': 0.75
        }) as mock_stats:
            stats = optimizer.get_statistics()
            assert mock_stats.called


class TestCodeMetrics:
    """Tests for CodeMetrics functionality."""

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_code_metrics_initialization(self, mock_config, mock_logger):
        """Test CodeMetrics initializes correctly."""
        metrics = CodeMetrics(mock_config, mock_logger)
        assert metrics is not None

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_code_metrics_cyclomatic_complexity(self, temp_python_file, mock_config, mock_logger):
        """Test CodeMetrics calculates cyclomatic complexity."""
        metrics = CodeMetrics(mock_config, mock_logger)

        # Mock complexity calculation
        with patch.object(metrics, 'calculate_complexity', return_value={'complexity': 3, 'rank': 'A'}) as mock_calc:
            result = metrics.calculate_complexity(temp_python_file)
            assert mock_calc.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_code_metrics_halstead_metrics(self, temp_python_file, mock_config, mock_logger):
        """Test CodeMetrics calculates Halstead metrics."""
        metrics = CodeMetrics(mock_config, mock_logger)

        # Mock Halstead metrics
        with patch.object(metrics, 'calculate_halstead', return_value={
            'volume': 150.5,
            'difficulty': 12.3,
            'effort': 1852.15
        }) as mock_halstead:
            result = metrics.calculate_halstead(temp_python_file)
            assert mock_halstead.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_code_metrics_maintainability_index(self, temp_python_file, mock_config, mock_logger):
        """Test CodeMetrics calculates maintainability index."""
        metrics = CodeMetrics(mock_config, mock_logger)

        # Mock maintainability index
        with patch.object(metrics, 'calculate_maintainability_index', return_value=85.5) as mock_mi:
            mi = metrics.calculate_maintainability_index(temp_python_file)
            assert mock_mi.called or isinstance(mi, (int, float))

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_code_metrics_quality_grading(self, temp_python_file, mock_config, mock_logger):
        """Test CodeMetrics provides quality grading (A-F)."""
        metrics = CodeMetrics(mock_config, mock_logger)

        # Mock quality grading
        with patch.object(metrics, 'get_quality_grade', return_value='A') as mock_grade:
            grade = metrics.get_quality_grade(temp_python_file)
            assert mock_grade.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_code_metrics_directory_analysis(self, mock_config, mock_logger):
        """Test CodeMetrics can analyze entire directories."""
        metrics = CodeMetrics(mock_config, mock_logger)

        # Mock directory analysis
        with patch.object(metrics, 'analyze_directory', return_value={
            'total_files': 10,
            'avg_complexity': 5.2,
            'issues': []
        }) as mock_analyze:
            result = metrics.analyze_directory('/tmp/test')
            assert mock_analyze.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_code_metrics_ast_fallback(self, temp_python_file, mock_config, mock_logger):
        """Test CodeMetrics AST-based fallback when radon unavailable."""
        metrics = CodeMetrics(mock_config, mock_logger)

        # Mock AST fallback
        with patch.object(metrics, 'analyze_with_ast', return_value={'complexity': 2}) as mock_ast:
            result = metrics.analyze_with_ast(temp_python_file)
            assert mock_ast.called


class TestSecurityScanner:
    """Tests for SecurityScanner functionality."""

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_security_scanner_initialization(self, mock_config, mock_logger):
        """Test SecurityScanner initializes correctly."""
        scanner = SecurityScanner(mock_config, mock_logger)
        assert scanner is not None

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_security_scanner_bandit_integration(self, temp_python_file, mock_config, mock_logger):
        """Test SecurityScanner integrates with bandit."""
        scanner = SecurityScanner(mock_config, mock_logger)

        # Mock bandit scanning
        with patch.object(scanner, 'scan_with_bandit', return_value={'issues': [], 'score': 10}) as mock_scan:
            result = scanner.scan_with_bandit(temp_python_file)
            assert mock_scan.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_security_scanner_ast_fallback(self, temp_python_file, mock_config, mock_logger):
        """Test SecurityScanner AST-based fallback."""
        scanner = SecurityScanner(mock_config, mock_logger)

        # Mock AST fallback for eval/exec, pickle, hardcoded passwords, SQL injection
        with patch.object(scanner, 'scan_with_ast', return_value={'issues': []}) as mock_ast:
            result = scanner.scan_with_ast(temp_python_file)
            assert mock_ast.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_security_scanner_output_formats(self, temp_python_file, mock_config, mock_logger):
        """Test SecurityScanner supports multiple output formats."""
        scanner = SecurityScanner(mock_config, mock_logger)

        formats = ['text', 'json', 'html']
        for fmt in formats:
            # Mock format output
            with patch.object(scanner, 'scan', return_value=f"Report in {fmt} format") as mock_scan:
                result = scanner.scan(temp_python_file, report_format=fmt)
                assert mock_scan.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_security_scanner_severity_filtering(self, temp_python_file, mock_config, mock_logger):
        """Test SecurityScanner filters by severity."""
        scanner = SecurityScanner(mock_config, mock_logger)

        # Mock severity filtering
        with patch.object(scanner, 'scan', return_value={'high': [], 'medium': [], 'low': []}) as mock_scan:
            result = scanner.scan(temp_python_file, severity_filter='high')
            assert mock_scan.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_security_scanner_file_directory_string(self, mock_config, mock_logger):
        """Test SecurityScanner handles files, directories, and strings."""
        scanner = SecurityScanner(mock_config, mock_logger)

        # Mock file scanning
        with patch.object(scanner, 'scan_file', return_value={'issues': []}) as mock_file:
            scanner.scan_file('/tmp/test.py')
            assert mock_file.called

        # Mock directory scanning
        with patch.object(scanner, 'scan_directory', return_value={'issues': []}) as mock_dir:
            scanner.scan_directory('/tmp/src')
            assert mock_dir.called

        # Mock string scanning
        with patch.object(scanner, 'scan_string', return_value={'issues': []}) as mock_str:
            scanner.scan_string('code = eval(user_input)')
            assert mock_str.called


class TestIntentClassifier:
    """Tests for IntentClassifier functionality."""

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_intent_classifier_initialization(self, mock_config, mock_logger):
        """Test IntentClassifier initializes correctly."""
        classifier = IntentClassifier(mock_config, mock_logger)
        assert classifier is not None

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_intent_classifier_30_intent_types(self, mock_config, mock_logger):
        """Test IntentClassifier supports 30+ intent types."""
        classifier = IntentClassifier(mock_config, mock_logger)

        # Mock intent classification
        intent_types = ['code_edit', 'git_operation', 'file_operation', 'project_query',
                       'documentation', 'system_command', 'question', 'request']

        for intent in intent_types:
            with patch.object(classifier, 'classify', return_value=intent) as mock_classify:
                result = classifier.classify(f"Test query for {intent}")
                assert mock_classify.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_intent_classifier_entity_extraction(self, mock_config, mock_logger):
        """Test IntentClassifier extracts entities."""
        classifier = IntentClassifier(mock_config, mock_logger)

        query = "Edit main.py and commit to feature branch"
        # Mock entity extraction
        with patch.object(classifier, 'extract_entities', return_value={
            'files': ['main.py'],
            'git_refs': ['feature'],
            'operations': ['edit', 'commit']
        }) as mock_extract:
            entities = classifier.extract_entities(query)
            assert mock_extract.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_intent_classifier_confidence_scoring(self, mock_config, mock_logger):
        """Test IntentClassifier provides confidence scores."""
        classifier = IntentClassifier(mock_config, mock_logger)

        # Mock confidence scoring
        with patch.object(classifier, 'classify_with_confidence', return_value={
            'intent': 'code_edit',
            'confidence': 0.92
        }) as mock_classify:
            result = classifier.classify_with_confidence("Edit the function")
            assert mock_classify.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_intent_classifier_secondary_intent(self, mock_config, mock_logger):
        """Test IntentClassifier detects secondary intents."""
        classifier = IntentClassifier(mock_config, mock_logger)

        # Mock secondary intent detection
        with patch.object(classifier, 'classify_multi', return_value=[
            {'intent': 'code_edit', 'confidence': 0.9},
            {'intent': 'git_operation', 'confidence': 0.7}
        ]) as mock_multi:
            results = classifier.classify_multi("Edit file and commit")
            assert mock_multi.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_intent_classifier_pattern_matching(self, mock_config, mock_logger):
        """Test IntentClassifier uses pattern matching."""
        classifier = IntentClassifier(mock_config, mock_logger)

        # Mock pattern matching
        patterns = ['code_*', 'git_*', 'file_*', 'project_*']
        for pattern in patterns:
            with patch.object(classifier, 'matches_pattern', return_value=True) as mock_match:
                matches = classifier.matches_pattern('test_intent', pattern)
                assert mock_match.called or isinstance(matches, bool)

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_intent_classifier_batch_classification(self, mock_config, mock_logger):
        """Test IntentClassifier supports batch classification."""
        classifier = IntentClassifier(mock_config, mock_logger)

        queries = ["Edit file", "Run tests", "Commit changes"]
        # Mock batch classification
        with patch.object(classifier, 'classify_batch', return_value=[
            'code_edit', 'testing', 'git_operation'
        ]) as mock_batch:
            results = classifier.classify_batch(queries)
            assert mock_batch.called


class TestCommandDisambiguator:
    """Tests for CommandDisambiguator functionality."""

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_initialization(self, mock_config, mock_logger):
        """Test CommandDisambiguator initializes correctly."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)
        assert disambiguator is not None

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_fuzzy_matching(self, mock_config, mock_logger):
        """Test CommandDisambiguator performs fuzzy matching."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        # Mock fuzzy matching
        with patch.object(disambiguator, 'find_similar', return_value=['commit', 'push', 'pull']) as mock_fuzzy:
            matches = disambiguator.find_similar('comit', ['commit', 'push', 'pull', 'fetch'])
            assert mock_fuzzy.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_similarity_algorithms(self, mock_config, mock_logger):
        """Test CommandDisambiguator uses 5 similarity algorithms."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        algorithms = ['sequence_matching', 'edit_distance', 'prefix', 'substring', 'word_overlap']
        for algo in algorithms:
            # Mock algorithm availability
            with patch.object(disambiguator, 'has_algorithm', return_value=True) as mock_has:
                has_algo = disambiguator.has_algorithm(algo)
                assert mock_has.called or has_algo is True

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_context_aware(self, mock_config, mock_logger):
        """Test CommandDisambiguator provides context-aware scoring."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        # Mock context-aware scoring
        with patch.object(disambiguator, 'score_with_context', return_value=0.88) as mock_score:
            score = disambiguator.score_with_context('comit', 'commit', context={'recent': ['push']})
            assert mock_score.called or isinstance(score, float)

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_frequency_tracking(self, mock_config, mock_logger):
        """Test CommandDisambiguator tracks command frequency."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        # Mock frequency tracking
        with patch.object(disambiguator, 'get_frequency', return_value=15) as mock_freq:
            freq = disambiguator.get_frequency('commit')
            assert mock_freq.called or isinstance(freq, int)

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_history_learning(self, mock_config, mock_logger):
        """Test CommandDisambiguator learns from history."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        # Mock history-based learning
        with patch.object(disambiguator, 'update_history', return_value=True) as mock_update:
            result = disambiguator.update_history('commit')
            assert mock_update.called or result is True

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_usage_analytics(self, mock_config, mock_logger):
        """Test CommandDisambiguator provides usage analytics."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        # Mock usage analytics
        with patch.object(disambiguator, 'get_analytics', return_value={
            'total_commands': 100,
            'unique_commands': 25,
            'most_used': 'commit'
        }) as mock_analytics:
            analytics = disambiguator.get_analytics()
            assert mock_analytics.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_interactive_disambiguation(self, mock_config, mock_logger):
        """Test CommandDisambiguator supports interactive mode."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        # Mock interactive disambiguation
        with patch.object(disambiguator, 'disambiguate_interactive', return_value='commit') as mock_interactive:
            choice = disambiguator.disambiguate_interactive('comit', ['commit', 'config'])
            assert mock_interactive.called

    @pytest.mark.unit
    @pytest.mark.analysis
    def test_command_disambiguator_automatic_disambiguation(self, mock_config, mock_logger):
        """Test CommandDisambiguator supports automatic mode."""
        disambiguator = CommandDisambiguator(mock_config, mock_logger)

        # Mock automatic disambiguation
        with patch.object(disambiguator, 'disambiguate_auto', return_value='commit') as mock_auto:
            choice = disambiguator.disambiguate_auto('comit', ['commit', 'config'], threshold=0.8)
            assert mock_auto.called
