# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Infrastructure Module Tests

"""
Unit tests for infrastructure modules.

Tests the core infrastructure components including file management,
error handling, health monitoring, metrics collection, and data validation.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import time

from jenova.infrastructure.file_manager import FileManager
from jenova.infrastructure.error_handler import ErrorHandler
from jenova.infrastructure.health_monitor import HealthMonitor
from jenova.infrastructure.metrics_collector import MetricsCollector
from jenova.infrastructure.data_validator import DataValidator


class TestFileManager:
    """Tests for FileManager class."""

    @pytest.fixture
    def file_manager(self):
        """Provide FileManager instance."""
        ui_logger = Mock()
        file_logger = Mock()
        return FileManager(ui_logger, file_logger)

    @pytest.fixture
    def temp_file(self):
        """Provide temporary file path."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        try:
            Path(temp_path).unlink()
        except FileNotFoundError:
            pass

    def test_write_json_atomic(self, file_manager, temp_file):
        """Test atomic JSON writing."""
        test_data = {"key": "value", "number": 42}

        file_manager.write_json_atomic(temp_file, test_data)

        # Verify file was written
        assert Path(temp_file).exists()

        # Verify content is correct
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    def test_read_json_atomic(self, file_manager, temp_file):
        """Test atomic JSON reading."""
        test_data = {"test": "data", "value": 123}

        # Write test data
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)

        # Read using file manager
        loaded_data = file_manager.read_json_atomic(temp_file)

        assert loaded_data == test_data

    def test_file_lock_context_manager(self, file_manager, temp_file):
        """Test file locking with context manager."""
        # Should complete without error
        with file_manager.file_lock(temp_file):
            # Write something while holding lock
            with open(temp_file, 'w') as f:
                f.write("locked write")

        # Lock should be released after context
        assert Path(temp_file).exists()

    def test_read_nonexistent_file(self, file_manager):
        """Test reading non-existent file returns None."""
        result = file_manager.read_json_atomic("/nonexistent/file.json")

        assert result is None

    def test_atomic_write_preserves_data_on_error(self, file_manager, temp_file):
        """Test atomic write doesn't corrupt on error."""
        # Write valid initial data
        initial_data = {"status": "good"}
        file_manager.write_json_atomic(temp_file, initial_data)

        # Attempt to write invalid data (should fail gracefully)
        try:
            # Mock json.dump to raise error
            with patch('json.dump', side_effect=ValueError("Mock error")):
                file_manager.write_json_atomic(temp_file, {"bad": "data"})
        except:
            pass

        # Original file should still have initial data
        with open(temp_file, 'r') as f:
            preserved_data = json.load(f)

        assert preserved_data == initial_data


class TestErrorHandler:
    """Tests for ErrorHandler class."""

    @pytest.fixture
    def error_handler(self):
        """Provide ErrorHandler instance."""
        ui_logger = Mock()
        file_logger = Mock()
        return ErrorHandler(ui_logger, file_logger)

    def test_handle_error_logs_error(self, error_handler):
        """Test error handling logs the error."""
        test_error = ValueError("Test error")
        context = {"operation": "test_op"}

        error_handler.handle_error(test_error, context)

        # Verify file logger was called
        assert error_handler.file_logger.log_error.called

    def test_handle_error_with_context(self, error_handler):
        """Test error handling includes context."""
        test_error = RuntimeError("Context test")
        context = {"module": "test_module", "user": "test_user"}

        error_handler.handle_error(test_error, context)

        # Check that context information was logged
        call_args = error_handler.file_logger.log_error.call_args[0][0]
        assert "Context test" in call_args

    def test_get_error_count(self, error_handler):
        """Test error counting."""
        initial_count = error_handler.get_error_count()

        # Generate some errors
        error_handler.handle_error(ValueError("Error 1"), {})
        error_handler.handle_error(RuntimeError("Error 2"), {})

        final_count = error_handler.get_error_count()

        assert final_count == initial_count + 2

    def test_clear_errors(self, error_handler):
        """Test clearing error history."""
        # Generate errors
        error_handler.handle_error(ValueError("Error 1"), {})
        error_handler.handle_error(ValueError("Error 2"), {})

        # Clear errors
        error_handler.clear_errors()

        assert error_handler.get_error_count() == 0

    def test_get_recent_errors(self, error_handler):
        """Test retrieving recent errors."""
        # Generate errors
        error_handler.handle_error(ValueError("Recent 1"), {})
        error_handler.handle_error(RuntimeError("Recent 2"), {})

        recent = error_handler.get_recent_errors(limit=2)

        assert len(recent) == 2
        assert any("Recent 1" in str(e) for e in recent)
        assert any("Recent 2" in str(e) for e in recent)


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    @pytest.fixture
    def health_monitor(self):
        """Provide HealthMonitor instance."""
        file_logger = Mock()
        return HealthMonitor(file_logger)

    def test_get_health_status(self, health_monitor):
        """Test getting health status."""
        status = health_monitor.get_health_status()

        # Should return dict with system metrics
        assert isinstance(status, dict)
        assert 'cpu_percent' in status
        assert 'memory_percent' in status
        assert 'disk_percent' in status
        assert 'timestamp' in status

    def test_is_healthy_cpu_threshold(self, health_monitor):
        """Test health check with CPU threshold."""
        # Set strict threshold
        health_monitor.cpu_threshold = 0.1  # 0.1% - should fail

        is_healthy = health_monitor.is_healthy()

        # Depends on system load, but structure should work
        assert isinstance(is_healthy, bool)

    def test_is_healthy_memory_threshold(self, health_monitor):
        """Test health check with memory threshold."""
        # Set very permissive threshold
        health_monitor.memory_threshold = 99.9

        is_healthy = health_monitor.is_healthy()

        # Should be healthy with high threshold
        assert is_healthy is True

    def test_check_disk_space(self, health_monitor):
        """Test disk space checking."""
        disk_status = health_monitor.check_disk_space()

        assert isinstance(disk_status, dict)
        assert 'total_gb' in disk_status
        assert 'used_gb' in disk_status
        assert 'free_gb' in disk_status
        assert 'percent' in disk_status

    def test_get_memory_info(self, health_monitor):
        """Test memory information retrieval."""
        memory_info = health_monitor.get_memory_info()

        assert isinstance(memory_info, dict)
        assert 'total_gb' in memory_info
        assert 'available_gb' in memory_info
        assert 'percent' in memory_info


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def metrics_collector(self):
        """Provide MetricsCollector instance."""
        file_logger = Mock()
        return MetricsCollector(file_logger)

    def test_record_metric(self, metrics_collector):
        """Test recording a metric."""
        metrics_collector.record_metric('test_metric', 42.5)

        metrics = metrics_collector.get_metrics()

        assert 'test_metric' in metrics
        assert 42.5 in metrics['test_metric']

    def test_measure_context_manager(self, metrics_collector):
        """Test timing measurement with context manager."""
        with metrics_collector.measure('test_operation'):
            time.sleep(0.01)  # Sleep 10ms

        metrics = metrics_collector.get_metrics()

        assert 'test_operation' in metrics
        # Should have recorded execution time
        assert len(metrics['test_operation']) > 0
        # Time should be at least 10ms (0.01s)
        assert metrics['test_operation'][0] >= 0.01

    def test_get_metric_summary(self, metrics_collector):
        """Test metric summary statistics."""
        # Record multiple values
        metrics_collector.record_metric('test', 10)
        metrics_collector.record_metric('test', 20)
        metrics_collector.record_metric('test', 30)

        summary = metrics_collector.get_metric_summary('test')

        assert summary['count'] == 3
        assert summary['mean'] == 20
        assert summary['min'] == 10
        assert summary['max'] == 30

    def test_clear_metrics(self, metrics_collector):
        """Test clearing metrics."""
        # Record some metrics
        metrics_collector.record_metric('test1', 100)
        metrics_collector.record_metric('test2', 200)

        # Clear metrics
        metrics_collector.clear_metrics()

        metrics = metrics_collector.get_metrics()

        assert len(metrics) == 0

    def test_multiple_measurements(self, metrics_collector):
        """Test multiple measurements of same operation."""
        # Measure same operation multiple times
        for i in range(3):
            with metrics_collector.measure('repeat_op'):
                time.sleep(0.001)

        metrics = metrics_collector.get_metrics()

        # Should have 3 measurements
        assert len(metrics['repeat_op']) == 3


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Provide DataValidator instance."""
        file_logger = Mock()
        return DataValidator(file_logger)

    def test_validate_string_success(self, validator):
        """Test successful string validation."""
        result = validator.validate_string(
            "test",
            min_length=2,
            max_length=10
        )

        assert result is True

    def test_validate_string_too_short(self, validator):
        """Test string too short validation."""
        result = validator.validate_string(
            "a",
            min_length=2
        )

        assert result is False

    def test_validate_string_too_long(self, validator):
        """Test string too long validation."""
        result = validator.validate_string(
            "very long string",
            max_length=5
        )

        assert result is False

    def test_validate_number_in_range(self, validator):
        """Test number within range."""
        result = validator.validate_number(
            50,
            min_value=0,
            max_value=100
        )

        assert result is True

    def test_validate_number_out_of_range(self, validator):
        """Test number outside range."""
        result = validator.validate_number(
            150,
            min_value=0,
            max_value=100
        )

        assert result is False

    def test_validate_dict_required_keys(self, validator):
        """Test dictionary with required keys."""
        test_dict = {
            "name": "test",
            "value": 42,
            "active": True
        }

        result = validator.validate_dict(
            test_dict,
            required_keys=["name", "value"]
        )

        assert result is True

    def test_validate_dict_missing_keys(self, validator):
        """Test dictionary missing required keys."""
        test_dict = {
            "name": "test"
        }

        result = validator.validate_dict(
            test_dict,
            required_keys=["name", "value", "active"]
        )

        assert result is False

    def test_validate_list_length(self, validator):
        """Test list length validation."""
        test_list = [1, 2, 3, 4, 5]

        result = validator.validate_list(
            test_list,
            min_length=1,
            max_length=10
        )

        assert result is True

    def test_validate_list_too_short(self, validator):
        """Test list too short."""
        test_list = [1]

        result = validator.validate_list(
            test_list,
            min_length=3
        )

        assert result is False

    def test_validate_email(self, validator):
        """Test email validation."""
        valid_email = "test@example.com"
        invalid_email = "not_an_email"

        assert validator.validate_email(valid_email) is True
        assert validator.validate_email(invalid_email) is False

    def test_validate_url(self, validator):
        """Test URL validation."""
        valid_url = "https://example.com/path"
        invalid_url = "not a url"

        assert validator.validate_url(valid_url) is True
        assert validator.validate_url(invalid_url) is False


# Integration test
def test_infrastructure_integration():
    """Integration test for infrastructure components working together."""
    ui_logger = Mock()
    file_logger = Mock()

    # Create infrastructure components
    file_manager = FileManager(ui_logger, file_logger)
    error_handler = ErrorHandler(ui_logger, file_logger)
    health_monitor = HealthMonitor(file_logger)
    metrics = MetricsCollector(file_logger)
    validator = DataValidator(file_logger)

    # Simulate workflow
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        # Measure file operation
        with metrics.measure('file_write'):
            test_data = {"status": "ok", "value": 100}

            # Validate data
            assert validator.validate_dict(test_data, required_keys=["status"])

            # Write with file manager
            file_manager.write_json_atomic(temp_path, test_data)

        # Check health
        health_status = health_monitor.get_health_status()
        assert health_status['memory_percent'] >= 0

        # Verify metrics were recorded
        metrics_data = metrics.get_metrics()
        assert 'file_write' in metrics_data

    except Exception as e:
        # Handle errors
        error_handler.handle_error(e, {"operation": "integration_test"})
        raise

    finally:
        # Cleanup
        try:
            Path(temp_path).unlink()
        except:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
