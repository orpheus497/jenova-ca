##Script function and purpose: Basic smoke tests for The JENOVA Cognitive Architecture
##Simple tests that verify basic imports and module structure

import pytest
import sys
from pathlib import Path

##Class purpose: Basic smoke tests
class TestBasicImports:
    ##Function purpose: Test that jenova module can be imported
    def test_jenova_import(self):
        """Test that the jenova module can be imported."""
        try:
            import jenova
            assert jenova is not None
        except ImportError as e:
            pytest.fail(f"Failed to import jenova module: {e}")

    ##Function purpose: Test that main module can be imported
    def test_main_import(self):
        """Test that the main module can be imported."""
        try:
            from jenova import main
            assert main is not None
            assert hasattr(main, 'main')
        except ImportError as e:
            pytest.fail(f"Failed to import main module: {e}")

    ##Function purpose: Test that config module can be imported
    def test_config_import(self):
        """Test that the config module can be imported."""
        try:
            from jenova.config import load_configuration
            assert load_configuration is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config module: {e}")

    ##Function purpose: Test Python path configuration
    def test_python_path(self):
        """Test that src directory is in Python path."""
        project_root = Path(__file__).parent.parent
        src_path = str(project_root / "src")
        assert src_path in sys.path or any(src_path in p for p in sys.path), \
            f"src directory ({src_path}) not found in Python path"

