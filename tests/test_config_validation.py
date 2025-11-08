# The JENOVA Cognitive Architecture - Configuration Validation Tests
# Copyright (c) 2024-2025, orpheus497. All rights reserved.
# Licensed under the MIT License

"""
Configuration validation tests for JENOVA.

Tests Pydantic schema validation for all configuration options,
ensuring type safety and constraint enforcement.
"""

import pytest
from pydantic import ValidationError

from jenova.config.config_schema import (
    JenovaConfig, ModelConfig, HardwareConfig,
    DeviceType, MemoryStrategy
)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_default_model_config_valid(self):
        """Test that default ModelConfig values are valid."""
        config = ModelConfig()
        assert config.threads == -1
        assert config.gpu_layers == 'auto'
        assert config.mlock is False
        assert config.context_size == 4096
        assert config.temperature == 0.7

    @pytest.mark.unit
    @pytest.mark.config
    def test_gpu_layers_accepts_auto(self):
        """Test that gpu_layers accepts 'auto' string value."""
        config = ModelConfig(gpu_layers='auto')
        assert config.gpu_layers == 'auto'

        config_upper = ModelConfig(gpu_layers='AUTO')
        assert config_upper.gpu_layers == 'auto'  # Should be lowercase

    @pytest.mark.unit
    @pytest.mark.config
    def test_gpu_layers_accepts_integers(self):
        """Test that gpu_layers accepts various integer values."""
        # Test -1 (all layers)
        config_all = ModelConfig(gpu_layers=-1)
        assert config_all.gpu_layers == -1

        # Test 0 (CPU only)
        config_cpu = ModelConfig(gpu_layers=0)
        assert config_cpu.gpu_layers == 0

        # Test specific layer count
        config_20 = ModelConfig(gpu_layers=20)
        assert config_20.gpu_layers == 20

        config_32 = ModelConfig(gpu_layers=32)
        assert config_32.gpu_layers == 32

    @pytest.mark.unit
    @pytest.mark.config
    def test_gpu_layers_rejects_invalid_string(self):
        """Test that gpu_layers rejects invalid string values."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(gpu_layers='invalid')
        assert "gpu_layers string value must be 'auto'" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.config
    def test_gpu_layers_rejects_out_of_range(self):
        """Test that gpu_layers rejects out-of-range integers."""
        # Test below minimum
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(gpu_layers=-2)
        assert "between -1 and 128" in str(exc_info.value)

        # Test above maximum
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(gpu_layers=200)
        assert "between -1 and 128" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.config
    def test_context_size_validation(self):
        """Test that context_size is validated and rounded."""
        # Common sizes should pass through
        for size in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            config = ModelConfig(context_size=size)
            assert config.context_size == size

        # Uncommon size should be rounded down
        config_odd = ModelConfig(context_size=5000)
        assert config_odd.context_size == 4096  # Rounded to nearest common

    @pytest.mark.unit
    @pytest.mark.config
    def test_temperature_bounds(self):
        """Test that temperature is bounded between 0 and 2."""
        config_min = ModelConfig(temperature=0.0)
        assert config_min.temperature == 0.0

        config_max = ModelConfig(temperature=2.0)
        assert config_max.temperature == 2.0

        # Out of range should fail
        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            ModelConfig(temperature=2.1)


class TestHardwareConfig:
    """Tests for HardwareConfig validation."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_default_hardware_config_valid(self):
        """Test that default HardwareConfig values are valid."""
        config = HardwareConfig()
        assert config.show_details is False
        assert config.prefer_device == DeviceType.AUTO
        assert config.memory_strategy == MemoryStrategy.AUTO
        assert config.enable_health_monitor is True
        assert config.pytorch_gpu_enabled is False  # New field

    @pytest.mark.unit
    @pytest.mark.config
    def test_pytorch_gpu_enabled_is_boolean(self):
        """Test that pytorch_gpu_enabled accepts boolean values."""
        config_false = HardwareConfig(pytorch_gpu_enabled=False)
        assert config_false.pytorch_gpu_enabled is False

        config_true = HardwareConfig(pytorch_gpu_enabled=True)
        assert config_true.pytorch_gpu_enabled is True

    @pytest.mark.unit
    @pytest.mark.config
    def test_prefer_device_enum_validation(self):
        """Test that prefer_device accepts valid enum values."""
        config_auto = HardwareConfig(prefer_device='auto')
        assert config_auto.prefer_device == DeviceType.AUTO

        config_cuda = HardwareConfig(prefer_device='cuda')
        assert config_cuda.prefer_device == DeviceType.CUDA

        config_cpu = HardwareConfig(prefer_device='cpu')
        assert config_cpu.prefer_device == DeviceType.CPU

        # Invalid device should fail
        with pytest.raises(ValidationError):
            HardwareConfig(prefer_device='invalid_device')

    @pytest.mark.unit
    @pytest.mark.config
    def test_memory_strategy_enum_validation(self):
        """Test that memory_strategy accepts valid enum values."""
        for strategy in ['auto', 'performance', 'balanced', 'minimal']:
            config = HardwareConfig(memory_strategy=strategy)
            assert config.memory_strategy.value == strategy


class TestFullConfigIntegration:
    """Integration tests for complete configuration loading."""

    @pytest.mark.integration
    @pytest.mark.config
    def test_config_with_auto_gpu_layers(self):
        """Test that full config accepts gpu_layers: auto."""
        config_dict = {
            'model': {
                'model_path': '/path/to/model.gguf',
                'threads': -1,
                'gpu_layers': 'auto',  # Test new feature
                'mlock': False,
                'n_batch': 256,
                'context_size': 4096,
                'max_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.95,
                'embedding_model': 'all-MiniLM-L6-v2',
                'timeout_seconds': 240
            }
        }

        model_config = ModelConfig(**config_dict['model'])
        assert model_config.gpu_layers == 'auto'

    @pytest.mark.integration
    @pytest.mark.config
    def test_config_with_pytorch_gpu_enabled(self):
        """Test that full config accepts pytorch_gpu_enabled."""
        config_dict = {
            'hardware': {
                'show_details': False,
                'prefer_device': 'cuda',
                'memory_strategy': 'balanced',
                'enable_health_monitor': True,
                'pytorch_gpu_enabled': True  # Test new feature
            }
        }

        hw_config = HardwareConfig(**config_dict['hardware'])
        assert hw_config.pytorch_gpu_enabled is True

    @pytest.mark.integration
    @pytest.mark.config
    def test_backward_compatibility_numeric_gpu_layers(self):
        """Test that old numeric gpu_layers configs still work."""
        # Old config with numeric value
        old_config = {
            'model': {
                'model_path': '/path/to/model.gguf',
                'threads': 8,
                'gpu_layers': 20,  # Old default for 4GB VRAM
                'mlock': False,
                'n_batch': 256,
                'context_size': 4096,
                'max_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.95,
                'embedding_model': 'all-MiniLM-L6-v2',
                'timeout_seconds': 240
            }
        }

        model_config = ModelConfig(**old_config['model'])
        assert model_config.gpu_layers == 20
        assert model_config.threads == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
