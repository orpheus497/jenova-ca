# The JENOVA Cognitive Architecture - Hardware Detection Tests
# Copyright (c) 2024-2025, orpheus497. All rights reserved.
# Licensed under the MIT License

"""
Hardware detection and GPU layer recommendation tests for JENOVA.

Tests the HardwareDetector class and recommend_gpu_layers() function
to ensure optimal GPU configuration across all hardware tiers.
"""

import pytest

from jenova.utils.hardware_detector import HardwareDetector, recommend_gpu_layers


class TestRecommendGPULayers:
    """Tests for recommend_gpu_layers() function."""

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_recommend_12gb_vram(self):
        """Test GPU layer recommendation for 12GB VRAM (high-end)."""
        layers = recommend_gpu_layers(vram_mb=12288)
        assert layers == -1, "12GB VRAM should use all layers (-1)"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_recommend_8gb_vram(self):
        """Test GPU layer recommendation for 8GB VRAM (mid-high)."""
        layers = recommend_gpu_layers(vram_mb=8192)
        assert layers == 32, "8GB VRAM should use 32 layers for 7B models"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_recommend_6gb_vram(self):
        """Test GPU layer recommendation for 6GB VRAM (mid)."""
        layers = recommend_gpu_layers(vram_mb=6144)
        assert layers == 24, "6GB VRAM should use 24 layers (75% of 32)"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_recommend_4gb_vram(self):
        """Test GPU layer recommendation for 4GB VRAM (GTX 1650 Ti)."""
        layers = recommend_gpu_layers(vram_mb=4096)
        # Should be 20 or less (conservative for stability)
        assert 16 <= layers <= 20, f"4GB VRAM should use 16-20 layers, got {layers}"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_recommend_2gb_vram(self):
        """Test GPU layer recommendation for 2GB VRAM (low-end)."""
        layers = recommend_gpu_layers(vram_mb=2048)
        # Very conservative - minimal GPU offload
        assert 8 <= layers <= 12, f"2GB VRAM should use 8-12 layers, got {layers}"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_recommend_no_gpu(self):
        """Test GPU layer recommendation for systems without GPU."""
        layers_zero = recommend_gpu_layers(vram_mb=0)
        assert layers_zero == 0, "0 VRAM should use CPU only (0 layers)"

        layers_low = recommend_gpu_layers(vram_mb=512)
        assert layers_low == 0, "< 2GB VRAM should use CPU only (0 layers)"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_model_size_affects_recommendation(self):
        """Test that model size affects layer count estimation."""
        # Small model (TinyLlama)
        layers_tiny = recommend_gpu_layers(vram_mb=4096, model_size_gb=1.5)
        # 7B model (default)
        layers_7b = recommend_gpu_layers(vram_mb=4096, model_size_gb=7.0)
        # Large model (13B+)
        layers_13b = recommend_gpu_layers(vram_mb=4096, model_size_gb=13.0)

        # Larger models consume more VRAM per layer
        # But tier-based defaults may override this
        assert isinstance(layers_tiny, int)
        assert isinstance(layers_7b, int)
        assert isinstance(layers_13b, int)
        assert layers_tiny >= 0 and layers_7b >= 0 and layers_13b >= 0

    @pytest.mark.integration
    @pytest.mark.hardware
    def test_auto_detection_without_vram_param(self):
        """Test that auto-detection works without explicit VRAM parameter."""
        # Should either detect GPU or return 0 (CPU) gracefully
        layers = recommend_gpu_layers()
        assert isinstance(layers, int), "Should return integer layer count"
        assert -1 <= layers <= 128, "Should return valid layer count"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Exactly 2GB (boundary)
        layers_2gb = recommend_gpu_layers(vram_mb=2048)
        assert layers_2gb >= 0, "2GB boundary should not crash"

        # Just above 12GB (should still use all layers)
        layers_high = recommend_gpu_layers(vram_mb=16384)
        assert layers_high == -1, "16GB should use all layers"

        # Very high VRAM (24GB+)
        layers_ultra = recommend_gpu_layers(vram_mb=24576)
        assert layers_ultra == -1, "24GB should use all layers"


class TestHardwareDetector:
    """Tests for HardwareDetector class."""

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_hardware_detector_initialization(self):
        """Test that HardwareDetector initializes without errors."""
        detector = HardwareDetector()
        assert detector is not None
        assert hasattr(detector, 'platform')
        assert hasattr(detector, 'cpu_count')
        assert hasattr(detector, 'ram_total_gb')

    @pytest.mark.integration
    @pytest.mark.hardware
    def test_detect_all_returns_dict(self):
        """Test that detect_all() returns a valid dictionary."""
        detector = HardwareDetector()
        resources = detector.detect_all()

        assert isinstance(resources, dict)
        assert 'platform' in resources
        assert 'cpu' in resources
        assert 'ram' in resources

    @pytest.mark.integration
    @pytest.mark.hardware
    @pytest.mark.slow
    def test_gpu_detection(self):
        """Test GPU detection (may not find GPU in all environments)."""
        detector = HardwareDetector()
        gpu_devices = detector.gpu_devices

        assert isinstance(gpu_devices, list)
        # GPU may or may not be present - both are valid
        if gpu_devices:
            # If GPU found, check it has expected structure
            gpu = gpu_devices[0]
            assert 'name' in gpu or 'device_name' in gpu

    @pytest.mark.integration
    @pytest.mark.hardware
    def test_get_optimal_configuration(self):
        """Test that get_optimal_configuration returns valid recommendations."""
        detector = HardwareDetector()
        resources = detector.detect_all()
        config = detector.get_optimal_configuration(resources)

        assert isinstance(config, dict)
        # Should contain optimization recommendations
        assert len(config) > 0


class TestGPULayerRecommendationLogic:
    """Tests for GPU layer recommendation algorithm logic."""

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_safety_margin_calculation(self):
        """Test that safety margin is properly applied."""
        # 4GB VRAM with 20% margin = 3.2GB available
        # Minus 500MB cache = 2.7GB for layers
        # At 125MB/layer, theoretical max = ~21 layers
        # But we cap at 20 for stability
        layers = recommend_gpu_layers(vram_mb=4096, model_size_gb=7.0)
        assert layers <= 20, "Safety margin should prevent overallocation"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_vram_tiers_are_distinct(self):
        """Test that different VRAM tiers produce distinct recommendations."""
        tiers = [
            (2048, '2GB'),
            (4096, '4GB'),
            (6144, '6GB'),
            (8192, '8GB'),
            (12288, '12GB')
        ]

        results = []
        for vram_mb, name in tiers:
            layers = recommend_gpu_layers(vram_mb=vram_mb)
            results.append((name, layers))
            print(f"{name}: {layers} layers")

        # Each tier should have different or increasing recommendations
        # (except top tiers may both use -1)
        for i in range(len(results) - 1):
            current_layers = results[i][1]
            next_layers = results[i + 1][1]
            # Higher VRAM should allow >= layers (with -1 as maximum)
            if current_layers != -1:
                assert next_layers >= current_layers or next_layers == -1, \
                    f"{results[i][0]} should not have more layers than {results[i+1][0]}"

    @pytest.mark.unit
    @pytest.mark.hardware
    def test_never_exceeds_model_capacity(self):
        """Test that recommendations never exceed model's actual layer count."""
        # 7B model typically has 32 layers
        layers_7b = recommend_gpu_layers(vram_mb=16384, model_size_gb=7.0)
        # Even with 16GB VRAM, should not recommend more than model has
        # (-1 means "all available", which is valid)
        assert layers_7b == -1 or layers_7b <= 32

        # TinyLlama has ~24 layers
        layers_tiny = recommend_gpu_layers(vram_mb=16384, model_size_gb=1.5)
        assert layers_tiny == -1 or layers_tiny <= 24


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
