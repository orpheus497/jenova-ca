# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Simple, robust CUDA management for JENOVA.

This module provides safe CUDA detection and management without
complex manipulation of environment variables or internal state.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import subprocess


@dataclass
class CUDAInfo:
    """Information about CUDA availability and capabilities."""
    available: bool
    device_count: int
    device_name: Optional[str] = None
    total_memory: Optional[int] = None  # In MB
    free_memory: Optional[int] = None   # In MB
    compute_capability: Optional[Tuple[int, int]] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None


class CUDAManager:
    """
    Simple CUDA manager that detects GPU capabilities without manipulation.

    Design principles:
    1. Detection only - no environment variable manipulation
    2. Clear error messages when CUDA is unavailable
    3. Safe fallback to CPU
    4. No monkey-patching or internal state changes
    """

    def __init__(self, file_logger=None):
        self.file_logger = file_logger
        self._cuda_info: Optional[CUDAInfo] = None

    def detect_cuda(self) -> CUDAInfo:
        """
        Detect CUDA availability and capabilities.

        Returns:
            CUDAInfo object with detection results
        """
        if self._cuda_info is not None:
            return self._cuda_info

        # Try PyTorch first (most common)
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else None

                # Get memory info for first device
                total_memory = None
                free_memory = None
                if device_count > 0:
                    try:
                        props = torch.cuda.get_device_properties(0)
                        total_memory = props.total_memory // (1024 * 1024)  # Convert to MB
                        free_memory = (torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) // (1024 * 1024)
                    except:
                        pass

                self._cuda_info = CUDAInfo(
                    available=True,
                    device_count=device_count,
                    device_name=device_name,
                    total_memory=total_memory,
                    free_memory=free_memory,
                    compute_capability=torch.cuda.get_device_capability(0) if device_count > 0 else None,
                    cuda_version=torch.version.cuda
                )

                if self.file_logger:
                    self.file_logger.log_info(
                        f"CUDA detected: {device_name} with {total_memory}MB total memory"
                    )

                return self._cuda_info
        except ImportError:
            if self.file_logger:
                self.file_logger.log_info("PyTorch not available for CUDA detection")
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_warning(f"PyTorch CUDA detection failed: {e}")

        # Try nvidia-smi as fallback
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    if len(parts) >= 4:
                        self._cuda_info = CUDAInfo(
                            available=True,
                            device_count=len(lines),
                            device_name=parts[0].strip(),
                            total_memory=int(float(parts[1].strip())),
                            free_memory=int(float(parts[2].strip())),
                            driver_version=parts[3].strip()
                        )

                        if self.file_logger:
                            self.file_logger.log_info(
                                f"CUDA detected via nvidia-smi: {self._cuda_info.device_name}"
                            )

                        return self._cuda_info
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            if self.file_logger:
                self.file_logger.log_info(f"nvidia-smi detection failed: {e}")

        # No CUDA available
        self._cuda_info = CUDAInfo(available=False, device_count=0)

        if self.file_logger:
            self.file_logger.log_info("CUDA not available - will use CPU")

        return self._cuda_info

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        info = self.detect_cuda()
        return info.available

    def get_recommended_layers(self, model_size_gb: float, safety_margin: float = 0.8) -> int:
        """
        Calculate recommended GPU layers based on available VRAM.

        Args:
            model_size_gb: Approximate model size in GB
            safety_margin: Use this fraction of available memory (default 0.8 = 80%)

        Returns:
            Recommended number of GPU layers (0 if CUDA unavailable)
        """
        info = self.detect_cuda()

        if not info.available or info.free_memory is None:
            return 0

        # Convert MB to GB
        available_gb = (info.free_memory / 1024) * safety_margin

        if available_gb < 1.0:
            # Less than 1GB free - use CPU
            return 0

        # Rough heuristic: each layer needs about model_size / 40
        # For a 7B model (~4GB), that's ~100MB per layer
        estimated_layers = int((available_gb * 1024) / (model_size_gb * 1024 / 40))

        # Common layer counts for different model sizes
        if estimated_layers >= 32:
            return 32  # Full offload for 7B models
        elif estimated_layers >= 20:
            return 20  # Partial offload
        elif estimated_layers >= 16:
            return 16
        elif estimated_layers >= 8:
            return 8
        elif estimated_layers >= 4:
            return 4
        else:
            return 0  # Too little VRAM, use CPU

    def validate_gpu_config(self, requested_layers: int) -> Tuple[bool, str]:
        """
        Validate if requested GPU layers are feasible.

        Args:
            requested_layers: Number of layers user wants to offload

        Returns:
            Tuple of (is_valid, message)
        """
        info = self.detect_cuda()

        if requested_layers == 0:
            return True, "CPU-only mode (no GPU layers)"

        if not info.available:
            return False, (
                "CUDA not available but gpu_layers > 0 requested. "
                "Set gpu_layers: 0 or install CUDA drivers."
            )

        if info.free_memory is None:
            return True, "Cannot verify VRAM but CUDA is available"

        # Warn if requesting many layers with limited VRAM
        if requested_layers > 20 and info.free_memory < 3000:  # Less than 3GB
            return False, (
                f"Requested {requested_layers} layers but only {info.free_memory}MB VRAM available. "
                f"Recommended: {self.get_recommended_layers(4.0)} layers or less."
            )

        return True, f"GPU configuration valid: {requested_layers} layers"

    def get_info_summary(self) -> str:
        """Get a human-readable summary of CUDA status."""
        info = self.detect_cuda()

        if not info.available:
            return "CUDA: Not available (CPU-only mode)"

        lines = [
            f"CUDA: Available",
            f"  Devices: {info.device_count}",
        ]

        if info.device_name:
            lines.append(f"  Device: {info.device_name}")

        if info.total_memory:
            lines.append(f"  Total VRAM: {info.total_memory}MB")

        if info.free_memory:
            lines.append(f"  Free VRAM: {info.free_memory}MB")

        if info.compute_capability:
            lines.append(f"  Compute: {info.compute_capability[0]}.{info.compute_capability[1]}")

        if info.driver_version:
            lines.append(f"  Driver: {info.driver_version}")

        if info.cuda_version:
            lines.append(f"  CUDA: {info.cuda_version}")

        return "\n".join(lines)
