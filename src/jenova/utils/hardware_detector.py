# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Comprehensive hardware detection and optimization module.

Supports:
- NVIDIA GPUs (discrete and mobile)
- AMD GPUs and APUs
- Intel GPUs (Iris, UHD, Arc, integrated)
- CPU-only systems
- ARM architectures (Apple Silicon, Android, etc.)
- Multi-platform: Linux, macOS, Windows, Android/Termux
"""

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class HardwareType(Enum):
    """Types of compute hardware."""

    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    AMD_APU = "amd_apu"
    INTEL_GPU = "intel_gpu"
    INTEL_INTEGRATED = "intel_integrated"
    APPLE_SILICON = "apple_silicon"
    ARM_CPU = "arm_cpu"
    X86_CPU = "x86_cpu"
    UNKNOWN = "unknown"


class Platform(Enum):
    """Operating system platforms."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    ANDROID = "android"
    UNKNOWN = "unknown"


@dataclass
class ComputeDevice:
    """Information about a compute device."""

    device_type: HardwareType
    name: str
    memory_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    compute_units: Optional[int] = None
    is_integrated: bool = False
    supports_cuda: bool = False
    supports_opencl: bool = False
    supports_vulkan: bool = False
    supports_metal: bool = False
    device_id: Optional[str] = None
    priority: int = 0  # Higher priority = preferred for compute


@dataclass
class SystemResources:
    """Overall system resource information."""

    platform: Platform
    architecture: str  # x86_64, aarch64, arm64, etc.
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_mb: int
    ram_available_mb: int
    swap_total_mb: int
    swap_free_mb: int
    compute_devices: List[ComputeDevice]


class HardwareDetector:
    """Comprehensive hardware detection for optimal resource allocation."""

    def __init__(self):
        self.platform = self._detect_platform()
        self.architecture = platform.machine().lower()

    def _detect_platform(self) -> Platform:
        """Detect the operating system platform."""
        system = platform.system().lower()
        if system == "linux":
            # Check if running on Android/Termux
            if os.path.exists("/system/build.prop") or "TERMUX_VERSION" in os.environ:
                return Platform.ANDROID
            return Platform.LINUX
        elif system == "darwin":
            return Platform.MACOS
        elif system == "windows":
            return Platform.WINDOWS
        return Platform.UNKNOWN

    def _run_command(self, cmd: List[str], timeout: int = 5) -> Optional[str]:
        """Run a shell command and return output, or None on failure."""
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return None

    def _detect_nvidia_gpus(self) -> List[ComputeDevice]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        devices = []

        # Try nvidia-smi
        output = self._run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,index",
                "--format=csv,noheader,nounits",
            ]
        )

        if output:
            for line in output.split("\n"):
                if line.strip():
                    try:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            device = ComputeDevice(
                                device_type=HardwareType.NVIDIA_GPU,
                                name=parts[0],
                                memory_mb=int(parts[1]),
                                memory_free_mb=int(parts[2]),
                                is_integrated=False,
                                supports_cuda=True,
                                supports_opencl=True,
                                supports_vulkan=True,
                                device_id=parts[3],
                                priority=100,  # High priority for CUDA support
                            )
                            devices.append(device)
                    except (ValueError, IndexError):
                        continue

        return devices

    def _detect_amd_gpus(self) -> List[ComputeDevice]:
        """Detect AMD GPUs and APUs."""
        devices = []

        if self.platform == Platform.LINUX:
            # Try rocm-smi for discrete AMD GPUs
            output = self._run_command(["rocm-smi", "--showmeminfo", "vram"])
            if output:
                # Parse rocm-smi output
                # This is a simplified parser - real implementation would be more robust
                gpu_match = re.search(r"GPU\[(\d+)\]", output)
                if gpu_match:
                    # Get GPU name from lspci
                    lspci_output = self._run_command(["lspci"])
                    if lspci_output:
                        for line in lspci_output.split("\n"):
                            if (
                                "VGA" in line
                                and "AMD" in line.upper()
                                or "ATI" in line.upper()
                            ):
                                name = (
                                    line.split(":", 1)[1].strip()
                                    if ":" in line
                                    else "AMD GPU"
                                )
                                device = ComputeDevice(
                                    device_type=HardwareType.AMD_GPU,
                                    name=name,
                                    is_integrated=False,
                                    supports_opencl=True,
                                    supports_vulkan=True,
                                    priority=90,
                                )
                                devices.append(device)

            # Detect AMD APUs (integrated graphics)
            lspci_output = self._run_command(["lspci"])
            if lspci_output:
                for line in lspci_output.split("\n"):
                    if "VGA" in line or "Display" in line:
                        line_upper = line.upper()
                        if any(
                            apu in line_upper
                            for apu in [
                                "RENOIR",
                                "CEZANNE",
                                "BARCELO",
                                "REMBRANDT",
                                "PHOENIX",
                                "PICASSO",
                                "RAVEN",
                            ]
                        ):
                            name = (
                                line.split(":", 1)[1].strip()
                                if ":" in line
                                else "AMD APU"
                            )
                            device = ComputeDevice(
                                device_type=HardwareType.AMD_APU,
                                name=name,
                                is_integrated=True,
                                supports_opencl=True,
                                supports_vulkan=True,
                                priority=70,
                            )
                            devices.append(device)

        elif self.platform == Platform.WINDOWS:
            # Use wmic on Windows
            output = self._run_command(
                ["wmic", "path", "win32_VideoController", "get", "name"]
            )
            if output:
                for line in output.split("\n"):
                    if "AMD" in line.upper() or "ATI" in line.upper():
                        is_apu = any(apu in line.upper() for apu in ["RYZEN", "APU"])
                        device = ComputeDevice(
                            device_type=(
                                HardwareType.AMD_APU if is_apu else HardwareType.AMD_GPU
                            ),
                            name=line.strip(),
                            is_integrated=is_apu,
                            supports_opencl=True,
                            supports_vulkan=True,
                            priority=70 if is_apu else 90,
                        )
                        devices.append(device)

        return devices

    def _detect_intel_gpus(self) -> List[ComputeDevice]:
        """Detect Intel GPUs (Iris, UHD, Arc, integrated graphics)."""
        devices = []

        if self.platform == Platform.LINUX:
            # Check for Intel GPUs via lspci
            lspci_output = self._run_command(["lspci"])
            if lspci_output:
                for line in lspci_output.split("\n"):
                    line_upper = line.upper()
                    if (
                        "VGA" in line_upper
                        or "DISPLAY" in line_upper
                        or "3D" in line_upper
                        or "GRAPHICS" in line_upper
                    ) and "INTEL" in line_upper:
                        # Extract the GPU name
                        # Format: "0000:00:02.0 VGA compatible controller: Intel Corporation ..."
                        # Find the part after "controller:" or "Controller:"
                        name = "Intel GPU"
                        controller_match = re.search(
                            r"(?:controller|Controller):\s*(.+)", line
                        )
                        if controller_match:
                            name = controller_match.group(1).strip()
                        elif " " in line:
                            # Fallback: get everything after the first space-separated section
                            parts = line.split(None, 1)
                            if len(parts) > 1:
                                name = parts[1]

                        # Determine if it's discrete (Arc) or integrated
                        is_arc = "ARC" in line_upper
                        is_integrated = not is_arc

                        device_type = (
                            HardwareType.INTEL_GPU
                            if is_arc
                            else HardwareType.INTEL_INTEGRATED
                        )

                        # Try to get memory info from /sys
                        memory_mb = None
                        try:
                            # Intel integrated graphics share system RAM
                            # We'll estimate based on system RAM
                            if is_integrated:
                                meminfo = self._run_command(["cat", "/proc/meminfo"])
                                if meminfo:
                                    for mem_line in meminfo.split("\n"):
                                        if mem_line.startswith("MemTotal:"):
                                            total_kb = int(mem_line.split()[1])
                                            # Allocate ~1/8 of system RAM for integrated GPU
                                            memory_mb = (total_kb // 1024) // 8
                                            break
                        except Exception:
                            pass

                        device = ComputeDevice(
                            device_type=device_type,
                            name=name,
                            memory_mb=memory_mb,
                            is_integrated=is_integrated,
                            supports_opencl=True,
                            supports_vulkan=True,
                            priority=80 if is_arc else 60,
                        )
                        devices.append(device)

        elif self.platform == Platform.WINDOWS:
            output = self._run_command(
                ["wmic", "path", "win32_VideoController", "get", "name"]
            )
            if output:
                for line in output.split("\n"):
                    if "INTEL" in line.upper() and (
                        "IRIS" in line.upper()
                        or "UHD" in line.upper()
                        or "ARC" in line.upper()
                        or "HD" in line.upper()
                    ):
                        is_arc = "ARC" in line.upper()
                        device = ComputeDevice(
                            device_type=(
                                HardwareType.INTEL_GPU
                                if is_arc
                                else HardwareType.INTEL_INTEGRATED
                            ),
                            name=line.strip(),
                            is_integrated=not is_arc,
                            supports_opencl=True,
                            supports_vulkan=True,
                            priority=80 if is_arc else 60,
                        )
                        devices.append(device)

        elif self.platform == Platform.MACOS:
            # Intel Macs might have Iris or UHD graphics
            output = self._run_command(["system_profiler", "SPDisplaysDataType"])
            if output:
                if "Intel" in output:
                    # Extract Intel GPU name
                    for line in output.split("\n"):
                        if "Chipset Model:" in line and "Intel" in line:
                            name = line.split(":", 1)[1].strip()
                            device = ComputeDevice(
                                device_type=HardwareType.INTEL_INTEGRATED,
                                name=name,
                                is_integrated=True,
                                supports_metal=True,
                                supports_opencl=True,
                                priority=60,
                            )
                            devices.append(device)

        return devices

    def _detect_apple_silicon(self) -> List[ComputeDevice]:
        """Detect Apple Silicon (M1, M2, M3, etc.)."""
        devices = []

        if self.platform == Platform.MACOS and self.architecture in [
            "arm64",
            "aarch64",
        ]:
            # Check for Apple Silicon
            output = self._run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
            if output and "Apple" in output:
                # Get GPU core count
                gpu_cores = None
                sysinfo = self._run_command(["system_profiler", "SPDisplaysDataType"])
                if sysinfo:
                    # Try to extract GPU core count
                    match = re.search(r"Total Number of Cores:\s*(\d+)", sysinfo)
                    if match:
                        gpu_cores = int(match.group(1))

                device = ComputeDevice(
                    device_type=HardwareType.APPLE_SILICON,
                    name=output,
                    is_integrated=True,
                    supports_metal=True,
                    compute_units=gpu_cores,
                    priority=95,  # Apple Silicon is very efficient
                )
                devices.append(device)

        return devices

    def _detect_cpu(self) -> Tuple[str, int, int]:
        """Detect CPU information: name, physical cores, logical cores."""
        cpu_name = "Unknown CPU"
        physical_cores = 1
        logical_cores = 1

        try:
            if self.platform == Platform.LINUX or self.platform == Platform.ANDROID:
                # Get CPU name
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_name = line.split(":", 1)[1].strip()
                            break

                # Get core counts
                logical_cores = os.cpu_count() or 1
                # Estimate physical cores (rough heuristic)
                physical_cores = logical_cores // 2 if logical_cores > 1 else 1

                # Try to get actual physical core count
                output = self._run_command(["lscpu"])
                if output:
                    for line in output.split("\n"):
                        if "Core(s) per socket:" in line:
                            try:
                                cores_per_socket = int(line.split(":")[1].strip())
                                sockets = 1
                                for l2 in output.split("\n"):
                                    if "Socket(s):" in l2:
                                        sockets = int(l2.split(":")[1].strip())
                                        break
                                physical_cores = cores_per_socket * sockets
                            except ValueError:
                                pass

            elif self.platform == Platform.MACOS:
                cpu_name = (
                    self._run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
                    or "Unknown CPU"
                )
                physical_cores = int(
                    self._run_command(["sysctl", "-n", "hw.physicalcpu"]) or "1"
                )
                logical_cores = int(
                    self._run_command(["sysctl", "-n", "hw.logicalcpu"]) or "1"
                )

            elif self.platform == Platform.WINDOWS:
                output = self._run_command(["wmic", "cpu", "get", "name"])
                if output:
                    lines = output.split("\n")
                    if len(lines) > 1:
                        cpu_name = lines[1].strip()

                logical_cores = os.cpu_count() or 1
                output = self._run_command(["wmic", "cpu", "get", "NumberOfCores"])
                if output:
                    try:
                        physical_cores = int(output.split("\n")[1].strip())
                    except (ValueError, IndexError):
                        physical_cores = logical_cores // 2

        except Exception:
            pass

        return cpu_name, max(1, physical_cores), max(1, logical_cores)

    def _detect_memory(self) -> Tuple[int, int, int, int]:
        """
        Detect system memory: RAM total, RAM available, swap total, swap free.
        Returns values in MB.
        """
        ram_total_mb = 0
        ram_available_mb = 0
        swap_total_mb = 0
        swap_free_mb = 0

        try:
            if self.platform == Platform.LINUX or self.platform == Platform.ANDROID:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                    for line in meminfo.split("\n"):
                        if line.startswith("MemTotal:"):
                            ram_total_mb = int(line.split()[1]) // 1024
                        elif line.startswith("MemAvailable:"):
                            ram_available_mb = int(line.split()[1]) // 1024
                        elif line.startswith("SwapTotal:"):
                            swap_total_mb = int(line.split()[1]) // 1024
                        elif line.startswith("SwapFree:"):
                            swap_free_mb = int(line.split()[1]) // 1024

            elif self.platform == Platform.MACOS:
                # RAM
                output = self._run_command(["sysctl", "-n", "hw.memsize"])
                if output:
                    ram_total_mb = int(output) // (1024 * 1024)

                # Available memory (approximate)
                vm_stat = self._run_command(["vm_stat"])
                if vm_stat:
                    # Parse vm_stat for free and inactive memory
                    free_pages = 0
                    for line in vm_stat.split("\n"):
                        if "Pages free:" in line:
                            free_pages += int(line.split(":")[1].strip().rstrip("."))
                        elif "Pages inactive:" in line:
                            free_pages += int(line.split(":")[1].strip().rstrip("."))

                    # macOS page size is typically 4KB
                    ram_available_mb = (free_pages * 4096) // (1024 * 1024)

                # Swap info
                output = self._run_command(["sysctl", "-n", "vm.swapusage"])
                if output:
                    # Parse: "total = X.XXM  used = X.XXM  free = X.XXM"
                    match = re.search(r"total = ([\d.]+)([MG])", output)
                    if match:
                        val = float(match.group(1))
                        swap_total_mb = (
                            int(val * 1024) if match.group(2) == "G" else int(val)
                        )

                    match = re.search(r"free = ([\d.]+)([MG])", output)
                    if match:
                        val = float(match.group(1))
                        swap_free_mb = (
                            int(val * 1024) if match.group(2) == "G" else int(val)
                        )

            elif self.platform == Platform.WINDOWS:
                # Total RAM
                output = self._run_command(
                    ["wmic", "OS", "get", "TotalVisibleMemorySize"]
                )
                if output:
                    try:
                        ram_total_mb = int(output.split("\n")[1].strip()) // 1024
                    except (ValueError, IndexError):
                        pass

                # Free RAM
                output = self._run_command(["wmic", "OS", "get", "FreePhysicalMemory"])
                if output:
                    try:
                        ram_available_mb = int(output.split("\n")[1].strip()) // 1024
                    except (ValueError, IndexError):
                        pass

                # Swap/Page file
                output = self._run_command(
                    ["wmic", "pagefile", "get", "AllocatedBaseSize"]
                )
                if output:
                    try:
                        swap_total_mb = int(output.split("\n")[1].strip())
                    except (ValueError, IndexError):
                        pass

        except Exception:
            pass

        return ram_total_mb, ram_available_mb, swap_total_mb, swap_free_mb

    def detect_all(self) -> SystemResources:
        """Perform comprehensive hardware detection."""
        # Detect all compute devices
        compute_devices = []

        # NVIDIA GPUs
        compute_devices.extend(self._detect_nvidia_gpus())

        # AMD GPUs and APUs
        compute_devices.extend(self._detect_amd_gpus())

        # Intel GPUs
        compute_devices.extend(self._detect_intel_gpus())

        # Apple Silicon
        compute_devices.extend(self._detect_apple_silicon())

        # Sort by priority (highest first)
        compute_devices.sort(key=lambda d: d.priority, reverse=True)

        # CPU info
        cpu_name, physical_cores, logical_cores = self._detect_cpu()

        # Memory info
        ram_total_mb, ram_available_mb, swap_total_mb, swap_free_mb = (
            self._detect_memory()
        )

        return SystemResources(
            platform=self.platform,
            architecture=self.architecture,
            cpu_name=cpu_name,
            cpu_cores_physical=physical_cores,
            cpu_cores_logical=logical_cores,
            ram_total_mb=ram_total_mb,
            ram_available_mb=ram_available_mb,
            swap_total_mb=swap_total_mb,
            swap_free_mb=swap_free_mb,
            compute_devices=compute_devices,
        )

    def get_optimal_configuration(
        self,
        resources: SystemResources,
        model_size_mb: int = 4000,
        context_size: int = 8192,
    ) -> Dict:
        """
        Calculate optimal configuration based on detected hardware.

        Args:
            resources: System resource information
            model_size_mb: Estimated model size in MB
            context_size: Desired context size in tokens

        Returns:
            Dictionary with optimal configuration
        """
        config = {
            "device": "cpu",
            "gpu_layers": 0,
            "threads": resources.cpu_cores_physical,
            "use_mlock": False,
            "use_mmap": True,
            "low_vram": False,
            "n_batch": 512,
            "offload_kqv": True,
            "recommended_context": context_size,
            "memory_strategy": "balanced",
            "backend": "cpu",
        }

        # Determine best compute device
        if resources.compute_devices:
            best_device = resources.compute_devices[0]

            if best_device.device_type == HardwareType.NVIDIA_GPU:
                config["device"] = "cuda"
                config["backend"] = "cuda"
                config["gpu_layers"] = self._calculate_gpu_layers_nvidia(
                    best_device, model_size_mb, context_size
                )

            elif best_device.device_type in [
                HardwareType.AMD_GPU,
                HardwareType.AMD_APU,
            ]:
                config["device"] = "opencl"
                config["backend"] = "opencl"
                config["gpu_layers"] = self._calculate_gpu_layers_amd(
                    best_device, model_size_mb, context_size, resources.ram_available_mb
                )

            elif best_device.device_type in [
                HardwareType.INTEL_GPU,
                HardwareType.INTEL_INTEGRATED,
            ]:
                config["device"] = "opencl"
                config["backend"] = "opencl"
                config["gpu_layers"] = self._calculate_gpu_layers_intel(
                    best_device, model_size_mb, context_size, resources.ram_available_mb
                )
                # Intel integrated benefits from lower batch size
                config["n_batch"] = 256

            elif best_device.device_type == HardwareType.APPLE_SILICON:
                config["device"] = "metal"
                config["backend"] = "metal"
                config["gpu_layers"] = -1  # Apple Silicon uses unified memory
                config["use_mmap"] = True

        # Memory management strategy
        total_required_mb = model_size_mb + self._estimate_kv_cache_mb(context_size)

        if resources.ram_available_mb >= total_required_mb * 1.5:
            # Plenty of RAM - use mlock for performance
            config["use_mlock"] = True
            config["memory_strategy"] = "performance"
        elif resources.ram_available_mb >= total_required_mb:
            # Enough RAM - use mmap without mlock
            config["use_mlock"] = False
            config["memory_strategy"] = "balanced"
        elif resources.ram_available_mb + resources.swap_free_mb >= total_required_mb:
            # Need swap - optimize for swap usage
            config["use_mlock"] = False
            config["use_mmap"] = True
            config["memory_strategy"] = "swap_optimized"
            # Reduce context size to fit better
            config["recommended_context"] = min(context_size, 4096)
        else:
            # Very tight on memory
            config["use_mlock"] = False
            config["memory_strategy"] = "minimal"
            config["recommended_context"] = 2048
            config["n_batch"] = 256

        return config

    def _calculate_gpu_layers_nvidia(
        self, device: ComputeDevice, model_size_mb: int, context_size: int
    ) -> int:
        """Calculate optimal GPU layers for NVIDIA GPUs."""
        if not device.memory_free_mb:
            return 0

        # Account for CUDA overhead and KV cache
        cuda_overhead = 100
        compute_buffer = 170 + (context_size / 1024 - 1) * 60
        kv_cache_per_layer = (context_size * 128 * 8 * 2 * 2) / (1024 * 1024)

        available_mb = device.memory_free_mb - cuda_overhead - compute_buffer

        if available_mb <= 0:
            return 0

        # ~125 MB per layer for Q4_K_M 7B models
        avg_mb_per_layer = 125
        total_mb_per_layer = avg_mb_per_layer + kv_cache_per_layer
        max_layers = int(available_mb / total_mb_per_layer)

        return min(max_layers, 32)

    def _calculate_gpu_layers_amd(
        self,
        device: ComputeDevice,
        model_size_mb: int,
        context_size: int,
        system_ram_mb: int,
    ) -> int:
        """Calculate optimal GPU layers for AMD GPUs/APUs."""
        if device.is_integrated:
            # APU shares system RAM
            # Use conservative allocation
            available_mb = system_ram_mb * 0.25  # Use 25% of RAM for GPU
        else:
            # Discrete GPU - estimate conservatively if memory unknown
            available_mb = device.memory_mb * 0.8 if device.memory_mb else 2048

        # ROCm overhead
        overhead = 150
        kv_cache_per_layer = (context_size * 128 * 8 * 2 * 2) / (1024 * 1024)

        available_mb -= overhead

        if available_mb <= 0:
            return 0

        avg_mb_per_layer = 125
        total_mb_per_layer = avg_mb_per_layer + kv_cache_per_layer
        max_layers = int(available_mb / total_mb_per_layer)

        # APUs benefit from partial offload
        if device.is_integrated:
            max_layers = min(max_layers, 16)

        return min(max_layers, 32)

    def _calculate_gpu_layers_intel(
        self,
        device: ComputeDevice,
        model_size_mb: int,
        context_size: int,
        system_ram_mb: int,
    ) -> int:
        """Calculate optimal GPU layers for Intel GPUs."""
        # Intel integrated GPUs share system RAM
        if device.is_integrated:
            # Use conservative allocation - Intel integrated is less powerful
            if device.memory_mb:
                available_mb = device.memory_mb * 0.8
            else:
                # Estimate based on system RAM
                available_mb = system_ram_mb * 0.15  # 15% of system RAM
        else:
            # Intel Arc discrete GPU
            available_mb = device.memory_mb * 0.8 if device.memory_mb else 4096

        overhead = 200  # Intel drivers have more overhead
        kv_cache_per_layer = (context_size * 128 * 8 * 2 * 2) / (1024 * 1024)

        available_mb -= overhead

        if available_mb <= 0:
            return 0

        avg_mb_per_layer = 125
        total_mb_per_layer = avg_mb_per_layer + kv_cache_per_layer
        max_layers = int(available_mb / total_mb_per_layer)

        # Intel integrated benefits from partial offload
        # Iris Xe can handle more than older UHD
        if device.is_integrated:
            if "IRIS" in device.name.upper() or "XE" in device.name.upper():
                max_layers = min(max_layers, 12)
            else:
                max_layers = min(max_layers, 8)

        return min(max_layers, 32)

    def _estimate_kv_cache_mb(self, context_size: int, num_layers: int = 32) -> int:
        """Estimate KV cache size for a 7B model."""
        # KV cache = n_ctx * n_layers * n_embd_head_k * n_head_kv * 2 (K+V) * 2 bytes (f16)
        return (context_size * num_layers * 128 * 8 * 2 * 2) // (1024 * 1024)


def print_system_info(resources: SystemResources):
    """Print formatted system information."""
    print(f"\n{'='*60}")
    print(f"System Hardware Detection")
    print(f"{'='*60}")
    print(f"Platform: {resources.platform.value}")
    print(f"Architecture: {resources.architecture}")
    print(f"\nCPU: {resources.cpu_name}")
    print(f"  Physical cores: {resources.cpu_cores_physical}")
    print(f"  Logical cores: {resources.cpu_cores_logical}")
    print(f"\nMemory:")
    print(f"  RAM: {resources.ram_available_mb}/{resources.ram_total_mb} MB available")
    print(f"  Swap: {resources.swap_free_mb}/{resources.swap_total_mb} MB free")

    if resources.compute_devices:
        print(f"\nCompute Devices ({len(resources.compute_devices)} found):")
        for i, device in enumerate(resources.compute_devices, 1):
            print(f"\n  {i}. {device.name}")
            print(f"     Type: {device.device_type.value}")
            print(f"     Integrated: {device.is_integrated}")
            if device.memory_mb:
                if device.memory_free_mb:
                    print(
                        f"     Memory: {device.memory_free_mb}/{device.memory_mb} MB free"
                    )
                else:
                    print(f"     Memory: {device.memory_mb} MB")
            print(f"     Backends: ", end="")
            backends = []
            if device.supports_cuda:
                backends.append("CUDA")
            if device.supports_opencl:
                backends.append("OpenCL")
            if device.supports_vulkan:
                backends.append("Vulkan")
            if device.supports_metal:
                backends.append("Metal")
            print(", ".join(backends) if backends else "None detected")
            print(f"     Priority: {device.priority}")
    else:
        print(f"\nCompute Devices: None detected (CPU-only)")

    print(f"{'='*60}\n")


def recommend_gpu_layers(vram_mb: int = None, model_size_gb: float = 7.0) -> int:
    """
    Recommend optimal GPU layer offloading based on available VRAM.

    This function provides intelligent defaults for gpu_layers configuration
    across different hardware tiers. It balances performance (more GPU layers)
    with stability (avoiding VRAM exhaustion).

    Args:
        vram_mb: Available VRAM in MB. Auto-detected if None.
        model_size_gb: Model size in GB (default 7.0 for 7B Q4 models).

    Returns:
        Recommended GPU layers:
        - -1: All layers (12GB+ VRAM)
        - 0: CPU only (<2GB VRAM or no GPU)
        - N: Specific layer count (2-32, based on VRAM tier)

    Heuristic:
        - 7B Q4 model typically has ~32 layers
        - Each layer requires ~125-150MB VRAM (varies by quantization)
        - Reserve 500MB for KV cache (context storage)
        - Reserve 20% safety margin for memory fragmentation
        - Lower tiers use conservative estimates to prevent OOM crashes

    Examples:
        >>> recommend_gpu_layers(12288)  # 12GB VRAM
        -1  # All layers
        >>> recommend_gpu_layers(4096)   # 4GB VRAM
        20  # Conservative for stability
        >>> recommend_gpu_layers(0)      # No GPU
        0   # CPU only
    """
    if vram_mb is None:
        # Auto-detect VRAM using HardwareDetector
        try:
            detector = HardwareDetector()
            if detector.gpu_devices and "vram_mb" in detector.gpu_devices[0]:
                vram_mb = detector.gpu_devices[0]["vram_mb"]
            else:
                # No GPU detected or VRAM info unavailable
                return 0
        except Exception:
            # Detection failed, default to CPU
            return 0

    # Estimate total layer count based on model size
    # These are typical values for common model families
    if model_size_gb <= 2:
        total_layers = 24  # TinyLlama, small models
    elif model_size_gb <= 4:
        total_layers = 28  # 3B models (Phi, StableLM)
    elif model_size_gb <= 8:
        total_layers = 32  # 7B models (Llama, Mistral, Gemma)
    else:
        total_layers = 40  # 13B+ models

    # VRAM allocation strategy
    reserved_for_cache_mb = 500  # KV cache for context window
    safety_margin = 0.20  # 20% safety margin for fragmentation/overhead
    available_for_layers = vram_mb * (1 - safety_margin) - reserved_for_cache_mb

    # Estimate MB per layer (varies by quantization)
    # Q4 quantization: ~125MB/layer for 7B models
    # Larger models: ~150MB/layer
    mb_per_layer = 150 if model_size_gb > 7 else 125

    # Calculate theoretical maximum layers we could fit
    calculated_layers = int(available_for_layers / mb_per_layer)
    calculated_layers = max(0, min(calculated_layers, total_layers))

    # Tier-based recommendations (conservative for stability)
    # These override calculations to provide tested, stable defaults
    if vram_mb >= 12288:  # 12GB+ (RTX 3060 12GB, RTX 4070, etc.)
        return -1  # All layers - plenty of VRAM
    elif vram_mb >= 8192:  # 8GB (RTX 3070, RTX 4060 Ti, etc.)
        return min(total_layers, 32)  # Near-maximum for 7B models
    elif vram_mb >= 6144:  # 6GB (RTX 2060, GTX 1660 Ti, etc.)
        return min(total_layers, 24)  # 75% of layers for 7B
    elif vram_mb >= 4096:  # 4GB (GTX 1650 Ti, RTX 3050 Mobile, etc.)
        # Conservative default tested on GTX 1650 Ti
        # Can go higher (24+) but 20 is stable across configs
        return min(calculated_layers, 20)
    elif vram_mb >= 2048:  # 2GB (Older GPUs, integrated graphics)
        # Very conservative - minimal GPU offload
        return min(calculated_layers, 12)
    else:  # <2GB or detection failed
        return 0  # CPU only - not worth the GPU overhead


if __name__ == "__main__":
    # Test hardware detection
    detector = HardwareDetector()
    resources = detector.detect_all()
    print_system_info(resources)

    # Get optimal configuration
    print("\nOptimal Configuration:")
    config = detector.get_optimal_configuration(resources)
    for key, value in config.items():
        print(f"  {key}: {value}")
