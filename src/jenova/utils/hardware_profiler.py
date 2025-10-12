import os
import platform
import subprocess
import psutil


class HardwareProfiler:
    """Detects detailed system hardware specifications."""
    
    def __init__(self):
        self.cpu_info = self._detect_cpu()
        self.gpu_info = self._detect_gpu()
        self.memory_info = self._detect_memory()
        self.swap_info = self._detect_swap()
        self._detect_gpu_capabilities()
        self._classify_hardware_profile()
    
    def _detect_cpu(self) -> dict:
        """Detects CPU architecture, vendor, physical core count, and model name."""
        cpu_info = {
            'architecture': platform.machine(),
            'vendor': 'Unknown',
            'physical_cores': psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
            'model_name': 'Unknown'
        }
        
        # Try to detect CPU vendor and model
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'GenuineIntel' in cpuinfo or 'Intel' in cpuinfo:
                        cpu_info['vendor'] = 'Intel'
                    elif 'AuthenticAMD' in cpuinfo or 'AMD' in cpuinfo:
                        cpu_info['vendor'] = 'AMD'
                    
                    # Extract model name
                    for line in cpuinfo.split('\n'):
                        if line.startswith('model name'):
                            cpu_info['model_name'] = line.split(':', 1)[1].strip()
                            break
                    
                    # Detect specific ARM SoC processors on aarch64
                    if cpu_info['architecture'] in ['aarch64', 'arm64']:
                        model_lower = cpu_info['model_name'].lower()
                        if 'snapdragon' in model_lower:
                            cpu_info['vendor'] = 'Qualcomm'
                            cpu_info['soc_type'] = 'Snapdragon'
                        elif 'apple' in model_lower:
                            cpu_info['vendor'] = 'Apple'
                            cpu_info['soc_type'] = 'Apple Silicon'
                        elif 'tensor' in model_lower:
                            cpu_info['vendor'] = 'Google'
                            cpu_info['soc_type'] = 'Tensor'
                        
            elif platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                        capture_output=True, text=True)
                brand = result.stdout.strip()
                cpu_info['model_name'] = brand
                if 'Intel' in brand:
                    cpu_info['vendor'] = 'Intel'
                elif 'Apple' in brand:
                    cpu_info['vendor'] = 'Apple'
                    cpu_info['soc_type'] = 'Apple Silicon'
            elif platform.system() == 'Windows':
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                     r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                brand = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                cpu_info['model_name'] = brand
                if 'Intel' in brand:
                    cpu_info['vendor'] = 'Intel'
                elif 'AMD' in brand:
                    cpu_info['vendor'] = 'AMD'
        except Exception:
            pass  # Keep as 'Unknown' if detection fails
        
        return cpu_info
    
    def _detect_gpu(self) -> dict:
        """Detects GPU vendor, available VRAM, and whether it's integrated or dedicated."""
        gpu_info = {
            'vendor': 'None',
            'vram_mb': 0,
            'is_apu': False,  # Integrated graphics (APU/iGPU)
            'is_dedicated': False
        }
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                vram = result.stdout.strip().split('\n')[0]
                gpu_info['vendor'] = 'NVIDIA'
                gpu_info['vram_mb'] = int(float(vram))
                gpu_info['is_dedicated'] = True
                return gpu_info
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        # Check for AMD GPU
        try:
            result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse rocm-smi output
                for line in result.stdout.split('\n'):
                    if 'Total Memory' in line or 'VRAM Total' in line:
                        # Extract memory value
                        parts = line.split(':')
                        if len(parts) > 1:
                            mem_str = parts[1].strip().split()[0]
                            gpu_info['vendor'] = 'AMD'
                            gpu_info['vram_mb'] = int(float(mem_str))
                            gpu_info['is_dedicated'] = True
                            return gpu_info
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        # Fallback: Check /sys/class/drm for GPU detection on Linux
        if platform.system() == 'Linux':
            try:
                drm_path = '/sys/class/drm'
                if os.path.exists(drm_path):
                    for device in os.listdir(drm_path):
                        device_path = os.path.join(drm_path, device, 'device')
                        if os.path.exists(device_path):
                            vendor_path = os.path.join(device_path, 'vendor')
                            if os.path.exists(vendor_path):
                                with open(vendor_path, 'r') as f:
                                    vendor_id = f.read().strip()
                                    if vendor_id == '0x10de':
                                        gpu_info['vendor'] = 'NVIDIA'
                                        # Try to read VRAM info
                                        mem_info_path = os.path.join(device_path, 'mem_info_vram_total')
                                        if os.path.exists(mem_info_path):
                                            with open(mem_info_path, 'r') as mem_f:
                                                vram_bytes = int(mem_f.read().strip())
                                                gpu_info['vram_mb'] = vram_bytes // (1024 * 1024)
                                        gpu_info['is_dedicated'] = True
                                        break
                                    elif vendor_id == '0x1002':
                                        gpu_info['vendor'] = 'AMD'
                                        # Check if it's an APU by looking for integrated graphics markers
                                        # APUs typically don't have dedicated VRAM in /sys/class/drm
                                        gpu_info['is_apu'] = self._is_amd_apu(device_path)
                                        if gpu_info['is_apu']:
                                            # For APUs, get UMA/GART memory size
                                            gpu_info['vram_mb'] = self._get_apu_memory_size()
                                        break
                                    elif vendor_id == '0x8086':
                                        gpu_info['vendor'] = 'Intel'
                                        gpu_info['is_apu'] = True
                                        gpu_info['vram_mb'] = self._get_apu_memory_size()
                                        break
            except Exception:
                pass
        
        return gpu_info
    
    def _is_amd_apu(self, device_path: str) -> bool:
        """Check if an AMD GPU is an integrated APU."""
        try:
            # Check if there's no dedicated VRAM
            mem_info_path = os.path.join(device_path, 'mem_info_vram_total')
            if not os.path.exists(mem_info_path):
                return True
            
            # Check device class - APUs typically show as VGA controller
            class_path = os.path.join(device_path, 'class')
            if os.path.exists(class_path):
                with open(class_path, 'r') as f:
                    device_class = f.read().strip()
                    # 0x030000 is VGA controller, often indicates integrated graphics
                    if device_class == '0x030000':
                        return True
        except Exception:
            pass
        return False
    
    def _get_apu_memory_size(self) -> int:
        """Get allocated GPU memory (UMA/GART size) for APUs/iGPUs."""
        try:
            # Try to read from /sys/kernel/debug/dri (requires root or debugfs mounted)
            if os.path.exists('/sys/kernel/debug/dri'):
                for entry in os.listdir('/sys/kernel/debug/dri'):
                    # Look for entries like "0" or "1"
                    if entry.isdigit():
                        uma_path = os.path.join('/sys/kernel/debug/dri', entry, 'amdgpu_gtt_mm')
                        if os.path.exists(uma_path):
                            with open(uma_path, 'r') as f:
                                content = f.read()
                                # Parse the size from the output
                                for line in content.split('\n'):
                                    if 'size:' in line:
                                        size_bytes = int(line.split(':')[1].strip())
                                        return size_bytes // (1024 * 1024)
            
            # Fallback: estimate based on system RAM (typically 512MB-2GB for APUs)
            total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
            if total_ram_mb >= 16000:
                return 2048  # 2GB for systems with 16GB+ RAM
            elif total_ram_mb >= 8000:
                return 1024  # 1GB for systems with 8GB+ RAM
            else:
                return 512   # 512MB for systems with less RAM
        except Exception:
            pass
        
        return 512  # Conservative default
    
    def _detect_memory(self) -> dict:
        """Detects total system RAM."""
        memory_info = {
            'total_mb': 0
        }
        
        try:
            mem = psutil.virtual_memory()
            memory_info['total_mb'] = mem.total // (1024 * 1024)
        except Exception:
            pass
        
        return memory_info
    
    def _detect_swap(self) -> dict:
        """Detects swap space configuration."""
        swap_info = {
            'total_mb': 0,
            'available': False
        }
        
        try:
            swap = psutil.swap_memory()
            swap_info['total_mb'] = swap.total // (1024 * 1024)
            swap_info['available'] = swap.total > 0
        except Exception:
            pass
        
        return swap_info
    
    def get_summary(self) -> dict:
        """Returns a summary of all detected hardware."""
        return {
            'cpu': self.cpu_info,
            'gpu': self.gpu_info,
            'memory': self.memory_info,
            'swap': self.swap_info,
            'gpu_capabilities': getattr(self, 'gpu_capabilities', {}),
            'hardware_profile': getattr(self, 'hardware_profile', 'Unknown')
        }
    
    def _detect_gpu_capabilities(self):
        """Check for actual GPU runtime support (CUDA/ROCm libraries)."""
        self.gpu_capabilities = {
            'cuda_available': False,
            'rocm_available': False
        }
        
        # Check for CUDA runtime library
        try:
            if platform.system() == 'Linux':
                # Check for libcudart.so in common locations
                cuda_paths = [
                    '/usr/local/cuda/lib64/libcudart.so',
                    '/usr/lib/x86_64-linux-gnu/libcudart.so',
                    '/usr/lib64/libcudart.so'
                ]
                for path in cuda_paths:
                    if os.path.exists(path):
                        self.gpu_capabilities['cuda_available'] = True
                        break
                
                # Also check via ldconfig
                if not self.gpu_capabilities['cuda_available']:
                    result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=5)
                    if 'libcudart.so' in result.stdout:
                        self.gpu_capabilities['cuda_available'] = True
            elif platform.system() == 'Windows':
                # Check for cudart64_*.dll
                import glob
                cuda_dlls = glob.glob('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin\\cudart64_*.dll')
                if cuda_dlls:
                    self.gpu_capabilities['cuda_available'] = True
        except Exception:
            pass
        
        # Check for ROCm runtime library
        try:
            if platform.system() == 'Linux':
                # Check for libamdhip64.so
                rocm_paths = [
                    '/opt/rocm/lib/libamdhip64.so',
                    '/usr/lib/libamdhip64.so',
                    '/usr/lib64/libamdhip64.so'
                ]
                for path in rocm_paths:
                    if os.path.exists(path):
                        self.gpu_capabilities['rocm_available'] = True
                        break
                
                # Also check via ldconfig
                if not self.gpu_capabilities['rocm_available']:
                    result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=5)
                    if 'libamdhip64.so' in result.stdout:
                        self.gpu_capabilities['rocm_available'] = True
        except Exception:
            pass
    
    def _classify_hardware_profile(self):
        """Classify the hardware profile for optimization strategy selection."""
        gpu_vendor = self.gpu_info.get('vendor', 'None')
        is_apu = self.gpu_info.get('is_apu', False)
        is_dedicated = self.gpu_info.get('is_dedicated', False)
        cpu_arch = self.cpu_info.get('architecture', '')
        soc_type = self.cpu_info.get('soc_type', None)
        
        if soc_type in ['Apple Silicon', 'Snapdragon', 'Tensor']:
            self.hardware_profile = f'High-Performance ARM SoC ({soc_type})'
        elif is_apu:
            if gpu_vendor == 'AMD':
                self.hardware_profile = 'AMD APU with shared memory'
            elif gpu_vendor == 'Intel':
                self.hardware_profile = 'Intel CPU with integrated graphics'
            else:
                self.hardware_profile = 'APU/iGPU with shared memory'
        elif is_dedicated:
            if gpu_vendor == 'NVIDIA':
                self.hardware_profile = 'Dedicated NVIDIA GPU'
            elif gpu_vendor == 'AMD':
                self.hardware_profile = 'Dedicated AMD GPU'
            else:
                self.hardware_profile = 'Dedicated GPU'
        else:
            self.hardware_profile = 'CPU-only system'

