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
    
    def _detect_cpu(self) -> dict:
        """Detects CPU architecture, vendor, and physical core count."""
        cpu_info = {
            'architecture': platform.machine(),
            'vendor': 'Unknown',
            'physical_cores': psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
        }
        
        # Try to detect CPU vendor
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'GenuineIntel' in cpuinfo or 'Intel' in cpuinfo:
                        cpu_info['vendor'] = 'Intel'
                    elif 'AuthenticAMD' in cpuinfo or 'AMD' in cpuinfo:
                        cpu_info['vendor'] = 'AMD'
            elif platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                        capture_output=True, text=True)
                brand = result.stdout.strip()
                if 'Intel' in brand:
                    cpu_info['vendor'] = 'Intel'
                elif 'Apple' in brand:
                    cpu_info['vendor'] = 'Apple'
            elif platform.system() == 'Windows':
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                     r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                brand = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                if 'Intel' in brand:
                    cpu_info['vendor'] = 'Intel'
                elif 'AMD' in brand:
                    cpu_info['vendor'] = 'AMD'
        except Exception:
            pass  # Keep as 'Unknown' if detection fails
        
        return cpu_info
    
    def _detect_gpu(self) -> dict:
        """Detects GPU vendor and available VRAM."""
        gpu_info = {
            'vendor': 'None',
            'vram_mb': 0
        }
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                vram = result.stdout.strip().split('\n')[0]
                gpu_info['vendor'] = 'NVIDIA'
                gpu_info['vram_mb'] = int(float(vram))
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
                                        break
                                    elif vendor_id == '0x1002':
                                        gpu_info['vendor'] = 'AMD'
                                        break
                                    elif vendor_id == '0x8086':
                                        gpu_info['vendor'] = 'Intel'
                                        break
            except Exception:
                pass
        
        return gpu_info
    
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
    
    def get_summary(self) -> dict:
        """Returns a summary of all detected hardware."""
        return {
            'cpu': self.cpu_info,
            'gpu': self.gpu_info,
            'memory': self.memory_info
        }
