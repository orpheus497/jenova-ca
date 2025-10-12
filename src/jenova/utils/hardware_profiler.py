import platform
import psutil


class HardwareProfiler:
    """Simple hardware detection - v3.0.1 stable logic."""
    
    def __init__(self):
        self.cpu_info = self._detect_cpu()
        self.gpu_info = self._detect_gpu()
        self.memory_info = self._detect_memory()
    
    def _detect_cpu(self) -> dict:
        """Detects basic CPU information."""
        cpu_info = {
            'architecture': platform.machine(),
            'vendor': 'Unknown',
            'physical_cores': psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
        }
        return cpu_info
    
    def _detect_gpu(self) -> dict:
        """Simple GPU detection."""
        gpu_info = {
            'vendor': 'None',
            'vram_mb': 0
        }
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

