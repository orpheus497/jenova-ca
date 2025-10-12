import os
import json
from jenova.utils.hardware_profiler import HardwareProfiler


class OptimizationEngine:
    """Calculates optimal performance settings based on hardware capabilities."""
    
    def __init__(self, user_data_root: str, ui_logger):
        self.user_data_root = user_data_root
        self.ui_logger = ui_logger
        self.optimization_file = os.path.join(user_data_root, 'optimization.json')
        self.profiler = HardwareProfiler()
        self.settings = {}
    
    def run(self) -> dict:
        """Profiles hardware and calculates optimal settings."""
        hardware = self.profiler.get_summary()
        
        # Calculate optimal settings
        self.settings = {
            'hardware': hardware,
            'optimal_settings': self._calculate_optimal_settings(hardware)
        }
        
        # Save settings to file
        self._save_settings()
        
        return self.settings
    
    def _calculate_optimal_settings(self, hardware: dict) -> dict:
        """Calculates optimal n_gpu_layers and n_threads based on hardware."""
        settings = {
            'n_gpu_layers': 0,
            'n_threads': 4  # Safe default
        }
        
        # Calculate optimal thread count
        cpu_cores = hardware['cpu']['physical_cores']
        # Use physical cores, leave 1-2 for system tasks
        if cpu_cores > 4:
            settings['n_threads'] = cpu_cores - 2
        elif cpu_cores > 2:
            settings['n_threads'] = cpu_cores - 1
        else:
            settings['n_threads'] = cpu_cores
        
        # Calculate optimal GPU layers
        gpu_vendor = hardware['gpu']['vendor']
        gpu_vram_mb = hardware['gpu']['vram_mb']
        
        if gpu_vendor in ['NVIDIA', 'AMD'] and gpu_vram_mb > 0:
            # Estimate how many layers can fit in VRAM
            # Rule of thumb: ~100-200MB per layer depending on model size
            # For safety, we'll be conservative
            if gpu_vram_mb >= 8000:  # 8GB+ VRAM
                settings['n_gpu_layers'] = -1  # Offload all layers
            elif gpu_vram_mb >= 6000:  # 6-8GB VRAM
                settings['n_gpu_layers'] = 35
            elif gpu_vram_mb >= 4000:  # 4-6GB VRAM
                settings['n_gpu_layers'] = 25
            elif gpu_vram_mb >= 2000:  # 2-4GB VRAM
                settings['n_gpu_layers'] = 15
            elif gpu_vram_mb >= 1000:  # 1-2GB VRAM
                settings['n_gpu_layers'] = 5
        elif gpu_vendor == 'Intel' and gpu_vram_mb > 0:
            # Intel integrated GPUs - be more conservative
            if gpu_vram_mb >= 2000:
                settings['n_gpu_layers'] = 10
            elif gpu_vram_mb >= 1000:
                settings['n_gpu_layers'] = 5
        
        return settings
    
    def _save_settings(self):
        """Saves optimization settings to file."""
        try:
            with open(self.optimization_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            self.ui_logger.system_message(f"Warning: Could not save optimization settings: {e}")
    
    def load_settings(self) -> dict:
        """Loads previously saved optimization settings."""
        if os.path.exists(self.optimization_file):
            try:
                with open(self.optimization_file, 'r') as f:
                    self.settings = json.load(f)
                return self.settings
            except Exception:
                pass
        return {}
    
    def apply_settings(self, config: dict) -> dict:
        """Applies optimal settings to the configuration."""
        if not self.settings:
            self.settings = self.run()
        
        optimal = self.settings.get('optimal_settings', {})
        
        # Apply settings to config
        if 'n_threads' in optimal:
            config['hardware']['threads'] = optimal['n_threads']
        if 'n_gpu_layers' in optimal:
            config['hardware']['gpu_layers'] = optimal['n_gpu_layers']
        
        return config
    
    def get_report(self) -> str:
        """Generates a detailed report of hardware and optimization settings."""
        if not self.settings:
            self.load_settings()
        
        if not self.settings:
            return "No optimization data available. Please restart the application to run optimization."
        
        hardware = self.settings.get('hardware', {})
        optimal = self.settings.get('optimal_settings', {})
        
        report = []
        report.append("\n╔══════════════════════════════════════════════════════╗")
        report.append("║     Hardware Profile & Optimization Report          ║")
        report.append("╚══════════════════════════════════════════════════════╝\n")
        
        # CPU Information
        cpu = hardware.get('cpu', {})
        report.append("[CPU Information]")
        report.append(f"  Architecture:    {cpu.get('architecture', 'Unknown')}")
        report.append(f"  Vendor:          {cpu.get('vendor', 'Unknown')}")
        report.append(f"  Physical Cores:  {cpu.get('physical_cores', 'Unknown')}")
        report.append("")
        
        # GPU Information
        gpu = hardware.get('gpu', {})
        report.append("[GPU Information]")
        if gpu.get('vendor', 'None') != 'None':
            report.append(f"  Vendor:          {gpu.get('vendor', 'None')}")
            report.append(f"  VRAM:            {gpu.get('vram_mb', 0)} MB")
        else:
            report.append("  No GPU detected or GPU not available")
        report.append("")
        
        # Memory Information
        memory = hardware.get('memory', {})
        report.append("[Memory Information]")
        total_mb = memory.get('total_mb', 0)
        total_gb = total_mb / 1024
        report.append(f"  Total RAM:       {total_gb:.2f} GB ({total_mb} MB)")
        report.append("")
        
        # Optimal Settings
        report.append("[Optimal Settings Applied]")
        report.append(f"  CPU Threads:     {optimal.get('n_threads', 'N/A')}")
        report.append(f"  GPU Layers:      {optimal.get('n_gpu_layers', 'N/A')}")
        if optimal.get('n_gpu_layers', 0) == -1:
            report.append("                   (All layers offloaded to GPU)")
        elif optimal.get('n_gpu_layers', 0) == 0:
            report.append("                   (CPU-only mode)")
        report.append("")
        report.append("═══════════════════════════════════════════════════════\n")
        
        return "\n".join(report)
