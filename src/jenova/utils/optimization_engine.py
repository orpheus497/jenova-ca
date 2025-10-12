import os
import json
from jenova.utils.hardware_profiler import HardwareProfiler


class OptimizationEngine:
    """Simple hardware optimization - v3.0.1 stable logic."""
    
    def __init__(self, user_data_root: str, ui_logger):
        self.user_data_root = user_data_root
        self.ui_logger = ui_logger
        self.optimization_file = os.path.join(user_data_root, 'optimization.json')
        self.profiler = HardwareProfiler()
        self.settings = {}
    
    def run(self) -> dict:
        """Profiles hardware and calculates basic optimal settings."""
        hardware = self.profiler.get_summary()
        
        # Calculate simple optimal settings
        self.settings = {
            'hardware': hardware,
            'optimal_settings': self._calculate_optimal_settings(hardware)
        }
        
        # Save settings to file
        self._save_settings()
        
        return self.settings
    
    def _calculate_optimal_settings(self, hardware: dict) -> dict:
        """Calculates simple optimal settings based on CPU cores."""
        cpu_cores = hardware['cpu']['physical_cores']
        
        # Simple, stable defaults: use available cores minus one for system
        n_threads = max(1, cpu_cores - 1)
        
        settings = {
            'n_gpu_layers': 0,  # Conservative default
            'n_threads': n_threads,
            'strategy': 'Simple Stable'
        }
        
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
