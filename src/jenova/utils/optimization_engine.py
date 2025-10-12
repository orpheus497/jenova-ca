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
        """Calculates optimal n_gpu_layers and n_threads based on hardware using multi-strategy approach."""
        settings = {
            'n_gpu_layers': 0,
            'n_threads': 4,  # Safe default
            'strategy': 'CPU-only fallback'
        }
        
        cpu_cores = hardware['cpu']['physical_cores']
        gpu_vendor = hardware['gpu']['vendor']
        gpu_vram_mb = hardware['gpu']['vram_mb']
        is_apu = hardware['gpu'].get('is_apu', False)
        is_dedicated = hardware['gpu'].get('is_dedicated', False)
        hardware_profile = hardware.get('hardware_profile', 'Unknown')
        gpu_capabilities = hardware.get('gpu_capabilities', {})
        cpu_arch = hardware['cpu'].get('architecture', '')
        soc_type = hardware['cpu'].get('soc_type', None)
        
        # Strategy 1: High-Performance ARM SoC (Apple Silicon, Snapdragon, Tensor)
        if soc_type in ['Apple Silicon', 'Snapdragon', 'Tensor']:
            settings['n_gpu_layers'] = -1  # Offload all layers to GPU
            settings['n_threads'] = cpu_cores  # Use all performance cores
            settings['strategy'] = f'High-Performance ARM SoC ({soc_type})'
            return settings
        
        # Strategy 2: APU-Specific Tuning (Balanced approach)
        if is_apu and gpu_vendor in ['AMD', 'Intel']:
            # Conservative calculation to avoid memory swapping
            # Reserve 1-2 physical cores for system/GPU data feeding
            if cpu_cores >= 8:
                settings['n_threads'] = cpu_cores - 2
            elif cpu_cores >= 4:
                settings['n_threads'] = cpu_cores - 1
            else:
                settings['n_threads'] = max(2, cpu_cores)
            
            # Calculate GPU layers conservatively for shared memory
            if gpu_vram_mb >= 2048:
                settings['n_gpu_layers'] = 20  # Conservative for 2GB+ shared
            elif gpu_vram_mb >= 1024:
                settings['n_gpu_layers'] = 12  # Conservative for 1GB+ shared
            elif gpu_vram_mb >= 512:
                settings['n_gpu_layers'] = 8   # Conservative for 512MB+ shared
            else:
                settings['n_gpu_layers'] = 0   # Too little shared memory
            
            settings['strategy'] = f'APU-Balanced (AMD APU)' if gpu_vendor == 'AMD' else f'APU-Balanced (Intel iGPU)'
            return settings
        
        # Strategy 3: Dedicated GPU Tuning (Aggressive approach)
        if is_dedicated and gpu_vendor in ['NVIDIA', 'AMD']:
            # Check for runtime support
            has_runtime = False
            if gpu_vendor == 'NVIDIA' and gpu_capabilities.get('cuda_available', False):
                has_runtime = True
            elif gpu_vendor == 'AMD' and gpu_capabilities.get('rocm_available', False):
                has_runtime = True
            
            if has_runtime:
                # Aggressive strategy - maximize GPU layers to fill dedicated VRAM
                if gpu_vram_mb >= 24000:  # 24GB+ VRAM
                    settings['n_gpu_layers'] = -1  # Offload all layers
                elif gpu_vram_mb >= 16000:  # 16-24GB VRAM
                    settings['n_gpu_layers'] = -1  # Offload all layers
                elif gpu_vram_mb >= 12000:  # 12-16GB VRAM
                    settings['n_gpu_layers'] = -1  # Offload all layers
                elif gpu_vram_mb >= 8000:  # 8-12GB VRAM
                    settings['n_gpu_layers'] = -1  # Offload all layers
                elif gpu_vram_mb >= 6000:  # 6-8GB VRAM
                    settings['n_gpu_layers'] = 40
                elif gpu_vram_mb >= 4000:  # 4-6GB VRAM
                    settings['n_gpu_layers'] = 30
                elif gpu_vram_mb >= 2000:  # 2-4GB VRAM
                    settings['n_gpu_layers'] = 20
                else:
                    settings['n_gpu_layers'] = 10
                
                # Reserve 1-2 cores for feeding the GPU
                if cpu_cores >= 8:
                    settings['n_threads'] = cpu_cores - 2
                elif cpu_cores > 2:
                    settings['n_threads'] = cpu_cores - 1
                else:
                    settings['n_threads'] = cpu_cores
                
                settings['strategy'] = f'Dedicated GPU-Aggressive ({gpu_vendor})'
                return settings
        
        # Strategy 4: CPU-Only Fallback
        # No capable GPU detected or no runtime support
        settings['n_gpu_layers'] = 0
        if cpu_cores > 4:
            settings['n_threads'] = cpu_cores - 2
        elif cpu_cores > 2:
            settings['n_threads'] = cpu_cores - 1
        else:
            settings['n_threads'] = cpu_cores
        
        settings['strategy'] = 'CPU-only fallback'
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
        report.append(f"  Strategy:        {optimal.get('strategy', 'N/A')}")
        report.append(f"  CPU Threads:     {optimal.get('n_threads', 'N/A')}")
        report.append(f"  GPU Layers:      {optimal.get('n_gpu_layers', 'N/A')}")
        if optimal.get('n_gpu_layers', 0) == -1:
            report.append("                   (All layers offloaded to GPU)")
        elif optimal.get('n_gpu_layers', 0) == 0:
            report.append("                   (CPU-only mode)")
        report.append("")
        
        # Hardware Profile
        hardware_profile = hardware.get('hardware_profile', 'Unknown')
        report.append(f"[Hardware Profile]: {hardware_profile}")
        report.append("")
        
        report.append("═══════════════════════════════════════════════════════\n")
        
        return "\n".join(report)
