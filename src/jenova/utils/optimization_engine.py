import os
import json
from jenova.utils.hardware_profiler import HardwareProfiler


class OptimizationEngine:
    """Aggressive Dynamic Resource Engine (ADRE) - Maximizes hardware utilization for AI performance."""
    
    def __init__(self, user_data_root: str, ui_logger):
        self.user_data_root = user_data_root
        self.ui_logger = ui_logger
        self.optimization_file = os.path.join(user_data_root, 'optimization.json')
        self.profiler = HardwareProfiler()
        self.settings = {}
        
        # ADRE Configuration
        self.SYSTEM_RESERVE_MB = 1536  # 1.5GB minimal reserve for OS
        self.MODEL_LAYER_SIZE_MB = 120  # Estimated size per layer (conservative estimate)
    
    def run(self) -> dict:
        """Profiles hardware and calculates optimal settings using ADRE."""
        hardware = self.profiler.get_summary()
        
        # Calculate optimal settings with ADRE
        self.settings = {
            'hardware': hardware,
            'optimal_settings': self._calculate_optimal_settings_adre(hardware)
        }
        
        # Save settings to file
        self._save_settings()
        
        return self.settings
    
    def _calculate_optimal_settings_adre(self, hardware: dict) -> dict:
        """Calculates optimal settings using Aggressive Dynamic Resource Engine (ADRE)."""
        settings = {
            'n_gpu_layers': 0,
            'n_threads': 4,  # Safe default
            'strategy': 'ADRE'
        }
        
        # Extract hardware info
        cpu_cores = hardware['cpu']['physical_cores']
        cpu_model = hardware['cpu'].get('model_name', '')
        gpu_vendor = hardware['gpu']['vendor']
        gpu_vram_mb = hardware['gpu']['vram_mb']
        is_apu = hardware['gpu'].get('is_apu', False)
        is_dedicated = hardware['gpu'].get('is_dedicated', False)
        gpu_capabilities = hardware.get('gpu_capabilities', {})
        soc_type = hardware['cpu'].get('soc_type', None)
        
        # HYPER-THREADED SYNERGISTIC ENGINE
        # Set n_threads to 32 for maximum throughput via hyper-threading over-subscription
        # This saturates worker cores (all except Core 0 which is reserved for main thread)
        settings['n_threads'] = 32
        
        # Set n_gpu_layers to -1 for intelligent maximum GPU offload
        # This instructs the backend to offload every possible layer to GPU
        settings['n_gpu_layers'] = -1
        settings['strategy'] = 'ADRE: Hyper-Threaded Synergistic Engine (32 threads, max GPU layers)'
        
        return settings
    
    def _calculate_optimal_gpu_layers(self, vram_budget_mb: int, ai_budget_mb: int, 
                                       ram_mb: int, max_layers: int) -> int:
        """
        Proactive-Reactive Loop: Iteratively calculate the maximum n_gpu_layers
        that fits within the AI Budget without exceeding system resources.
        
        Args:
            vram_budget_mb: Target VRAM allocation
            ai_budget_mb: Total AI Budget (TSM - System Reserve)
            ram_mb: Total system RAM
            max_layers: Maximum layers to consider
        
        Returns:
            Optimal number of GPU layers
        """
        # Start with maximum possible layers
        n_layers = max_layers
        
        while n_layers > 0:
            # Calculate memory footprint
            vram_used_mb = n_layers * self.MODEL_LAYER_SIZE_MB
            
            # Estimate RAM needed for remaining layers (if any)
            # Assuming remaining layers stay in RAM
            remaining_layers = max(0, max_layers - n_layers)
            ram_for_layers_mb = remaining_layers * self.MODEL_LAYER_SIZE_MB
            
            # Total memory footprint
            total_footprint_mb = vram_used_mb + ram_for_layers_mb
            
            # Check if it fits within AI Budget
            if total_footprint_mb <= ai_budget_mb:
                # Additional check: ensure VRAM usage doesn't exceed budget
                if vram_used_mb <= vram_budget_mb:
                    return n_layers
            
            # Reduce layers and try again
            n_layers -= 1
        
        # If no layers fit, return 0 (CPU-only)
        return 0
    
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
        
        # Swap Information
        swap = hardware.get('swap', {})
        swap_mb = swap.get('total_mb', 0)
        swap_gb = swap_mb / 1024
        if swap.get('available', False):
            report.append(f"  Swap Space:      {swap_gb:.2f} GB ({swap_mb} MB)")
        else:
            report.append(f"  Swap Space:      Not configured")
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
