# Aggressive Dynamic Resource Engine (ADRE)

## Overview

The **Aggressive Dynamic Resource Engine (ADRE)** is Jenova AI's revolutionary hardware optimization system introduced in version 3.4.0. Built on an "AI-First" principle, ADRE maximizes hardware utilization to deliver peak AI performance by dedicating the vast majority of system resources to AI operations.

## Philosophy

ADRE operates on the principle that when users run Jenova AI, they want maximum AI performance. Unlike traditional conservative approaches that leave substantial system resources idle "just in case," ADRE aggressively allocates resources with intelligent safeguards to ensure stability while maximizing performance.

### Key Principles

1. **AI-First Resource Allocation**: Dedicate the majority of system resources to AI operations
2. **Minimal System Reserve**: Only 1.5GB reserved for the operating system
3. **Intelligent Memory Management**: Use Total System Memory (RAM + Swap) as the foundation
4. **Proactive-Reactive Optimization**: Dynamically calculate optimal settings through iterative refinement
5. **Hardware-Specific Tuning**: Tailored optimization curves for different hardware types

## How ADRE Works

### 1. Total System Memory (TSM) Calculation

```
TSM = Total RAM + Total Swap
AI Budget = TSM - System Reserve (1.5GB)
```

The AI Budget represents the maximum memory available for AI operations, ensuring the OS has just enough resources to remain stable while maximizing AI performance.

**Example:**
- System with 16GB RAM + 4GB Swap
- TSM = 20GB
- AI Budget = 20GB - 1.5GB = 18.5GB available for AI

### 2. Aggressive VRAM Budgeting

ADRE uses hardware-specific VRAM allocation strategies:

#### High-Performance ARM SoCs
- **Target Hardware**: Apple Silicon, Qualcomm Snapdragon, Google Tensor
- **VRAM Budget**: 70% of AI Budget
- **Rationale**: Unified memory architecture allows aggressive allocation
- **GPU Layers**: -1 (offload all layers)
- **Threads**: All physical cores

#### Dedicated GPUs
- **Target Hardware**: NVIDIA (with CUDA) or AMD (with ROCm)
- **VRAM Budget**: 95% of dedicated VRAM
- **Rationale**: Dedicated VRAM can be maximally utilized
- **GPU Layers**: Calculated via proactive-reactive loop
- **Threads**: Physical Cores - 1

#### APUs/iGPUs
- **Target Hardware**: AMD APUs, Intel integrated graphics
- **VRAM Budget**: 50% of system RAM
- **Rationale**: Shared memory requires balanced allocation
- **GPU Layers**: Calculated via proactive-reactive loop
- **Threads**: Physical Cores - 1

#### CPU-Only Systems
- **VRAM Budget**: N/A
- **GPU Layers**: 0
- **Threads**: Physical Cores - 1

### 3. Maximal Thread Allocation

```
n_threads = Physical CPU Cores - 1
```

ADRE leaves only **one physical core** for all non-AI system tasks. This aggressive approach ensures:
- Maximum CPU utilization for AI inference
- Minimal context switching overhead
- One core available for critical system operations and I/O

**Examples:**
- 4-core CPU → 3 threads for AI
- 8-core CPU → 7 threads for AI
- 16-core CPU → 15 threads for AI
- 32-core CPU → 31 threads for AI

### 4. Proactive-Reactive Loop

ADRE's most innovative feature is its **proactive-reactive loop** for calculating optimal `n_gpu_layers`:

#### Algorithm

```python
1. Start with maximum possible layers (e.g., 80 for typical models)
2. Calculate memory footprint:
   - VRAM Used = n_layers × Layer Size (120MB estimate)
   - RAM for Remaining = (max_layers - n_layers) × Layer Size
   - Total Footprint = VRAM Used + RAM for Remaining
3. Check if Total Footprint ≤ AI Budget AND VRAM Used ≤ VRAM Budget
4. If yes: Use this n_layers (optimal found)
5. If no: Reduce n_layers by 1 and repeat from step 2
6. Continue until optimal configuration found or n_layers = 0
```

#### Benefits

- **Dynamic**: Adapts to actual system configuration
- **Optimal**: Finds the absolute maximum GPU offload that fits
- **Stable**: Ensures memory footprint never exceeds available resources
- **Intelligent**: Considers both VRAM and RAM constraints

#### Example Calculation

**System Configuration:**
- 16GB RAM, 4GB Swap, 8GB dedicated VRAM
- TSM = 20GB, AI Budget = 18.5GB
- VRAM Budget (95%) = 7.6GB

**Iteration Process:**
1. Try 80 layers: 9.6GB VRAM + 0GB RAM = 9.6GB → Exceeds VRAM budget
2. Try 70 layers: 8.4GB VRAM + 1.2GB RAM = 9.6GB → Exceeds VRAM budget
3. Try 64 layers: 7.68GB VRAM + 1.92GB RAM = 9.6GB → Exceeds VRAM budget slightly
4. Try 63 layers: 7.56GB VRAM + 2.04GB RAM = 9.6GB → Within VRAM budget! ✓

**Result**: ADRE selects 63 GPU layers for optimal performance

## Hardware-Specific Optimizations

### Apple Silicon (M1, M2, M3, etc.)

```
Strategy: ADRE: High-Performance ARM SoC (Apple Silicon) - Aggressive 70% AI Budget
n_gpu_layers: -1 (all layers on GPU)
n_threads: All physical cores
VRAM Budget: 70% of AI Budget
```

**Rationale**: Unified memory architecture allows treating all memory as VRAM while maintaining system stability.

### Qualcomm Snapdragon

```
Strategy: ADRE: High-Performance ARM SoC (Snapdragon) - Aggressive 70% AI Budget
n_gpu_layers: -1 (all layers on GPU)
n_threads: All physical cores
VRAM Budget: 70% of AI Budget
```

**Rationale**: Similar to Apple Silicon, optimized for ARM SoC unified memory.

### Google Tensor

```
Strategy: ADRE: High-Performance ARM SoC (Tensor) - Aggressive 70% AI Budget
n_gpu_layers: -1 (all layers on GPU)
n_threads: All physical cores
VRAM Budget: 70% of AI Budget
```

**Rationale**: Tensor SoCs benefit from full GPU offload with unified memory.

### NVIDIA GPUs (with CUDA)

```
Strategy: ADRE: Dedicated NVIDIA GPU - Aggressive 95% VRAM Utilization
n_gpu_layers: Calculated via proactive-reactive loop
n_threads: Physical Cores - 1
VRAM Budget: 95% of dedicated VRAM
```

**Rationale**: Dedicated VRAM allows near-complete utilization with proactive-reactive optimization.

### AMD GPUs (with ROCm)

```
Strategy: ADRE: Dedicated AMD GPU - Aggressive 95% VRAM Utilization
n_gpu_layers: Calculated via proactive-reactive loop
n_threads: Physical Cores - 1
VRAM Budget: 95% of dedicated VRAM
```

**Rationale**: Same as NVIDIA, maximizes dedicated VRAM usage.

### AMD APUs

```
Strategy: ADRE: AMD APU - Aggressive 50% RAM for VRAM
n_gpu_layers: Calculated via proactive-reactive loop
n_threads: Physical Cores - 1
VRAM Budget: 50% of system RAM
```

**Rationale**: Shared memory requires balanced allocation between CPU and GPU operations.

### Intel Integrated Graphics

```
Strategy: ADRE: Intel APU - Aggressive 50% RAM for VRAM
n_gpu_layers: Calculated via proactive-reactive loop
n_threads: Physical Cores - 1
VRAM Budget: 50% of system RAM
```

**Rationale**: Similar to AMD APUs, balances shared memory allocation.

## User Experience

### First-Run Warning

When ADRE is activated for the first time, users see a one-time warning:

```
======================================================================
⚡ AGGRESSIVE DYNAMIC RESOURCE ENGINE (ADRE) ACTIVE
======================================================================

The new Aggressive Dynamic Resource Engine is active.

While Jenova AI is running, a majority of your system's resources
will be dedicated to maximizing AI performance.

System responsiveness may be reduced during AI operations.

======================================================================
```

This warning:
- Appears only once per user
- Informs users of ADRE's aggressive resource usage
- Sets expectations about system responsiveness
- Is stored in `.adre_warning_shown` file

### Optimization Report

Users can view detailed optimization settings using the `/optimize` command:

```
╔══════════════════════════════════════════════════════╗
║     Hardware Profile & Optimization Report          ║
╚══════════════════════════════════════════════════════╝

[CPU Information]
  Architecture:    x86_64
  Vendor:          AMD
  Physical Cores:  16

[GPU Information]
  Vendor:          NVIDIA
  VRAM:            8192 MB

[Memory Information]
  Total RAM:       16.00 GB (16384 MB)
  Swap Space:      4.00 GB (4096 MB)

[Optimal Settings Applied]
  Strategy:        ADRE: Dedicated NVIDIA GPU - Aggressive 95% VRAM Utilization
  CPU Threads:     15
  GPU Layers:      64

[Hardware Profile]: Dedicated NVIDIA GPU

═══════════════════════════════════════════════════════
```

## Performance Benefits

### Benchmark Scenarios

#### Scenario 1: 8-Core CPU, 8GB NVIDIA GPU
- **Before ADRE**: 6 threads, 40 GPU layers
- **With ADRE**: 7 threads, 64 GPU layers
- **Improvement**: +16% CPU utilization, +60% GPU offload

#### Scenario 2: AMD Ryzen APU, 16GB RAM
- **Before ADRE**: 6 threads, 12 GPU layers
- **With ADRE**: 7 threads, calculated optimal layers
- **Improvement**: +16% CPU utilization, optimized shared memory

#### Scenario 3: Apple M2 Max, 32GB Unified Memory
- **Before ADRE**: Full cores, -1 layers
- **With ADRE**: Full cores, -1 layers (optimized budget)
- **Improvement**: Better memory management, reduced system reserve

## Comparison: Old System vs ADRE

| Feature | Old System | ADRE (v3.4.0) |
|---------|-----------|---------------|
| **Philosophy** | Conservative | AI-First Aggressive |
| **System Reserve** | ~4-8GB implicit | 1.5GB explicit |
| **Memory Calculation** | RAM only | TSM (RAM + Swap) |
| **VRAM Strategy** | Fixed tiers | Dynamic % allocation |
| **Thread Allocation** | Cores - 2 | Cores - 1 |
| **GPU Layers** | Fixed by tier | Proactive-reactive loop |
| **Swap Awareness** | Binary (yes/no) | Integrated in TSM |
| **Optimization** | Static strategies | Dynamic calculation |
| **Hardware-Specific** | Basic | Hyper-granular |

## Technical Implementation

### Constants

```python
SYSTEM_RESERVE_MB = 1536  # 1.5GB for OS
MODEL_LAYER_SIZE_MB = 120  # Conservative estimate per layer
```

### Main ADRE Function

```python
def _calculate_optimal_settings_adre(self, hardware: dict) -> dict:
    # Extract hardware info
    cpu_cores = hardware['cpu']['physical_cores']
    ram_mb = hardware['memory']['total_mb']
    swap_mb = hardware.get('swap', {}).get('total_mb', 0)
    
    # Calculate TSM and AI Budget
    tsm_mb = ram_mb + swap_mb
    ai_budget_mb = tsm_mb - SYSTEM_RESERVE_MB
    
    # Aggressive thread allocation
    n_threads = max(1, cpu_cores - 1)
    
    # Hardware-specific optimization...
```

### Proactive-Reactive Loop

```python
def _calculate_optimal_gpu_layers(self, vram_budget_mb, ai_budget_mb, 
                                   ram_mb, max_layers):
    n_layers = max_layers
    
    while n_layers > 0:
        vram_used_mb = n_layers * MODEL_LAYER_SIZE_MB
        remaining_layers = max(0, max_layers - n_layers)
        ram_for_layers_mb = remaining_layers * MODEL_LAYER_SIZE_MB
        total_footprint_mb = vram_used_mb + ram_for_layers_mb
        
        if (total_footprint_mb <= ai_budget_mb and 
            vram_used_mb <= vram_budget_mb):
            return n_layers
        
        n_layers -= 1
    
    return 0
```

## Best Practices

### For Users

1. **Ensure Adequate Swap**: ADRE benefits from swap space as it increases TSM
2. **Close Unnecessary Applications**: ADRE assumes AI is the priority
3. **Monitor First Run**: Observe system behavior during initial AI session
4. **Use `/optimize` Command**: Review optimization settings for your hardware

### For System Administrators

1. **Configure Swap**: Especially important for ARM systems
2. **Dedicated GPU Systems**: Ensure CUDA/ROCm runtimes are installed
3. **Multi-User Environments**: Be aware of aggressive resource usage
4. **Performance Monitoring**: Monitor system under ADRE with production workloads

## ARM-Specific Considerations

### Enhanced Swap Detection

ADRE includes fortified swap detection for ARM systems using multiple methods:

1. `swapon` command
2. `/proc/swaps` parsing
3. `free` command fallback

This ensures reliable detection across different Linux distributions.

### ARM Swap Guidance

If no swap is detected on ARM systems, Jenova displays comprehensive setup instructions:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent in /etc/fstab:
/swapfile none swap sw 0 0
```

## Troubleshooting

### System Unresponsive During AI Operations

**Expected Behavior**: ADRE dedicates most resources to AI. Minor system slowdown is normal.

**If Severe**:
1. Increase system reserve (modify `SYSTEM_RESERVE_MB`)
2. Add more swap space
3. Close background applications

### Out of Memory Errors

**Rare but Possible**:
1. Check swap configuration: `swapon --show`
2. Verify TSM calculation: Use `/optimize` command
3. Review system logs for OOM killer activity

**Resolution**:
- Add swap space if none exists
- Increase swap size if too small
- Report issue with hardware details

### GPU Not Detected

**Check**:
1. CUDA/ROCm runtime installed: `ldconfig -p | grep cuda` or `rocm`
2. GPU drivers functional: `nvidia-smi` or `rocm-smi`
3. GPU accessible to user

**ADRE Behavior**: Falls back to CPU-only mode automatically

## Future Enhancements

Planned improvements for future versions:

1. **Dynamic Layer Size Detection**: Detect actual model layer size instead of estimation
2. **Real-Time Adjustment**: Adjust `n_gpu_layers` based on actual memory usage
3. **Multi-GPU Support**: Optimize across multiple GPUs
4. **User Profiles**: Allow "balanced" vs "maximum" ADRE modes
5. **Telemetry Integration**: Learn from usage patterns to improve defaults

## Version History

- **v3.4.0 (2025-10-12)**: Initial ADRE release
  - Replaced old optimization system
  - Implemented proactive-reactive loop
  - Added one-time user warning
  - Enhanced ARM swap detection

## Support and Feedback

For issues, feedback, or suggestions:
- GitHub Issues: https://github.com/orpheus497/jenova-ai/issues
- Report hardware-specific optimization opportunities
- Share benchmark results with ADRE

---

**ADRE**: Aggressive by design, intelligent by implementation, maximum performance by default.
