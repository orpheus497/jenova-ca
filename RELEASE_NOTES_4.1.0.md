# JENOVA Cognitive Architecture v4.1.0 Release Notes

**Release Date:** October 29, 2025
**Author:** orpheus497
**License:** MIT

---

## 🚀 Major Features

### Universal Hardware Detection & Multi-Platform Support

Version 4.1.0 introduces comprehensive hardware detection and optimization, making JENOVA truly universal across all computing platforms and GPU types.

**Supported Hardware:**
- ✅ **NVIDIA GPUs** (GeForce, RTX, Quadro) - Full CUDA support
- ✅ **Intel GPUs** (Iris Xe, UHD, Arc) - OpenCL/Vulkan support
- ✅ **AMD GPUs & APUs** (Radeon, Ryzen with graphics) - OpenCL/ROCm support
- ✅ **Apple Silicon** (M1/M2/M3/M4) - Native Metal support
- ✅ **ARM CPUs** - Including Android/Termux compatibility
- ✅ **Multi-GPU Systems** - Automatic detection and intelligent prioritization

**Supported Platforms:**
- ✅ Linux (all distributions)
- ✅ macOS (Intel and Apple Silicon)
- ✅ Windows 10/11
- ✅ Android via Termux

### Intelligent Resource Management

The new hardware detection system automatically configures optimal settings based on your specific hardware:

- **Automatic GPU Detection:** Identifies all available compute devices with detailed specifications
- **Memory Strategies:** Performance, balanced, swap-optimized, and minimal modes
- **Multi-GPU Prioritization:** Automatically selects the best GPU in hybrid systems (e.g., laptop with integrated + discrete GPU)
- **RAM & Swap Optimization:** Intelligent memory management to keep your system responsive
- **Platform-Specific Tuning:** Optimizations tailored to your OS and architecture

---

## 📋 Complete Feature List

### Added
- Comprehensive hardware detection system (`utils/hardware_detector.py`)
- Multi-GPU support with automatic prioritization
- Platform-specific memory management strategies
- Hardware configuration section in `main_config.yaml`
- Comprehensive hardware support guide (`docs/HARDWARE_SUPPORT.md`)
- Security: Configurable whitelist for shell command execution
- Granular error handling in application startup
- Retry logic with exponential backoff for LLM calls
- Dynamic cognitive scheduler considering conversation velocity
- Advanced context re-ranking in memory search
- Data validation layer for all memory modules
- Recursive text chunking for document processing
- Dictionary-based command dispatcher for terminal UI
- Comprehensive FOSS dependency attribution in README

### Changed
- Applied `autopep8` and `isort` for consistent code formatting
- Added module-level docstrings to all Python files
- Standardized creator attribution in all source files
- Reconfigured GPU offload strategy for maximum utilization
- Enhanced error reporting for model loading failures
- Updated environment variables (PYTORCH_ALLOC_CONF)
- Strengthened installation scripts with robust error handling
- Optimized cognitive prompts for better accuracy
- Optimized Cortex reflection for large graphs
- Enhanced fine-tuning data generation
- Integrated hardware detection into model loader

### Fixed
- Startup crash from llama-cpp-python cleanup method
- VRAM allocation conflicts between PyTorch and llama-cpp-python
- Out of memory errors during GPU model loading
- Context size optimization for 4GB VRAM constraints
- CUDA detection without triggering PyTorch initialization
- Logger availability checks across all modules
- Testing framework initialization issues
- FileTools sandbox path traversal vulnerabilities
- Web search tool error handling

### Removed
- Deprecated `finetuning` section from config (use `finetune/` directory)
- Unused `reorganize_insights` method
- Development artifacts in `enhancement_plan/` directory

---

## 🎯 Upgrade Instructions

### From 4.0.0 to 4.1.0

The upgrade is **100% backward compatible**. No configuration changes are required, but you can take advantage of new features:

**Option 1: Keep Existing Configuration**
```bash
cd jenova-ca
git pull
source venv/bin/activate
pip install -e . --upgrade
```

Your existing `main_config.yaml` will continue to work without modification.

**Option 2: Enable Hardware Detection Features**

Add this new section to your `src/jenova/config/main_config.yaml`:

```yaml
hardware:
  # Show detailed hardware detection info at startup
  show_details: false

  # Device preference: 'auto', 'cuda', 'opencl', 'vulkan', 'metal', 'cpu'
  prefer_device: 'auto'

  # Which GPU to use in multi-GPU systems (0 = first, 1 = second, etc.)
  device_index: 0

  # Memory strategy: 'auto', 'performance', 'balanced', 'swap_optimized', 'minimal'
  memory_strategy: 'auto'
```

### For Intel/AMD GPU Users

To use Intel Iris/Arc or AMD GPUs, you need llama-cpp-python with OpenCL support:

```bash
source venv/bin/activate

# For OpenCL support (Intel/AMD)
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Or for Vulkan support
CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Test Hardware Detection

Run the standalone hardware detector to see what JENOVA detects:

```bash
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'src')
from jenova.utils.hardware_detector import HardwareDetector, print_system_info

detector = HardwareDetector()
resources = detector.detect_all()
print_system_info(resources)
"
```

---

## 📚 Documentation

- **Hardware Support Guide:** `docs/HARDWARE_SUPPORT.md` - Complete guide for all hardware types
- **README:** Updated with hardware detection information
- **CHANGELOG:** Full details of all changes

---

## 🐛 Known Issues

None reported for this release.

---

## 🔗 Links

- **Repository:** https://github.com/orpheus497/jenova-ca
- **Issues:** https://github.com/orpheus497/jenova-ca/issues
- **Documentation:** See README.md and docs/ directory

---

## 🙏 Acknowledgments

This release includes contributions and improvements to:
- llama-cpp-python (MIT License)
- PyTorch (BSD License)
- ChromaDB (Apache 2.0)
- sentence-transformers (Apache 2.0)
- Rich (MIT License)
- And all other FOSS dependencies

Special thanks to the open-source community for making this project possible.

---

## 📊 Statistics

- **Version:** 4.1.0
- **Release Date:** 2025-10-29
- **Files Changed:** 10+
- **New Files:** 2 (hardware_detector.py, HARDWARE_SUPPORT.md)
- **Lines Added:** 2,000+
- **Backward Compatible:** ✅ Yes
- **Breaking Changes:** ❌ None

---

**The JENOVA Cognitive Architecture** - Designed and developed by orpheus497
Licensed under the MIT License
