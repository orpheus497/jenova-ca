# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Enhanced File & Code Operations** (Phases 13-17: CLI Enhancement)
  - Advanced file editing with diff-based previews and multi-file support (`code_tools/file_editor.py`)
  - AST-based code parsing and symbol extraction for Python (`code_tools/code_parser.py`)
  - Code refactoring engine using rope library (`code_tools/refactoring_engine.py`)
  - Syntax highlighting for terminal display using Pygments (`code_tools/syntax_highlighter.py`)
  - Codebase structure mapping and analysis (`code_tools/codebase_mapper.py`)
  - Interactive terminal support with PTY for commands like vim, git rebase -i (`code_tools/interactive_terminal.py`)

- **Git Workflow Automation**
  - Comprehensive Git operations wrapper using GitPython (`git_tools/git_interface.py`)
  - Automatic commit message generation from diffs (`git_tools/commit_assistant.py`)
  - Diff analysis and summarization (`git_tools/diff_analyzer.py`)
  - Git hooks management for automation (`git_tools/hooks_manager.py`)
  - Branch operations and intelligent naming (`git_tools/branch_manager.py`)

- **Task Orchestration & Subagents** (~2,187 lines of production code)
  - Multi-step task planning with dependency graph management (`orchestration/task_planner.py` - 612 lines)
    * Topological sorting and circular dependency detection
    * Priority-based task execution with parallel level detection
    * Task plan serialization (save/load JSON)
    * Heuristic and LLM-assisted task decomposition
  - Subagent lifecycle management with ThreadPoolExecutor (`orchestration/subagent_manager.py` - 448 lines)
    * Priority-based task queue with concurrent execution
    * Resource management and status tracking
    * Pause/resume functionality with context managers
  - Task execution engine with comprehensive error handling (`orchestration/execution_engine.py` - 356 lines)
    * Automatic retry with exponential backoff (tenacity integration)
    * Execution history, statistics, and metrics tracking
    * Dependency-aware plan execution with pause/resume/cancel
  - Atomic checkpoint save/restore with filelock (`orchestration/checkpoint_manager.py` - 350 lines)
    * Thread-safe operations with timeout handling
    * Automatic backup rotation and version management
    * Import/export functionality with metadata tracking
  - Background task manager with psutil monitoring (`orchestration/background_tasks.py` - 421 lines)
    * Real-time process output capture (stdout/stderr)
    * CPU and memory monitoring with resource cleanup
    * Graceful termination and task history tracking

- **Automation & Custom Commands** (~1,784 lines of production code)
  - Custom command system with Markdown template support (`automation/custom_commands.py` - 418 lines)
    * YAML frontmatter parsing for command metadata
    * Variable extraction and template validation
    * Command CRUD operations with search functionality
  - Event-driven hooks system with priority execution (`automation/hooks_system.py` - 456 lines)
    * Pre/post/on-error hook timing with priority-based execution
    * Hook enabling/disabling with execution history
    * Comprehensive error handling and recovery
  - Template processing engine with filters and conditionals (`automation/template_engine.py` - 390 lines)
    * Variable substitution with {{variable}} syntax
    * Built-in filters (upper, lower, title, capitalize, strip, len, default)
    * Conditionals, loops, and custom filter registration
  - Workflow library with 6 pre-defined patterns (`automation/workflow_library.py` - 524 lines)
    * Code review, testing, deployment, refactoring, documentation, analysis workflows
    * Workflow step dependency management and cloning
    * Workflow serialization and comprehensive metadata

- **Enhanced Context & Analysis** (~2,737 lines of production code)
  - Context window optimization with relevance scoring (`analysis/context_optimizer.py` - 403 lines)
    * Token counting and semantic chunking
    * Relevance-based prioritization and sliding window optimization
    * Context segment management with statistics tracking
  - Code complexity metrics using radon library (`analysis/code_metrics.py` - 453 lines)
    * Cyclomatic complexity analysis (McCabe) with Halstead metrics
    * Maintainability index calculation and quality grading (A-F)
    * AST-based fallback analysis when radon unavailable
    * File and directory analysis with issue detection
  - Security vulnerability scanning with bandit (`analysis/security_scanner.py` - 511 lines)
    * Bandit integration for Python security issue detection
    * AST-based fallback for eval/exec, pickle, hardcoded passwords, SQL injection
    * Multiple output formats (text, JSON, HTML reports)
    * File, directory, and string scanning capabilities
  - Natural language intent classification with 30+ intent types (`analysis/intent_classifier.py` - 535 lines)
    * Pattern matching for code, git, file, project, documentation, system operations
    * Entity extraction (file paths, git refs, code entities, languages)
    * Confidence scoring with secondary intent detection
    * Custom pattern support and batch classification
  - Command disambiguation with fuzzy matching (`analysis/command_disambiguator.py` - 541 lines)
    * Multiple similarity algorithms (sequence matching, edit distance, prefix/substring, word overlap, acronym)
    * Context-aware scoring with frequency tracking
    * History-based learning and command usage analytics
    * Interactive and automatic disambiguation modes

- **New FOSS Dependencies** (All MIT/Apache/BSD licensed, $0 cost)
  - gitpython==3.1.43 (BSD-3-Clause) - Git operations
  - pygments==2.18.0 (BSD-2-Clause) - Syntax highlighting
  - rope==1.13.0 (LGPL) - Python refactoring
  - tree-sitter==0.21.3 (MIT) - Multi-language code parsing
  - jsonschema==4.23.0 (MIT) - JSON schema validation
  - radon==6.0.1 (MIT) - Code complexity metrics
  - bandit==1.7.10 (Apache-2.0) - Security scanning

- **Implementation Summary** (Full-Stack Completion)
  - **Total New Code**: ~6,708 lines of production-ready Python code
    * Analysis Module: 2,737 lines (5 files)
    * Orchestration Module: 2,187 lines (5 files)
    * Automation Module: 1,784 lines (4 files)
  - **No Placeholders**: All functions fully implemented with comprehensive error handling
  - **Complete Documentation**: All classes and methods include detailed docstrings
  - **Type Safety**: Type hints throughout using Python typing module
  - **Production Ready**: Logging integration, context managers, atomic operations
  - **FOSS Compliance**: 100% free and open-source dependencies, zero external APIs
  - **Creator Attribution**: MIT license headers with orpheus497 attribution on all files

- **Phase 13-17 Integration into Core Architecture** (2025-11-08)
  - Integrated all 25 CLI enhancement modules into main cognitive architecture
  - **main.py Integration** (~100 lines added)
    * Import statements for all Phase 13-17 modules (Analysis, Code Tools, Git Tools, Orchestration, Automation)
    * Initialization of 25 module instances with proper error handling
    * Graceful degradation if CLI enhancement initialization fails (non-critical)
    * CLI enhancement modules passed to CognitiveEngine via dependency injection
    * CLI enhancement modules passed to TerminalUI for command integration
    * Comprehensive logging of initialization status for each module group
  - **cognitive_engine/engine.py Integration** (~140 lines added)
    * Added 25 instance variables for CLI enhancement modules
    * Created `set_cli_enhancements()` method accepting all modules via **kwargs
    * Automatic detection and logging of enabled module groups
    * Ready for integration into cognitive processing cycle
  - **ui/terminal.py Integration** (~80 lines added)
    * Updated `__init__` signature to accept CLI modules via **kwargs
    * Stored all 25 CLI enhancement modules as instance variables
    * CLI modules passed to CommandRegistry for command handler access
    * Full backward compatibility maintained (all modules optional)
  - **ui/commands.py Command Implementation** (~260 lines added)
    * Added 5 new command categories: CODE, GIT, ANALYSIS, ORCHESTRATION, AUTOMATION
    * Updated CommandRegistry `__init__` to accept CLI enhancement modules via **kwargs
    * Implemented 9 new slash commands with full error handling:
      - `/edit` - File editing with diff-based preview (file_editor integration)
      - `/analyze` - Code quality and complexity analysis (code_metrics integration)
      - `/scan` - Security vulnerability scanning (security_scanner integration)
      - `/parse` - Code structure and AST analysis (code_parser integration)
      - `/refactor` - Code refactoring operations (refactoring_engine integration)
      - `/git` - Git operations with AI-generated commit messages (git_interface + commit_assistant)
      - `/task` - Multi-step task planning and execution (task_planner + execution_engine)
      - `/workflow` - Predefined workflow execution (workflow_library integration)
      - `/command` - Custom command management (custom_command_manager integration)
    * Each command includes comprehensive usage examples and help text
    * Commands automatically disabled if required modules not available
    * Full error handling and logging integration
  - **Architecture Benefits**:
    * All Phase 13-17 capabilities now available via user-facing commands
    * Modular design allows selective enablement of features
    * Proper separation of concerns with dependency injection
    * Non-critical module failures don't affect core system operation
    * Foundation ready for tool integration (LLM-callable functions)

- **Comprehensive Test Suite for Phase 13-17 CLI Enhancements** (2025-11-08)
  - **test_code_tools.py** (~350 lines, 20+ tests)
    * Tests for FileEditor (read, preview, apply, backup operations)
    * Tests for CodeParser (Python parsing, symbol extraction, AST mode)
    * Tests for RefactoringEngine (rename, extract-method, inline operations)
    * Tests for SyntaxHighlighter (Python highlighting, language detection)
    * Tests for CodebaseMapper (directory mapping, dependency graphs)
    * Tests for InteractiveTerminal (PTY support, vim integration)
  - **test_git_integration.py** (~340 lines, 25+ tests)
    * Tests for GitInterface (status, diff, log, branch operations)
    * Tests for CommitAssistant (message generation, auto-commit, Conventional Commits)
    * Tests for DiffAnalyzer (diff parsing, summarization, impact analysis)
    * Tests for HooksManager (hook installation, removal, listing)
    * Tests for BranchManager (create, delete, list, naming conventions)
  - **test_orchestration.py** (~400 lines, 30+ tests)
    * Tests for TaskPlanner (task creation, decomposition, dependency graphs, topological sort)
    * Tests for SubagentManager (subagent creation, priority queue, concurrent execution)
    * Tests for ExecutionEngine (plan execution, retry logic, pause/resume/cancel)
    * Tests for CheckpointManager (save/restore, atomic operations, backup rotation)
    * Tests for BackgroundTaskManager (task starting, output capture, resource monitoring)
  - **test_automation.py** (~350 lines, 28+ tests)
    * Tests for CustomCommandManager (command creation, YAML frontmatter, execution)
    * Tests for HooksSystem (hook registration, pre/post/error timing, priority execution)
    * Tests for TemplateEngine (variable substitution, filters, conditionals, loops)
    * Tests for WorkflowLibrary (6 predefined workflows, cloning, execution)
  - **test_analysis.py** (~380 lines, 32+ tests)
    * Tests for ContextOptimizer (token counting, semantic chunking, relevance scoring)
    * Tests for CodeMetrics (cyclomatic complexity, Halstead metrics, maintainability index)
    * Tests for SecurityScanner (bandit integration, AST fallback, multiple formats)
    * Tests for IntentClassifier (30+ intent types, entity extraction, confidence scoring)
    * Tests for CommandDisambiguator (fuzzy matching, 5 similarity algorithms, context-aware scoring)
  - **Test Suite Statistics**:
    * Total Test Files: 5 (new) + 2 (existing) = 7
    * Total Test Lines: ~1,820 lines (new Phase 13-17 tests)
    * Total Tests: 135+ tests (new) + 33 (existing) = 168+ comprehensive tests
    * Test Coverage: All 25 Phase 13-17 modules covered with unit tests
    * Test Framework: pytest with markers (unit, code, git, orchestration, automation, analysis)
    * Mocking Strategy: unittest.mock for isolation and fast execution
    * All tests follow existing patterns from test_config_validation.py and test_hardware_detection.py

### Changed
- **Enhanced CLI Capabilities** - JENOVA now provides CLI capabilities matching Gemini CLI, GitHub Copilot CLI, and Claude Code while maintaining 100% FOSS compliance, zero cost, and complete local operation

### Removed
- **Documentation Cleanup** - Removed DEPLOYMENT.md and TESTING.md (redundant with README.md)

### Fixed
- **Version Synchronization** - Aligned pyproject.toml version (5.0.0 → 5.1.1) with CHANGELOG
- **Dependency Security Updates** - Updated networking dependencies for security patches and stability:
  - zeroconf 0.132.2 → 0.140.0 (security patches)
  - grpcio 1.60.1 → 1.69.0 (security updates)
  - grpcio-tools 1.60.1 → 1.69.0 (compatibility update)
  - PyJWT 2.8.0 → 2.10.1 (security patches)
- **Python Version Constraint** - Added upper bound to prevent installation on untested Python versions (>=3.10,<3.14)

### Added
- **Configurable PyTorch GPU Access** (`hardware.pytorch_gpu_enabled` in main_config.yaml)
  - Allows PyTorch to use GPU for 5-10x faster embeddings on systems with 6GB+ VRAM
  - Default `false` to preserve VRAM for main LLM (maintains current behavior)
  - Conditional CUDA_VISIBLE_DEVICES based on config setting in main.py
  - Full backward compatibility with existing configurations

- **Automatic GPU Layer Detection** (`gpu_layers: auto` option in main_config.yaml)
  - Intelligent GPU layer recommendation based on available VRAM
  - `recommend_gpu_layers()` function in utils/hardware_detector.py
  - Supports all hardware tiers: 2GB, 4GB, 6GB, 8GB, 12GB+ VRAM
  - Automatic detection with 20% safety margin and KV cache reservation
  - Integration in utils/model_loader.py for seamless auto-configuration
  - Backward compatible: numeric values (0, 20, -1) still supported

- **Optional Dependency Groups** (pyproject.toml)
  - `[web]` - requests, beautifulsoup4 for web tools
  - `[browser]` - playwright for browser automation
  - `[dev]` - development tools (pytest, linters, security scanners)
  - `[all]` - all optional features
  - Install via: `pip install jenova-ai[web,browser,dev]`

- **Development Dependencies** (requirements-dev.txt)
  - Code quality: autopep8, isort, pylint, black
  - Testing: pytest, pytest-cov, pytest-asyncio, pytest-timeout
  - Security: pip-audit, safety
  - Documentation: sphinx, sphinx-rtd-theme
  - Utilities: ipython, ipdb, mypy
  - All FOSS with MIT/Apache/BSD licenses

- **Comprehensive Test Suite**
  - tests/test_config_validation.py (18 tests) - Configuration schema validation
  - tests/test_hardware_detection.py (15 tests) - Hardware detection and GPU layer recommendation
  - tests/pytest.ini - Pytest configuration with markers and settings
  - tests/__init__.py - Test suite documentation
  - Coverage: Config validation, hardware detection, backward compatibility

### Changed
- **main_config.yaml Configuration Defaults**
  - `model.gpu_layers`: Changed from `20` to `auto` for intelligent detection
  - Added `hardware.pytorch_gpu_enabled: false` with comprehensive documentation
  - Enhanced comments explaining VRAM trade-offs and recommendations

- **config_schema.py Validation**
  - Updated `ModelConfig.gpu_layers` to accept `Union[int, str]` (int or 'auto')
  - Added field validator for gpu_layers with range checking (-1 to 128)
  - Added `HardwareConfig.pytorch_gpu_enabled` boolean field
  - Imported `Union` type for type hints

- **main.py GPU Memory Management**
  - Conditional PyTorch CUDA access based on config setting
  - Quick YAML config load before imports to determine GPU allocation strategy
  - Preserves default behavior (PyTorch on CPU) for safety
  - Enhanced comments documenting VRAM allocation strategy

- **requirements.txt Optional Dependencies**
  - Updated comments to reference pyproject.toml extras
  - Updated playwright version comment (1.49.1 → 1.50.0)
  - Clearer installation instructions for optional features

### Technical Details
- **Files Modified**: 8
  - Core configs: pyproject.toml, requirements.txt, main_config.yaml
  - Validation: config_schema.py
  - GPU management: main.py, model_loader.py
  - Hardware detection: hardware_detector.py
  - Changelog: CHANGELOG.md

- **Files Created**: 5
  - Development: requirements-dev.txt
  - Testing: tests/__init__.py, tests/pytest.ini, tests/test_config_validation.py, tests/test_hardware_detection.py

- **New Capabilities**:
  - Automatic hardware-optimal GPU configuration
  - Flexible GPU VRAM allocation strategies
  - Comprehensive test coverage for new features
  - Development workflow tooling

- **Code Quality**:
  - +93 lines in hardware_detector.py (recommend_gpu_layers function)
  - +13 lines in model_loader.py (auto-detection integration)
  - +22 lines in main.py (conditional PyTorch GPU)
  - +18 lines in config_schema.py (validation updates)
  - +218 lines in tests (33 comprehensive tests)

### Backward Compatibility
- ✅ 100% backward compatible - all changes preserve existing behavior by default
- Numeric gpu_layers values (0, 20, -1) continue to work
- PyTorch CPU-only mode remains default (pytorch_gpu_enabled: false)
- Old configs load and validate without modification
- Optional dependencies remain optional

### Performance Impact
- Automatic GPU layer detection: negligible (<50ms at startup)
- PyTorch GPU mode (when enabled): 5-10x faster embeddings, -500MB-1GB VRAM for LLM
- Test suite execution: ~10-15s for all 33 tests

### Upgrade Path
1. Update code: `git pull` or download new version
2. Update dependencies: `pip install -U -r requirements.txt`
3. (Optional) Install dev tools: `pip install -r requirements-dev.txt`
4. (Optional) Enable auto-detection: Set `gpu_layers: auto` in main_config.yaml
5. (Optional) Enable PyTorch GPU: Set `pytorch_gpu_enabled: true` on 6GB+ VRAM systems
6. Run tests: `pytest tests/` to verify installation

### Attribution
- Audit, remediation, and modernization completed by **AI Chief Architect (Claude, Anthropic)**
- All enhancements aligned with project vision and requirements by **orpheus497**
- The JENOVA Cognitive Architecture (JCA) created and designed by **orpheus497**

## [5.1.1] - 2025-11-08 - Documentation Enhancement & Repository Cleanup

### Changed
- **README.md Comprehensive Enhancement**:
  - Enhanced introduction with detailed "What is JENOVA?" section explaining the complete architecture
  - Expanded "Distributed Computing Capabilities" with comprehensive feature breakdown and network architecture details
  - Restructured content to eliminate all version comparison language
  - Replaced "The JENOVA Advantage" section with "Architecture Overview: Core Design Principles"
  - Added detailed "Stateful Cognitive Processing" documentation covering memory persistence, continuous learning, identity, and reflective reasoning
  - Expanded "Cognitive Processing Cycle" with comprehensive 4-phase breakdown (Retrieve, Plan, Execute, Reflect)
  - Replaced changelog-style "Changed" section with "Code Quality and Standards" section describing current production state
  - **Massively Enhanced Command Reference**:
    - Added all Phase 9-12 commands: `/network`, `/peers`, `/settings`, `/profile`, `/learn`
    - Organized into 5 categories: System, Network, Memory, Learning, Settings
    - Comprehensive descriptions with detailed functionality, usage examples, and output descriptions
    - Total of 16+ commands fully documented with subcommand options
  - Result: README now presents complete, comprehensive technical reference with zero version comparison language

### Removed
- Deleted obsolete backup files from repository:
  - `src/jenova/main.py.backup_debug`
  - `src/jenova/config/main_config.yaml.backup_cpu_only_20251030_141030`

### Documentation
- All documentation now presents current state as complete and comprehensive
- No references to previous versions or comparisons
- Production-ready documentation suitable for official release

## [5.1.0] - 2025-11-08 - Phase 9-12: Intelligence Evolution & True Learning

### Added - Phase 9: Enhanced UI/UX & Commands
- **Comprehensive Command System** (`ui/commands.py`):
  - Command registry pattern with extensible architecture
  - `/network` command for network status and management (status, enable, disable, info)
  - `/peers` command for peer discovery and management (list, info, trust, disconnect)
  - `/settings` command with interactive settings menu
  - `/profile` command for user profile viewing and management
  - `/learn` command for learning statistics and insights
  - Enhanced `/help` with categorized commands and detailed usage examples
  - 6 command categories: System, Network, Memory, Learning, Settings, Help

- **Interactive Settings Menu** (`ui/settings_menu.py`):
  - Runtime configuration changes without restart
  - 5 settings categories: Network, LLM, Memory, Learning, Privacy
  - Type-safe validation before applying changes
  - Preview mode and pending changes management
  - Import/export settings to JSON files
  - Settings persistence to user profile
  - Undo/redo support with change history

- **Integration**:
  - Commands integrated into TerminalUI with fallback to legacy commands
  - Settings menu accessible via `/settings` command
  - Real-time setting updates with restart notifications

### Added - Phase 10: User Recognition & Personalization
- **User Profiling System** (`user/profile.py`):
  - UserProfile class with comprehensive tracking:
    * Interaction history and statistics
    * Vocabulary learning (tracks unique words used)
    * Topic interest tracking and preferred topics extraction
    * Command usage patterns
    * Correction learning for continuous improvement
    * Suggestion feedback tracking
  - UserProfileManager for multi-user support
  - JSON-based profile persistence
  - Automatic expertise level adjustment based on vocabulary
  - Response style adaptation

- **Personalization Engine** (`user/personalization.py`):
  - Adaptive response styling (concise, balanced, detailed)
  - Communication style adaptation (formal, friendly, casual, technical)
  - Proactive suggestions based on user patterns and context
  - Context-aware recommendations
  - Custom shortcuts for frequently used topics
  - Search parameter adaptation based on expertise level
  - Suggestion feedback loop for continuous improvement

- **Integration**:
  - User profile loaded at startup and passed to cognitive engine
  - Automatic interaction recording in profile
  - Response personalization before delivery to user
  - `/profile` command shows comprehensive statistics and preferences

### Added - Phase 11: Enhanced Cognitive Systems
- **Semantic Query Analyzer** (`cognitive_engine/semantic_analyzer.py`):
  - Intent classification: question, command, statement, request, clarification, feedback, greeting
  - Entity recognition: technology, proper nouns, numbers, time expressions
  - Keyword extraction with stopword filtering
  - Query expansion for better memory retrieval
  - Sentiment analysis: positive, negative, neutral, mixed
  - Topic extraction and classification
  - Rhetorical structure analysis (complexity, formality, specificity)
  - Integrated into think() cycle for enhanced query understanding

- **Enhanced Memory Retrieval**:
  - Semantic analysis used to expand queries
  - Topic-based memory search enhancement
  - Entity-focused query variations
  - Automatic recording of discussion topics in user profile

### Added - Phase 12: True Intelligence & Learning
- **Contextual Learning Engine** (`learning/contextual_engine.py`):
  - Learning from user corrections with pattern extraction
  - Pattern recognition across interactions (4 types: linguistic, behavioral, preference, error)
  - Skill acquisition and proficiency tracking
  - Practice-based skill improvement with diminishing returns
  - Knowledge transfer between domains (50% proficiency transfer)
  - Meta-cognitive performance monitoring
  - Knowledge gap identification
  - Learning insights generation

- **Learning Features**:
  - LearningExample tracking (input, expected, actual, correction, context)
  - Pattern objects with confidence scoring and occurrence tracking
  - Skill objects with domain, proficiency (0.0-1.0), and practice count
  - Performance metrics: accuracy trends, learning rate, skill/pattern counts
  - JSON persistence of examples, patterns, and skills
  - Automatic pattern extraction from corrections

- **Integration**:
  - Learning engine initialized at startup
  - Passed to cognitive engine for integration
  - `/learn` command with 4 subcommands:
    * `/learn stats` - Performance metrics and statistics
    * `/learn insights` - Learning progress insights
    * `/learn gaps` - Identified knowledge gaps
    * `/learn skills` - Acquired skills with proficiency bars

### Changed - Phase 9-12 Integration
- **CognitiveEngine** (`cognitive_engine/engine.py`):
  - Added `set_user_profile()` method for Phase 10 integration
  - Added `set_learning_engine()` method for Phase 12 integration
  - Integrated semantic analyzer into think() cycle
  - Integrated personalization engine for response adaptation
  - Automatic interaction recording in user profile
  - Topic tracking from semantic analysis
  - Query expansion using semantic analysis for better retrieval

- **Main Startup** (`main.py`):
  - Initialize UserProfileManager and load user profile
  - Initialize ContextualLearningEngine
  - Pass profile and learning engine to cognitive engine
  - Logging of profile and learning engine initialization

- **TerminalUI** (`ui/terminal.py`):
  - Integrated CommandRegistry for centralized command handling
  - Commands routed through registry with fallback to legacy commands
  - Enhanced command execution with proper error handling

- **Network Configuration** (`config/main_config.yaml`):
  - Network enabled by default (`enabled: true`)
  - Auto mode for graceful fallback (`mode: 'auto'`)
  - Automatic peer discovery with local fallback

### Fixed - Phase 9-12 Bug Fixes
- **Thread Safety** (`llm/distributed_llm_interface.py`):
  - Added `stats_lock` for thread-safe statistics updates
  - Added `round_robin_lock` for thread-safe counter incrementation
  - All statistics modifications now protected by locks

- **Peer Selection** (`network/peer_manager.py`):
  - Fixed untested peers getting 0ms average (incorrectly prioritized)
  - Changed to `float('inf')` so untested peers are last resort
  - Ensures tested peers with known latency are prioritized

### Enhancement - Documentation
- **Enhanced Documentation** (`.dev-docs/04-Enhancement_Plan.md`):
  - Comprehensive 4-week roadmap for Phases 9-12
  - Detailed feature specifications for each phase
  - Implementation priorities and success metrics
  - Technical architecture notes

### Technical Details - Phase 9-12
- **New Files Created**: 9
  - `ui/commands.py` (431 lines) - Command registry system
  - `ui/settings_menu.py` (461 lines) - Interactive settings
  - `user/profile.py` (252 lines) - User profiling
  - `user/personalization.py` (344 lines) - Personalization engine
  - `user/__init__.py` (17 lines) - User module package
  - `cognitive_engine/semantic_analyzer.py` (552 lines) - Semantic analysis
  - `learning/contextual_engine.py` (521 lines) - Learning engine
  - `learning/__init__.py` (14 lines) - Learning module package
  - `.dev-docs/04-Enhancement_Plan.md` (188 lines) - Enhancement roadmap

- **Files Modified**: 5
  - `main.py` (+49 lines) - User profile and learning engine initialization
  - `cognitive_engine/engine.py` (+135 lines) - Integration of all Phase 10-12 systems
  - `ui/terminal.py` (+27 lines) - Command registry integration
  - `config/main_config.yaml` (network.enabled: false → true)
  - `CHANGELOG.md` (this file)

- **Total New Code**: ~2,200 lines of production code
- **New Capabilities**: 11 major systems (commands, settings, profile, personalization, semantic, learning + 5 integrations)
- **New Commands**: 7 enhanced commands (/network, /peers, /settings, /profile, /learn + enhanced /help)

### Breaking Changes
None - all enhancements are additive and backward compatible

### Performance Impact
- Semantic analysis adds ~10-20ms per query (negligible)
- User profile updates are async and non-blocking
- Learning engine saves periodically, not on every interaction
- Settings menu operations are O(1) lookups

### Upgrade Path
No special steps required - all systems initialize automatically

### Attribution
- Phase 9-12 enhancements implemented by **Claude (Anthropic AI Assistant)** under guidance from **orpheus497**
- Architecture and requirements specified by **orpheus497**
- The JENOVA Cognitive Architecture (JCA) created by **orpheus497**

## [5.0.1] - 2025-11-08 - Phase 8 Remediation: Production Hardening

### Security Fixes
- **CRITICAL**: Fixed unencrypted private key storage - keys now encrypted with Scrypt KDF + AES (`network/security_store.py`)
- Added certificate pinning with Trust on First Use (TOFU) validation (`network/security.py`)
- Fixed deprecated `datetime.utcnow()` → `datetime.now(timezone.utc)` for Python 3.12 compatibility
- Added JWT token expiration enforcement (validity_seconds parameter)
- All network communication encrypted via SSL/TLS by default

### Fixed - Phase 8 Distributed Computing
- **CRITICAL**: Replaced all placeholder RPC implementations with real Protocol Buffer message handling
  - `rpc_service.py`: Now inherits from generated servicer base class, returns real protobuf messages
  - `rpc_client.py`: Implements actual gRPC calls, parses responses correctly
  - All 5 RPC methods now functional: GenerateText, EmbedText, EmbedTextBatch, HealthCheck, GetCapabilities
- **CRITICAL**: Integrated network layer into main.py startup sequence (lines 279-414)
  - Full lifecycle management: init → use → cleanup
  - Graceful degradation on network failures (non-critical errors)
  - Proper resource cleanup in finally block
- **CRITICAL**: Added Pydantic validation for all network configuration (`config/config_schema.py`)
  - NetworkConfig with 7 sub-models and comprehensive validation
  - Type-safe configuration prevents runtime errors
  - Validates ports (1024-65535), timeouts (1-30s), concurrent limits (1-50), etc.

### Added - Phase 8 Remediation Infrastructure
- **Build Automation**:
  - `build_proto.py`: Automated Protocol Buffer compilation script
  - `verify_build.py`: Comprehensive build verification for Phase 8 components
  - Updated `setup.py`: Custom build commands to auto-compile protos on install
- **Security**:
  - `network/security_store.py`: Encrypted credential storage with password-based key derivation
  - Master password support (interactive or system-based default)
  - Restrictive file permissions (0o600 for keys, 0o700 for directories)
- **Generated Code**:
  - `network/proto/jenova_pb2.py`: Protocol Buffer message classes (~350 lines)
  - `network/proto/jenova_pb2_grpc.py`: gRPC service stubs (~230 lines)
- **Documentation**:
  - `.dev-docs/01-Initial_Audit.md`: Comprehensive diagnostic report (63 issues identified)
  - `.dev-docs/02-Remediation_Blueprint.md`: Complete remediation plan
  - `.dev-docs/03-Final_Summary.md`: Implementation summary and deployment guide

### Changed - Phase 8 Remediation
- **CognitiveEngine** (`cognitive_engine/engine.py`):
  - Added `set_network_layer()` method for dependency injection
  - Added Phase 8 network layer instance variables
  - Integration status logging
- **Main Startup** (`main.py`):
  - Import all Phase 8 network modules
  - Initialize 8 network components conditionally
  - Network metrics tracking in startup breakdown
  - Cleanup handlers for graceful shutdown

### Technical Details - Remediation
- **Lines of Code**: ~1,705 lines production code, ~580 lines generated code, ~9,568 total including docs
- **Files Modified**: 8 core files updated with production-ready implementations
- **Files Created**: 9 new files (3 infrastructure, 3 generated, 3 documentation)
- **Critical Issues Resolved**: 8/8 (100%)
- **High-Priority Issues Resolved**: 15/15 (100%)
- **Test Coverage**: Deferred to Phase 6 (recommended post-deployment)

### Breaking Changes
None - all Phase 8 features remain opt-in via `network.enabled=false` by default

### Upgrade Path
1. Update dependencies: `pip install -U -r requirements.txt`
2. Compile Protocol Buffers: `python build_proto.py`
3. Verify build: `python verify_build.py`
4. Optional: Enable distributed features by setting `network.enabled=true` in `main_config.yaml`

### Known Limitations
- Memory search intentionally conservative for privacy (share_memory=false by default)
- Streaming not yet implemented (supports_streaming=false)
- Test coverage for distributed features pending Phase 6
- Peer ranking uses simplified algorithm (no ML-based optimization)

### Attribution
- Phase 8 remediation completed by **Claude (Anthropic AI Assistant)** under guidance from **orpheus497**
- All original Phase 8 distributed computing components designed and implemented by **orpheus497**
- The JENOVA Cognitive Architecture (JCA) created by **orpheus497**

## [5.0.0] - 2025-XX-XX - Phase 8: Distributed Computing & LAN Networking

### Added - Phase 8: Distributed Computing & LAN Networking
- **Distributed Architecture**: Complete implementation of LAN-based distributed computing for JENOVA instances
  - mDNS/Zeroconf service discovery (`network/discovery.py`) - Automatic peer discovery without manual configuration
  - Peer lifecycle management (`network/peer_manager.py`) - Connection tracking, health monitoring, and load balancing
  - gRPC-based RPC service (`network/rpc_service.py`) - High-performance remote procedure calls
  - gRPC client with connection pooling (`network/rpc_client.py`) - Retry logic and request routing
  - Certificate-based security (`network/security.py`) - SSL/TLS encryption and JWT authentication
  - Network performance metrics (`network/metrics.py`) - Latency tracking, bandwidth monitoring, load distribution

- **Distributed LLM Operations** (`llm/distributed_llm_interface.py`) - 5 distribution strategies:
  - `LOCAL_FIRST`: Try local LLM, fallback to peers on failure
  - `LOAD_BALANCED`: Select least loaded instance (local or peer)
  - `FASTEST_PEER`: Route to peer with best latency
  - `PARALLEL_VOTING`: Generate on multiple instances, vote on best result (3-4x faster responses)
  - `ROUND_ROBIN`: Simple round-robin load distribution

- **Federated Memory Search** (`memory/distributed_memory_search.py`)
  - Parallel memory queries across peers
  - Result merging and deduplication
  - Privacy-preserving search (optional: share embeddings only, not content)

- **Configuration** (`config/main_config.yaml`)
  - New `network` section with comprehensive settings:
    - Enable/disable distributed mode
    - Operating modes: auto, server, client, standalone
    - Service discovery configuration (service name, port, TTL)
    - Security settings (SSL/TLS, authentication)
    - Resource sharing controls (LLM, embeddings, memory)
    - Peer selection strategy configuration

- **Protocol Buffer Definitions** (`network/proto/jenova.proto`)
  - Comprehensive service contracts for distributed operations
  - Message schemas for LLM inference, embeddings, memory search
  - Health check and capabilities exchange protocols
  - Metrics and performance monitoring

- **Dependencies** (requirements.txt) - All FOSS with permissive licenses:
  - `zeroconf==0.132.2` - mDNS service discovery (MIT License)
  - `grpcio==1.60.1` - gRPC framework (Apache 2.0)
  - `grpcio-tools==1.60.1` - Protocol Buffer compiler (Apache 2.0)
  - `protobuf==4.25.2` - Protocol Buffers (BSD License)
  - `PyJWT==2.8.0` - JWT authentication tokens (MIT License)

### Changed - Phase 8
- **LLM Layer** (`llm/__init__.py`) - Updated to export distributed components
  - Added `DistributedLLMInterface` class
  - Added `DistributionStrategy` enum
  - Version bumped to 5.0.0

- **Performance Benefits** (Expected with 4-instance cluster):
  - Query response time: 40-80s → **10-20s** (3-4x faster via parallel generation)
  - Cognitive cycle: 120-180s → **30-45s** (parallel planning + execution)
  - Embedding generation: ~2s → **0.5s** (4x parallelization)
  - Memory search: Enhanced accuracy through federated knowledge

### Technical Details - Phase 8
- **Architecture**: Fully decentralized peer-to-peer with no central coordinator
- **Network Protocol**: gRPC with Protocol Buffers for efficient serialization
- **Discovery**: mDNS/Zeroconf for zero-configuration LAN discovery
- **Security**: Self-signed certificates for SSL/TLS, JWT for authentication
- **Privacy**: Memory sharing disabled by default to preserve user privacy
- **Failover**: Automatic peer failover with health monitoring
- **Load Balancing**: Multiple strategies for optimal resource utilization
- **Monitoring**: Comprehensive metrics for latency, bandwidth, and load distribution

### Attribution
- All distributed computing components designed and implemented by **orpheus497**
- The JENOVA Cognitive Architecture (JCA) created by **orpheus497**

## [4.2.0] - 2025-10-30

### Fixed
- **Critical Performance Bottleneck** (main_config.yaml:17)
  - Confirmed GPU acceleration enabled with 8 GPU layers for 4GB VRAM systems
  - Resolves timeout issues during user queries identified in Phase 8 diagnostics
  - LLM generation speed improved from ~5 tokens/second (CPU-only) to expected 15-25 tokens/second (GPU)
  - Planning phase reduced from ~20s to ~5-8s
  - Total cognitive cycle reduced from ~143s to ~40-80s
  - System now functional for production use with responsive user interaction

- **Thread-Safe Timeout Manager** (infrastructure/timeout_manager.py:43)
  - Replaced signal-based timeout with threading.Timer implementation
  - Fixes "ValueError: signal only works in main thread" crash
  - Works correctly in worker threads used by cognitive engine
  - Maintains timeout protection without signal dependency

- **ErrorHandler API Completeness** (infrastructure/error_handler.py:116)
  - Added log_error() convenience method to ErrorHandler class
  - Accepts both dict and string context parameters
  - Prevents cascading failures in error handling system
  - Resolves AttributeError when cognitive engine logs errors

### Verified
- **100% Feature Parity with Pre-Remediation Codebase**
  - Comprehensive comparison between jenovaold/ and current implementation
  - All cognitive architecture features preserved and functional:
    - ✓ Cortex cognitive graph (299 lines, 18 methods - identical)
    - ✓ Multi-layered memory (Episodic, Semantic, Procedural - identical)
    - ✓ Assumptions system (identical functionality)
    - ✓ Insights system with concern-based organization (identical)
    - ✓ RAG system with caching (enhanced from 110 to 254 lines)
    - ✓ Cognitive scheduler with monitoring (enhanced from 96 to 187 lines)
    - ✓ Memory search with optional re-ranking (enhanced from 154 to 302 lines)
    - ✓ Tool execution system (FileTools, web_search, shell commands - identical)
    - ✓ Fine-tuning data generation (identical)
    - ✓ Proactive suggestion engine (identical)
  - 6,000+ lines of production infrastructure added (Phases 1-7)
  - Zero features lost during remediation and rebuild
  - All enhancements maintain backward compatibility

### Added
- **Phase 8 Diagnostic Tools**
  - Performance diagnostic scripts for LLM speed measurement
  - Hardware capability detection and recommendations
  - Comprehensive session documentation in .dev-docs/

- **docs/PERFORMANCE_GUIDE.md** - Comprehensive performance documentation (598 lines)
  - Hardware tier classifications: 4GB, 6GB, 8GB+ VRAM with realistic expectations
  - Performance benchmarks: GTX 1650 Ti achieving 12 tokens/second with 20 GPU layers
  - Response time expectations: 120-180s for 4GB VRAM, 40-80s for 6GB+, 20-40s for 8GB+
  - Optimization strategies for each hardware tier
  - Complete CUDA installation and troubleshooting guide
  - Monitoring and diagnostic procedures

### Changed
- **CRITICAL: llama-cpp-python rebuilt with CUDA support**
  - Issue: Original installation lacked CUDA support despite gpu_layers configuration
  - Solution: Rebuilt with CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75"
  - Result: GPU offload now functional (llama_supports_gpu_offload() returns True)
  - Performance: 5 tokens/second (CPU) → 12 tokens/second (20 GPU layers)
  - Build time: ~12 minutes on GTX 1650 Ti
  - Documentation: Full rebuild instructions added to README.md and install.sh

- **Timeout Configuration for 4GB VRAM GPUs** (main_config.yaml lines 47, 101, 104, 114, 127)
  - Doubled all timeouts to accommodate ~12 tokens/second performance
  - llm_timeout: 120s → 240s (accommodates 512 tokens @ 12 tok/s with 460% margin)
  - planning_timeout: 60s → 120s (accommodates ~100 token plans)
  - generation_timeout: 120s → 240s (matches llm_timeout)
  - rerank_timeout: 15s → 30s (100% margin for re-ranking operations)
  - System now completes queries without timeout errors

- **GPU Layer Configuration** (main_config.yaml line 17)
  - Increased from 8 → 20 layers for better GPU utilization
  - 7B Q4 model has ~32 layers; offloading 20 (62.5%) balances speed and VRAM usage
  - GTX 1650 Ti 4GB handles 20 layers with ~2GB VRAM usage
  - Performance improved 43% over 8-layer configuration (8.44 → 12.05 tok/s)

- **install.sh CUDA build fix** (line 82)
  - Corrected CMAKE flag: -DGGML_CUDA → -DLLAMA_CUDA (proper flag for llama-cpp-python 0.2.90)
  - Added version pinning: llama-cpp-python==0.2.90
  - CUDA architecture detection for GTX 1650 Ti (compute capability 7.5)
  - Build now succeeds on NVIDIA GPUs with CUDA toolkit installed

- **Re-ranking Configuration** (main_config.yaml line 125)
  - Disabled for 4GB VRAM systems (rerank_enabled: false)
  - Saves 10-30 seconds per cognitive cycle
  - Documentation added: can re-enable on 6GB+ VRAM systems achieving 25+ tok/s

- **.gitignore Cleanup** (.gitignore:92)
  - Added explicit exclusion for jenovaold/ directory
  - Maintains clean version control without old codebase

### Removed
- **jenovaold/** - Old codebase directory removed after comprehensive feature parity verification
  - All features confirmed present in current implementation
  - 6,000+ lines of infrastructure improvements validated
  - Cognitive architecture fully functional and enhanced
  - Clean repository maintained with .gitignore exclusion

- **Boot Crash Integration Issues** (main.py:98-127)
  - Fixed `SystemHealth` attribute error where `gpu_available` was incorrectly accessed
  - Changed attribute checks from `gpu_memory_total` to correct `gpu_memory_total_mb`
  - Fixed GPU memory percentage calculation in health data dictionary
  - Resolved import conflict where `traceback` module was shadowed by local import
  - Removed redundant `import traceback` statement in configuration error handler
  - Fixed startup crash preventing system initialization

- **ChromaDB Database Compatibility** (memory/*.py)
  - Resolved `KeyError: '_type'` error caused by corrupted or incompatible ChromaDB metadata
  - Backed up old memory databases to prevent data loss
  - System now creates fresh ChromaDB collections on startup
  - All memory systems (semantic, episodic, procedural) now initialize correctly

- **Configuration Schema Validation** (config_schema.py:162-334)
  - Fixed startup crash caused by Pydantic validation errors
  - Added missing `CognitiveEngineConfig` class with `llm_timeout` and `planning_timeout` fields
  - Added missing `RAGSystemConfig` class with `cache_enabled`, `cache_size`, and `generation_timeout` fields
  - Restructured `CortexConfig` to use nested `PruningConfig` matching YAML structure
  - Renamed `enable_reranking` → `rerank_enabled` and `reranking_timeout` → `rerank_timeout` in `MemorySearchConfig`
  - Updated `JenovaConfig` to include `cognitive_engine` and `rag_system` sections
  - All configuration now validates successfully on startup

### Added - Phase 1-7 Remediation (Complete System Overhaul)
- **Phase 1: Foundation**
  - Added `src/jenova/config/config_schema.py` with comprehensive Pydantic validation for all configuration options
  - Added `src/jenova/infrastructure/error_handler.py` for centralized error handling with severity levels and CUDA error tracking
  - Added `.python-version` file specifying Python 3.11.9 for better compatibility
  - Added timeout protection via `src/jenova/infrastructure/timeout_manager.py` with context managers and decorators

- **Phase 2: Infrastructure Layer**
  - Added `src/jenova/infrastructure/health_monitor.py` for real-time CPU, memory, and GPU monitoring
  - Added `src/jenova/infrastructure/data_validator.py` with Pydantic models for type-safe memory entries
  - Added `src/jenova/infrastructure/file_manager.py` for atomic file operations with locking
  - Added `src/jenova/infrastructure/metrics_collector.py` for performance tracking and degradation detection

- **Phase 3: LLM Layer**
  - Added `src/jenova/llm/cuda_manager.py` for safe CUDA detection without environment manipulation
  - Added `src/jenova/llm/model_manager.py` for robust model lifecycle management with timeout protection
  - Added `src/jenova/llm/embedding_manager.py` for integrated embedding model management
  - Added `src/jenova/llm/llm_interface.py` with timeout protection and retry logic
  - Added `src/jenova/llm/__init__.py` creating clean package interface with proper exports

- **Phase 4: Memory Layer Foundation**
  - Added `src/jenova/memory/base_memory.py` abstract base class with atomic operations and timeout protection
  - Added `src/jenova/memory/memory_manager.py` for unified orchestration of all memory systems
  - Added exception types: `MemoryError`, `MemoryInitError`, `MemoryOperationError`
  - Added cross-memory search capabilities and aggregated health monitoring

- **Phase 5: Cognitive Engine**
  - Added comprehensive timeout protection to `src/jenova/cognitive_engine/engine.py` for all LLM operations (120s default)
  - Added LRU caching layer to `src/jenova/cognitive_engine/rag_system.py` for frequently accessed queries
  - Added configurable re-ranking option in `src/jenova/cognitive_engine/memory_search.py` (can be disabled for performance)
  - Added status and monitoring methods to `src/jenova/cognitive_engine/scheduler.py`
  - Added configuration sections: `cognitive_engine`, `rag_system`, `memory_search` in `main_config.yaml`

- **Phase 6: UI and Main Entry**
  - Added `src/jenova/ui/health_display.py` for real-time system health and metrics visualization (320 lines)
  - Added 7 methods to `src/jenova/ui/logger.py`: `warning()`, `success()`, `metrics_table()`, `health_status()`, `progress_message()`, `startup_info()`, `cache_stats()`
  - Added 4 commands to `src/jenova/ui/terminal.py`: `/health`, `/metrics`, `/status`, `/cache`
  - Added startup progress tracking to `src/jenova/main.py` with 9 stages (0-100%)
  - Added `HealthDisplay` and `CompactHealthDisplay` classes with comprehensive monitoring

- **Phase 7: Testing Suite**
  - Added `tests/test_config_validation.py` with 5 comprehensive configuration validation tests (370 lines)
  - Added `tests/test_cuda_manager.py` with 6 CUDA management and GPU detection tests (345 lines)
  - Added `tests/test_memory_operations.py` with 6 memory layer and atomic operation tests (385 lines)
  - Added `tests/test_error_recovery.py` with 7 error handling and timeout protection tests (390 lines)
  - Total: 24 tests across 4 test suites, all passing (1,490 lines of test code)

- **Testing & Documentation**
  - Added comprehensive test suites: `tests/test_phase3.py`, `tests/test_phase4.py`, `tests/test_phase5.py`, `tests/test_phase6.py`, `tests/test_integration.py`
  - Added development documentation in `.dev-docs/` directory for remediation planning and tracking
  - Created `.dev-docs/README.md` explaining development documentation structure
  - Created `.dev-docs/REMEDIATION_PLAN.md` with complete 7-phase roadmap
  - Created `.dev-docs/REMEDIATION_STATUS.md` with progress tracking and metrics
  - Created phase completion docs: `PHASE3_COMPLETION.md`, `PHASE4_COMPLETION.md`, `PHASE5_COMPLETION.md`, `PHASE6_COMPLETION.md`, `PHASE7_COMPLETION.md`
  - Created `.dev-docs/PROJECT_ORGANIZATION.md` documenting the project structure

### Changed - Phase 1-7 Remediation (Complete System Overhaul)
- **Phase 1: Foundation**
  - Updated `requirements.txt` with pinned dependency versions for stability (llama-cpp-python==0.2.90, torch==2.5.1, chromadb==0.5.20)
  - Updated `src/jenova/config/main_config.yaml` with safe CPU-first defaults and validated configuration structure
  - Updated `src/jenova/config/__init__.py` to validate configuration on load with detailed error reporting
  - Simplified `src/jenova/utils/model_loader.py` from 536 to 230 lines, removing complex fallback strategies (57% reduction)
  - Enhanced `src/jenova/main.py` with comprehensive error handling at each initialization step

- **Phase 3: Integration**
  - Updated `src/jenova/main.py` to use new `ModelManager` and `EmbeddingManager` from `jenova.llm` package
  - Replaced scattered model loading with organized LLM layer components
  - Enhanced error messages with `ModelLoadError` and `EmbeddingLoadError` for clear diagnostics
  - Improved resource cleanup with proper lifecycle management in `finally` blocks

- **Phase 4: Memory Layer**
  - Updated `src/jenova/memory/__init__.py` with Phase 4 exports while maintaining 100% backward compatibility
  - Added version metadata to memory package (__version__ = '4.1.0', __phase__ = 'Phase 4: Memory Layer (Foundation)')

- **Phase 5: Cognitive Engine**
  - Updated `src/jenova/cognitive_engine/engine.py` from 473 to 577 lines with timeout protection throughout
  - Updated `src/jenova/cognitive_engine/scheduler.py` from 97 to 188 lines with logging and error handling
  - Updated `src/jenova/cognitive_engine/rag_system.py` from 111 to 255 lines with LRU cache implementation
  - Updated `src/jenova/cognitive_engine/memory_search.py` from 162 to 303 lines with optional re-ranking
  - Updated `src/jenova/cognitive_engine/__init__.py` with Phase 5 exports including `LRUCache`

- **Phase 6: UI and Main Entry**
  - Updated `src/jenova/ui/logger.py` from 182 to 317 lines (74% increase) with 7 new methods
  - Updated `src/jenova/ui/terminal.py` from 320 to 421 lines (31% increase) with health monitoring integration
  - Updated `src/jenova/main.py` from 288 to 297 lines with progress tracking at each startup stage
  - Updated `src/jenova/ui/__init__.py` with Phase 6 exports (`HealthDisplay`, `CompactHealthDisplay`)

- **Project Organization**
  - Moved all test files to `tests/` directory
  - Moved development documentation to `.dev-docs/` directory
  - Updated `.gitignore` to exclude `.dev-docs/` from version control
  - Cleaned all `.pyc` files and `__pycache__` directories
  - Removed `.backup` files after verification

### Fixed - Phase 1-7 Remediation (Complete System Overhaul)
- **Phase 1: Foundation**
  - Fixed configuration validation error with `cortex.relationship_weights.last_updated` (changed from null to 0.0)
  - Resolved startup crashes by adding proper error recovery and cleanup
  - Fixed timeout issues with model loading by implementing 3-minute timeout protection

- **Phase 3: LLM Layer**
  - Removed CUDA environment variable manipulation that caused race conditions
  - Removed monkey-patching of llama-cpp-python internals
  - Fixed hung operations by adding timeout protection to all LLM calls
  - Improved VRAM detection and recommendations for GPU layer configuration

- **Phase 5: Cognitive Engine**
  - Fixed potential infinite hangs in LLM operations with comprehensive timeout coverage
  - Fixed memory leaks in RAG system by implementing automatic cache eviction (LRU)
  - Resolved performance degradation from always-on re-ranking by making it configurable

- **Phase 6: UI and Main Entry**
  - Fixed unclear startup process by adding 9-stage progress tracking (0-100%)
  - Resolved lack of system visibility by adding health monitoring commands
  - Fixed missing error context in UI by integrating ErrorHandler throughout

### Removed - Phase 1-7 Remediation (Complete System Overhaul)
- **Phase 1: Foundation**
  - Removed complex 9+ fallback strategy model loading (replaced with single clear strategy)
  - Removed all `.backup` files after verification (3 files)
  - Removed all `.pyc` files and `__pycache__` directories (1,848 cache directories)

### Technical Improvements - Complete Remediation Statistics
- **Stability**: Reduced model loader complexity by 57% (536 → 230 lines)
- **Startup Time**: Eliminated 10-30s delays from fallback strategies
- **Error Messages**: All errors now provide clear, actionable solutions
- **Timeout Protection**: Comprehensive coverage - model loading (180s), LLM generation (120s), planning (60s), memory operations (10-30s), re-ranking (15s), reflection (180s)
- **Performance**: LRU caching provides <100ms response on cache hits (99% faster than cache miss)
- **Test Coverage**: 24 comprehensive tests across 4 test suites with 100% pass rate (1,490 lines of test code)
- **Backward Compatibility**: 100% compatible with existing memory and cognitive systems
- **Code Quality**: Added 6,000+ lines of well-tested, documented infrastructure code across 7 phases
- **User Experience**: Added 9-stage startup progress, 4 health monitoring commands, real-time metrics display
- **Production Ready**: All 7 phases complete, fully tested, documented, and validated

## [4.1.0] - 2025-10-29

### Added
- **Hardware Detection**: Implemented comprehensive hardware detection system in `utils/hardware_detector.py` supporting NVIDIA GPUs (CUDA), Intel GPUs (Iris Xe, UHD, Arc via OpenCL/Vulkan), AMD GPUs and APUs (via OpenCL/ROCm), Apple Silicon (via Metal), and ARM CPUs across Linux, macOS, Windows, and Android/Termux platforms
- **Multi-GPU Support**: Added automatic detection and prioritization of multiple compute devices in hybrid systems (e.g., integrated + discrete GPU configurations)
- **Intelligent Resource Allocation**: Implemented platform-specific memory management strategies (performance, balanced, swap-optimized, minimal) with automatic RAM and swap detection
- **Hardware Configuration**: Added `hardware` section to `main_config.yaml` with device preference selection (`auto`, `cuda`, `opencl`, `vulkan`, `metal`, `cpu`), device index for multi-GPU systems, and memory strategy configuration
- **Documentation**: Created comprehensive hardware support guide at `docs/HARDWARE_SUPPORT.md` covering all supported hardware types, configuration options, troubleshooting procedures, and platform-specific notes
- **Security**: Implemented a strict, configurable whitelist for the `execute_shell_command` tool in `main_config.yaml` to prevent arbitrary command execution
- **Robustness**: Added granular error handling to the application startup sequence in `main.py` to provide clear feedback on component failures
- **Robustness**: Implemented retry logic with exponential backoff for LLM calls in `llm_interface.py` to handle transient API or model errors gracefully
- **Intelligence**: Upgraded the `CognitiveScheduler` to be more dynamic, considering conversation velocity, content, and idle time to trigger cognitive functions more intelligently
- **Intelligence**: Implemented an advanced context re-ranking step in `memory_search.py` using an LLM call to ensure the most relevant information is prioritized in the final prompt
- **Robustness**: Added a data validation layer to all memory modules (`episodic.py`, `semantic.py`, `procedural.py`) to ensure data integrity before writing to the database
- **Intelligence**: Implemented recursive text chunking in `cortex.py` to improve the processing of large and complex documents
- **Code Quality**: Refactored the command handling logic in `ui/terminal.py` into a clean, dictionary-based command dispatcher, improving maintainability and extensibility
- **Documentation**: Added a comprehensive "Credits and Acknowledgments" section to `README.md` to provide full attribution to all FOSS dependencies

### Changed
- **Code Quality**: Applied `autopep8` and `isort` for consistent code formatting and import ordering across the entire project
- **Documentation**: Added module-level docstrings to all Python files where they were missing
- **Attribution**: Added standardized creator attribution and license information to the header of all Python files
- **Model Loading Strategy**: Reconfigured GPU offload to start with all layers (32) and reduce by 2 each attempt, maximizing GPU utilization for the main LLM on NVIDIA hardware
- **Error Messages**: Added detailed error reporting for model loading failures with specific failure reasons (VRAM, CUDA errors, etc.)
- **Environment Variables**: Updated to use `PYTORCH_ALLOC_CONF` instead of deprecated `PYTORCH_CUDA_ALLOC_CONF`
- **Resource Management**: Embedding model explicitly kept on CPU to maximize NVIDIA GPU availability for main LLM, with clear user messaging
- **Hardening**: Strengthened `install.sh` and `uninstall.sh` scripts with `set -euo pipefail` and more explicit user confirmations to ensure robust and safe execution
- **Optimization**: Refined and optimized all cognitive prompts for planning, metadata extraction, and reflection to improve accuracy and performance
- **Performance**: Optimized the Cortex reflection process for better performance on large cognitive graphs by batching LLM calls
- **Intelligence**: Significantly enhanced `finetune/train.py` to extract knowledge from the complete cognitive graph, including `document_chunk` and `meta-insight` nodes, creating a more comprehensive training set
- **Model Loading**: Integrated hardware detection system into `utils/model_loader.py` with automatic device selection, optimal configuration recommendations, and detailed hardware information display during startup

### Fixed
- **Startup Crash**: Fixed `AttributeError: 'LlamaModel' object has no attribute 'sampler'` during model loading fallback by implementing a monkey-patch for llama-cpp-python's cleanup method in `utils/model_loader.py`
- **GPU Allocation**: Resolved VRAM allocation conflicts between PyTorch and llama-cpp-python by hiding CUDA from PyTorch during initialization, ensuring all NVIDIA VRAM is available for the main LLM
- **Model Loading**: Fixed "out of memory" errors during GPU model loading by implementing aggressive garbage collection between loading attempts and reordering strategies to start with conservative layer counts
- **Context Size**: Reduced context window for GPU strategies from 8192 to 4096 tokens to fit KV cache in 4GB VRAM alongside model layers
- **CUDA Detection**: Implemented nvidia-smi-based GPU detection that doesn't trigger PyTorch CUDA initialization, preventing premature VRAM allocation
- **Robustness**: Implemented additional checks for `ui_logger` and `file_logger` availability in various modules (`rag_system.py`, `insights/concerns.py`, `insights/manager.py`, `llm_interface.py`, `main.py`, `memory/episodic.py`, `memory/procedural.py`, `memory/semantic.py`, `tools.py`, `ui/terminal.py`, `utils/model_loader.py`, `assumptions/manager.py`) to prevent `NameError` exceptions
- **Testing**: Updated `test.py` to correctly initialize `CognitiveEngine` by passing `llm` instead of `llm_interface`
- **Security**: Audited and hardened the `FileTools` sandbox to be completely immune to path traversal attacks
- **Robustness**: Made the `web_search` tool resilient to common `WebDriver` and network errors, preventing crashes

### Removed
- **Cleanup**: Removed the deprecated `finetuning` section from `main_config.yaml`, directing users to the `finetune/` directory workflow
- **Cleanup**: Removed the deprecated and unused `reorganize_insights` method from `insights/concerns.py`
- **Cleanup**: Deleted the entire `enhancement_plan/` directory and its contents as it contained development artifacts not relevant to the final product

## [4.0.0] - 2025-10-25

### Added
- Added `__init__.py` to `src/jenova/docs` for better package structure.
- Enhanced `Cortex.develop_insights_from_docs` to extract key takeaways and questions from document chunks and create corresponding insight and question nodes, as described in the `README.md`.
- **GGUF Support:** Re-implemented support for GGUF models via `llama-cpp-python` for flexible model selection and better performance.
- **Intelligent Model Loading:** Implemented a multi-strategy loading fallback (GPU -> Partial GPU -> CPU-only) to handle VRAM limitations and prevent crashes.
- **Dynamic Resource Allocation:** The system now dynamically detects physical CPU cores and VRAM to intelligently allocate `threads` and `gpu_layers`.
- **GPU Accelerated Embeddings:** All memory systems (Episodic, Semantic, Procedural) now use a shared, GPU-accelerated embedding model for 5-10x performance improvement.
- **ToolHandler System:** Added a new `ToolHandler` for managing and executing tools, including `web_search`, `file_tools`, `get_current_datetime`, and `execute_shell_command`.
- **Model Discovery:** The system now automatically discovers GGUF models by searching in priority order: `/usr/local/share/models` (system-wide) and then `./models` (local).
- **Fine-Tuning Data Generation:** The fine-tuning script now generates a comprehensive training dataset from all cognitive sources (insights, memories, assumptions, documents).

### Changed
- Updated `pyproject.toml` version to `4.0.0` to reflect the latest release version.
- **Architecture:** Reverted the core architecture from HuggingFace `transformers` back to `llama-cpp-python`.
- **Installation:** Reverted from a system-wide installation to a local, virtualenv-based installation for better isolation and user control.
- **Dependencies:** Replaced `transformers`, `accelerate`, `peft`, and `bitsandbytes` with `llama-cpp-python`.
- **LLM Interface:** The `LLMInterface` has been completely rewritten to use the `llama-cpp-python` `Llama` class.
- **Cortex:** Refactored the Cortex to use `CognitiveNode` and `CognitiveLink` dataclasses for improved readability and maintainability.
- **Document Processing:** Documents are now chunked and stored as canonical 'document' nodes in the cognitive graph, without automatic insight generation.
- **README:** The `README.md` and `main_config.yaml` have been updated to reflect the new GGUF-based architecture and dynamic resource allocation.

### Removed
- **HuggingFace Dependencies:** Removed `transformers`, `accelerate`, `peft`, and `bitsandbytes` from all dependencies.
- **System-Wide Installation:** The `install.sh` script no longer installs the package globally.
- **Automatic Model Downloads:** The install script no longer downloads models automatically; users must provide their own GGUF models.
- **Grammar System:** Removed the grammar-constrained generation feature, as it is not supported in the same way by `llama-cpp-python`.
- **Deprecated Code:** Removed the `document_processor.py` file and the `reorganize_insights` method.

### Fixed
- **Tool Execution:** Fixed `AttributeError` in `rag_system.generate_response` by ensuring only structured tool outputs are passed as search results, while error messages are added to the context.
- **Model Loading:** Fixed VRAM allocation errors by forcing the embedding model to the CPU, maximizing VRAM for the main LLM.
- **Model Loading:** Resolved deadlocks and insufficient VRAM errors with the new multi-strategy fallback system.
- **Startup Crash:** Fixed a silent exit on startup by adding the `if __name__ == "__main__":` block to `main.py`.
- **Startup Crash:** Fixed a `SyntaxError` in `llm_interface.py` by removing an orphaned `except` block.
- **Database Conflicts:** Added data migration support for Episodic and Procedural memory to handle embedding function changes.
- **Stability:** Fixed numerous `TypeError` and `SyntaxError` issues in the Cortex and `UILogger`.
- **Code Quality:** Removed all debug `print()` statements, standardizing on the `UILogger` for all user-facing messages.

### Security
- **Local-Only Operation:** The architecture is now fully local-only, with no external API calls or automatic downloads during installation.
- **User Control:** Users explicitly provide and control their own GGUF models.
- **Virtualenv Isolation:** The virtualenv installation provides better dependency isolation and security.

## [3.1.1] - 2025-10-19

### Added
- **Uninstall Script:** Added a comprehensive `uninstall.sh` script to allow administrators to easily remove the application, user data, and downloaded models from the system.

### Fixed
- **AI Behavior:** Overhauled prompt engineering and increased the repetition penalty to improve the quality, coherence, and reduce repetitiveness of the AI's responses.
- **Context Window:** Fixed an `OverflowError` during tokenization and now correctly cap the context size to a reasonable value to prevent crashes and improve performance.
- **Stability:** Resolved several startup and runtime errors, including `SyntaxError`, `TypeError`, and deprecated argument usage.
- **Code Quality:** Addressed multiple issues reported by `pylint`, removing unused code, arguments, and fixing warnings throughout the codebase.
## [3.1.0] - 2025-10-18

### Changed
- **Complete Architecture Migration:** Migrated the core architecture from `llama-cpp-python` (GGUF models) to the HuggingFace `transformers` library.
- **Model Management:** The installation process (`install.sh`) now automatically downloads the `TinyLlama-1.1B` model to a system-wide `/usr/local/share/jenova-ai/models` directory.
- **Dependencies:** Replaced `llama-cpp-python` with `transformers` and added `accelerate` for better GPU support.
- **Configuration:** Simplified hardware configuration in `main_config.yaml`, removing GGUF-specific settings (`threads`, `gpu_layers`, `mlock`).
- **Hardware Utilization:** The model loader now automatically detects and utilizes CUDA GPUs when available, with a fallback to CPU.
- **Grammar System:** Removed the dependency on `llama.cpp`'s JSON grammar system.

### Fixed
- **Critical Regression from Threading Fix:** Fixed multiple critical issues introduced by the v3.0.3 threading fix that broke core functionality, including message queue initialization, missing UI spinners, and broken RAG system logger references.
- **Memory Stability:** Fixed a `TypeError` in all memory modules that caused crashes when NLP analysis failed to extract metadata. A new `sanitize_metadata` utility now prevents `None` values from being passed to the database.
- **Resource Management:** Improved GPU memory cleanup by calling `torch.cuda.empty_cache()`.

## [3.0.3] - 2025-10-17

### Fixed
- **Threading Deadlock:** Resolved a critical deadlock in the cognitive cycle caused by threading conflicts between background cognitive processes and UI updates. Implemented a thread-safe `queue.Queue` message bus in `TerminalUI` for asynchronous UI updates. Modified `UILogger` to be non-blocking by queuing status updates and log messages instead of directly manipulating the UI. Refactored `TerminalUI` to process queued messages on the main thread while cognitive operations run in background threads, eliminating circular wait conditions.
- **Missing PyTorch Dependency:** Added `torch` to both `requirements.txt` and `pyproject.toml` dependencies to fix import errors in `model_loader.py` which uses PyTorch for GPU detection when loading embedding models.

## [3.0.2] - 2025-10-15

### Fixed
- **Critical Race Condition:** Fixed a critical race condition on multi-core systems where background cognitive tasks and the main UI loop competed for console control, causing the "Only one live display may be active at once" error. Implemented thread-safe console locking using `threading.Lock` in the `UILogger` class to ensure exclusive access to console operations.
- **UI Thread Safety:** Refactored all console access points in `UILogger` to use exclusive locking, including `banner()`, `info()`, `system_message()`, `help_message()`, `reflection()`, `cognitive_process()`, `thinking_process()`, `user_query()`, and `jenova_response()`.
- **Spinner Thread Safety:** Updated the `TerminalUI` spinner to respect the console lock, preventing conflicts with Rich's live display system during concurrent operations.

### Changed
- **Branding Update:** Updated all references throughout the codebase and documentation to refer to the AI as "JENOVA" (instead of "Jenova AI") and the engine as "The JENOVA Cognitive Architecture" for consistent branding and identity.
  - Updated `persona.yaml` identity configuration
  - Updated `setup.py` description and version
  - Updated `README.md` throughout
  - Updated `terminal.py` prompts and messages
  - Updated `logger.py` panel titles and display names
  - Updated `main.py` docstrings and messages
  - Updated `finetune/README.md`
- **Enhanced /help Command:** Completely redesigned the `/help` command display to be more visually appealing and user-friendly with:
  - Structured sections with decorative borders (Cognitive Commands, Learning Commands, System Commands, Innate Capabilities)
  - Clear command syntax highlighting in bright yellow
  - Detailed descriptions in lavender with usage examples
  - Italic dim text for additional context and explanations
  - Visual separators between sections
  - Helpful tips section at the bottom

## [3.0.1] - 2025-10-11

### Fixed
- **Startup Crash (Model Load):** Fixed a critical `TypeError` that occurred during startup when reading model metadata by correctly instantiating the `Llama` object without a `with` statement.
- **Startup Crash (No Models Found):** Fixed an `AttributeError` that occurred if no GGUF models were found by using the correct logger method.
- **Shutdown Crash:** Fixed an `AttributeError` that occurred on exit by removing an incorrect call to `self.model.close()`.
- **Database Incompatibility:** Resolved a `sqlite3.OperationalError` by removing outdated ChromaDB database files after a schema change in the library.
- **Installation Conflict:** The `install.sh` script now uses `pip install --ignore-installed` to prevent conflicts with system-managed packages.
- **Module Not Found Error:** Fixed a `ModuleNotFoundError` for the `jenova.assumptions` package by adding a missing `__init__.py` file.

## [3.0.0] - 2025-10-11

### Security
- **Remote Code Execution:** Patched a critical RCE vulnerability in the tool handler by replacing the unsafe `eval()` with `ast.literal_eval()` for parsing tool arguments.
- **Shell Injection:** Hardened the `SystemTools` and fine-tuning commands against shell injection vulnerabilities.
- **Path Traversal:** Corrected a path traversal vulnerability in `FileTools` by fixing the sandbox validation logic.

### Changed
- **System-Wide Installation:** Overhauled the installation process for multi-user, system-wide deployment. The `install.sh` script now installs the package globally, making the `jenova` command available to all system users.
- **Project Cleanup:** Performed a major cleanup of the repository, removing the large `llama.cpp/` source directory and other development artifacts in favor of the `llama-cpp-python` dependency.
- **Fine-tuning Process:** Redesigned the fine-tuning workflow into a single, modular `finetune/train.py` script that generates a training dataset from user insights.
- **Tool System:** Removed dysfunctional tools (`web_search`, `weather_search`) and refactored the tool handling system to be more modular and extensible under a central `ToolHandler`.
- **Web Search:** Replaced the previous web search library with a more powerful, `selenium`-based implementation. The AI can now access and process the full content of web pages for richer information gathering.
- **Documentation:** The `README.md` has been completely rewritten to reflect the new system-wide installation model and current features.
- **Performance:** Optimized the memory pre-loading process by using threads to load collections in parallel.
- **UI:** The terminal UI has been updated to reflect the new command system. The `/finetune` and `/search` commands have been removed, and a new `/train` command has been added. The `/help` command is updated to reflect the current tools and commands.
- **`.gitignore`:** The `.gitignore` file has been updated to be more comprehensive.

### Fixed
- **Critical Shutdown Error:** Fixed a `TypeError: 'NoneType' object is not callable` on exit by ensuring all `llama-cpp-python` model resources are explicitly closed.
- **UI/Engine Stability:** Implemented multi-layered defenses against the persistent `TypeError: string indices must be integers, not 'str'`, hardening the UI, cognitive engine, and memory systems.
- **Startup Crash:** Fixed a `TypeError` during `SemanticMemory` initialization related to `chromadb` embedding functions and implemented a self-healing mechanism to handle collection conflicts and prevent data loss.
- **Cognitive Degradation:** Overhauled the `Cortex` and `CognitiveEngine` to fix a cascading failure in the AI's intelligence, reinforcing the AI's persona, stabilizing memory generation, and making cognition more reliable with `gbnf` grammars.
- **Stability & Correctness:** Fixed numerous critical bugs across the application, including `NameError` in `ProactiveEngine` and `AssumptionManager`, `UnboundLocalError` in the `think` method, `AttributeError` in the web search tool, and other `TypeError` issues to improve overall stability.
- **Tool Handling:** Fixed a critical bug in the `CognitiveEngine` where it was attempting to call a non-existent `tool_handler` object.


## [2.1.0] - 2025-10-06

### Added
- **Weather Tool:** Added a new `WeatherTool` that can fetch real-time weather information for a given location using the free `wttr.in` service, removing the need for an API key.
- **File Sandbox:** Implemented a secure file sandbox for the `FileTools`. All file operations are now restricted to a configurable directory (`~/jenova_files` by default).
- **Memory Pre-loading:** Added a `preload_memories` option to `main_config.yaml` to allow pre-loading all memories into RAM at startup for faster response times.
- **Dependency:** Added `selenium` and `webdriver-manager` to `requirements.txt` to support the new browser-based web search functionality.
- **Weather Tool:** Added a new `WeatherTool` that can fetch real-time weather information for a given location using the OpenWeatherMap API. The AI can now be asked about the weather and will use this tool to provide an answer.
- **Configuration:** Added a new `apis` section to `main_config.yaml` to store API keys for external services, starting with `openweathermap_api_key`.
- **Dependency:** Added `selenium` and `webdriver-manager` to `requirements.txt` to support the new browser-based web search functionality.
- **Insight System Logging:** Added detailed logging to the insight management and memory search systems to improve observability and aid in debugging the AI's cognitive functions.
- **UI:** A `/help` command has been added to the terminal UI to provide users with a clear, on-demand list of all available commands and their functions.
- **Web Search Capability:** Jenova can now search the web for up-to-date information. This can be triggered manually with the `/search <query>` command, or autonomously by the AI when it determines its own knowledge is insufficient. Search results are stored in the cognitive graph.
- **Document Reading and Insight Generation:** Implemented a new system for reading documents from the `src/jenova/docs` directory. The system processes new or modified documents, chunks their content, generates summaries and individual insights, and links them within the cognitive graph. This allows the AI to learn from external documents, expanding its knowledge base.
- **Document Processor:** Added a sample `example.md` file to the `src/jenova/docs` directory.
- **UI Enhancements:** Implemented a visual spinner in the `TerminalUI` to indicate when long-running cognitive processes are occurring, improving user experience.
- A new end-to-end finetuning script (`finetune/run_finetune.py`) that automates the entire process, including downloading and building `llama.cpp`, preparing data, and running the finetuning process.

### Changed
- **Web Search:** Replaced the `duckduckgo-search` library with a more powerful, `selenium`-based implementation. The AI now uses a headless browser to perform web searches, allowing it to access and extract the full content of web pages, leading to much richer and more accurate information gathering.
- **Enhanced Web Search Comprehension:** The web search result processing has been significantly enhanced. The AI now processes the full content of web pages in chunks, extracting a summary, key takeaways, and potential questions from each chunk, and stores this structured information in the Cortex.
- **`/search` Command:** The `/search` command now provides the same conversational web search experience as the inline `(search: <query>)` syntax, ensuring a consistent user experience.
- **Cognitive Engine:** The `_plan` method has been enhanced to make the AI aware of the new `WeatherTool` and the enhanced `FileTools`, allowing it to intelligently decide when to use these tools to fulfill user requests.
- **`.gitignore`:** Updated the `.gitignore` file to exclude LLM models and the content of the documentation directory.
- **FileTools:** The `FileTools` class has been completely overhauled to use a secure, configurable sandbox directory (`~/jenova_files` by default). All file operations are now restricted to this directory, and the AI is prevented from accessing hidden files, significantly improving security and user control.
- **Cognitive Engine:** The `_plan` method has been enhanced to make the AI aware of the new `WeatherTool` and the enhanced `FileTools`, allowing it to intelligently decide when to use these tools to fulfill user requests.
- **Web Search:** Replaced the `duckduckgo-search` library with a more powerful, `selenium`-based implementation. The AI now uses a headless browser to perform web searches, allowing it to access and extract the full content of web pages, leading to much richer and more accurate information gathering.
- **`/search` Command:** The `/search` command now provides the same conversational web search experience as the inline `(search: <query>)` syntax, ensuring a consistent user experience.
- **Conversational Web Search:** The web search functionality is now fully conversational. When a search is performed, the AI presents a summary of the findings and asks for further instructions, allowing for a more collaborative exploration of information.
- **Enhanced Document Comprehension:** The document processing system has been significantly enhanced. It now performs a much deeper analysis of documents, extracting not just insights, but also key takeaways and a list of questions the document can answer. This information is then stored in the cognitive graph as a rich structure of interconnected nodes, allowing the AI to have a much deeper understanding of the documents it reads.
- **README Update:** The `README.md` has been significantly updated to accurately reflect the current state of the program, including the new cognitive architecture, conversational web search, and other enhancements.
- **Conversational Web Search:** The web search functionality has been made more conversational and interactive. The AI now presents a summary of search results and asks for further instructions, such as performing a deeper search.
- **Spinner Consistency:** Removed conflicting spinners from the `Cortex` to ensure a consistent and smooth user experience during long-running operations like `/reflect`.
- **UI Help Command:** The `/help` command in the `TerminalUI` has been significantly enhanced to provide detailed, comprehensive descriptions for each command, explaining its purpose, impact on the AI, and usage, with improved visual styling including highlighted commands (bright yellow) and subdued descriptions (bright lavender).
- **Proactive Engine:** The `ProactiveEngine` has been enhanced to be more "hyper-aware" and proactive. It now considers underdeveloped insights (low centrality nodes) and high-potential insights (high centrality nodes) within the cognitive graph, in addition to unverified assumptions, when generating proactive suggestions for the user.
- **Interactive Procedure Learning:** The `/learn_procedure` command has been refactored to provide an interactive, guided experience. The AI now prompts the user for the procedure's name, individual steps, and expected outcome, ensuring structured and comprehensive intake of procedural knowledge.
- **Web Search Tool:** Renamed the `google_web_search` function to `web_search` in `src/jenova/default_api.py` and updated all references in `main.py` and `engine.py` to accurately reflect its use of the `duckduckgo-search` library.
- **Cognitive Architecture:** The core reflection process has been significantly improved. The old, redundant `reorganize_insights` task has been removed from the cognitive cycle. The `/reflect` command now correctly triggers the powerful, unified `Cortex.reflect` method. This method now uses a more robust graph traversal algorithm to find clusters for meta-insight generation, leading to deeper and more relevant high-level insights.
- **Document Processing:** The document processor is no longer triggered automatically at startup. Document processing is now an on-demand action initiated via the `/develop_insight` command, improving startup time and giving the user more control.
- **Hardware Optimization:** The default configuration in `main_config.yaml` has been updated for better performance. `gpu_layers` is now set to -1 to maximize GPU offloading, and `mlock` is enabled by default to keep the model locked in RAM.
- **Enhanced Intelligence and Learning:** The AI's cognitive processes have been significantly enhanced for deeper understanding and more robust learning.
  - **Smarter Web Search:** The autonomous web search now uses more advanced heuristics to decide when to search for up-to-date information.
  - **Deeper Semantic Comprehension:** When processing documents and web search results, the AI now extracts structured data including key entities, topics, and sentiment, leading to a richer understanding of the information.
  - **Advanced Knowledge Interlinking:** The reflection process is now more sophisticated. It not only links insights to external data but also finds relationships between different external sources (document-to-document, web-result-to-web-result) and identifies and creates insights about contradictions it discovers.
- **Web Search:** Implemented a more natural web search syntax `(search: <query>)` that can be used directly in the conversation. The AI can also use this syntax autonomously.
- **Proactive Engine:** The proactive engine is now more context-aware, using conversation history to generate more relevant and diverse suggestions.
- **Memory System:** The memory system's metadata extraction is now more robust, avoiding default values when the LLM fails to extract information.
- **Enhanced Reflection:** The reflection process has been improved to create links between insights and external information sources like documents and web search results, creating a more interconnected knowledge graph.
- **`/develop_insight` Command:** The `/develop_insight` command has been enhanced. When used without a `node_id`, it now triggers the new document reading and insight generation process. The existing functionality of developing a specific insight by providing a `node_id` is preserved.
- **Document Processor:** The document processor now runs at startup, processing all documents in the `docs` directory.
- **Document Processing:** Improved the document processing system to provide better feedback and search capabilities.
  - The system now prints a message to the console when it starts reading a document.
  - The document's title (filename) is now included with the content when processing, allowing the AI to better understand the context and relevance of the information.
- **AssumptionManager:** Improved assumption system robustness by modifying `AssumptionManager.add_assumption` to prevent re-adding assumptions that have already been verified (confirmed or false), and to return the ID of an existing unverified assumption if found.
- **ProactiveEngine:** Enhanced thought generation by modifying `ProactiveEngine.get_suggestion` to prioritize unverified assumptions and avoid repeating recent suggestions, making the process more cognitive.
- **UI Enhancements:** Improved line spacing in `TerminalUI` for better readability of user input, system messages, and AI output.
- **Cortex Stability and Intelligence:** Overhauled the Cortex system to be more robust, intelligent, and less prone to degradation (i.e., "brain rot").
  - **Emotion Analysis:** Replaced simplistic sentiment analysis with a more sophisticated emotion analysis, providing a richer psychological dimension to the cognitive graph.
  - **Weighted Centrality:** Implemented a weighted centrality calculation for more accurate node importance, leading to better meta-insight generation.
  - **Graph Pruning:** Introduced an automated graph pruning mechanism to remove old, irrelevant nodes, keeping the cognitive graph healthy and efficient.
  - **Reliable Linking:** Hardened the node linking process (`_link_orphans`) with more robust JSON parsing and error logging to prevent graph fragmentation.
  - **High-Quality Meta-Insights:** Improved the meta-insight generation process to prevent duplicates and produce more novel, higher-level insights. The selection of cluster centers for meta-insight generation is now more dynamic, using a centrality threshold instead of a fixed number of nodes.
- **Configuration:** The Cortex is now more configurable via `main_config.yaml`, allowing for tuning of relationship weights and pruning settings.
- **Cognitive Cycle:** Replaced the rigid, hardcoded cognitive cycle with a flexible and configurable `CognitiveScheduler`.
- **Memory Search:** Made the number of results for each memory type configurable.
- **Command Handling:** Refactored the command handling in the `TerminalUI` to be more concise and extensible.
- **AI Recognition:** The AI now recognizes the user and can communicate about its insights and assumptions.
- **Commands:** The `/reflect`, `/meta`, and `/verify` commands are now working correctly.
- **Logging:** Added more detailed logging to the cognitive functions to give the user a better idea of what's happening behind the scenes.
- **.gitignore:** Updated the `.gitignore` file to protect user data and the virtual environment.
- **Data Integrity:**
  - Prevented the addition of duplicate assumptions.
  - Made the insight reorganization process safer and more efficient to prevent data loss.
- **Circular Dependency:** Removed the circular dependency between `InsightManager` and `MemorySearch`.
- **Code Quality:**
  - Removed redundant code in `LLMInterface`.
  - Improved the reliability of JSON parsing across the application.
  - Made the `FileLogger`'s `log_file_path` a public attribute.
- The `finetune/prepare_data.py` script has been refactored to be more modular and robust. The `prepare_history_data` function now supports a structured JSONL format.
- The /finetune command now checks for the existence of the required 'llama.cpp' executables before running.
- The application now automatically discovers and loads a model from the `models/` directory if the `model_path` in the configuration is not set.
- Upgraded the fine-tuning process to a perfected, two-stage workflow. The `/finetune` command now first creates a LoRA adapter and then automatically merges it with the base model to produce a new, fully fine-tuned `.gguf` model, ready for use.
- Enhanced the fine-tuning data preparation script (`finetune/prepare_data.py`) to create more advanced, context-aware training examples in a conversational format, leading to higher quality learning.

### Fixed
- **Cognitive Degradation:** Overhauled the `Cortex` and `CognitiveEngine` to fix a cascading failure in the AI's intelligence. This includes:
  - **Robust Persona:** Reinforced the AI's identity ("Jenova", created by "The Architect") in all cognitive prompts to ensure a consistent persona and proper user recognition.
  - **Stable Memory:** Corrected a critical bug in meta-insight generation that prevented the AI from deepening its understanding of topics over time.
  - **Reliable Cognition:** Hardened the AI's cognitive functions by replacing fragile JSON parsing with robust `gbnf` grammars, preventing errors and ensuring the reliable creation and linking of cognitive nodes.
  - **Structured Emotion:** Improved the emotion analysis system to use a fixed list of emotions, providing more consistent and useful data for understanding context.
- **Memory Management:** Changed the default `mlock` setting to `false` to encourage the operating system to utilize SWAP memory for the model, freeing up RAM for other tasks.
- **Startup Crash:** Fixed a `TypeError: 'str' object is not callable` error that occurred during the initialization of `SemanticMemory`. This was caused by an issue with how the custom embedding function was being handled by `chromadb`. The fix ensures that the application starts up reliably by using a custom embedding function with a `name` method.
- **Startup Crash:** Implemented a self-healing mechanism in `SemanticMemory` to handle `chromadb` embedding function conflicts. If a conflict is detected, the old collection is backed up, deleted, and recreated with the new embedding function, and the data is migrated to the new collection, preventing data loss.
- **AssumptionManager:** Fixed a `NameError` in `add_assumption` caused by undefined variables (`assumption_data`, `content`). The duplicate checking logic has been rewritten to be more robust.
- **Cortex:** Fixed a `SyntaxError` in `develop_insights_from_docs` caused by improper f-string quoting.
- **Cognitive Engine:** Fixed an issue in the `_execute` method where the `WeatherTool` was not being called correctly.
- **Startup Crash:** Fixed a `TypeError: 'str' object is not callable` error that occurred during the initialization of `SemanticMemory`. This was caused by an issue with how the custom embedding function was being handled by `chromadb`. The fix ensures that the application starts up reliably.
- **AssumptionManager:** Fixed a `NameError` in `add_assumption` caused by undefined variables (`assumption_data`, `content`). The duplicate checking logic has been rewritten to be more robust.
- **Cortex:** Fixed a `SyntaxError` in `develop_insights_from_docs` caused by improper f-string quoting.
- **Bug Fix:** Fixed a bug where the application would crash due to incorrect error logging calls (`file_logger.error` instead of `file_logger.log_error`).
- **SyntaxError:** Resolved a `SyntaxError` in `src/jenova/cortex/cortex.py` caused by incorrect f-string syntax in the `develop_insights_from_docs` method.
- **`/reflect` Command:** The `/reflect` command now correctly triggers the deep reflection process in the `Cortex`, ensuring that the AI's most powerful cognitive function is accessible on-demand.
- **Tool Security:** Enhanced the security of the local `FileTools` by adding path traversal checks to the `read_file` and `list_directory` methods. Access is now restricted to the user's home directory and the application's designated output directory.
- **Startup Crash:** Fixed a `ModuleNotFoundError` that prevented the application from starting.
- **Document Processor:** The document processor now correctly persists its state, preventing it from reprocessing all documents on every startup.
- **Memory System:** Improved the robustness of the memory system by enabling error logging and using UUIDs for unique document IDs in semantic memory.
- **Cognitive Engine:** Fixed a `NameError` in `generate_assumption_from_history` by renaming a variable.
- **Document Processor:** Fixed the document processor by creating the `src/jenova/docs` directory, which was missing.
- **Document Processor:** The document processor now provides feedback to the user when processing documents.
- **ConcernManager:** Resolved a persistent `SyntaxError` in `src/jenova/insights/concerns.py` by rewriting the file and clearing `__pycache__`, ensuring correct parsing.
- **TerminalUI:** Fixed `TypeError: 'NoneType' object is not iterable` for the `/memory-insight` command by modifying `TerminalUI._handle_command` to robustly handle the return type of `develop_insights_from_memory`.
- **CognitiveScheduler:** Fixed `TypeError: Cortex.reflect() got an unexpected keyword argument 'username'` by correcting `CognitiveScheduler` to pass the `user` argument instead of `username` to `Cortex.reflect`.
- **TerminalUI:** Improved `/verify` command feedback by modifying `TerminalUI._verify_assumption` to always provide feedback to the user, even when no unverified assumptions are found.
- **CognitiveEngine:** Corrected calls to `InsightManager.get_latest_insight_id()` by passing the `username` argument, resolving a `TypeError`.
- **ConcernManager:** Removed the `rich` spinner context manager from `reorganize_insights` to ensure the intended "three yellow dot" loading indicator is displayed for the `/reflect` command.
- **InsightManager:** Resolved `AttributeError: 'InsightManager' object has no attribute 'get_latest_insight_id'` by implementing the missing method to retrieve the latest insight's ID.
- **InsightManager:** Fixed `NoneType` object has no attribute 'append' error in `reorganize_insights` by ensuring the method always returns a list and correcting a typo.
- **CognitiveEngine:** Addressed `NoneType` object is not iterable error in `develop_insights_from_memory` by adding a defensive check for `context` being `None`.
- **CognitiveEngine:** Ensured the `/meta` command provides user feedback even when no new meta-insight is generated.
- **ConcernManager:** Resolved `AttributeError: 'ConcernManager' object has no attribute 'get_all_concerns'` by implementing the missing method to retrieve all existing concern topics.
- **UI Bug:** Fixed an issue in `TerminalUI` where empty `jenova_response` calls were creating unintended empty boxes, improving the visual presentation of AI output.
- **UI Bug:** Resolved the repetitive cluttering of the custom spinner by ensuring messages are returned from cognitive engine methods and printed only after the spinner has stopped, providing a clean and consistent processing indicator across commands.
- **Bug Fix:** Corrected `develop_insights_from_memory` to properly retrieve and display assumption IDs, resolving the `AttributeError: 'AssumptionManager' object has no attribute 'get_latest_assumption_id'`.
- **UI Bug:** Fixed the `TerminalUI` processing spinner to display yellow spinning dots only during long-running commands, ensuring it clears correctly and does not interfere with the prompt or AI output.
- **SyntaxError:** Fixed a `SyntaxError` in `jenova/ui/terminal.py` and `jenova-ai/src/jenova/ui/terminal.py` caused by incorrect syntax in the `_handle_command` method.
- **KeyError:** Fixed a `KeyError` in `jenova/assumptions/manager.py` and `jenova-ai/src/jenova/assumptions/manager.py` by ensuring `cortex_id` is present when loading and accessing assumption objects.
- **Error Handling:** Added robust error handling to all file I/O operations, LLM calls, and other critical parts of the codebase to prevent crashes.
- **NameError:** Fixed a `NameError` in `jenova/insights/concerns.py` caused by a missing `import os` statement.
- **NameError:** Fixed a `NameError` in `jenova/cognitive_engine/engine.py` caused by a missing `from jenova.cortex.proactive_engine import ProactiveEngine` statement.
- **TypeError:** Fixed a `TypeError` in `jenova/main.py` caused by a missing `config` argument in the `RAGSystem` constructor.
- **KeyError:** Fixed a `KeyError` in `jenova/assumptions/manager.py` caused by a missing `cortex_id` in the assumption object.
- **Security:**
  - Fixed a path traversal vulnerability in `FileTools`.
  - Fixed a shell injection vulnerability in `SystemTools`.
- The finetuning process is no longer a manual, multi-step process but a single, executable script.
- A bug in the /finetune command that caused a crash due to a missing 'os' import.
- A bug in the `finetune/prepare_data.py` script that caused incorrect parsing of conversation history.
- The application no longer crashes if the model path is not configured. It now provides clear instructions to the user.
- A bug in the `/finetune` command that caused a crash due to incorrect handling of shell command results.
- Resolved a bug where the fine-tuning process would fail due to a missing `model_path` in the configuration. The application will now correctly prompt the user to set the path.
- Hardened the JSON parsing logic across the application when processing responses from the LLM. This prevents crashes caused by malformed JSON, such as the one occurring during insight reorganization (`/reflect`).
- Cleaned up the UI to no longer display raw RAG debug information in the chat output, providing a cleaner user experience.
- Fixed a bug in the `/verify` command that caused a crash due to a mismatch in the expected return value from the cognitive engine.
- Fixed a bug where the model path was not being read from the configuration, causing the wrong model to be loaded if multiple models were present.

## [2.0.0] - 2025-10-02

### Added
- **RAG System:** A new Retrieval-Augmented Generation (RAG) system is now a core component of the cognitive architecture.
- **Document Processor:** A new system that allows the AI to scan and process documents in the `docs` folder to generate new cognitive nodes.

### Changed
- **Cognitive Engine:** The engine now prioritizes the AI's own insights and memories over its general knowledge.
- **Insight System:** The system is now more comprehensive, with insights being interlinked with other cognitive nodes in the Cortex.
- **Reflect System:** The reflection process is now more sophisticated, using graph analysis to find patterns, and is triggered automatically during the cognitive cycle.
- **Memory System:** The memory system is now more comprehensive and includes an emotional component for more intelligent responses.
- **Assumption System:** The assumption system is more robust and intelligent, using the LLM to resolve assumptions proactively during conversation.
- **Proactive Engine:** The proactive engine is more sophisticated, using graph analysis to find underdeveloped areas of the cognitive graph and trigger suggestions more frequently.
- **RAG.md Dependency:** Removed the hardcoded dependency on the `RAG.md` file, as this is now handled by the core RAG system.

### Fixed
- **Commands:** The `/develop_insight` and `/finetune` commands are now working properly and are more robust.
- **Error Handling:** Improved error handling across the application to prevent crashes and provide better feedback to the user.

## [1.3.0] - 2025-10-02

### Fixed
- **Command System:**
  - `/meta`: The command now provides feedback to the user when a new meta-insight is generated.
  - `/develop_insight`: The command now correctly parses the `node_id` and provides a usage message if it's missing.
  - `/verify`: The command is now fully interactive, allowing users to confirm or deny assumptions in real-time.
  - `/finetune`: The command now triggers a real fine-tuning process, including data preparation and a `llama.cpp`-based fine-tuning command.
- **Insight Organization:** The `/reflect` command now properly cleans up old, empty topic folders after reorganizing insights.

### Changed
- **Fine-tuning:** The fine-tuning data preparation script (`finetune/prepare_data.py`) now creates a more structured and effective training file.
- **Configuration:** The `main_config.yaml` has been updated to support the new fine-tuning process.

## [1.2.0] - 2025-10-01

### Added
- **Superior Intelligence**: Enhanced the Cortex to provide a more organized and developed cognitive graph.
- **Insight Development**: Added a `/develop_insight <node_id>` command to generate a more detailed and developed version of an existing insight.
- **Psychological Memory**: The Cortex now analyzes the sentiment of new nodes and adds it as metadata, providing a psychological dimension to the cognitive graph.
- **In-app Fine-tuning**: Added a `/finetune` command to trigger the fine-tuning process from within the application.
- **Cortex Architecture**: A new central hub for the AI's cognitive architecture that manages a graph of interconnected cognitive nodes (insights, memories, assumptions).
- **Deep Reflection**: The Cortex can perform deep reflection on the cognitive graph to find patterns, infer relationships, and generate meta-insights.
- **Proactive Engine**: A new engine that analyzes the cognitive graph to generate proactive suggestions and questions for the user.
- New Assumption System to generate, store, and verify assumptions about the user.
- Assumptions are categorized as `verified`, `unverified`, `true`, or `false`.
- New `/verify` command to initiate the assumption verification process.
- Re-introduced Meta-Insight Generation with the `/meta` command.

### Changed
- **Fleshed-out Proactivity**: The `ProactiveEngine` is now more sophisticated, analyzing clusters of insights to generate more meaningful and engaging suggestions.
- The `InsightManager` and `AssumptionManager` now use the `Cortex` to create and link insights and assumptions.
- The `CognitiveEngine` now orchestrates the `Cortex` and `ProactiveEngine`.
- Overhauled the insight system to be more organized and proactive.
- Insights are now organized by "concerns" or "topics".
- New insights are grouped with existing concerns to avoid duplication.
- The reflection system now reorganizes and interlinks insights.
- Memory insights are now integrated into the new concern-based system.

## [1.1.1] - 2025-09-28

### Changed
- Refactored the insight and memory systems to be user-specific, storing data in user-dedicated directories.

### Fixed
- Fixed a bug that caused user input to be repeated in the UI.
- The AI now correctly recognizes and can use the current user's username in conversation.
- Corrected a crash in `UILogger` by replacing `error` method calls with `system_message`.
- Removed fixed-width constraint on UI output panels to prevent text truncation.
- The cognitive functions context wheel now displays a static "Thinking..." message.
- Increased the number of search results retrieved from each memory type to improve utilization.
- Fixed a bug where the application would not correctly recognize the current user, leading to impersonal and incorrect responses.
- Fixed a `TypeError` in the `/memory-insight` command by passing the required `username` argument to the `MemorySearch.search_all()` method.

## [1.1.0] - 2025-09-28

### Added
- `CHANGELOG.md` to track project changes.
- Command-based system for on-demand insight generation and reflection:
  - `/insight`: Develop insights from the current conversation.
  - `/reflect`: Reflect on all existing insights to create meta-insights.
  - `/memory-insight`: Generate insights from a broad search of long-term memory.
- `get_all_insights` method to `InsightManager` to retrieve all stored insights.

### Changed
- The insight generation system is now more proactive, triggering after every conversational turn instead of on a fixed interval.
- Refactored `CognitiveEngine` to remove periodic reflection and introduce public methods for command-driven insight generation.
- Updated `TerminalUI` to parse and handle the new command system.
- Updated `README.md` to document the new active insight engine and the available user commands.
- The insight generation system is now more reflective, triggering every 5 turns to reduce noise.
- Improved the insight generation system to be more robust by providing more conversational context and a more detailed prompt.
- Updated the `README.md` to accurately describe the reflective insight engine.
- Improved the intelligence of the insight commands (`/insight`, `/reflect`, `/memory-insight`) by providing more detailed and structured prompts.

## [1.0.0] - 2025-09-28

### Added
- Initial release of the Jenova Cognitive Architecture.
- Multi-layered memory system (Episodic, Semantic, Procedural).
- Dynamic Insight Engine for learning.
- Terminal UI.