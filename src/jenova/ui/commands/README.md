# JENOVA Command System Refactoring

**Phase 23: Command Refactoring - Modular Architecture**

## Overview

This directory contains the refactored command system for The JENOVA Cognitive Architecture. The original monolithic `commands.py` (1,330 lines) has been redesigned into a modular architecture with specialized command handlers.

## Architecture

### Base Classes (`base.py` - 198 lines)

**CommandCategory** - Enum defining command categories:
- SYSTEM - System information and help
- NETWORK - Network and peer management
- MEMORY - Backup and memory operations
- LEARNING - Learning statistics and profiles
- SETTINGS - Configuration management
- CODE - Code editing and analysis
- GIT - Git operations
- ORCHESTRATION - Task and workflow management
- AUTOMATION - Custom commands and automation

**Command** - Command definition class with:
- name, description, category
- handler function
- aliases, usage string, examples

**BaseCommandHandler** - Abstract base class for all handlers providing:
- Consistent initialization with cognitive_engine, loggers
- `register_commands()` abstract method
- `_register(command)` for command registration
- `_format_error()`, `_format_success()` for consistent messaging
- `_log_command_execution()` for audit trail

## Specialized Handlers (Planned)

### 1. SystemCommandHandler
**Commands**: help, profile, learn
**Purpose**: System information, user profiles, learning statistics
**Dependencies**: cognitive_engine, user_profile

### 2. NetworkCommandHandler
**Commands**: network, peers
**Purpose**: Network status, distributed mode, peer management
**Dependencies**: cognitive_engine.network (if available)

### 3. SettingsCommandHandler
**Commands**: settings
**Purpose**: Interactive settings menu, configuration management
**Dependencies**: settings_menu, cognitive_engine.config

### 4. MemoryCommandHandler
**Commands**: backup, export, import, backups
**Purpose**: Cognitive architecture backup and restoration
**Dependencies**: backup_manager

### 5. CodeToolsCommandHandler
**Commands**: edit, analyze, scan, parse, refactor
**Purpose**: Code manipulation and analysis tools
**Dependencies**: file_editor, code_metrics, security_scanner, code_parser, refactoring_engine

### 6. OrchestrationCommandHandler
**Commands**: git, task, workflow, command
**Purpose**: Git operations, task planning, workflows, custom commands
**Dependencies**: git_interface, commit_assistant, task_planner, execution_engine, custom_command_manager, workflow_library

## Registry (`registry.py`)

**CommandRegistry** - Central command router that:
- Instantiates all handler classes
- Aggregates commands from all handlers
- Routes command execution to appropriate handler
- Provides unified `execute(command_string)` interface
- Maintains backwards compatibility with existing code

## Benefits

**Modularity**: Each handler is self-contained with clear responsibilities

**Maintainability**: Easier to locate and modify specific command functionality

**Testability**: Individual handlers can be unit tested in isolation

**Extensibility**: New command categories can be added by creating new handlers

**Separation of Concerns**: Commands grouped by functional area

**Reduced Complexity**: Smaller files easier to understand and navigate

## Migration Strategy

1. ✅ **Phase 1**: Create base classes and module structure
2. **Phase 2**: Implement all 6 specialized handlers
3. **Phase 3**: Create new CommandRegistry router
4. **Phase 4**: Update imports in main.py and other modules
5. **Phase 5**: Deprecate old commands.py
6. **Phase 6**: Add comprehensive unit tests for each handler

## Usage Example

```python
from jenova.ui.commands import CommandRegistry

# Initialize registry (auto-registers all handlers)
registry = CommandRegistry(
    cognitive_engine=engine,
    ui_logger=ui_logger,
    file_logger=file_logger,
    # Handler-specific dependencies
    backup_manager=backup_mgr,
    settings_menu=settings,
    # ... other dependencies
)

# Execute command
result = registry.execute("/help")
result = registry.execute("/network status")
result = registry.execute("/backup full")
```

## File Structure

```
src/jenova/ui/commands/
├── __init__.py              # Module exports
├── README.md                # This file
├── base.py                  # Base classes (Command, CommandCategory, BaseCommandHandler)
├── registry.py              # CommandRegistry router
├── system_handler.py        # System commands (help, profile, learn)
├── network_handler.py       # Network commands (network, peers)
├── settings_handler.py      # Settings commands (settings)
├── memory_handler.py        # Memory commands (backup, export, import, backups)
├── code_tools_handler.py    # Code tools (edit, analyze, scan, parse, refactor)
└── orchestration_handler.py # Orchestration (git, task, workflow, command)
```

## Backwards Compatibility

The new `CommandRegistry` maintains the same public interface as the original, ensuring existing code continues to work without modification:

- `execute(command_string)` - Execute command
- `get_command(name)` - Get command by name
- `list_commands()` - List all commands
- `get_commands_by_category(category)` - Filter by category

## Status

**Phase 23 Foundation**: ✅ Complete
- Base classes implemented
- Module structure created
- Architecture documented

**Remaining Work**:
- Implement 6 specialized handlers (~1,100 lines)
- Create CommandRegistry router (~150 lines)
- Add unit tests
- Update imports in main application

---

**Author**: Claude Code (for orpheus497)
**License**: MIT
**Phase**: 23 - Command Refactoring
