# Example Plugin

Demonstration plugin for JENOVA Cognitive Architecture showing plugin capabilities and best practices.

## Features

- **Tool Registration**: Custom tools for word counting, searching, and summarization
- **Command Registration**: Custom commands for interaction
- **Memory Integration**: Query semantic memory
- **File I/O**: Read/write files in sandboxed environment
- **LLM Access**: Generate text using rate-limited LLM access

## Installation

1. Copy this directory to `~/.jenova-ai/plugins/example_plugin`
2. Restart JENOVA or reload plugins

## Usage

### Tools

The plugin registers three tools:

**example_tool**: Basic demonstration
```python
result = api.call_tool("example_plugin:example_tool", {
    "message": "Hello world"
})
# Returns: {"response": "Received: Hello world", "length": 11, "call_count": 1}
```

**word_count**: Text analysis
```python
result = api.call_tool("example_plugin:word_count", {
    "text": "The quick brown fox jumps"
})
# Returns: {"word_count": 5, "char_count": 25, "line_count": 1, "unique_words": 5}
```

**search_and_summarize**: Memory search with LLM summarization
```python
result = api.call_tool("example_plugin:search_and_summarize", {
    "query": "Python programming",
    "api": api_instance
})
# Returns: {"results": [...], "result_count": 3, "summary": "..."}
```

### Commands

The plugin registers two commands:

**/example [message]**: Example command
```
/example Hello from the plugin!
```

**/stats**: Display plugin statistics
```
/stats
```

## Permissions

This plugin requires the following permissions:
- `memory:read` - Query semantic memory
- `memory:write` - Save insights
- `tools:register` - Register custom tools
- `commands:register` - Register custom commands
- `file:read` - Read files from sandbox
- `file:write` - Write files to sandbox
- `llm:inference` - Generate text using LLM

## Resource Limits

- Max CPU time: 30 seconds per call
- Max memory: 256 MB
- Max file size: 10 MB

## File Structure

```
example_plugin/
├── __init__.py      # Main plugin module
├── plugin.yaml      # Plugin manifest
└── README.md        # This file
```

## Development

### Plugin Lifecycle

1. **initialize(api)**: Called once on plugin load
   - Register tools and commands
   - Initialize resources

2. **activate(api)**: Called when plugin is activated
   - Start background tasks (if needed)

3. **deactivate(api)**: Called when plugin is deactivated
   - Stop background tasks

4. **cleanup(api)**: Called on plugin unload
   - Save state
   - Release resources

### Best Practices

1. **Always check permissions** before using API features
2. **Handle errors gracefully** in tool/command handlers
3. **Log important events** using `api.log()`
4. **Stay within resource limits** (CPU, memory)
5. **Use sandboxed file I/O** for persistence
6. **Respect rate limits** for LLM and memory access

## Testing

Test the plugin manually:

```python
from jenova.plugins.plugin_manager import PluginManager

manager = PluginManager(...)
manager.discover_plugins()
manager.load_plugin("example_plugin")
manager.initialize_plugin("example_plugin")
manager.activate_plugin("example_plugin")

# Test tools
api = manager.plugins["example_plugin"].api
result = api.registered_tools["word_count"]({"text": "Hello world"})
print(result)
```

## License

MIT License - Copyright (c) 2024, orpheus497
