# Example Plugin for JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Example Plugin - Demonstrates JENOVA plugin capabilities.

Shows:
- Tool registration
- Command registration
- Memory integration
- File I/O
- LLM access
"""

from typing import Dict, Any, Optional
from jenova.plugins.plugin_api import PluginAPI


# Plugin state
plugin_state = {
    "initialized": False,
    "activation_count": 0,
    "tool_calls": 0,
    "command_calls": 0,
}


def initialize(api: PluginAPI) -> None:
    """
    Initialize plugin (called once on load).

    Args:
        api: Plugin API instance

    Example:
        This is called automatically by the plugin manager.
    """
    api.log("info", "Initializing example plugin")

    # Register custom tools
    api.register_tool("example_tool", example_tool_handler)
    api.register_tool("word_count", word_count_tool)
    api.register_tool("search_and_summarize", search_and_summarize_tool)

    # Register custom commands
    api.register_command("example", example_command_handler)
    api.register_command("stats", stats_command_handler)

    # Save initialization marker
    api.write_file("init.txt", "Plugin initialized successfully")

    plugin_state["initialized"] = True
    api.log("info", "Example plugin initialized - 3 tools and 2 commands registered")


def activate(api: PluginAPI) -> None:
    """
    Activate plugin (called when plugin is activated).

    Args:
        api: Plugin API instance
    """
    plugin_state["activation_count"] += 1
    api.log("info", f"Example plugin activated (count: {plugin_state['activation_count']})")


def deactivate(api: PluginAPI) -> None:
    """
    Deactivate plugin (called when plugin is deactivated).

    Args:
        api: Plugin API instance
    """
    api.log("info", "Example plugin deactivated")


def cleanup(api: PluginAPI) -> None:
    """
    Cleanup plugin resources (called on unload).

    Args:
        api: Plugin API instance
    """
    # Save statistics
    stats = {
        "activation_count": plugin_state["activation_count"],
        "tool_calls": plugin_state["tool_calls"],
        "command_calls": plugin_state["command_calls"],
    }
    api.write_file("stats.txt", str(stats))

    api.log("info", "Example plugin cleanup complete")


# Tool Handlers


def example_tool_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example tool demonstrating basic functionality.

    Args:
        args: Tool arguments with 'message' key

    Returns:
        Dict with result

    Example:
        >>> result = example_tool_handler({"message": "Hello"})
        >>> print(result["response"])  # "Received: Hello"
    """
    plugin_state["tool_calls"] += 1

    message = args.get("message", "")
    response = f"Received: {message}"

    return {
        "response": response,
        "length": len(message),
        "call_count": plugin_state["tool_calls"],
    }


def word_count_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Count words in text.

    Args:
        args: Tool arguments with 'text' key

    Returns:
        Dict with word count statistics

    Example:
        >>> result = word_count_tool({"text": "Hello world"})
        >>> print(result["word_count"])  # 2
    """
    plugin_state["tool_calls"] += 1

    text = args.get("text", "")
    words = text.split()

    return {
        "word_count": len(words),
        "char_count": len(text),
        "line_count": text.count("\n") + 1,
        "unique_words": len(set(words)),
    }


def search_and_summarize_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search memory and generate summary using LLM.

    Args:
        args: Tool arguments with 'query' and 'api' keys

    Returns:
        Dict with search results and summary

    Example:
        >>> result = search_and_summarize_tool({
        ...     "query": "Python programming",
        ...     "api": api_instance
        ... })
    """
    plugin_state["tool_calls"] += 1

    api: PluginAPI = args.get("api")
    query = args.get("query", "")

    if not api:
        return {"error": "API instance required"}

    # Search memory
    results = api.query_memory(query, memory_type="semantic", n_results=3)

    if not results:
        return {
            "results": [],
            "summary": "No results found",
        }

    # Generate summary using LLM
    results_text = "\n\n".join([str(r) for r in results])
    prompt = f"Summarize these search results for '{query}':\n\n{results_text}"

    summary = api.generate_text(prompt, max_tokens=256, temperature=0.3)

    return {
        "results": results,
        "result_count": len(results),
        "summary": summary,
    }


# Command Handlers


def example_command_handler(args: Dict[str, Any]) -> str:
    """
    Example command handler.

    Args:
        args: Command arguments with 'api' and 'message' keys

    Returns:
        Command response message

    Example:
        /example Hello world
    """
    plugin_state["command_calls"] += 1

    api: PluginAPI = args.get("api")
    message = args.get("message", "")

    if api:
        api.log("info", f"Example command called with: {message}")

    return f"Example command executed: {message}\nCall count: {plugin_state['command_calls']}"


def stats_command_handler(args: Dict[str, Any]) -> str:
    """
    Display plugin statistics.

    Args:
        args: Command arguments with 'api' key

    Returns:
        Statistics message

    Example:
        /stats
    """
    api: PluginAPI = args.get("api")

    if api:
        api_stats = api.get_stats()
        stats_msg = (
            f"Example Plugin Statistics:\n"
            f"- Activations: {plugin_state['activation_count']}\n"
            f"- Tool calls: {plugin_state['tool_calls']}\n"
            f"- Command calls: {plugin_state['command_calls']}\n"
            f"- API calls: {api_stats['api_calls_total']}\n"
            f"- Tools registered: {api_stats['tools_registered']}\n"
            f"- Commands registered: {api_stats['commands_registered']}\n"
        )
        return stats_msg

    return "API not available"
