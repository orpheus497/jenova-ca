# Bubble Tea UI for JENOVA

JENOVA now features a modern, beautiful terminal UI built with [Bubble Tea](https://github.com/charmbracelet/bubbletea) - a powerful TUI framework for Go.

## Features

The new Bubble Tea UI provides:

- **Modern, responsive terminal interface** with smooth rendering
- **Real-time message updates** with proper viewport scrolling
- **Loading indicators** for long-running operations
- **Rich text formatting** with colors and styles
- **Command support** - all existing JENOVA commands work seamlessly
- **Better keyboard navigation** and input handling

## Building the UI

The Bubble Tea UI requires Go 1.24 or later to be installed. To build the TUI:

```bash
./build_tui.sh
```

This script will:
1. Check for Go installation
2. Download required Go dependencies
3. Build the `jenova-tui` binary in the `tui/` directory

## Running JENOVA with Bubble Tea UI

### Using the Bubble Tea UI (default)

The Bubble Tea UI is now the default interface:

```bash
./jenova
```

Or explicitly set the UI mode:

```bash
export JENOVA_UI=bubbletea
./jenova
```

**Note:** BubbleTea is now the sole supported UI interface.
The classic Python terminal UI has been removed to streamline the codebase.

## Architecture

The Bubble Tea UI uses an IPC (Inter-Process Communication) architecture:

- **Go TUI Process**: Handles all UI rendering and user input using Bubble Tea
- **Python Backend**: Runs the cognitive engine and processes user queries
- **JSON Protocol**: Messages are exchanged between processes via stdin/stdout

```
┌─────────────────┐         JSON Messages         ┌──────────────────┐
│   Go TUI        │ ◄────────────────────────────► │  Python Backend  │
│  (Bubble Tea)   │   stdin/stdout pipes           │  (Cognitive Eng) │
└─────────────────┘                                └──────────────────┘
```

## Message Types

The TUI communicates with the Python backend using these message types:

**From Python → TUI:**
- `banner` - Display startup banner
- `info` - Information messages
- `system_message` - System notifications
- `ai_response` - JENOVA's responses
- `help` - Help text
- `start_loading` / `stop_loading` - Loading state control

**From TUI → Python:**
- `user_input` - User's message or command
- `exit` - User wants to quit

## Development

### TUI Source Code

The TUI is implemented in Go and located at:
- `tui/main.go` - Main Bubble Tea application
- `tui/go.mod` - Go module dependencies

### Python Wrapper

The Python wrapper handles communication with the TUI:
- `src/jenova/ui/bubbletea.py` - BubbleTeaUI class
- `src/jenova/main.py` - Main entry point (BubbleTea is the sole UI)

### Modifying the UI

1. Edit `tui/main.go` to change UI appearance or behavior
2. Rebuild with `./build_tui.sh`
3. Test with `JENOVA_UI=bubbletea ./jenova`

### Adding New Message Types

1. Add message type to the `Message` struct in `main.go`
2. Handle the message in the `Update()` function
3. Update `bubbletea.py` to send the new message type
4. Rebuild and test

## Dependencies

### Go Dependencies
- `github.com/charmbracelet/bubbletea` - TUI framework
- `github.com/charmbracelet/bubbles` - TUI components
- `github.com/charmbracelet/lipgloss` - Styling library

### Python Requirements
No additional Python dependencies are required - the existing dependencies remain unchanged.

## Troubleshooting

### "TUI binary not found" error

Run the build script:
```bash
./build_tui.sh
```

### Go not installed

Install Go 1.24 or later:
- Ubuntu/Debian: `sudo apt install golang-go`
- Fedora: `sudo dnf install golang`
- macOS: `brew install go`
- Or download from: https://golang.org/dl/

### UI not responding

The TUI process may have crashed. Check:
1. Terminal size (must be at least 80x24)
2. Python backend errors in logs
3. Restart JENOVA

## Benefits of Bubble Tea UI

1. **Better Performance**: Native Go rendering is faster than Python-based UIs
2. **Modern Look**: Beautiful, consistent styling across all terminal types
3. **Maintainability**: Clear separation between UI and business logic
4. **Cross-platform**: Works consistently across Linux, macOS, and Windows
5. **Extensibility**: Easy to add new UI components using Bubble Tea's ecosystem
