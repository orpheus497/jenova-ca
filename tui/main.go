// Script function and purpose: Bubble Tea TUI for The JENOVA Cognitive Architecture
// This module provides a modern terminal UI using the Bubble Tea framework in Go

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Message purpose: Define IPC message types for communication with Python backend
type Message struct {
	Type    string                 `json:"type"`
	Content string                 `json:"content,omitempty"`
	Data    map[string]interface{} `json:"data,omitempty"`
}

// Block purpose: Define lipgloss styles for terminal UI elements
var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FF00FF")).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#FF00FF")).
			Padding(0, 1)

	headerStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#00FFFF"))

	userStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#44FF44"))

	aiStyle = lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FF44FF"))

	systemStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF4444"))

	inputStyle = lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			BorderForeground(lipgloss.Color("#888888"))

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#888888"))
)

// Model purpose: Define the application state model for Bubble Tea
type model struct {
	viewport    viewport.Model
	textarea    textarea.Model
	spinner     spinner.Model
	messages    []string
	username    string
	ready       bool
	loading     bool
	width       int
	height      int
	pythonInput io.Writer
}

// Function purpose: Create and return initial model with default configuration
func initialModel(pythonInput io.Writer) model {
	ta := textarea.New()
	ta.Placeholder = "Type your message or command..."
	ta.Focus()
	ta.CharLimit = 2000
	ta.SetWidth(80)
	ta.SetHeight(1)
	ta.ShowLineNumbers = false

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFFF00"))

	vp := viewport.New(80, 20)

	return model{
		textarea:    ta,
		spinner:     s,
		viewport:    vp,
		messages:    []string{},
		username:    os.Getenv("USER"),
		pythonInput: pythonInput,
	}
}

// Function purpose: Initialize Bubble Tea commands on startup
func (m model) Init() tea.Cmd {
	return tea.Batch(
		textarea.Blink,
		m.spinner.Tick,
	)
}

// Function purpose: Handle messages and update model state
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
		spCmd tea.Cmd
	)

	m.textarea, tiCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)
	m.spinner, spCmd = m.spinner.Update(msg)

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		if !m.ready {
			m.viewport = viewport.New(msg.Width, msg.Height-6)
			m.viewport.YPosition = 0
			m.ready = true
		} else {
			m.viewport.Width = msg.Width
			m.viewport.Height = msg.Height - 6
		}

		m.textarea.SetWidth(msg.Width - 4)

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			// Send exit message to Python
			exitMsg := Message{Type: "exit"}
			jsonData, _ := json.Marshal(exitMsg)
			fmt.Fprintln(m.pythonInput, string(jsonData))
			return m, tea.Quit

		case tea.KeyEnter:
			input := strings.TrimSpace(m.textarea.Value())
			if input != "" {
				// Add user message to display
				userMsg := userStyle.Render(m.username+"@JENOVA> ") + input
				m.messages = append(m.messages, userMsg)
				m.updateViewport()

				// Send message to Python backend
				msg := Message{Type: "user_input", Content: input}
				jsonData, _ := json.Marshal(msg)
				fmt.Fprintln(m.pythonInput, string(jsonData))

				// Clear input and set loading state
				m.textarea.Reset()
				m.loading = true
			}
			return m, nil
		}

	case Message:
		// Handle messages from Python backend
		switch msg.Type {
		case "banner":
			banner := headerStyle.Render(msg.Content)
			m.messages = append(m.messages, banner)
			if attribution, ok := msg.Data["attribution"].(string); ok {
				m.messages = append(m.messages, headerStyle.Render(attribution))
			}
			m.updateViewport()

		case "info":
			m.messages = append(m.messages, systemStyle.Render(">> "+msg.Content))
			m.updateViewport()

		case "system_message":
			m.messages = append(m.messages, systemStyle.Render(">> "+msg.Content))
			m.updateViewport()

		case "ai_response":
			aiMsg := aiStyle.Render("JENOVA> ") + msg.Content
			m.messages = append(m.messages, "", aiMsg, "")
			m.loading = false
			m.updateViewport()

		case "help":
			m.messages = append(m.messages, msg.Content)
			m.updateViewport()

		case "start_loading":
			m.loading = true

		case "stop_loading":
			m.loading = false
		}
	}

	return m, tea.Batch(tiCmd, vpCmd, spCmd)
}

// Function purpose: Update viewport content with chat messages
func (m *model) updateViewport() {
	content := strings.Join(m.messages, "\n")
	m.viewport.SetContent(content)
	m.viewport.GotoBottom()
}

// Function purpose: Render the terminal UI view
func (m model) View() string {
	if !m.ready {
		return "\n  Initializing..."
	}

	// Build the UI
	var sections []string

	// Title
	title := titleStyle.Render("JENOVA Cognitive Architecture")
	sections = append(sections, title, "")

	// Viewport with chat history
	sections = append(sections, m.viewport.View())

	// Loading indicator
	if m.loading {
		sections = append(sections, "", m.spinner.View()+" Processing...")
	}

	// Input area
	sections = append(sections, "", inputStyle.Render(m.textarea.View()))

	// Help text
	helpText := helpStyle.Render("Enter: send • /help: commands • Ctrl+C/Esc: quit")
	sections = append(sections, helpText)

	return strings.Join(sections, "\n")
}

// Function purpose: Listen for JSON messages from Python backend via stdin
// Listen for messages from Python backend via stdin
func listenForMessages(program *tea.Program) {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := scanner.Text()
		var msg Message
		if err := json.Unmarshal([]byte(line), &msg); err == nil {
			program.Send(msg)
		}
	}
}

// Function purpose: Main entry point that initializes and runs the Bubble Tea program
func main() {
	// Create a pipe to communicate with Python
	// Python reads from our stdout, we read from stdin
	pythonInput := os.Stdout

	// Create the model and program
	m := initialModel(pythonInput)
	p := tea.NewProgram(m, tea.WithAltScreen())

	// Start listening for messages from Python in background
	go listenForMessages(p)

	// Run the program
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running program: %v\n", err)
		os.Exit(1)
	}
}
