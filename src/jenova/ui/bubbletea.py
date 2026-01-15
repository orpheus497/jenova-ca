##Script function and purpose: Bubble Tea UI Wrapper for The JENOVA Cognitive Architecture
##This module provides a bridge between the Python backend and the Go-based Bubble Tea TUI
##Handles IPC communication, command processing, and interactive multi-step flows

"""
Bubble Tea UI wrapper for JENOVA Cognitive Architecture.
This module provides a bridge between the Python backend and the Go-based Bubble Tea TUI.
"""

import json
import os
import subprocess
import threading
import queue
from typing import Optional, Any
from jenova.ui.logger import UILogger
from jenova.cognitive_engine.engine import CognitiveEngine


##Class purpose: Wraps Go Bubble Tea TUI and manages Python-Go IPC communication
##Handles all command processing, interactive flows, and state management
class BubbleTeaUI:
    """
    Wrapper for the Bubble Tea terminal UI.
    Manages communication between Python backend and Go TUI process.
    """

    ##Function purpose: Initialize UI wrapper with engine, communication channels, and state tracking
    def __init__(self, cognitive_engine: CognitiveEngine, logger: UILogger):
        self.engine = cognitive_engine
        self.logger = logger
        self.tui_process: Optional[subprocess.Popen] = None
        self.message_queue = queue.Queue()
        self.running = False
        self.username = os.getenv("USER", "user")
        
        ##Block purpose: Initialize interactive mode state tracking
        ##Modes: 'normal', 'verify', 'learn_procedure_name', 'learn_procedure_steps', 'learn_procedure_outcome'
        self.interactive_mode = 'normal'
        self.pending_assumption = None
        self.procedure_data: dict[str, Any] = {}
        ##Block purpose: Thread lock for synchronizing interactive mode state access
        self.state_lock = threading.Lock()
        
        ##Block purpose: Input processing queue for serializing user input handling
        self.input_queue = queue.Queue()
        
        ##Block purpose: Find the TUI binary path
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.tui_path = os.path.join(script_dir, "tui", "jenova-tui")
        
        if not os.path.exists(self.tui_path):
            tui_dir = os.path.dirname(self.tui_path)
            raise FileNotFoundError(
                f"TUI binary not found at {self.tui_path}. "
                f"Please build it by running 'go build -o jenova-tui .' in the 'tui' directory located at: "
                f"{tui_dir}"
            )

    ##Function purpose: Send a JSON message to the TUI process via stdin
    def send_message(self, msg_type: str, content: str = "", data: Optional[dict[str, Any]] = None):
        """Send a message to the TUI."""
        message = {
            "type": msg_type,
            "content": content,
        }
        if data:
            message["data"] = data
        
        try:
            json_msg = json.dumps(message)
            if self.tui_process and self.tui_process.stdin:
                self.tui_process.stdin.write((json_msg + "\n").encode('utf-8'))
                self.tui_process.stdin.flush()
        except Exception as e:
            self.logger.info(f"Error sending message to TUI: {e}")

    ##Function purpose: Read messages from TUI stdout in separate thread for async communication
    def read_messages(self):
        """Read messages from the TUI in a separate thread."""
        if not self.tui_process or not self.tui_process.stdout:
            return
        
        try:
            for line in iter(self.tui_process.stdout.readline, b''):
                if not line:
                    break
                
                try:
                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        message = json.loads(line_str)
                        self.message_queue.put(message)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.info(f"Error reading message: {e}")
        except Exception as e:
            self.logger.info(f"Error in read thread: {e}")

    ##Function purpose: Worker thread that serially processes user input from queue
    def _input_worker(self) -> None:
        """Process user input serially from the input queue."""
        while self.running:
            try:
                user_input = self.input_queue.get(timeout=0.1)
                if user_input is None:  ##Block purpose: None signals worker shutdown
                    self.input_queue.task_done()
                    break
                try:
                    self.process_user_input(user_input)
                finally:
                    ##Block purpose: Mark task as done even if processing fails
                    self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.info(f"Error in input worker: {e}")

    ##Function purpose: Process user input based on current interactive mode
    def process_user_input(self, user_input: str):
        """Process user input based on current interactive mode."""
        try:
            ##Block purpose: Synchronize access to interactive mode state
            with self.state_lock:
                current_mode = self.interactive_mode
            
            ##Block purpose: Handle input based on current interactive mode
            if current_mode == 'verify':
                self._handle_verify_response(user_input)
            elif current_mode == 'learn_procedure_name':
                self._handle_procedure_name(user_input)
            elif current_mode == 'learn_procedure_steps':
                self._handle_procedure_step(user_input)
            elif current_mode == 'learn_procedure_outcome':
                self._handle_procedure_outcome(user_input)
            elif user_input.startswith('/'):
                ##Block purpose: Handle slash commands in normal mode
                self._handle_command(user_input)
            else:
                ##Block purpose: Handle regular conversation in normal mode
                self.send_message("start_loading")
                response = self.engine.think(user_input, self.username)
                
                if not isinstance(response, str):
                    response = str(response)
                
                self.send_message("ai_response", response)
                self.send_message("stop_loading")
        except Exception as e:
            self.send_message("system_message", f"Error: {e}")
            self.send_message("stop_loading")
            ##Block purpose: Reset all interactive state on error to prevent stuck states
            with self.state_lock:
                self.interactive_mode = 'normal'
                self.pending_assumption = None
                self.procedure_data = {}

    ##Function purpose: Handle slash commands from user input
    def _handle_command(self, user_input: str) -> None:
        """Handle user commands.
        
        Note: args is a list of individual space-separated arguments with case preserved.
        For example, '/develop_insight NodeID_123' yields args=['NodeID_123'].
        """
        ##Block purpose: Parse command while preserving argument case-sensitivity
        parts = user_input.split(' ', 1)
        command = parts[0].lower()
        ##Block purpose: Split remaining text into individual arguments, preserving case
        args = parts[1].split() if len(parts) > 1 else []
        
        self.send_message("start_loading")
        
        try:
            if command == '/help':
                self._show_help()
            elif command == '/insight':
                messages = self.engine.develop_insights_from_conversation(self.username)
                self._send_messages(messages)
            elif command == '/reflect':
                messages = self.engine.reflect_on_insights(self.username)
                self._send_messages(messages)
            elif command == '/memory-insight':
                messages = self.engine.develop_insights_from_memory(self.username)
                self._send_messages(messages)
            elif command == '/meta':
                messages = self.engine.generate_meta_insight(self.username)
                self._send_messages(messages)
            elif command == '/verify':
                self._start_verify()
            elif command == '/train':
                self.send_message("system_message", "To create a training file for fine-tuning, run: python3 finetune/train.py")
            elif command == '/develop_insight':
                self._develop_insight(args)
            elif command == '/learn_procedure':
                self._start_learn_procedure()
            else:
                self.send_message("system_message", f"Unknown command: {command}")
        except Exception as e:
            self.send_message("system_message", f"Error executing command: {e}")
        finally:
            self.send_message("stop_loading")

    ##Function purpose: Display comprehensive help information with formatted sections
    def _show_help(self) -> None:
        """Display formatted help information."""
        help_text = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         JENOVA COMMAND REFERENCE                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

COGNITIVE COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  /help
    Displays this comprehensive command reference guide.
    Shows all available commands with detailed descriptions.

  /insight
    Analyzes the current conversation and generates new insights.
    JENOVA will extract key takeaways from your recent interactions and
    store them as structured insights in long-term memory, contributing
    to its evolving understanding of topics and patterns.

  /reflect
    Initiates deep reflection within The JENOVA Cognitive Architecture.
    This powerful command reorganizes and interlinks all cognitive nodes
    (insights, memories, assumptions), identifies patterns, and generates
    higher-level meta-insights, significantly enhancing intelligence and
    cognitive coherence across the entire knowledge graph.

  /memory-insight
    Performs a comprehensive search across all memory layers.
    JENOVA will scan episodic, semantic, and procedural memory to
    develop new insights or assumptions based on accumulated knowledge
    from past interactions and learned information.

  /meta
    Generates higher-level meta-insights from existing knowledge.
    Analyzes clusters of related insights within the cognitive graph
    to form abstract conclusions and identify overarching themes,
    enabling more sophisticated pattern recognition and understanding.

  /verify
    Starts the assumption verification process.
    JENOVA will present an unverified assumption about your preferences
    or knowledge and ask for clarification, allowing you to confirm or
    deny it. This refines JENOVA's understanding of your context.

LEARNING COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  /develop_insight [node_id]
    Dual-purpose insight development and document learning command.
    • With node_id: Expands an existing insight with more context
      and connections, generating a more detailed version.
    • Without node_id: Scans the src/jenova/docs directory for new
      or updated documents, processes their content, and integrates
      insights and summaries into the cognitive graph, enabling
      learning from external documentation.

  /learn_procedure
    Interactive guided process to teach JENOVA a new procedure.
    JENOVA will prompt you for the procedure's name, individual steps,
    and expected outcome. This structured approach ensures comprehensive
    intake of procedural knowledge, stored in procedural memory for
    future recall and application in relevant contexts.

  /train
    Provides instructions for creating fine-tuning training data.
    Shows how to generate a training dataset from your interactions
    for fine-tuning the underlying language model with personalized
    knowledge and conversation patterns.

SYSTEM COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  exit or quit
    Exits The JENOVA Cognitive Architecture application.
    All current session data will be automatically saved.

INNATE CAPABILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  JENOVA can intelligently use its built-in tools to answer your
  questions without requiring specific commands. For example:

    • Ask for the current time
    • Request file operations in the sandbox directory (~/jenova_files)
    • Perform calculations and data analysis
    • Access and process information from its knowledge graph

╔═══════════════════════════════════════════════════════════════════════════════╗
║  Tip: Commands are not stored in conversational memory and can be used       ║
║  freely to manage JENOVA's cognitive processes.                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        self.send_message("help", help_text)

    ##Function purpose: Send multiple system messages to the TUI
    def _send_messages(self, messages):
        """Send multiple messages to the TUI."""
        if messages:
            if isinstance(messages, str):
                messages = [messages]
            for msg in messages:
                self.send_message("system_message", str(msg))

    ##Function purpose: Start the assumption verification interactive flow
    def _start_verify(self) -> None:
        """Start assumption verification flow."""
        assumption, question = self.engine.verify_assumptions(self.username)
        if question:
            self.send_message("system_message", f"JENOVA is asking for clarification: {question}")
            if assumption:
                ##Block purpose: Set interactive mode to capture next input as verification response
                with self.state_lock:
                    self.pending_assumption = assumption
                    self.interactive_mode = 'verify'
                self.send_message("system_message", "Please respond with 'yes' or 'no':")
        else:
            self.send_message("system_message", "No unverified assumptions to check.")

    ##Function purpose: Handle user response to assumption verification
    def _handle_verify_response(self, user_input: str) -> None:
        """Handle yes/no response for assumption verification."""
        response = user_input.strip().lower()
        
        ##Block purpose: Validate that response is yes/no (or y/n)
        valid_yes = ['yes', 'y']
        valid_no = ['no', 'n']
        
        if response not in valid_yes + valid_no:
            self.send_message("system_message", "Please respond with 'yes' or 'no':")
            ##Block purpose: Keep in verification mode, don't clear pending assumption
            return
        
        ##Block purpose: Normalize response to 'yes' or 'no'
        normalized_response = 'yes' if response in valid_yes else 'no'
        
        try:
            self.send_message("start_loading")
            
            ##Block purpose: Atomically read and clear pending assumption
            with self.state_lock:
                pending_assumption = self.pending_assumption
                self.pending_assumption = None
                self.interactive_mode = 'normal'
            
            if pending_assumption:
                self.engine.assumption_manager.resolve_assumption(
                    pending_assumption, 
                    normalized_response, 
                    self.username
                )
                self.send_message("system_message", "Assumption verification recorded. Thank you!")
            else:
                ##Block purpose: Log unexpected case where no assumption was pending
                self.logger.info("Warning: _handle_verify_response called with no pending assumption")
                self.send_message("system_message", "No pending assumption to verify.")
            
        except Exception as e:
            self.send_message("system_message", f"Error during verification: {e}")
            ##Block purpose: State already cleared above, no additional cleanup needed
        finally:
            self.send_message("stop_loading")

    ##Function purpose: Handle insight development command with optional node_id
    def _develop_insight(self, args: list[str]) -> None:
        """Handle insight development."""
        try:
            if args:
                node_id = args[0]
                messages = self.engine.cortex.develop_insight(node_id, self.username)
            else:
                messages = self.engine.cortex.develop_insights_from_docs(self.username)
            self._send_messages(messages)
        except Exception as e:
            self.send_message("system_message", f"Error: {e}")

    ##Function purpose: Start the multi-step procedure learning interactive flow
    def _start_learn_procedure(self) -> None:
        """Start interactive procedure learning flow."""
        ##Block purpose: Initialize procedure data storage and set interactive mode
        with self.state_lock:
            self.procedure_data = {
                "name": "",
                "steps": [],
                "outcome": ""
            }
            self.interactive_mode = 'learn_procedure_name'
        self.send_message("system_message", "Initiating interactive procedure learning...")
        self.send_message("system_message", "Please enter the procedure name:")

    ##Function purpose: Handle procedure name input in multi-step flow
    def _handle_procedure_name(self, user_input: str) -> None:
        """Handle procedure name input."""
        name = user_input.strip()
        
        if not name:
            self.send_message("system_message", "Procedure name cannot be empty. Please enter a name:")
            return
        
        ##Block purpose: Store name and transition to steps collection mode
        with self.state_lock:
            self.procedure_data["name"] = name
            self.interactive_mode = 'learn_procedure_steps'
            steps_count = len(self.procedure_data['steps'])
        
        self.send_message("system_message", f"Procedure name set to: {name}")
        self.send_message("system_message", "Enter procedure steps one by one. Type 'done' when finished.")
        self.send_message("system_message", f"Step {steps_count + 1}:")

    ##Function purpose: Handle individual step input in multi-step flow
    def _handle_procedure_step(self, user_input: str) -> None:
        """Handle procedure step input."""
        step = user_input.strip()
        
        ##Block purpose: Check if user is done entering steps
        if step.lower() == 'done':
            with self.state_lock:
                steps_count = len(self.procedure_data["steps"])
                if not steps_count:
                    self.send_message("system_message", "No steps entered. Please enter at least one step:")
                    return
                
                ##Block purpose: Transition to outcome collection mode
                self.interactive_mode = 'learn_procedure_outcome'
            
            self.send_message("system_message", f"Recorded {steps_count} steps.")
            self.send_message("system_message", "Please enter the expected outcome:")
            return
        
        ##Block purpose: Handle empty step input with feedback
        if not step:
            self.send_message("system_message", "Empty step entered. Please enter a step or type 'done':")
            return
        
        ##Block purpose: Add step to list and prompt for next
        with self.state_lock:
            self.procedure_data["steps"].append(step)
            steps_count = len(self.procedure_data['steps'])
            next_step = steps_count + 1
        
        self.send_message("system_message", f"Step {steps_count} recorded: {step}")
        self.send_message("system_message", f"Step {next_step} (or type 'done'):")

    ##Function purpose: Handle expected outcome input and complete procedure learning
    def _handle_procedure_outcome(self, user_input: str) -> None:
        """Handle procedure outcome input and complete the learning process."""
        outcome = user_input.strip()
        
        if not outcome:
            self.send_message("system_message", "Expected outcome cannot be empty. Please enter an outcome:")
            return
        
        try:
            self.send_message("start_loading")
            
            ##Block purpose: Store outcome and call engine to learn the procedure
            with self.state_lock:
                self.procedure_data["outcome"] = outcome
                procedure_data_copy = self.procedure_data.copy()
            
            messages = self.engine.learn_procedure(procedure_data_copy, self.username)
            self._send_messages(messages)
            
            ##Block purpose: Reset to normal mode after successful learning
            with self.state_lock:
                self.procedure_data = {}
                self.interactive_mode = 'normal'
            
        except Exception as e:
            self.send_message("system_message", f"Error learning procedure: {e}")
            ##Block purpose: Reset to normal mode and clear stale data after error
            with self.state_lock:
                self.procedure_data = {}
                self.interactive_mode = 'normal'
        finally:
            self.send_message("stop_loading")

    ##Function purpose: Start TUI process and run main message processing loop
    def run(self):
        """Start the TUI and process messages."""
        try:
            ##Block purpose: Start the TUI subprocess with stdin/stdout pipes for IPC
            self.tui_process = subprocess.Popen(
                [self.tui_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.running = True
            
            ##Block purpose: Start background thread to read messages from TUI
            read_thread = threading.Thread(target=self.read_messages, daemon=True)
            read_thread.start()
            
            ##Block purpose: Start single worker thread to process user inputs serially
            input_worker_thread = threading.Thread(target=self._input_worker, daemon=True)
            input_worker_thread.start()
            
            ##Block purpose: Send initial banner and welcome messages
            banner = """
     ██╗███████╗███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ 
     ██║██╔════╝████╗  ██║██╔═══██╗██║   ██║██╔══██╗
     ██║█████╗  ██╔██╗ ██║██║   ██║██║   ██║███████║
██   ██║██╔══╝  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
╚█████╔╝███████╗██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
 ╚════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
            """
            self.send_message("banner", banner, {"attribution": "Designed by orpheus497 - https://github.com/orpheus497"})
            self.send_message("info", "Initialized and Ready.")
            self.send_message("info", "Type your message, use a command, or type 'exit' to quit.")
            self.send_message("info", "Type /help to see available commands.")
            
            ##Block purpose: Main event loop - process messages from TUI
            while self.running:
                try:
                    message = self.message_queue.get(timeout=0.1)
                    
                    if message.get("type") == "user_input":
                        user_input = message.get("content", "").strip()
                        
                        ##Block purpose: Handle exit in any mode
                        if user_input.lower() in ['exit', 'quit']:
                            ##Block purpose: Reset interactive mode before exit
                            with self.state_lock:
                                if self.interactive_mode != 'normal':
                                    self.send_message("system_message", "Exiting interactive mode...")
                                    self.interactive_mode = 'normal'
                            break
                        
                        ##Block purpose: Queue input for serial processing by worker thread
                        self.input_queue.put(user_input)
                    
                    elif message.get("type") == "exit":
                        break
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.info(f"Error processing message: {e}")
            
        except Exception as e:
            print(f"Error running TUI: {e}")
        finally:
            ##Block purpose: Signal input worker thread to shut down before clearing running flag
            self.input_queue.put(None)
            ##Block purpose: Clean shutdown of TUI process and wait for worker thread
            self.running = False
            ##Block purpose: Join input worker thread for orderly shutdown
            if 'input_worker_thread' in locals():
                input_worker_thread.join(timeout=2.0)
            if self.tui_process:
                self.tui_process.terminate()
                self.tui_process.wait()
