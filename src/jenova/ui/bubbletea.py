"""
Bubble Tea UI wrapper for JENOVA Cognitive Architecture.
This module provides a bridge between the Python backend and the Go-based Bubble Tea TUI.
"""

import json
import os
import subprocess
import threading
import queue
from typing import Optional, Dict, Any
from jenova.ui.logger import UILogger
from jenova.cognitive_engine.engine import CognitiveEngine


class BubbleTeaUI:
    """
    Wrapper for the Bubble Tea terminal UI.
    Manages communication between Python backend and Go TUI process.
    """

    def __init__(self, cognitive_engine: CognitiveEngine, logger: UILogger):
        self.engine = cognitive_engine
        self.logger = logger
        self.tui_process: Optional[subprocess.Popen] = None
        self.message_queue = queue.Queue()
        self.running = False
        self.username = os.getenv("USER", "user")
        
        # Find the TUI binary
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.tui_path = os.path.join(script_dir, "tui", "jenova-tui")
        
        if not os.path.exists(self.tui_path):
            raise FileNotFoundError(f"TUI binary not found at {self.tui_path}. Please build it first.")

    def send_message(self, msg_type: str, content: str = "", data: Optional[Dict[str, Any]] = None):
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

    def process_user_input(self, user_input: str):
        """Process user input and generate response."""
        try:
            if user_input.startswith('/'):
                # Handle commands
                self._handle_command(user_input)
            else:
                # Regular conversation
                self.send_message("start_loading")
                response = self.engine.think(user_input, self.username)
                
                if not isinstance(response, str):
                    response = str(response)
                
                self.send_message("ai_response", response)
                self.send_message("stop_loading")
        except Exception as e:
            self.send_message("system_message", f"Error: {e}")
            self.send_message("stop_loading")

    def _handle_command(self, user_input: str):
        """Handle user commands."""
        command, *args = user_input.lower().split(' ', 1)
        
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
                self._verify_assumption()
            elif command == '/train':
                self.send_message("system_message", "To create a training file, run: python3 finetune/train.py")
            elif command == '/develop_insight':
                self._develop_insight(args)
            elif command == '/learn_procedure':
                self.send_message("system_message", "Interactive procedure learning not yet implemented in Bubble Tea UI")
            else:
                self.send_message("system_message", f"Unknown command: {command}")
        except Exception as e:
            self.send_message("system_message", f"Error executing command: {e}")
        finally:
            self.send_message("stop_loading")

    def _show_help(self):
        """Display help information."""
        help_text = """
╔═══════════════════════════════════════════════════════════════╗
║                   JENOVA COMMAND REFERENCE                    ║
╚═══════════════════════════════════════════════════════════════╝

COGNITIVE COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/help       - Display this help message
/insight    - Analyze conversation and generate insights
/reflect    - Deep reflection on cognitive architecture
/memory-insight - Search all memory layers for insights
/meta       - Generate higher-level meta-insights
/verify     - Verify assumptions about you

LEARNING COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/develop_insight [node_id] - Develop insights or process documents
/learn_procedure - Teach JENOVA a new procedure
/train      - Instructions for fine-tuning

SYSTEM COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

exit or quit - Exit the application
        """
        self.send_message("help", help_text)

    def _send_messages(self, messages):
        """Send multiple messages to the TUI."""
        if messages:
            if isinstance(messages, str):
                messages = [messages]
            for msg in messages:
                self.send_message("system_message", str(msg))

    def _verify_assumption(self):
        """Handle assumption verification."""
        assumption, question = self.engine.verify_assumptions(self.username)
        if question:
            self.send_message("system_message", f"Verification: {question}")
            # Note: Full verification flow would require more complex state management

    def _develop_insight(self, args):
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

    def run(self):
        """Start the TUI and process messages."""
        try:
            # Start the TUI process
            self.tui_process = subprocess.Popen(
                [self.tui_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.running = True
            
            # Start reading messages in a separate thread
            read_thread = threading.Thread(target=self.read_messages, daemon=True)
            read_thread.start()
            
            # Send initial banner
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
            
            # Process messages from TUI
            while self.running:
                try:
                    message = self.message_queue.get(timeout=0.1)
                    
                    if message.get("type") == "user_input":
                        user_input = message.get("content", "").strip()
                        
                        if user_input.lower() in ['exit', 'quit']:
                            break
                        
                        # Process in a separate thread to avoid blocking
                        threading.Thread(
                            target=self.process_user_input,
                            args=(user_input,),
                            daemon=True
                        ).start()
                    
                    elif message.get("type") == "exit":
                        break
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.info(f"Error processing message: {e}")
            
        except Exception as e:
            print(f"Error running TUI: {e}")
        finally:
            self.running = False
            if self.tui_process:
                self.tui_process.terminate()
                self.tui_process.wait()
