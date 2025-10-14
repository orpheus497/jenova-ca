import os
import getpass
import threading
import itertools
import time
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from jenova.ui.logger import UILogger
from jenova.cognitive_engine.engine import CognitiveEngine
from jenova import tools

BANNER = """
     ██╗███████╗███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ 
     ██║██╔════╝████╗  ██║██╔═══██╗██║   ██║██╔══██╗
     ██║█████╗  ██╔██╗ ██║██║   ██║██║   ██║███████║
██   ██║██╔══╝  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
╚█████╔╝███████╗██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
 ╚════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
"""
ATTRIBUTION = "Designed and Developed by orpheus497 - https://github.com/orpheus497"

class TerminalUI:
    def __init__(self, cognitive_engine: CognitiveEngine, logger: UILogger):
        self.engine = cognitive_engine
        self.logger = logger
        self.username = getpass.getuser()
        history_path = os.path.join(self.engine.config['user_data_root'], ".jenova_history")
        self.session = PromptSession(history=FileHistory(history_path), auto_suggest=AutoSuggestFromHistory())
        self.prompt_style = Style.from_dict({'username': '#44ff44 bold', 'at': '#888888', 'hostname': '#ff00ff bold', 'prompt': '#888888'})
        self.verifying_assumption = None
        self._spinner_running = False
        self._spinner_thread = None

    def _spinner(self):
        spinner_chars = itertools.cycle(['   ', '.  ', '.. ', '...'])
        color_code = '\033[93m' # Yellow color
        reset_code = '\033[0m'
        while self._spinner_running:
            with self.logger._console_lock:
                sys.stdout.write(f'{color_code}\r{next(spinner_chars)}{reset_code}')
                sys.stdout.flush()
            time.sleep(0.2)
        with self.logger._console_lock:
            sys.stdout.write('\r' + ' ' * 5 + '\r') # Clear spinner line completely
            sys.stdout.flush()

    def start_spinner(self):
        self._spinner_running = True
        self._spinner_thread = threading.Thread(target=self._spinner)
        self._spinner_thread.daemon = True
        self._spinner_thread.start()

    def stop_spinner(self):
        self._spinner_running = False
        if self._spinner_thread and self._spinner_thread.is_alive():
            self._spinner_thread.join()
        self._spinner_thread = None



    def run(self):
        self.logger.banner(BANNER, ATTRIBUTION)
        self.logger.info("Initialized and Ready.")
        self.logger.info("Type your message, use a command, or type 'exit' to quit.")
        self.logger.info("Type /help to see a list of available commands.\n")

        while True:
            try:
                if self.verifying_assumption:
                    prompt_message = [('class:prompt', 'Your answer (yes/no): ')]
                    user_input = self.session.prompt(prompt_message, style=self.prompt_style).strip()
                    self.engine.assumption_manager.resolve_assumption(self.verifying_assumption, user_input, self.username)
                    self.verifying_assumption = None
                    self.logger.system_message("") # Add line space after system message
                    continue

                prompt_message = [('class:username', self.username), ('class:at', '@'), ('class:hostname', 'Jenova'), ('class:prompt', '> ')]
                user_input = self.session.prompt(prompt_message, style=self.prompt_style).strip()

                if not user_input:
                    self.logger.system_message("") # Add line space after system message
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    break
                
                # Command Handling
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    self.logger.system_message("") # Add line space after system message
                else:
                    # Regular conversation
                    response = self.engine.think(user_input, self.username)
                    if not isinstance(response, str):
                        response = str(response)
                    self.logger.system_message("") # Add line space before AI output
                    self.logger.jenova_response(response)
                    self.logger.system_message("") # Add line space after AI output

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                self.logger.system_message(f"An unexpected error occurred in the UI loop: {e}")

    def _process_and_log_messages(self, returned_messages):
        if returned_messages:
            processed_messages = []
            if isinstance(returned_messages, str):
                returned_messages = [returned_messages]
            
            if returned_messages:
                for msg in returned_messages:
                    if not isinstance(msg, str):
                        processed_messages.append(str(msg))
                    else:
                        processed_messages.append(msg)
                
                for msg in processed_messages:
                    self.logger.system_message(msg)

    def _handle_command(self, user_input: str):
        """Handles user commands."""
        command, *args = user_input.lower().split(' ', 1)
        
        returned_messages = []
        if command == '/insight':
            self.start_spinner()
            returned_messages = self.engine.develop_insights_from_conversation(self.username)
            self.stop_spinner()
        elif command == '/reflect':
            self.start_spinner()
            returned_messages = self.engine.reflect_on_insights(self.username)
            self.stop_spinner()
        elif command == '/memory-insight':
            self.start_spinner()
            returned_messages = self.engine.develop_insights_from_memory(self.username)
            self.stop_spinner()
        elif command == '/meta':
            self.start_spinner()
            returned_messages = self.engine.generate_meta_insight(self.username)
            self.stop_spinner()
        elif command == '/verify':
            self._verify_assumption()
        elif command == '/train':
            self.logger.system_message("To create a training file for fine-tuning, run the following command in your terminal: python3 finetune/train.py")
        elif command == '/develop_insight':
            self._develop_insight(args)
        elif command == '/learn_procedure':
            self._learn_procedure(args)
        elif command == '/help':
            self._show_help()
        else:
            self.logger.system_message(f"Unknown command: {command}")

        self._process_and_log_messages(returned_messages)

    def _show_help(self):
        """Displays a detailed list of available commands and their functions."""
        self.logger.help_message("\n[bright_yellow]╔══════════════════════════════════════════════════════════════════╗[/bright_yellow]")
        self.logger.help_message("[bright_yellow]║[/bright_yellow]           [bold bright_cyan]JENOVA AI v3.1.0 - COMMAND REFERENCE[/bold bright_cyan]              [bright_yellow]║[/bright_yellow]")
        self.logger.help_message("[bright_yellow]║[/bright_yellow]      [italic]Cognitive Process Accelerator (CPA) Active[/italic]           [bright_yellow]║[/bright_yellow]")
        self.logger.help_message("[bright_yellow]╚══════════════════════════════════════════════════════════════════╝[/bright_yellow]\n")
        
        self.logger.help_message("[bright_yellow]═══ CORE COMMANDS ═══[/bright_yellow]")
        self.logger.help_message("[bright_yellow]  /help[/bright_yellow]                            - [#BDB2FF]Displays this comprehensive help message, detailing each command's purpose and impact.[/]")
        self.logger.help_message("")
        
        self.logger.help_message("[bright_yellow]═══ INSIGHT & REFLECTION COMMANDS ═══[/bright_yellow]")
        self.logger.help_message("[bright_yellow]  /insight[/bright_yellow]                         - [#BDB2FF]Triggers the AI to analyze the current conversation history and generate new, high-quality insights. These insights are stored in Jenova's long-term memory and contribute to its evolving understanding.[/]")
        self.logger.help_message("")
        self.logger.help_message("[bright_yellow]  /reflect[/bright_yellow]                         - [#BDB2FF]Initiates a deep reflection process within Jenova's Cortex. This command reorganizes and interlinks all existing cognitive nodes (insights, memories, assumptions), identifies patterns, and generates higher-level meta-insights, significantly enhancing Jenova's overall intelligence and coherence.[/]")
        self.logger.help_message("")
        self.logger.help_message("[bright_yellow]  /memory-insight[/bright_yellow]                  - [#BDB2FF]Prompts Jenova to perform a broad search across its multi-layered long-term memory (episodic, semantic, procedural) to develop new insights or assumptions based on its accumulated knowledge.[/]")
        self.logger.help_message("")
        self.logger.help_message("[bright_yellow]  /meta[/bright_yellow]                            - [#BDB2FF]Generates a new, higher-level meta-insight by analyzing clusters of existing insights within the Cortex. This helps Jenova to form more abstract conclusions and identify overarching themes.[/]")
        self.logger.help_message("")
        
        self.logger.help_message("[bright_yellow]═══ LEARNING & VERIFICATION COMMANDS ═══[/bright_yellow]")
        self.logger.help_message("[bright_yellow]  /verify[/bright_yellow]                          - [#BDB2FF]Starts the assumption verification process. Jenova will present an unverified assumption it has made about you and ask for clarification, allowing you to confirm or deny it. This refines Jenova's understanding of your preferences and knowledge.[/]")
        self.logger.help_message("")
        self.logger.help_message("[bright_yellow]  /develop_insight [node_id][/bright_yellow]       - [#BDB2FF]This command has dual functionality:[/]")
        self.logger.help_message("[#BDB2FF]                                       - If a `node_id` is provided: Jenova will take an existing insight and generate a more detailed and developed version of it, adding more context or connections.[/]")
        self.logger.help_message("[#BDB2FF]                                       - If no `node_id` is provided: Jenova will scan the `src/jenova/docs` directory for new or updated documents, process their content, and integrate new insights and summaries into its cognitive graph. This is how Jenova learns from external documentation.[/]")
        self.logger.help_message("")
        self.logger.help_message("[bright_yellow]  /learn_procedure[/bright_yellow]                 - [#BDB2FF]Initiates an interactive, guided process to teach Jenova a new procedure. Jenova will prompt you for the procedure's name, individual steps, and expected outcome, ensuring structured and comprehensive intake of procedural knowledge. This information is stored in Jenova's procedural memory, allowing it to recall and apply the procedure in relevant contexts.[/]")
        self.logger.help_message("")
        
        self.logger.help_message("[bright_yellow]═══ ADVANCED FEATURES ═══[/bright_yellow]")
        self.logger.help_message("[bright_yellow]  /train[/bright_yellow]                           - [#BDB2FF]Provides instructions on how to create a training file for fine-tuning the model with your own data.[/]")
        self.logger.help_message("")
        self.logger.help_message("[bright_yellow]  /exit[/bright_yellow] or [bright_yellow]/quit[/bright_yellow]                  - [#BDB2FF]Exits the Jenova AI application. All current session data, including CPA state (learned patterns, query history, hot functions, compilation history), will be automatically saved for continuity in future sessions.[/]")
        self.logger.help_message("")
        
        self.logger.help_message("[bright_yellow]═══ COGNITIVE PROCESS ACCELERATOR (CPA) ═══[/bright_yellow]")
        self.logger.help_message("[#BDB2FF]The CPA is ALWAYS ACTIVE, continuously optimizing performance:[/]")
        self.logger.help_message("[#BDB2FF]  • Persistent State: Automatically saves learned patterns every 5 minutes[/]")
        self.logger.help_message("[#BDB2FF]  • Proactive Caching: Pre-warms model for instant responses[/]")
        self.logger.help_message("[#BDB2FF]  • Profile-Guided JIT: Compiles hot functions with numba for speed[/]")
        self.logger.help_message("[#BDB2FF]  • Adaptive Timing: Adjusts optimization cycles (0.5s-8s) based on load[/]")
        self.logger.help_message("[#BDB2FF]  • Predictive Pre-Loading: Analyzes 50 recent queries for patterns[/]")
        self.logger.help_message("[#BDB2FF]  • Smart Memory: Prioritizes frequently accessed data[/]")
        self.logger.help_message("[#BDB2FF]  • Background Insights: Generates insights from conversation patterns[/]")
        self.logger.help_message("[#BDB2FF]  • Proactive Testing: Tests assumptions during idle time[/]")
        self.logger.help_message("[#BDB2FF]  • Deep Reflection: Analyzes thought patterns for themes[/]")
        self.logger.help_message("[#BDB2FF]  • Enhanced Prediction: Builds predictive models from query sequences[/]")
        self.logger.help_message("[#BDB2FF]  • 2.5x Faster: Base cycle reduced from 5s to 2s[/]")
        self.logger.help_message("[#BDB2FF]  • Memory Across Sessions: True continuity - AI remembers between runs[/]")
        self.logger.help_message("")
        
        self.logger.help_message("[bright_yellow]═══ JENOVA'S INNATE ABILITIES ═══[/bright_yellow]")
        self.logger.help_message("[#BDB2FF]Jenova can intelligently use its tools to answer your questions without[/]")
        self.logger.help_message("[#BDB2FF]needing specific commands. The AI is TRULY ALIVE - continuously thinking,[/]")
        self.logger.help_message("[#BDB2FF]learning, testing assumptions, and optimizing during idle time.[/]")
        self.logger.help_message("")
        self.logger.help_message("[#BDB2FF]Examples of innate capabilities:[/]")
        self.logger.help_message("[#BDB2FF]  • Ask for the current time, date, or system information[/]")
        self.logger.help_message("[#BDB2FF]  • Request file operations in the sandbox directory (~/.jenova-ai/)[/]")
        self.logger.help_message("[#BDB2FF]  • Search and retrieve information from the web (when available)[/]")
        self.logger.help_message("[#BDB2FF]  • Perform calculations and logical reasoning[/]")
        self.logger.help_message("[#BDB2FF]  • Analyze documents and extract insights[/]")
        self.logger.help_message("")
        self.logger.help_message("[bright_yellow]═══════════════════════════════════════════════════════════════════[/bright_yellow]\n")
        self.logger.help_message("[italic dim]CPA Status: Active | State: Persistent | Performance: Optimized[/italic dim]\n")

    def _verify_assumption(self):
        """Handles the /verify command."""
        self.start_spinner()
        assumption, question = self.engine.verify_assumptions(self.username)
        self.stop_spinner()
        if question: # Check if there's a message to display
            self.logger.system_message(f"Jenova is asking for clarification: {question}")
            if assumption: # Only set verifying_assumption if there's an actual assumption
                self.verifying_assumption = assumption

    def _develop_insight(self, args: list):
        """Handles the /develop_insight command."""
        if args:
            node_id = args[0]
            self.start_spinner()
            returned_messages = self.engine.cortex.develop_insight(node_id, self.username)
            self.stop_spinner()
            self._process_and_log_messages(returned_messages)
        else:
            self.start_spinner()
            returned_messages = self.engine.cortex.develop_insights_from_docs(self.username)
            self.stop_spinner()
            self._process_and_log_messages(returned_messages)



    def _learn_procedure(self, args: list):
        """Handles the /learn_procedure command interactively."""
        self.logger.system_message("Initiating interactive procedure learning...")
        
        procedure_name = self.session.prompt([('class:prompt', 'Procedure Name: ')]).strip()
        if not procedure_name:
            self.logger.system_message("Procedure name cannot be empty. Aborting.")
            return

        steps = []
        self.logger.system_message("Enter procedure steps one by one. Type 'done' when finished.")
        while True:
            step = self.session.prompt([('class:prompt', f'Step {len(steps) + 1}: ')]).strip()
            if step.lower() == 'done':
                break
            if step:
                steps.append(step)
        
        if not steps:
            self.logger.system_message("No steps entered. Aborting.")
            return

        expected_outcome = self.session.prompt([('class:prompt', 'Expected Outcome: ')]).strip()
        if not expected_outcome:
            self.logger.system_message("Expected outcome cannot be empty. Aborting.")
            return

        procedure_data = {
            "name": procedure_name,
            "steps": steps,
            "outcome": expected_outcome
        }

        self.start_spinner()
        returned_messages = self.engine.learn_procedure(procedure_data, self.username)
        self.stop_spinner()
        self._process_and_log_messages(returned_messages)