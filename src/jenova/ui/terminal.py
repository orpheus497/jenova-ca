import os
import getpass
import threading
import itertools
import time
import sys
import queue
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
        
        # Use the existing message queue from logger (already set in main.py)
        self.message_queue = self.logger.message_queue

    def _spinner(self):
        spinner_chars = itertools.cycle(['   ', '.  ', '.. ', '...'])
        color_code = '\033[93m' # Yellow color
        reset_code = '\033[0m'
        while self._spinner_running:
            # No lock needed - spinner runs independently and TerminalUI processes queue
            sys.stdout.write(f'{color_code}\r{next(spinner_chars)}{reset_code}')
            sys.stdout.flush()
            time.sleep(0.2)
        # Clear spinner line completely
        sys.stdout.write('\r' + ' ' * 5 + '\r')
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
        # Process startup messages
        self.logger.process_queued_messages()

        while True:
            try:
                if self.verifying_assumption:
                    prompt_message = [('class:prompt', 'Your answer (yes/no): ')]
                    user_input = self.session.prompt(prompt_message, style=self.prompt_style).strip()
                    self.engine.assumption_manager.resolve_assumption(self.verifying_assumption, user_input, self.username)
                    self.verifying_assumption = None
                    self.logger.system_message("") # Add line space after system message
                    self.logger.process_queued_messages()
                    continue

                prompt_message = [('class:username', self.username), ('class:at', '@'), ('class:hostname', 'JENOVA'), ('class:prompt', '> ')]
                user_input = self.session.prompt(prompt_message, style=self.prompt_style).strip()

                if not user_input:
                    self.logger.system_message("") # Add line space after system message
                    self.logger.process_queued_messages()
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    break
                
                # Command Handling
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    self.logger.system_message("") # Add line space after system message
                    self.logger.process_queued_messages()
                else:
                    # Regular conversation - run in background thread with spinner
                    self.start_spinner()
                    response_container = []
                    def task():
                        response_container.append(self.engine.think(user_input, self.username))
                    
                    thread = threading.Thread(target=task)
                    thread.start()
                    
                    # Process queue messages while waiting
                    while thread.is_alive():
                        self.logger.process_queued_messages()
                        time.sleep(0.1)
                    thread.join()
                    
                    self.stop_spinner()
                    
                    # Get response
                    response = response_container[0] if response_container else "Error: No response generated"
                    
                    # Process any remaining messages
                    self.logger.process_queued_messages()
                    
                    if not isinstance(response, str):
                        response = str(response)
                    self.logger.system_message("") # Add line space before AI output
                    self.logger.jenova_response(response)
                    # Process the jenova_response message
                    self.logger.process_queued_messages()
                    self.logger.system_message("") # Add line space after AI output

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                self.logger.system_message(f"An unexpected error occurred in the UI loop: {e}")
                self.logger.process_queued_messages()

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
        
        # Start spinner for long-running operations
        should_spin = command in ['/insight', '/reflect', '/memory-insight', '/meta']
        
        if should_spin:
            self.start_spinner()
        
        try:
            if command == '/insight':
                # Run in background thread
                result_container = []
                def task():
                    result_container.append(self.engine.develop_insights_from_conversation(self.username))
                
                thread = threading.Thread(target=task)
                thread.start()
                
                # Process queue messages while waiting
                while thread.is_alive():
                    self.logger.process_queued_messages()
                    time.sleep(0.1)
                thread.join()
                
                returned_messages = result_container[0] if result_container else []
                
            elif command == '/reflect':
                result_container = []
                def task():
                    result_container.append(self.engine.reflect_on_insights(self.username))
                
                thread = threading.Thread(target=task)
                thread.start()
                
                # Process queue messages while waiting
                while thread.is_alive():
                    self.logger.process_queued_messages()
                    time.sleep(0.1)
                thread.join()
                
                returned_messages = result_container[0] if result_container else []
                
            elif command == '/memory-insight':
                result_container = []
                def task():
                    result_container.append(self.engine.develop_insights_from_memory(self.username))
                
                thread = threading.Thread(target=task)
                thread.start()
                
                # Process queue messages while waiting
                while thread.is_alive():
                    self.logger.process_queued_messages()
                    time.sleep(0.1)
                thread.join()
                
                returned_messages = result_container[0] if result_container else []
                
            elif command == '/meta':
                result_container = []
                def task():
                    result_container.append(self.engine.generate_meta_insight(self.username))
                
                thread = threading.Thread(target=task)
                thread.start()
                
                # Process queue messages while waiting
                while thread.is_alive():
                    self.logger.process_queued_messages()
                    time.sleep(0.1)
                thread.join()
                
                returned_messages = result_container[0] if result_container else []
                
            elif command == '/verify':
                self._verify_assumption()
            elif command == '/train':
                self.logger.system_message("To create fine-tuning data from your insights:")
                self.logger.system_message("  Prepare data only: python3 finetune/train.py --prepare-only")
                self.logger.system_message("  Full LoRA fine-tune: python3 finetune/train.py --epochs 3 --batch-size 4")
                self.logger.system_message("Note: Fine-tuning with LoRA requires GPU and additional packages (peft, bitsandbytes)")
            elif command == '/develop_insight':
                self._develop_insight(args)
            elif command == '/learn_procedure':
                self._learn_procedure(args)
            elif command == '/help':
                self._show_help()
            else:
                self.logger.system_message(f"Unknown command: {command}")
        finally:
            if should_spin:
                self.stop_spinner()
            # Process any remaining queued messages
            self.logger.process_queued_messages()

        self._process_and_log_messages(returned_messages)

    def _show_help(self):
        """Displays a detailed list of available commands and their functions."""
        self.logger.help_message("\n[bold bright_cyan]╔═══════════════════════════════════════════════════════════════════════════════╗[/bold bright_cyan]")
        self.logger.help_message("[bold bright_cyan]║                        JENOVA COMMAND REFERENCE                               ║[/bold bright_cyan]")
        self.logger.help_message("[bold bright_cyan]╚═══════════════════════════════════════════════════════════════════════════════╝[/bold bright_cyan]\n")
        
        self.logger.help_message("[bold bright_yellow]COGNITIVE COMMANDS[/bold bright_yellow]")
        self.logger.help_message("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
        
        self.logger.help_message("  [bright_yellow]/help[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Displays this comprehensive command reference guide.[/]")
        self.logger.help_message("    [dim italic]Shows all available commands with detailed descriptions.[/dim italic]\n")
        
        self.logger.help_message("  [bright_yellow]/insight[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Analyzes the current conversation and generates new insights.[/]")
        self.logger.help_message("    [dim italic]JENOVA will extract key takeaways from your recent interactions and[/dim italic]")
        self.logger.help_message("    [dim italic]store them as structured insights in long-term memory, contributing[/dim italic]")
        self.logger.help_message("    [dim italic]to its evolving understanding of topics and patterns.[/dim italic]\n")
        
        self.logger.help_message("  [bright_yellow]/reflect[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Initiates deep reflection within The JENOVA Cognitive Architecture.[/]")
        self.logger.help_message("    [dim italic]This powerful command reorganizes and interlinks all cognitive nodes[/dim italic]")
        self.logger.help_message("    [dim italic](insights, memories, assumptions), identifies patterns, and generates[/dim italic]")
        self.logger.help_message("    [dim italic]higher-level meta-insights, significantly enhancing intelligence and[/dim italic]")
        self.logger.help_message("    [dim italic]cognitive coherence across the entire knowledge graph.[/dim italic]\n")
        
        self.logger.help_message("  [bright_yellow]/memory-insight[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Performs a comprehensive search across all memory layers.[/]")
        self.logger.help_message("    [dim italic]JENOVA will scan episodic, semantic, and procedural memory to[/dim italic]")
        self.logger.help_message("    [dim italic]develop new insights or assumptions based on accumulated knowledge[/dim italic]")
        self.logger.help_message("    [dim italic]from past interactions and learned information.[/dim italic]\n")
        
        self.logger.help_message("  [bright_yellow]/meta[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Generates higher-level meta-insights from existing knowledge.[/]")
        self.logger.help_message("    [dim italic]Analyzes clusters of related insights within the cognitive graph[/dim italic]")
        self.logger.help_message("    [dim italic]to form abstract conclusions and identify overarching themes,[/dim italic]")
        self.logger.help_message("    [dim italic]enabling more sophisticated pattern recognition and understanding.[/dim italic]\n")
        
        self.logger.help_message("  [bright_yellow]/verify[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Starts the assumption verification process.[/]")
        self.logger.help_message("    [dim italic]JENOVA will present an unverified assumption about your preferences[/dim italic]")
        self.logger.help_message("    [dim italic]or knowledge and ask for clarification, allowing you to confirm or[/dim italic]")
        self.logger.help_message("    [dim italic]deny it. This refines JENOVA's understanding of your context.[/dim italic]\n")
        
        self.logger.help_message("[bold bright_yellow]LEARNING COMMANDS[/bold bright_yellow]")
        self.logger.help_message("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
        
        self.logger.help_message("  [bright_yellow]/develop_insight[/bright_yellow] [dim][node_id][/dim]")
        self.logger.help_message("    [#BDB2FF]Dual-purpose insight development and document learning command.[/]")
        self.logger.help_message("    [dim italic]• With node_id: Expands an existing insight with more context[/dim italic]")
        self.logger.help_message("    [dim italic]  and connections, generating a more detailed version.[/dim italic]")
        self.logger.help_message("    [dim italic]• Without node_id: Scans the src/jenova/docs directory for new[/dim italic]")
        self.logger.help_message("    [dim italic]  or updated documents, processes their content, and integrates[/dim italic]")
        self.logger.help_message("    [dim italic]  insights and summaries into the cognitive graph, enabling[/dim italic]")
        self.logger.help_message("    [dim italic]  learning from external documentation.[/dim italic]\n")
        
        self.logger.help_message("  [bright_yellow]/learn_procedure[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Interactive guided process to teach JENOVA a new procedure.[/]")
        self.logger.help_message("    [dim italic]JENOVA will prompt you for the procedure's name, individual steps,[/dim italic]")
        self.logger.help_message("    [dim italic]and expected outcome. This structured approach ensures comprehensive[/dim italic]")
        self.logger.help_message("    [dim italic]intake of procedural knowledge, stored in procedural memory for[/dim italic]")
        self.logger.help_message("    [dim italic]future recall and application in relevant contexts.[/dim italic]\n")
        
        self.logger.help_message("  [bright_yellow]/train[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Provides instructions for creating LoRA fine-tuning training data.[/]")
        self.logger.help_message("    [dim italic]Shows how to generate a training dataset from your insights and[/dim italic]")
        self.logger.help_message("    [dim italic]fine-tune TinyLlama with LoRA for personalized knowledge.[/dim italic]\n")
        
        self.logger.help_message("[bold bright_yellow]SYSTEM COMMANDS[/bold bright_yellow]")
        self.logger.help_message("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
        
        self.logger.help_message("  [bright_yellow]/exit[/bright_yellow] [dim]or[/dim] [bright_yellow]quit[/bright_yellow]")
        self.logger.help_message("    [#BDB2FF]Exits The JENOVA Cognitive Architecture application.[/]")
        self.logger.help_message("    [dim italic]All current session data will be automatically saved.[/dim italic]\n")
        
        self.logger.help_message("[bold bright_yellow]INNATE CAPABILITIES[/bold bright_yellow]")
        self.logger.help_message("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
        
        self.logger.help_message("  [#BDB2FF]JENOVA can intelligently use its built-in tools to answer your[/]")
        self.logger.help_message("  [#BDB2FF]questions without requiring specific commands. For example:[/]\n")
        self.logger.help_message("    [dim]• Ask for the current time[/dim]")
        self.logger.help_message("    [dim]• Request file operations in the sandbox directory (~/jenova_files)[/dim]")
        self.logger.help_message("    [dim]• Perform calculations and data analysis[/dim]")
        self.logger.help_message("    [dim]• Access and process information from its knowledge graph[/dim]\n")
        
        self.logger.help_message("[bold bright_cyan]╔═══════════════════════════════════════════════════════════════════════════════╗[/bold bright_cyan]")
        self.logger.help_message("[bold bright_cyan]║  Tip: Commands are not stored in conversational memory and can be used       ║[/bold bright_cyan]")
        self.logger.help_message("[bold bright_cyan]║  freely to manage JENOVA's cognitive processes.                              ║[/bold bright_cyan]")
        self.logger.help_message("[bold bright_cyan]╚═══════════════════════════════════════════════════════════════════════════════╝[/bold bright_cyan]\n")

    def _verify_assumption(self):
        """Handles the /verify command."""
        self.start_spinner()
        
        result_container = []
        def task():
            result_container.append(self.engine.verify_assumptions(self.username))
        
        thread = threading.Thread(target=task)
        thread.start()
        
        # Process queue messages while waiting
        while thread.is_alive():
            self.logger.process_queued_messages()
            time.sleep(0.1)
        thread.join()
        
        self.stop_spinner()
        self.logger.process_queued_messages()
        
        assumption, question = result_container[0] if result_container else (None, None)
        
        if question: # Check if there's a message to display
            self.logger.system_message(f"Jenova is asking for clarification: {question}")
            self.logger.process_queued_messages()
            if assumption: # Only set verifying_assumption if there's an actual assumption
                self.verifying_assumption = assumption

    def _develop_insight(self, args: list):
        """Handles the /develop_insight command."""
        self.start_spinner()
        
        result_container = []
        def task():
            if args:
                node_id = args[0]
                result_container.append(self.engine.cortex.develop_insight(node_id, self.username))
            else:
                result_container.append(self.engine.cortex.develop_insights_from_docs(self.username))
        
        thread = threading.Thread(target=task)
        thread.start()
        
        # Process queue messages while waiting
        while thread.is_alive():
            self.logger.process_queued_messages()
            time.sleep(0.1)
        thread.join()
        
        self.stop_spinner()
        self.logger.process_queued_messages()
        
        returned_messages = result_container[0] if result_container else []
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
        
        result_container = []
        def task():
            result_container.append(self.engine.learn_procedure(procedure_data, self.username))
        
        thread = threading.Thread(target=task)
        thread.start()
        
        # Process queue messages while waiting
        while thread.is_alive():
            self.logger.process_queued_messages()
            time.sleep(0.1)
        thread.join()
        
        self.stop_spinner()
        self.logger.process_queued_messages()
        
        returned_messages = result_container[0] if result_container else []
        self._process_and_log_messages(returned_messages)