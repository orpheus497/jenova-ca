# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for the terminal UI of the JENOVA Cognitive Architecture.
"""

import getpass
import itertools
import os

import sys
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.ui.logger import UILogger

BANNER = """
     ██╗███████╗███╗   ██╗ ██████╗ ██╗   ██╗ █████╗
     ██║██╔════╝████╗  ██║██╔═══██╗██║   ██║██╔══██╗
     ██║█████╗  ██╔██╗ ██║██║   ██║██║   ██║███████║
██   ██║██╔══╝  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
╚█████╔╝███████╗██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
 ╚════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝"""
ATTRIBUTION = "Designed and Developed by orpheus497 - https://github.com/orpheus497"

class TerminalUI:
    def __init__(self, cognitive_engine: CognitiveEngine, logger: UILogger):
        self.engine = cognitive_engine
        self.logger = logger
        self.username = getpass.getuser()
        history_path = os.path.join(
            self.engine.config['user_data_root'], ".jenova_history")
        self.session = PromptSession(history=FileHistory(
            history_path), auto_suggest=AutoSuggestFromHistory())
        self.prompt_style = Style.from_dict(
            {'username': '#44ff44 bold', 'at': '#888888', 'hostname': '#ff00ff bold', 'prompt': '#888888'})
        self.verifying_assumption = None
        self._spinner_running = False
        self._spinner_thread = None
        self.message_queue = self.logger.message_queue
        self.commands = self._register_commands()

    def _register_commands(self):
        return {
            '/insight': self._run_command_in_thread(self.engine.develop_insights_from_conversation, needs_spinner=True),
            '/reflect': self._run_command_in_thread(self.engine.reflect_on_insights, needs_spinner=True),
            '/memory-insight': self._run_command_in_thread(self.engine.develop_insights_from_memory, needs_spinner=True),
            '/meta': self._run_command_in_thread(self.engine.generate_meta_insight, needs_spinner=True),
            '/verify': self._verify_assumption,
            '/train': self._show_train_help,
            '/develop_insight': self._develop_insight,
            '/learn_procedure': self._learn_procedure,
            '/help': self._show_help,
        }

    def _spinner(self):
        spinner_chars = itertools.cycle(['   ', '.  ', '.. ', '...'])
        color_code = '\033[93m' # Yellow color
        reset_code = '\033[0m'
        while self._spinner_running:
            sys.stdout.write(f'{color_code}\r{next(spinner_chars)}{reset_code}')
            sys.stdout.flush()
            time.sleep(0.2)
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
        if self.logger:
            self.logger.banner(BANNER, ATTRIBUTION)
            self.logger.info("Initialized and Ready.")
            self.logger.info(
                "Type your message, use a command, or type 'exit' to quit.")
            self.logger.info(
                "Type /help to see a list of available commands.\n")
            self.logger.process_queued_messages()

        while True:
            try:
                if self.verifying_assumption:
                    prompt_message = [
                        ('class:prompt', 'Your answer (yes/no): ')]
                    user_input = self.session.prompt(
                        prompt_message, style=self.prompt_style).strip()
                    if self.engine:
                        self.engine.assumption_manager.resolve_assumption(
                            self.verifying_assumption, user_input, self.username)
                    self.verifying_assumption = None
                    if self.logger:
                        self.logger.system_message("")
                        self.logger.process_queued_messages()
                    continue

                prompt_message = [('class:username', self.username), ('class:at', '@'),
                                   ('class:hostname', 'JENOVA'), ('class:prompt', '> ')]
                user_input = self.session.prompt(
                    prompt_message, style=self.prompt_style).strip()

                if not user_input:
                    if self.logger:
                        self.logger.system_message("")
                        self.logger.process_queued_messages()
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    break

                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    self._handle_conversation(user_input)

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                if self.logger:
                    self.logger.system_message(
                        f"An unexpected error occurred in the UI loop: {e}")
                    self.logger.process_queued_messages()

    def _handle_command(self, user_input: str):
        command_name, *args = user_input.split(' ', 1)
        command_func = self.commands.get(command_name.lower())

        if command_func:
            command_func(args)
        else:
            if self.logger:
                self.logger.system_message(f"Unknown command: {command_name}")

        if self.logger:
            self.logger.system_message("")
            self.logger.process_queued_messages()

    def _handle_conversation(self, user_input: str):
        self.start_spinner()
        response_container = []
        def task():
            response_container.append(
                self.engine.think(user_input, self.username))

        thread = threading.Thread(target=task)
        thread.start()

        while thread.is_alive():
            if self.logger:
                self.logger.process_queued_messages()
            time.sleep(0.1)
        thread.join()

        self.stop_spinner()

        response = response_container[0] if response_container else "Error: No response generated"

        if self.logger:
            self.logger.process_queued_messages()

        if not isinstance(response, str):
            response = str(response)
        if self.logger:
            self.logger.system_message("")
            self.logger.jenova_response(response)
            self.logger.process_queued_messages()
            self.logger.system_message("")

    def _process_and_log_messages(self, returned_messages):
        if not returned_messages:
            return
        messages = returned_messages if isinstance(
            returned_messages, list) else [returned_messages]
        for msg in messages:
            if self.logger:
                self.logger.system_message(str(msg))

    def _run_command_in_thread(self, target_func, needs_spinner=False):
        def wrapper(args):
            if needs_spinner:
                self.start_spinner()

            result_container = []
            # Pass username for functions that need it
            func_args = (self.username,) + tuple(args)

            def task():
                result_container.append(target_func(*func_args))

            thread = threading.Thread(target=task)
            thread.start()

            while thread.is_alive():
                if self.logger:
                    self.logger.process_queued_messages()
                time.sleep(0.1)
            thread.join()

            if needs_spinner:
                self.stop_spinner()

            if self.logger:
                self.logger.process_queued_messages()
            returned_messages = result_container[0] if result_container else []
            self._process_and_log_messages(returned_messages)
        return wrapper


    def _verify_assumption(self, args):
        self.start_spinner()
        result_container = []
        def task():
            result_container.append(self.engine.verify_assumptions(self.username))
        
        thread = threading.Thread(target=task)
        thread.start()
        
        while thread.is_alive():
            if self.logger:
                self.logger.process_queued_messages()
            time.sleep(0.1)
        thread.join()
        
        self.stop_spinner()
        if self.logger:
            self.logger.process_queued_messages()
        
        assumption, question = result_container[0] if result_container else (None, None)
        
        if question:
            if self.logger:
                self.logger.system_message(f"Jenova is asking for clarification: {question}")
                self.logger.process_queued_messages()
            if assumption:
                self.verifying_assumption = assumption

    def _develop_insight(self, args):
        target_func = self.engine.cortex.develop_insight if args else self.engine.cortex.develop_insights_from_docs
        func_args = (args[0], self.username) if args else (self.username,)
        self._run_command_in_thread(target_func, needs_spinner=True)(func_args)

    def _learn_procedure(self, args):
        if self.logger:
            self.logger.system_message("Initiating interactive procedure learning...")
        
        procedure_name = self.session.prompt([('class:prompt', 'Procedure Name: ')]).strip()
        if not procedure_name:
            if self.logger:
                self.logger.system_message("Procedure name cannot be empty. Aborting.")
            return

        steps = []
        if self.logger:
            self.logger.system_message("Enter procedure steps one by one. Type 'done' when finished.")
        while True:
            step = self.session.prompt([('class:prompt', f'Step {len(steps) + 1}: ')]).strip()
            if step.lower() == 'done':
                break
            if step:
                steps.append(step)
        
        if not steps:
            if self.logger:
                self.logger.system_message("No steps entered. Aborting.")
            return

        expected_outcome = self.session.prompt([('class:prompt', 'Expected Outcome: ')]).strip()
        if not expected_outcome:
            if self.logger:
                self.logger.system_message("Expected outcome cannot be empty. Aborting.")
            return

        procedure_data = {"name": procedure_name, "steps": steps, "outcome": expected_outcome}
        self._run_command_in_thread(self.engine.learn_procedure, needs_spinner=True)((procedure_data, self.username))

    def _show_help(self, args):
        if self.logger:
            self.logger.system_message("JENOVA Available Commands:")
            self.logger.system_message("")
            self.logger.system_message("/help              - Show this help message")
            self.logger.system_message("/insight           - Develop insights from conversation history")
            self.logger.system_message("/reflect           - Reflect on existing insights")
            self.logger.system_message("/memory-insight    - Develop insights from memory")
            self.logger.system_message("/meta              - Generate meta-insight from all insights")
            self.logger.system_message("/verify            - Verify pending assumptions")
            self.logger.system_message("/develop_insight   - Develop insight from docs or specific input")
            self.logger.system_message("/learn_procedure   - Learn a new procedure interactively")
            self.logger.system_message("/train             - Show fine-tuning data generation help")
            self.logger.system_message("")

    def _show_train_help(self, args):
        if self.logger:
            self.logger.system_message("Fine-Tuning Data Generation")
            self.logger.system_message("")
            self.logger.system_message("To generate comprehensive training data from your cognitive architecture, run:")
            self.logger.system_message("  python finetune/train.py")
            self.logger.system_message("")
            self.logger.system_message("This creates 'finetune_train.jsonl' from all knowledge sources.")
            self.logger.system_message("Use this file with external fine-tuning tools like llama.cpp or Axolotl.")
            self.logger.system_message("")
            self.logger.system_message(
                "See finetune/README.md for detailed instructions.")
