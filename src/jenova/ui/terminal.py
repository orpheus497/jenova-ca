import os
import getpass
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from jenova.ui.logger import UILogger
from jenova.cognitive_engine.engine import CognitiveEngine

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

    def run(self):
        self.logger.banner(BANNER, ATTRIBUTION)
        self.logger.info("Initialized and Ready.")
        self.logger.info("Type your message, use a command, or type 'exit' to quit.")
        self.logger.info("Commands: /insight, /reflect, /memory-insight\n")

        while True:
            try:
                prompt_message = [('class:username', self.username), ('class:at', '@'), ('class:hostname', 'Jenova'), ('class:prompt', '> ')]
                user_input = self.session.prompt(prompt_message, style=self.prompt_style).strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    break
                
                # Command Handling
                if user_input.startswith('/'):
                    command = user_input.lower()
                    if command == '/insight':
                        self.engine.develop_insights_from_conversation(self.username)
                    elif command == '/reflect':
                        self.engine.reflect_on_insights(self.username)
                    elif command == '/memory-insight':
                        self.engine.develop_insights_from_memory(self.username)
                    else:
                        self.logger.system_message(f"Unknown command: {command}")
                else:
                    # Regular conversation
                    response = self.engine.think(user_input, self.username)
                    self.logger.jenova_response(response)

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                self.logger.system_message(f"An unexpected error occurred in the UI loop: {e}")