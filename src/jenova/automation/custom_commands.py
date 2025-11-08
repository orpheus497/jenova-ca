# Custom command loader
import os
class CustomCommandManager:
    def __init__(self, commands_dir=".jenova/commands"):
        self.commands_dir = commands_dir
        self.commands = {}
    
    def load_command(self, name: str) -> str:
        """Load custom command template."""
        path = os.path.join(self.commands_dir, f"{name}.md")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return None
