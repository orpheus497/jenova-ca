# Conversation checkpoint management
import json
class CheckpointManager:
    def save(self, data: dict, path: str):
        """Save checkpoint."""
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> dict:
        """Load checkpoint."""
        with open(path, 'r') as f:
            return json.load(f)
