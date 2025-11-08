# Background task management
class BackgroundTaskManager:
    def __init__(self):
        self.tasks = {}
    
    def start(self, command: list) -> int:
        """Start background task."""
        return len(self.tasks)
