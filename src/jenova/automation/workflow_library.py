# Workflow templates
class WorkflowLibrary:
    def __init__(self):
        self.workflows = {}
    
    def get(self, name: str) -> dict:
        """Get workflow template."""
        return self.workflows.get(name)
