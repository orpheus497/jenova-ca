# Hooks system
class HooksSystem:
    def __init__(self):
        self.hooks = {'pre': {}, 'post': {}}
    
    def register(self, event: str, callback, when='post'):
        """Register hook callback."""
        self.hooks[when][event] = callback
