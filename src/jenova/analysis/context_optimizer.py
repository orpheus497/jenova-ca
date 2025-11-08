# Context window optimization
class ContextOptimizer:
    def optimize(self, context: str, max_tokens: int) -> str:
        """Optimize context to fit within token limit."""
        return context[:max_tokens*4]  # Rough estimate
