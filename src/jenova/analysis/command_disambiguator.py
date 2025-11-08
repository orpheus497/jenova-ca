# Command disambiguation
class CommandDisambiguator:
    def disambiguate(self, command: str, options: list) -> str:
        """Disambiguate ambiguous command."""
        return options[0] if options else command
