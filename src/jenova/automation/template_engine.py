# Template processing
import re
class TemplateEngine:
    def render(self, template: str, variables: dict) -> str:
        """Render template with variables."""
        for key, value in variables.items():
            template = re.sub(r'\{\{' + key + r'\}\}', str(value), template)
        return template
