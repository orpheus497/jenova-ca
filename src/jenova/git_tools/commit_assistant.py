# The JENOVA Cognitive Architecture - Commit Assistant
# Copyright (c) 2024, orpheus497. All rights reserved.
# Licensed under the MIT License

"""Auto-generate commit messages from changes."""

class CommitAssistant:
    """Generate commit messages from diffs using LLM."""

    def __init__(self, git_interface, llm_interface=None, ui_logger=None, file_logger=None):
        self.git = git_interface
        self.llm = llm_interface
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def generate_message(self, diff_text: str = None) -> str:
        """Generate commit message from diff."""
        if diff_text is None:
            diff_text = self.git.diff(staged=True)

        if not diff_text:
            return "No staged changes"

        # Analyze diff
        lines_added = diff_text.count('\n+')
        lines_removed = diff_text.count('\n-')

        # Generate message using LLM if available
        if self.llm:
            prompt = f"Generate a concise commit message for these changes:\n\n{diff_text[:2000]}"
            try:
                message = self.llm.generate(prompt, max_tokens=100)
                return message.strip()
            except:
                pass

        # Fallback: simple analysis
        if "def " in diff_text or "class " in diff_text:
            return f"Add new code (+{lines_added} -{lines_removed} lines)"
        elif lines_added > lines_removed:
            return f"Add functionality (+{lines_added} lines)"
        elif lines_removed > lines_added:
            return f"Remove code (-{lines_removed} lines)"
        else:
            return f"Update code (~{lines_added} lines)"
