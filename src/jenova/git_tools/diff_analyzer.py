# The JENOVA Cognitive Architecture - Diff Analyzer
# Copyright (c) 2024, orpheus497. All rights reserved.
# Licensed under the MIT License

"""Analyze and summarize Git diffs."""


class DiffAnalyzer:
    """Analyze diffs and extract meaningful information."""

    def __init__(self, ui_logger=None, file_logger=None):
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def analyze(self, diff_text: str) -> dict:
        """Analyze diff and return statistics."""
        lines = diff_text.split("\n")

        stats = {"files_changed": 0, "additions": 0, "deletions": 0, "files": []}

        current_file = None

        for line in lines:
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3].lstrip("b/")
                    stats["files_changed"] += 1
                    stats["files"].append(
                        {"name": current_file, "additions": 0, "deletions": 0}
                    )

            elif line.startswith("+") and not line.startswith("+++"):
                stats["additions"] += 1
                if stats["files"]:
                    stats["files"][-1]["additions"] += 1

            elif line.startswith("-") and not line.startswith("---"):
                stats["deletions"] += 1
                if stats["files"]:
                    stats["files"][-1]["deletions"] += 1

        return stats

    def summarize(self, diff_text: str) -> str:
        """Generate human-readable summary."""
        stats = self.analyze(diff_text)

        summary = [
            f"Changed {stats['files_changed']} file(s)",
            f"+{stats['additions']} additions, -{stats['deletions']} deletions",
        ]

        return "\n".join(summary)
