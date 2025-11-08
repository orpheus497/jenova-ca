# The JENOVA Cognitive Architecture - Branch Manager
# Copyright (c) 2024, orpheus497. All rights reserved.
# Licensed under the MIT License

"""Git branch operations and naming."""


class BranchManager:
    """Manage Git branches."""

    def __init__(self, git_interface, ui_logger=None, file_logger=None):
        self.git = git_interface
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def suggest_name(self, description: str) -> str:
        """Suggest branch name from description."""
        # Clean and format description
        name = description.lower()
        name = name.replace(" ", "-")
        # Remove special characters
        import re

        name = re.sub(r"[^a-z0-9-]", "", name)
        # Limit length
        if len(name) > 50:
            name = name[:50]

        return f"feature/{name}"

    def create_and_checkout(self, name: str) -> bool:
        """Create branch and check it out."""
        if self.git.create_branch(name):
            return self.git.checkout(name)
        return False

    def merge(self, branch_name: str, message: str = None) -> bool:
        """Merge branch into current branch."""
        if not self.git.repo:
            return False

        try:
            self.git.repo.git.merge(
                branch_name, m=message or f"Merge branch '{branch_name}'"
            )
            return True
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error merging: {e}")
            return False

    def delete(self, branch_name: str, force: bool = False) -> bool:
        """Delete branch."""
        if not self.git.repo:
            return False

        try:
            if force:
                self.git.repo.delete_head(branch_name, force=True)
            else:
                self.git.repo.delete_head(branch_name)
            return True
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error deleting branch: {e}")
            return False
