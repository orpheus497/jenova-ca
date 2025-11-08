# The JENOVA Cognitive Architecture - Hooks Manager
# Copyright (c) 2024, orpheus497. All rights reserved.
# Licensed under the MIT License

"""Manage Git hooks and automated workflows."""

import os


class HooksManager:
    """Manage pre-commit and post-commit hooks."""

    def __init__(self, repo_path: str, ui_logger=None, file_logger=None):
        self.repo_path = repo_path
        self.hooks_dir = os.path.join(repo_path, ".git", "hooks")
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def install_hook(self, hook_name: str, script: str) -> bool:
        """Install a git hook script."""
        hook_path = os.path.join(self.hooks_dir, hook_name)

        try:
            with open(hook_path, "w") as f:
                f.write("#!/bin/sh\n")
                f.write(script)

            os.chmod(hook_path, 0o755)
            return True

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error installing hook: {e}")
            return False

    def remove_hook(self, hook_name: str) -> bool:
        """Remove a git hook."""
        hook_path = os.path.join(self.hooks_dir, hook_name)

        try:
            if os.path.exists(hook_path):
                os.remove(hook_path)
            return True

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error removing hook: {e}")
            return False

    def list_hooks(self) -> list:
        """List installed hooks."""
        if not os.path.exists(self.hooks_dir):
            return []

        hooks = []
        for name in os.listdir(self.hooks_dir):
            path = os.path.join(self.hooks_dir, name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                hooks.append(name)

        return hooks
