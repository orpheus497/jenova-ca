# The JENOVA Cognitive Architecture - Git Interface
# Copyright (c) 2024, orpheus497. All rights reserved.
# Licensed under the MIT License

"""Git operations wrapper using GitPython."""

from typing import Optional, List, Dict
import os


class GitInterface:
    """Git operations interface using GitPython library."""

    def __init__(self, repo_path: str = ".", ui_logger=None, file_logger=None):
        self.repo_path = os.path.abspath(repo_path)
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.repo = None
        self._init_repo()

    def _init_repo(self):
        """Initialize git repository."""
        try:
            from git import Repo, InvalidGitRepositoryError
            self.repo = Repo(self.repo_path)
        except ImportError:
            if self.file_logger:
                self.file_logger.log_error("gitpython not installed")
        except InvalidGitRepositoryError:
            if self.file_logger:
                self.file_logger.log_error(f"Not a git repository: {self.repo_path}")

    def status(self) -> Dict:
        """Get repository status."""
        if not self.repo:
            return {"error": "Repository not initialized"}

        try:
            return {
                "branch": self.repo.active_branch.name,
                "modified": [item.a_path for item in self.repo.index.diff(None)],
                "staged": [item.a_path for item in self.repo.index.diff("HEAD")],
                "untracked": self.repo.untracked_files,
                "ahead": len(list(self.repo.iter_commits('origin/HEAD..HEAD'))) if 'origin/HEAD' in self.repo.refs else 0
            }
        except Exception as e:
            return {"error": str(e)}

    def diff(self, staged: bool = False) -> str:
        """Get diff output."""
        if not self.repo:
            return "Error: Repository not initialized"

        try:
            if staged:
                return self.repo.git.diff('--cached')
            return self.repo.git.diff()
        except Exception as e:
            return f"Error: {e}"

    def log(self, max_count: int = 10) -> List[Dict]:
        """Get commit log."""
        if not self.repo:
            return []

        try:
            commits = []
            for commit in self.repo.iter_commits(max_count=max_count):
                commits.append({
                    "hash": commit.hexsha[:8],
                    "author": str(commit.author),
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message.strip()
                })
            return commits
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error getting log: {e}")
            return []

    def add(self, files: List[str]) -> bool:
        """Stage files."""
        if not self.repo:
            return False

        try:
            self.repo.index.add(files)
            return True
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error staging files: {e}")
            return False

    def commit(self, message: str) -> Optional[str]:
        """Create commit."""
        if not self.repo:
            return None

        try:
            commit = self.repo.index.commit(message)
            return commit.hexsha[:8]
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error committing: {e}")
            return None

    def branch_list(self) -> List[str]:
        """List all branches."""
        if not self.repo:
            return []

        try:
            return [branch.name for branch in self.repo.branches]
        except Exception as e:
            return []

    def create_branch(self, name: str) -> bool:
        """Create new branch."""
        if not self.repo:
            return False

        try:
            self.repo.create_head(name)
            return True
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error creating branch: {e}")
            return False

    def checkout(self, branch_name: str) -> bool:
        """Checkout branch."""
        if not self.repo:
            return False

        try:
            self.repo.git.checkout(branch_name)
            return True
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error checking out branch: {e}")
            return False
