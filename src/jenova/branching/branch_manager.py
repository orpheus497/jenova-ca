# The JENOVA Cognitive Architecture - Branch Manager
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 29: Branch Manager - Manages conversation branching and navigation.

Tracks conversation branches, allows creating new branches from any point,
switching between branches, and maintaining branch tree structure.
100% offline, local state management.
"""

import time
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class ConversationTurn:
    """
    Single conversation turn.

    Attributes:
        turn_id: Unique turn identifier
        timestamp: Unix timestamp
        user_message: User's message
        assistant_response: Assistant's response
        metadata: Additional metadata (tokens, duration, etc.)
    """

    turn_id: str
    timestamp: float
    user_message: str
    assistant_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Branch:
    """
    Conversation branch.

    Attributes:
        branch_id: Unique branch identifier
        parent_branch_id: Parent branch ID (None for root)
        branch_point_turn: Turn ID where branch diverged
        name: Human-readable branch name
        description: Branch description
        created_at: Creation timestamp
        turns: List of conversation turns in this branch
    """

    branch_id: str
    parent_branch_id: Optional[str]
    branch_point_turn: Optional[str]
    name: str
    description: str
    created_at: float
    turns: List[ConversationTurn] = field(default_factory=list)


class BranchManager:
    """
    Manage conversation branching and navigation.

    Allows creating branches from any conversation point, switching between
    branches, and maintaining complete branch tree structure.

    Example:
        >>> manager = BranchManager()
        >>> manager.add_turn("user msg", "assistant response")
        >>> branch_id = manager.create_branch("Explore alternative", turn_id)
        >>> manager.switch_branch(branch_id)
    """

    def __init__(self):
        """Initialize branch manager."""
        # All branches (branch_id -> Branch)
        self.branches: Dict[str, Branch] = {}

        # Current active branch
        self.current_branch_id: str = "main"

        # Create main/root branch
        self.branches["main"] = Branch(
            branch_id="main",
            parent_branch_id=None,
            branch_point_turn=None,
            name="Main",
            description="Main conversation branch",
            created_at=time.time(),
            turns=[],
        )

        # Turn counter for unique IDs
        self._turn_counter = 0

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add turn to current branch.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            metadata: Optional metadata

        Returns:
            Turn ID

        Example:
            >>> turn_id = manager.add_turn("Hello", "Hi there!")
        """
        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter}"

        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=time.time(),
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata or {},
        )

        current_branch = self.branches[self.current_branch_id]
        current_branch.turns.append(turn)

        return turn_id

    def create_branch(
        self,
        name: str,
        branch_from_turn: Optional[str] = None,
        description: str = ""
    ) -> str:
        """
        Create new branch from specified turn.

        Args:
            name: Branch name
            branch_from_turn: Turn ID to branch from (None = current end)
            description: Branch description

        Returns:
            New branch ID

        Example:
            >>> branch_id = manager.create_branch(
            ...     "Alternative path",
            ...     turn_id,
            ...     "Exploring different approach"
            ... )
        """
        # Generate unique branch ID
        branch_id = f"branch_{int(time.time() * 1000)}"

        # Get current branch
        current_branch = self.branches[self.current_branch_id]

        # Determine branch point
        if branch_from_turn is None:
            # Branch from end of current branch
            if current_branch.turns:
                branch_point = current_branch.turns[-1].turn_id
            else:
                branch_point = None
        else:
            branch_point = branch_from_turn

        # Copy turns up to branch point
        copied_turns = []
        if branch_point:
            for turn in current_branch.turns:
                copied_turns.append(turn)
                if turn.turn_id == branch_point:
                    break

        # Create new branch
        new_branch = Branch(
            branch_id=branch_id,
            parent_branch_id=self.current_branch_id,
            branch_point_turn=branch_point,
            name=name,
            description=description,
            created_at=time.time(),
            turns=copied_turns,
        )

        self.branches[branch_id] = new_branch

        return branch_id

    def switch_branch(self, branch_id: str) -> bool:
        """
        Switch to different branch.

        Args:
            branch_id: Target branch ID

        Returns:
            True if switched successfully

        Example:
            >>> manager.switch_branch("branch_12345")
        """
        if branch_id not in self.branches:
            return False

        self.current_branch_id = branch_id
        return True

    def get_current_branch(self) -> Branch:
        """
        Get current active branch.

        Returns:
            Current Branch object

        Example:
            >>> branch = manager.get_current_branch()
            >>> print(branch.name, len(branch.turns))
        """
        return self.branches[self.current_branch_id]

    def get_branch(self, branch_id: str) -> Optional[Branch]:
        """
        Get branch by ID.

        Args:
            branch_id: Branch ID

        Returns:
            Branch object or None

        Example:
            >>> branch = manager.get_branch("main")
        """
        return self.branches.get(branch_id)

    def list_branches(self) -> List[Dict[str, Any]]:
        """
        List all branches with metadata.

        Returns:
            List of branch summaries

        Example:
            >>> branches = manager.list_branches()
            >>> for b in branches:
            ...     print(b["name"], b["turn_count"])
        """
        summaries = []

        for branch_id, branch in self.branches.items():
            summaries.append({
                "branch_id": branch_id,
                "name": branch.name,
                "description": branch.description,
                "parent_branch_id": branch.parent_branch_id,
                "branch_point_turn": branch.branch_point_turn,
                "turn_count": len(branch.turns),
                "created_at": branch.created_at,
                "is_current": branch_id == self.current_branch_id,
            })

        return summaries

    def get_branch_tree(self) -> Dict[str, Any]:
        """
        Get branch tree structure.

        Returns:
            Nested dict representing branch hierarchy

        Example:
            >>> tree = manager.get_branch_tree()
            >>> print(json.dumps(tree, indent=2))
        """
        def build_tree(branch_id: str) -> Dict[str, Any]:
            branch = self.branches[branch_id]

            # Find child branches
            children = [
                build_tree(bid)
                for bid, b in self.branches.items()
                if b.parent_branch_id == branch_id
            ]

            return {
                "branch_id": branch_id,
                "name": branch.name,
                "turn_count": len(branch.turns),
                "is_current": branch_id == self.current_branch_id,
                "children": children,
            }

        return build_tree("main")

    def get_conversation_history(
        self,
        branch_id: Optional[str] = None,
        max_turns: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Get conversation history for branch.

        Args:
            branch_id: Branch ID (None = current)
            max_turns: Maximum turns to return (None = all)

        Returns:
            List of conversation turns

        Example:
            >>> history = manager.get_conversation_history(max_turns=10)
        """
        if branch_id is None:
            branch_id = self.current_branch_id

        branch = self.branches.get(branch_id)
        if not branch:
            return []

        turns = branch.turns
        if max_turns:
            turns = turns[-max_turns:]

        return turns

    def delete_branch(self, branch_id: str, delete_children: bool = False) -> bool:
        """
        Delete branch.

        Args:
            branch_id: Branch to delete
            delete_children: Also delete child branches

        Returns:
            True if deleted successfully

        Example:
            >>> manager.delete_branch("branch_12345", delete_children=True)
        """
        if branch_id == "main":
            return False  # Cannot delete main branch

        if branch_id not in self.branches:
            return False

        # Cannot delete current branch
        if branch_id == self.current_branch_id:
            return False

        # Find children
        children = [
            bid for bid, b in self.branches.items()
            if b.parent_branch_id == branch_id
        ]

        if children and not delete_children:
            # Reparent children to deleted branch's parent
            deleted_parent = self.branches[branch_id].parent_branch_id
            for child_id in children:
                self.branches[child_id].parent_branch_id = deleted_parent
        elif children and delete_children:
            # Recursively delete children
            for child_id in children:
                self.delete_branch(child_id, delete_children=True)

        # Delete branch
        del self.branches[branch_id]
        return True

    def save_to_file(self, file_path: Path) -> None:
        """
        Save branch state to file.

        Args:
            file_path: Output file path

        Example:
            >>> manager.save_to_file(Path("branches.json"))
        """
        data = {
            "current_branch_id": self.current_branch_id,
            "turn_counter": self._turn_counter,
            "branches": {
                branch_id: {
                    "branch_id": branch.branch_id,
                    "parent_branch_id": branch.parent_branch_id,
                    "branch_point_turn": branch.branch_point_turn,
                    "name": branch.name,
                    "description": branch.description,
                    "created_at": branch.created_at,
                    "turns": [asdict(turn) for turn in branch.turns],
                }
                for branch_id, branch in self.branches.items()
            },
        }

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: Path) -> bool:
        """
        Load branch state from file.

        Args:
            file_path: Input file path

        Returns:
            True if loaded successfully

        Example:
            >>> manager.load_from_file(Path("branches.json"))
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Restore branches
            self.branches = {}
            for branch_id, branch_data in data["branches"].items():
                turns = [
                    ConversationTurn(**turn_data)
                    for turn_data in branch_data["turns"]
                ]

                self.branches[branch_id] = Branch(
                    branch_id=branch_data["branch_id"],
                    parent_branch_id=branch_data["parent_branch_id"],
                    branch_point_turn=branch_data["branch_point_turn"],
                    name=branch_data["name"],
                    description=branch_data["description"],
                    created_at=branch_data["created_at"],
                    turns=turns,
                )

            # Restore state
            self.current_branch_id = data["current_branch_id"]
            self._turn_counter = data["turn_counter"]

            return True

        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get branch manager statistics.

        Returns:
            Statistics dictionary

        Example:
            >>> stats = manager.get_stats()
            >>> print(stats["total_branches"], stats["total_turns"])
        """
        total_turns = sum(len(b.turns) for b in self.branches.values())

        return {
            "total_branches": len(self.branches),
            "total_turns": total_turns,
            "current_branch": self.current_branch_id,
            "current_branch_turns": len(self.branches[self.current_branch_id].turns),
        }
