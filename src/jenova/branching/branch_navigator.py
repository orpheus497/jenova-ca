# The JENOVA Cognitive Architecture - Branch Navigator
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 29: Branch Navigator - Visualize and navigate conversation branches.

Provides ASCII tree visualization of branch structure and navigation
helpers for exploring conversation branches. 100% terminal-based.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from jenova.branching.branch_manager import BranchManager, Branch


class BranchNavigator:
    """
    Navigate and visualize conversation branches.

    Provides terminal-based visualization of branch tree and navigation
    helpers for switching between branches and exploring history.

    Example:
        >>> navigator = BranchNavigator(branch_manager)
        >>> print(navigator.render_tree())
        >>> print(navigator.render_timeline())
    """

    def __init__(self, branch_manager: BranchManager):
        """
        Initialize branch navigator.

        Args:
            branch_manager: BranchManager instance
        """
        self.manager = branch_manager

    def render_tree(self, show_turn_counts: bool = True) -> str:
        """
        Render branch tree as ASCII art.

        Args:
            show_turn_counts: Show turn counts for each branch

        Returns:
            ASCII tree representation

        Example:
            >>> tree = navigator.render_tree()
            >>> print(tree)
        """
        lines = []
        lines.append("CONVERSATION BRANCH TREE")
        lines.append("=" * 60)

        def render_branch(branch_id: str, prefix: str, is_last: bool):
            branch = self.manager.branches[branch_id]

            # Branch indicator
            if branch_id == "main":
                connector = ""
            elif is_last:
                connector = "└── "
            else:
                connector = "├── "

            # Current branch marker
            current_marker = "* " if branch_id == self.manager.current_branch_id else "  "

            # Branch info
            turn_info = f" ({len(branch.turns)} turns)" if show_turn_counts else ""
            branch_line = f"{current_marker}{prefix}{connector}{branch.name}{turn_info}"
            lines.append(branch_line)

            # Find children
            children = [
                bid for bid, b in self.manager.branches.items()
                if b.parent_branch_id == branch_id
            ]

            # Render children
            for i, child_id in enumerate(children):
                is_last_child = (i == len(children) - 1)

                if branch_id == "main":
                    child_prefix = ""
                elif is_last:
                    child_prefix = prefix + "    "
                else:
                    child_prefix = prefix + "│   "

                render_branch(child_id, child_prefix, is_last_child)

        render_branch("main", "", True)

        lines.append("")
        lines.append("* = current branch")

        return "\n".join(lines)

    def render_timeline(
        self,
        branch_id: Optional[str] = None,
        max_turns: int = 10
    ) -> str:
        """
        Render conversation timeline for branch.

        Args:
            branch_id: Branch ID (None = current)
            max_turns: Maximum turns to show

        Returns:
            Formatted timeline

        Example:
            >>> timeline = navigator.render_timeline(max_turns=5)
            >>> print(timeline)
        """
        if branch_id is None:
            branch_id = self.manager.current_branch_id

        branch = self.manager.branches.get(branch_id)
        if not branch:
            return "Branch not found"

        lines = []
        lines.append(f"TIMELINE: {branch.name}")
        lines.append("=" * 60)

        # Get turns (limit to max_turns)
        turns = branch.turns[-max_turns:] if len(branch.turns) > max_turns else branch.turns

        if not turns:
            lines.append("No conversation turns yet")
        else:
            for i, turn in enumerate(turns):
                # Turn header
                timestamp = datetime.fromtimestamp(turn.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"\n[Turn {i + 1}] {timestamp} (ID: {turn.turn_id})")
                lines.append("-" * 60)

                # User message
                lines.append(f"User: {turn.user_message}")

                # Assistant response (truncate if too long)
                response = turn.assistant_response
                if len(response) > 200:
                    response = response[:200] + "..."
                lines.append(f"Assistant: {response}")

        if len(branch.turns) > max_turns:
            lines.append(f"\n... showing last {max_turns} of {len(branch.turns)} turns")

        return "\n".join(lines)

    def render_branch_info(self, branch_id: str) -> str:
        """
        Render detailed branch information.

        Args:
            branch_id: Branch ID

        Returns:
            Formatted branch details

        Example:
            >>> info = navigator.render_branch_info("branch_12345")
            >>> print(info)
        """
        branch = self.manager.branches.get(branch_id)
        if not branch:
            return "Branch not found"

        lines = []
        lines.append(f"BRANCH: {branch.name}")
        lines.append("=" * 60)

        lines.append(f"ID: {branch_id}")
        lines.append(f"Description: {branch.description or '(none)'}")

        # Parent info
        if branch.parent_branch_id:
            parent = self.manager.branches[branch.parent_branch_id]
            lines.append(f"Parent: {parent.name} ({branch.parent_branch_id})")
            if branch.branch_point_turn:
                lines.append(f"Branched from turn: {branch.branch_point_turn}")
        else:
            lines.append("Parent: (root)")

        # Timestamps
        created_time = datetime.fromtimestamp(branch.created_at).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"Created: {created_time}")

        # Turn count
        lines.append(f"Turns: {len(branch.turns)}")

        # Child branches
        children = [
            bid for bid, b in self.manager.branches.items()
            if b.parent_branch_id == branch_id
        ]

        if children:
            lines.append(f"\nChild branches ({len(children)}):")
            for child_id in children:
                child = self.manager.branches[child_id]
                lines.append(f"  - {child.name} ({child_id})")

        # Current branch marker
        if branch_id == self.manager.current_branch_id:
            lines.append("\n[CURRENT BRANCH]")

        return "\n".join(lines)

    def render_stats(self) -> str:
        """
        Render branch manager statistics.

        Returns:
            Formatted statistics

        Example:
            >>> stats = navigator.render_stats()
            >>> print(stats)
        """
        stats = self.manager.get_stats()

        lines = []
        lines.append("BRANCH STATISTICS")
        lines.append("=" * 60)

        lines.append(f"Total branches: {stats['total_branches']}")
        lines.append(f"Total turns (all branches): {stats['total_turns']}")
        lines.append(f"Current branch: {stats['current_branch']}")
        lines.append(f"Current branch turns: {stats['current_branch_turns']}")

        # Branch distribution
        lines.append("\nBranch distribution:")
        for branch_id, branch in self.manager.branches.items():
            marker = "*" if branch_id == self.manager.current_branch_id else " "
            lines.append(f"  {marker} {branch.name}: {len(branch.turns)} turns")

        return "\n".join(lines)

    def find_branch_by_name(self, name: str) -> Optional[str]:
        """
        Find branch ID by name.

        Args:
            name: Branch name (case-insensitive)

        Returns:
            Branch ID or None

        Example:
            >>> branch_id = navigator.find_branch_by_name("Alternative")
        """
        name_lower = name.lower()
        for branch_id, branch in self.manager.branches.items():
            if branch.name.lower() == name_lower:
                return branch_id
        return None

    def get_branch_path(self, branch_id: str) -> List[str]:
        """
        Get path from root to branch.

        Args:
            branch_id: Target branch ID

        Returns:
            List of branch IDs from root to target

        Example:
            >>> path = navigator.get_branch_path("branch_12345")
            >>> print(" -> ".join(path))
        """
        path = []
        current_id = branch_id

        while current_id is not None:
            path.insert(0, current_id)
            branch = self.manager.branches.get(current_id)
            if not branch:
                break
            current_id = branch.parent_branch_id

        return path

    def render_branch_path(self, branch_id: str) -> str:
        """
        Render path from root to branch.

        Args:
            branch_id: Target branch ID

        Returns:
            Formatted path

        Example:
            >>> path = navigator.render_branch_path("branch_12345")
            >>> print(path)
        """
        path_ids = self.get_branch_path(branch_id)

        if not path_ids:
            return "Branch not found"

        lines = []
        lines.append(f"PATH TO: {self.manager.branches[branch_id].name}")
        lines.append("=" * 60)

        for i, bid in enumerate(path_ids):
            branch = self.manager.branches[bid]
            indent = "  " * i
            arrow = " → " if i < len(path_ids) - 1 else ""
            marker = "*" if bid == self.manager.current_branch_id else " "

            lines.append(f"{marker}{indent}{branch.name} ({len(branch.turns)} turns){arrow}")

        return "\n".join(lines)

    def suggest_branches(self, current_turn_index: int) -> List[Dict[str, Any]]:
        """
        Suggest branches user might want to explore.

        Args:
            current_turn_index: Current position in conversation

        Returns:
            List of branch suggestions

        Example:
            >>> suggestions = navigator.suggest_branches(5)
        """
        suggestions = []

        # Find branches not visited yet
        current_branch = self.manager.get_current_branch()

        for branch_id, branch in self.manager.branches.items():
            if branch_id == self.manager.current_branch_id:
                continue

            # Calculate divergence point
            common_turns = 0
            for i, turn in enumerate(branch.turns):
                if i < len(current_branch.turns) and turn.turn_id == current_branch.turns[i].turn_id:
                    common_turns += 1
                else:
                    break

            suggestions.append({
                "branch_id": branch_id,
                "name": branch.name,
                "description": branch.description,
                "turns": len(branch.turns),
                "divergence_point": common_turns,
                "is_parent": branch_id == current_branch.parent_branch_id,
                "is_child": branch.parent_branch_id == self.manager.current_branch_id,
            })

        # Sort by relevance (children first, then parents, then others)
        def sort_key(s):
            if s["is_child"]:
                return (0, -s["turns"])
            elif s["is_parent"]:
                return (1, -s["turns"])
            else:
                return (2, -s["divergence_point"])

        suggestions.sort(key=sort_key)

        return suggestions
