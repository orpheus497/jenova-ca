# The JENOVA Cognitive Architecture - Branching Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 29: Conversation Branching - Branch and navigate conversation paths.

Provides conversation branching, allowing users to explore alternative
conversation paths, rewind to earlier states, and maintain complete
conversation tree. 100% offline, local state management.

Components:
    - BranchManager: Manages branches, turns, and branch tree
    - BranchNavigator: Visualizes and navigates branch structure

Example:
    >>> from jenova.branching import BranchManager, BranchNavigator
    >>>
    >>> # Create and manage branches
    >>> manager = BranchManager()
    >>> turn_id = manager.add_turn("Hello", "Hi there!")
    >>> branch_id = manager.create_branch("Alternative", turn_id)
    >>> manager.switch_branch(branch_id)
    >>>
    >>> # Navigate and visualize
    >>> navigator = BranchNavigator(manager)
    >>> print(navigator.render_tree())
    >>> print(navigator.render_timeline())
    >>>
    >>> # Save/load state
    >>> manager.save_to_file(Path("branches.json"))
    >>> manager.load_from_file(Path("branches.json"))
"""

from jenova.branching.branch_manager import (
    ConversationTurn,
    Branch,
    BranchManager,
)
from jenova.branching.branch_navigator import BranchNavigator

__all__ = [
    "ConversationTurn",
    "Branch",
    "BranchManager",
    "BranchNavigator",
]
