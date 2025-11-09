# The JENOVA Cognitive Architecture - Branching Tests
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 29: Tests for Conversation Branching.

Tests branch management, navigation, and state persistence with
comprehensive coverage.
"""

import pytest
import tempfile
from pathlib import Path

from jenova.branching import (
    ConversationTurn,
    Branch,
    BranchManager,
    BranchNavigator,
)


class TestBranchManager:
    """Test suite for BranchManager."""

    @pytest.fixture
    def manager(self):
        """Fixture providing BranchManager instance."""
        return BranchManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.current_branch_id == "main"
        assert "main" in manager.branches
        assert len(manager.branches["main"].turns) == 0

    def test_add_turn(self, manager):
        """Test adding conversation turn."""
        turn_id = manager.add_turn("Hello", "Hi there!")

        assert turn_id.startswith("turn_")
        current_branch = manager.get_current_branch()
        assert len(current_branch.turns) == 1
        assert current_branch.turns[0].user_message == "Hello"
        assert current_branch.turns[0].assistant_response == "Hi there!"

    def test_add_multiple_turns(self, manager):
        """Test adding multiple turns."""
        for i in range(5):
            manager.add_turn(f"Message {i}", f"Response {i}")

        current_branch = manager.get_current_branch()
        assert len(current_branch.turns) == 5

    def test_create_branch(self, manager):
        """Test creating new branch."""
        # Add some turns
        turn1 = manager.add_turn("Turn 1", "Response 1")
        turn2 = manager.add_turn("Turn 2", "Response 2")

        # Create branch from turn1
        branch_id = manager.create_branch("Alternative", turn1, "Test branch")

        assert branch_id in manager.branches
        branch = manager.branches[branch_id]
        assert branch.name == "Alternative"
        assert branch.description == "Test branch"
        assert branch.parent_branch_id == "main"
        assert branch.branch_point_turn == turn1

        # Branch should have copied turns up to branch point
        assert len(branch.turns) == 1
        assert branch.turns[0].turn_id == turn1

    def test_create_branch_from_end(self, manager):
        """Test creating branch from end of current branch."""
        manager.add_turn("Turn 1", "Response 1")
        manager.add_turn("Turn 2", "Response 2")

        # Create branch without specifying turn (defaults to end)
        branch_id = manager.create_branch("From End")

        branch = manager.branches[branch_id]
        # Should copy all turns
        assert len(branch.turns) == 2

    def test_switch_branch(self, manager):
        """Test switching between branches."""
        manager.add_turn("Main turn", "Main response")

        # Create new branch
        branch_id = manager.create_branch("Alternative")

        # Switch to new branch
        assert manager.switch_branch(branch_id)
        assert manager.current_branch_id == branch_id

        # Add turn to new branch
        manager.add_turn("Alt turn", "Alt response")

        # Verify turns are separate
        assert len(manager.branches["main"].turns) == 1
        assert len(manager.branches[branch_id].turns) == 2  # Copied + new

    def test_switch_to_nonexistent_branch(self, manager):
        """Test switching to nonexistent branch."""
        result = manager.switch_branch("nonexistent")
        assert not result
        assert manager.current_branch_id == "main"

    def test_get_current_branch(self, manager):
        """Test getting current branch."""
        branch = manager.get_current_branch()
        assert branch.branch_id == "main"
        assert isinstance(branch, Branch)

    def test_get_branch(self, manager):
        """Test getting branch by ID."""
        branch = manager.get_branch("main")
        assert branch is not None
        assert branch.branch_id == "main"

        nonexistent = manager.get_branch("nonexistent")
        assert nonexistent is None

    def test_list_branches(self, manager):
        """Test listing all branches."""
        manager.add_turn("Turn 1", "Response 1")
        manager.create_branch("Branch 1")
        manager.create_branch("Branch 2")

        branches = manager.list_branches()
        assert len(branches) == 3  # main + 2 created
        assert all("branch_id" in b for b in branches)
        assert all("name" in b for b in branches)
        assert sum(1 for b in branches if b["is_current"]) == 1

    def test_get_branch_tree(self, manager):
        """Test getting branch tree structure."""
        manager.add_turn("Turn 1", "Response 1")

        branch1 = manager.create_branch("Branch 1")
        manager.switch_branch(branch1)
        manager.add_turn("Turn 2", "Response 2")

        branch2 = manager.create_branch("Branch 2")

        tree = manager.get_branch_tree()
        assert tree["branch_id"] == "main"
        assert len(tree["children"]) >= 1
        assert tree["name"] == "Main"

    def test_get_conversation_history(self, manager):
        """Test getting conversation history."""
        manager.add_turn("Turn 1", "Response 1")
        manager.add_turn("Turn 2", "Response 2")
        manager.add_turn("Turn 3", "Response 3")

        # Get all history
        history = manager.get_conversation_history()
        assert len(history) == 3

        # Get limited history
        history = manager.get_conversation_history(max_turns=2)
        assert len(history) == 2
        assert history[0].user_message == "Turn 2"  # Last 2

    def test_delete_branch(self, manager):
        """Test deleting branch."""
        manager.add_turn("Turn 1", "Response 1")
        branch_id = manager.create_branch("To Delete")

        # Delete branch
        result = manager.delete_branch(branch_id)
        assert result
        assert branch_id not in manager.branches

    def test_delete_main_branch(self, manager):
        """Test that main branch cannot be deleted."""
        result = manager.delete_branch("main")
        assert not result
        assert "main" in manager.branches

    def test_delete_current_branch(self, manager):
        """Test that current branch cannot be deleted."""
        branch_id = manager.create_branch("Current")
        manager.switch_branch(branch_id)

        result = manager.delete_branch(branch_id)
        assert not result
        assert branch_id in manager.branches

    def test_delete_branch_with_children(self, manager):
        """Test deleting branch with children."""
        parent_id = manager.create_branch("Parent")
        manager.switch_branch(parent_id)
        child_id = manager.create_branch("Child")

        manager.switch_branch("main")

        # Delete without children (reparent)
        result = manager.delete_branch(parent_id, delete_children=False)
        assert result
        assert parent_id not in manager.branches
        assert child_id in manager.branches
        # Child should be reparented to main
        assert manager.branches[child_id].parent_branch_id == "main"

    def test_delete_branch_cascade(self, manager):
        """Test cascading delete with children."""
        parent_id = manager.create_branch("Parent")
        manager.switch_branch(parent_id)
        child_id = manager.create_branch("Child")

        manager.switch_branch("main")

        # Delete with children
        result = manager.delete_branch(parent_id, delete_children=True)
        assert result
        assert parent_id not in manager.branches
        assert child_id not in manager.branches

    def test_save_and_load(self, manager):
        """Test saving and loading branch state."""
        # Create some state
        manager.add_turn("Turn 1", "Response 1")
        branch1 = manager.create_branch("Branch 1")
        manager.switch_branch(branch1)
        manager.add_turn("Turn 2", "Response 2")

        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = Path(f.name)

        try:
            manager.save_to_file(file_path)
            assert file_path.exists()

            # Load into new manager
            new_manager = BranchManager()
            result = new_manager.load_from_file(file_path)

            assert result
            assert len(new_manager.branches) == len(manager.branches)
            assert new_manager.current_branch_id == manager.current_branch_id
            assert len(new_manager.branches["main"].turns) == len(manager.branches["main"].turns)
        finally:
            file_path.unlink()

    def test_load_nonexistent_file(self, manager):
        """Test loading from nonexistent file."""
        result = manager.load_from_file(Path("/nonexistent/file.json"))
        assert not result

    def test_get_stats(self, manager):
        """Test getting statistics."""
        manager.add_turn("Turn 1", "Response 1")
        manager.add_turn("Turn 2", "Response 2")
        manager.create_branch("Branch 1")

        stats = manager.get_stats()
        assert stats["total_branches"] == 2
        assert stats["total_turns"] >= 2
        assert stats["current_branch"] == "main"


class TestBranchNavigator:
    """Test suite for BranchNavigator."""

    @pytest.fixture
    def manager_with_branches(self):
        """Fixture providing BranchManager with multiple branches."""
        manager = BranchManager()

        # Create main conversation
        manager.add_turn("Hello", "Hi there!")
        turn1 = manager.add_turn("How are you?", "I'm doing well!")

        # Create alternative branch
        branch1 = manager.create_branch("Alternative", turn1, "Different approach")
        manager.switch_branch(branch1)
        manager.add_turn("What's the weather?", "It's sunny!")

        # Create another branch from main
        manager.switch_branch("main")
        branch2 = manager.create_branch("Another path", turn1)

        manager.switch_branch("main")

        return manager

    @pytest.fixture
    def navigator(self, manager_with_branches):
        """Fixture providing BranchNavigator instance."""
        return BranchNavigator(manager_with_branches)

    def test_initialization(self, manager_with_branches):
        """Test navigator initialization."""
        navigator = BranchNavigator(manager_with_branches)
        assert navigator.manager is manager_with_branches

    def test_render_tree(self, navigator):
        """Test branch tree rendering."""
        tree = navigator.render_tree()
        assert isinstance(tree, str)
        assert "CONVERSATION BRANCH TREE" in tree
        assert "Main" in tree
        assert "* = current branch" in tree

    def test_render_tree_with_turn_counts(self, navigator):
        """Test tree rendering with turn counts."""
        tree = navigator.render_tree(show_turn_counts=True)
        assert "turns)" in tree

    def test_render_timeline(self, navigator):
        """Test timeline rendering."""
        timeline = navigator.render_timeline()
        assert isinstance(timeline, str)
        assert "TIMELINE:" in timeline
        assert "User:" in timeline or "No conversation turns" in timeline

    def test_render_timeline_with_limit(self, navigator):
        """Test timeline rendering with turn limit."""
        timeline = navigator.render_timeline(max_turns=1)
        assert isinstance(timeline, str)

    def test_render_branch_info(self, navigator):
        """Test branch info rendering."""
        info = navigator.render_branch_info("main")
        assert isinstance(info, str)
        assert "BRANCH: Main" in info
        assert "ID: main" in info

    def test_render_branch_info_nonexistent(self, navigator):
        """Test branch info for nonexistent branch."""
        info = navigator.render_branch_info("nonexistent")
        assert "not found" in info.lower()

    def test_render_stats(self, navigator):
        """Test statistics rendering."""
        stats = navigator.render_stats()
        assert isinstance(stats, str)
        assert "BRANCH STATISTICS" in stats
        assert "Total branches:" in stats

    def test_find_branch_by_name(self, navigator):
        """Test finding branch by name."""
        branch_id = navigator.find_branch_by_name("Main")
        assert branch_id == "main"

        branch_id = navigator.find_branch_by_name("alternative")  # Case insensitive
        assert branch_id is not None

        nonexistent = navigator.find_branch_by_name("Nonexistent")
        assert nonexistent is None

    def test_get_branch_path(self, navigator, manager_with_branches):
        """Test getting branch path."""
        # Create nested branch
        manager_with_branches.switch_branch("main")
        turn = manager_with_branches.add_turn("New turn", "New response")
        branch1 = manager_with_branches.create_branch("Level 1")
        manager_with_branches.switch_branch(branch1)
        branch2 = manager_with_branches.create_branch("Level 2")

        path = navigator.get_branch_path(branch2)
        assert len(path) >= 2
        assert path[0] == "main"
        assert path[-1] == branch2

    def test_render_branch_path(self, navigator, manager_with_branches):
        """Test rendering branch path."""
        manager_with_branches.switch_branch("main")
        branch = manager_with_branches.create_branch("Test Branch")

        path_str = navigator.render_branch_path(branch)
        assert isinstance(path_str, str)
        assert "PATH TO:" in path_str

    def test_suggest_branches(self, navigator):
        """Test branch suggestions."""
        suggestions = navigator.suggest_branches(0)
        assert isinstance(suggestions, list)
        assert all("branch_id" in s for s in suggestions)
        assert all("name" in s for s in suggestions)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_branching_workflow(self):
        """Test complete branching workflow."""
        manager = BranchManager()
        navigator = BranchNavigator(manager)

        # Main conversation
        manager.add_turn("What is AI?", "AI is artificial intelligence...")
        turn1 = manager.add_turn("Tell me more", "AI includes machine learning...")

        # Create alternative exploration
        alt_branch = manager.create_branch("Deep dive", turn1, "Explore ML deeply")
        manager.switch_branch(alt_branch)
        manager.add_turn("Explain neural networks", "Neural networks are...")

        # Switch back to main
        manager.switch_branch("main")
        manager.add_turn("What about robotics?", "Robotics combines...")

        # Verify structure
        assert len(manager.branches) == 2
        assert len(manager.branches["main"].turns) == 3
        assert len(manager.branches[alt_branch].turns) == 3  # 2 copied + 1 new

        # Visualize
        tree = navigator.render_tree()
        assert "Main" in tree
        assert "Deep dive" in tree

    def test_save_load_and_navigate(self):
        """Test save, load, and navigation."""
        manager = BranchManager()

        # Create branched conversation
        manager.add_turn("Turn 1", "Response 1")
        branch1 = manager.create_branch("Branch 1")
        manager.switch_branch(branch1)
        manager.add_turn("Turn 2", "Response 2")

        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = Path(f.name)

        try:
            manager.save_to_file(file_path)

            # Load into new manager
            new_manager = BranchManager()
            new_manager.load_from_file(file_path)

            # Navigate loaded state
            navigator = BranchNavigator(new_manager)
            tree = navigator.render_tree()

            assert "Main" in tree
            assert "Branch 1" in tree

            # Verify can switch branches
            assert new_manager.switch_branch("main")
            assert new_manager.current_branch_id == "main"
        finally:
            file_path.unlink()
