# The JENOVA Cognitive Architecture - Collaboration Tests
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 30: Tests for Multi-User Collaboration.

Tests user session management, collaborative sessions, synchronization
protocol, and access control with comprehensive coverage.
"""

import pytest
import tempfile
import time
from pathlib import Path

from jenova.collaboration import (
    # User session
    UserRole,
    SessionStatus,
    UserProfile,
    UserSession,
    UserSessionManager,
    # Collaboration
    CollaborationMode,
    ConversationState,
    CollaborativeSession,
    CollaborationManager,
    # Synchronization
    MessageType,
    SyncMessage,
    SyncProtocol,
    SyncClient,
    # Access control
    Permission,
    AccessPolicy,
    AccessController,
    ROLE_PERMISSIONS,
)


class TestUserSessionManager:
    """Test suite for UserSessionManager."""

    @pytest.fixture
    def manager(self):
        """Fixture providing UserSessionManager instance."""
        return UserSessionManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.profiles) == 0
        assert len(manager.sessions) == 0
        assert len(manager.user_sessions) == 0

    def test_create_user(self, manager):
        """Test user creation."""
        profile = manager.create_user("alice", "Alice")

        assert profile.user_id == "alice"
        assert profile.username == "Alice"
        assert profile.user_id in manager.profiles
        assert profile.user_id in manager.user_sessions

    def test_create_duplicate_user(self, manager):
        """Test creating duplicate user returns existing."""
        profile1 = manager.create_user("alice", "Alice")
        profile2 = manager.create_user("alice", "Alice Smith")

        assert profile1.user_id == profile2.user_id
        assert profile1.username == profile2.username  # Not updated

    def test_get_user(self, manager):
        """Test getting user profile."""
        manager.create_user("alice", "Alice")

        profile = manager.get_user("alice")
        assert profile is not None
        assert profile.username == "Alice"

        nonexistent = manager.get_user("bob")
        assert nonexistent is None

    def test_update_user(self, manager):
        """Test updating user profile."""
        manager.create_user("alice", "Alice")

        result = manager.update_user("alice", username="Alice Smith", display_color="#ff0000")
        assert result

        profile = manager.get_user("alice")
        assert profile.username == "Alice Smith"
        assert profile.display_color == "#ff0000"

    def test_create_session(self, manager):
        """Test session creation."""
        profile = manager.create_user("alice", "Alice")
        session = manager.create_session(profile, UserRole.OWNER)

        assert session.session_id in manager.sessions
        assert session.user_profile.user_id == "alice"
        assert session.role == UserRole.OWNER
        assert session.status == SessionStatus.ACTIVE

    def test_get_session(self, manager):
        """Test getting session."""
        profile = manager.create_user("alice", "Alice")
        session = manager.create_session(profile, UserRole.CONTRIBUTOR)

        retrieved = manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_end_session(self, manager):
        """Test ending session."""
        profile = manager.create_user("alice", "Alice")
        session = manager.create_session(profile, UserRole.CONTRIBUTOR)

        result = manager.end_session(session.session_id)
        assert result
        assert session.session_id not in manager.sessions

    def test_update_activity(self, manager):
        """Test updating session activity."""
        profile = manager.create_user("alice", "Alice")
        session = manager.create_session(profile, UserRole.CONTRIBUTOR)

        initial_activity = session.last_activity
        time.sleep(0.01)

        result = manager.update_activity(session.session_id, turn_contributed=True)
        assert result
        assert session.last_activity > initial_activity
        assert session.turns_contributed == 1

    def test_get_active_sessions(self, manager):
        """Test getting active sessions."""
        profile1 = manager.create_user("alice", "Alice")
        profile2 = manager.create_user("bob", "Bob")

        session1 = manager.create_session(profile1, UserRole.OWNER)
        session2 = manager.create_session(profile2, UserRole.CONTRIBUTOR)
        session2.status = SessionStatus.IDLE

        active = manager.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == session1.session_id

    def test_check_idle_sessions(self, manager):
        """Test checking for idle sessions."""
        profile = manager.create_user("alice", "Alice")
        session = manager.create_session(profile, UserRole.CONTRIBUTOR)

        # Manually set last activity to past
        session.last_activity = time.time() - 400

        idle = manager.check_idle_sessions(idle_timeout=300)
        assert len(idle) == 1
        assert session.session_id in idle
        assert session.status == SessionStatus.IDLE

    def test_kick_session(self, manager):
        """Test kicking session."""
        profile = manager.create_user("alice", "Alice")
        session = manager.create_session(profile, UserRole.CONTRIBUTOR)

        result = manager.kick_session(session.session_id, "Violation")
        assert result
        assert session.status == SessionStatus.KICKED
        assert "kick_reason" in session.metadata

    def test_get_session_stats(self, manager):
        """Test getting session statistics."""
        profile = manager.create_user("alice", "Alice")
        session = manager.create_session(profile, UserRole.CONTRIBUTOR)
        manager.update_activity(session.session_id, turn_contributed=True, message_sent=True)

        stats = manager.get_session_stats(session.session_id)
        assert stats["user_id"] == "alice"
        assert stats["turns_contributed"] == 1
        assert stats["messages_sent"] == 1


class TestCollaborationManager:
    """Test suite for CollaborationManager."""

    @pytest.fixture
    def manager(self):
        """Fixture providing CollaborationManager instance."""
        return CollaborationManager()

    @pytest.fixture
    def manager_with_users(self, manager):
        """Fixture with pre-created users and sessions."""
        # Create users
        alice_profile = manager.user_manager.create_user("alice", "Alice")
        bob_profile = manager.user_manager.create_user("bob", "Bob")

        # Create user sessions
        alice_session = manager.user_manager.create_session(alice_profile, UserRole.OWNER)
        bob_session = manager.user_manager.create_session(bob_profile, UserRole.CONTRIBUTOR)

        return manager, alice_session, bob_session

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.sessions) == 0
        assert len(manager.turn_locks) == 0

    def test_create_session(self, manager_with_users):
        """Test creating collaborative session."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", "Discussion")

        assert session.session_id in manager.sessions
        assert session.name == "Team Chat"
        assert session.owner_id == "alice"
        assert session.mode == CollaborationMode.SEQUENTIAL

    def test_add_participant(self, manager_with_users):
        """Test adding participant to session."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat")

        result = manager.add_participant(session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)
        assert result
        assert bob_session.session_id in session.participants

    def test_remove_participant(self, manager_with_users):
        """Test removing participant."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat")
        manager.add_participant(session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)

        result = manager.remove_participant(session.session_id, bob_session.session_id)
        assert result
        assert bob_session.session_id not in session.participants

    def test_request_turn_sequential(self, manager_with_users):
        """Test requesting turn in sequential mode."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", mode=CollaborationMode.SEQUENTIAL)
        manager.add_participant(session.session_id, alice_session.session_id, UserRole.OWNER)
        manager.add_participant(session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)

        # Alice requests turn
        result1 = manager.request_turn(session.session_id, alice_session.session_id)
        assert result1

        # Bob tries to request turn (should fail)
        result2 = manager.request_turn(session.session_id, bob_session.session_id)
        assert not result2

    def test_release_turn(self, manager_with_users):
        """Test releasing turn."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", mode=CollaborationMode.SEQUENTIAL)
        manager.add_participant(session.session_id, alice_session.session_id, UserRole.OWNER)

        manager.request_turn(session.session_id, alice_session.session_id)
        result = manager.release_turn(session.session_id, alice_session.session_id)

        assert result
        assert session.session_id not in manager.turn_locks

    def test_add_turn(self, manager_with_users):
        """Test adding conversation turn."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", mode=CollaborationMode.CONCURRENT)
        manager.add_participant(session.session_id, alice_session.session_id, UserRole.OWNER)

        turn_id = manager.add_turn(session.session_id, alice_session.session_id, "Hello!")

        assert turn_id is not None
        assert len(session.conversation_turns) == 1
        assert session.conversation_turns[0]["user_message"] == "Hello!"

    def test_add_turn_moderated(self, manager_with_users):
        """Test adding turn in moderated mode."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", mode=CollaborationMode.MODERATED)
        manager.add_participant(session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)

        contrib_id = manager.add_turn(session.session_id, bob_session.session_id, "Hello!")

        assert contrib_id is not None
        assert len(session.pending_contributions) == 1
        assert len(session.conversation_turns) == 0

    def test_approve_contribution(self, manager_with_users):
        """Test approving pending contribution."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", mode=CollaborationMode.MODERATED)
        manager.add_participant(session.session_id, alice_session.session_id, UserRole.OWNER)
        manager.add_participant(session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)

        contrib_id = manager.add_turn(session.session_id, bob_session.session_id, "Hello!")
        result = manager.approve_contribution(session.session_id, contrib_id, alice_session.session_id)

        assert result
        assert len(session.conversation_turns) == 1
        assert len(session.pending_contributions) == 0

    def test_reject_contribution(self, manager_with_users):
        """Test rejecting pending contribution."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", mode=CollaborationMode.MODERATED)
        manager.add_participant(session.session_id, alice_session.session_id, UserRole.OWNER)
        manager.add_participant(session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)

        contrib_id = manager.add_turn(session.session_id, bob_session.session_id, "Spam!")
        result = manager.reject_contribution(session.session_id, contrib_id, alice_session.session_id, "Off-topic")

        assert result
        assert len(session.pending_contributions) == 0
        assert len(session.conversation_turns) == 0

    def test_get_conversation_history(self, manager_with_users):
        """Test getting conversation history."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat", mode=CollaborationMode.CONCURRENT)
        manager.add_participant(session.session_id, alice_session.session_id, UserRole.OWNER)

        manager.add_turn(session.session_id, alice_session.session_id, "Turn 1")
        manager.add_turn(session.session_id, alice_session.session_id, "Turn 2")
        manager.add_turn(session.session_id, alice_session.session_id, "Turn 3")

        history = manager.get_conversation_history(session.session_id, max_turns=2)
        assert len(history) == 2
        assert history[0]["user_message"] == "Turn 2"

    def test_save_session(self, manager_with_users):
        """Test saving session to file."""
        manager, alice_session, bob_session = manager_with_users

        session = manager.create_session("alice", "Team Chat")
        manager.add_turn(session.session_id, alice_session.session_id, "Hello!")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            file_path = Path(f.name)

        try:
            result = manager.save_session(session.session_id, file_path)
            assert result
            assert file_path.exists()
        finally:
            file_path.unlink()


class TestSyncProtocol:
    """Test suite for SyncProtocol."""

    def test_initialization(self):
        """Test protocol initialization."""
        protocol = SyncProtocol(port=5555)

        assert protocol.port == 5555
        assert not protocol.running

    def test_register_handler(self):
        """Test registering message handler."""
        protocol = SyncProtocol()

        def handler(msg):
            pass

        protocol.register_handler(MessageType.TURN_ADD, handler)
        assert MessageType.TURN_ADD in protocol.handlers
        assert handler in protocol.handlers[MessageType.TURN_ADD]

    def test_unregister_handler(self):
        """Test unregistering handler."""
        protocol = SyncProtocol()

        def handler(msg):
            pass

        protocol.register_handler(MessageType.TURN_ADD, handler)
        protocol.unregister_handler(MessageType.TURN_ADD, handler)

        assert handler not in protocol.handlers.get(MessageType.TURN_ADD, [])


class TestSyncClient:
    """Test suite for SyncClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = SyncClient("alice_session", port=5556)

        assert client.session_id == "alice_session"
        assert client.protocol.port == 5556

    def test_message_creation(self):
        """Test creating sync message."""
        client = SyncClient("alice_session")

        message = client._create_message(
            MessageType.TURN_ADD,
            "session_123",
            {"data": "test"}
        )

        assert message.message_type == MessageType.TURN_ADD
        assert message.sender_id == "alice_session"
        assert message.session_id == "session_123"
        assert message.payload["data"] == "test"


class TestAccessController:
    """Test suite for AccessController."""

    @pytest.fixture
    def controller(self):
        """Fixture providing AccessController instance."""
        return AccessController()

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert len(controller.policies) == 0
        assert len(controller.custom_permissions) == 0

    def test_check_permission_owner(self, controller):
        """Test checking owner permissions."""
        has_perm = controller.check_permission(UserRole.OWNER, Permission.SESSION_DELETE)
        assert has_perm

    def test_check_permission_viewer(self, controller):
        """Test checking viewer permissions."""
        has_perm = controller.check_permission(UserRole.VIEWER, Permission.CONTENT_ADD)
        assert not has_perm

        has_view = controller.check_permission(UserRole.VIEWER, Permission.CONTENT_VIEW)
        assert has_view

    def test_create_policy(self, controller):
        """Test creating access policy."""
        policy = controller.create_policy("resource_1", {UserRole.OWNER, UserRole.ADMIN})

        assert policy.resource_id == "resource_1"
        assert UserRole.OWNER in policy.allowed_roles

    def test_check_access(self, controller):
        """Test checking resource access."""
        controller.create_policy("resource_1", {UserRole.OWNER})

        has_access = controller.check_access("resource_1", "alice", UserRole.OWNER)
        assert has_access

        no_access = controller.check_access("resource_1", "bob", UserRole.VIEWER)
        assert not no_access

    def test_grant_access(self, controller):
        """Test granting user access."""
        controller.create_policy("resource_1", {UserRole.OWNER})
        controller.grant_access("resource_1", "alice")

        # Alice should have access even with VIEWER role
        has_access = controller.check_access("resource_1", "alice", UserRole.VIEWER)
        assert has_access

    def test_revoke_access(self, controller):
        """Test revoking user access."""
        controller.create_policy("resource_1", {UserRole.OWNER})
        controller.grant_access("resource_1", "alice")
        controller.revoke_access("resource_1", "alice")

        # Alice should not have access even with OWNER role
        has_access = controller.check_access("resource_1", "alice", UserRole.OWNER)
        assert not has_access

    def test_grant_permission_to_role(self, controller):
        """Test granting custom permission to role."""
        controller.grant_permission_to_role(UserRole.VIEWER, Permission.CONTENT_ADD)

        has_perm = controller.check_permission(UserRole.VIEWER, Permission.CONTENT_ADD)
        assert has_perm

    def test_revoke_permission_from_role(self, controller):
        """Test revoking permission from role."""
        controller.revoke_permission_from_role(UserRole.CONTRIBUTOR, Permission.CONTENT_ADD)

        has_perm = controller.check_permission(UserRole.CONTRIBUTOR, Permission.CONTENT_ADD)
        assert not has_perm

    def test_get_role_permissions(self, controller):
        """Test getting role permissions."""
        perms = controller.get_role_permissions(UserRole.ADMIN)

        assert Permission.CONTENT_ADD in perms
        assert Permission.SESSION_DELETE not in perms


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_collaboration_workflow(self):
        """Test complete multi-user collaboration workflow."""
        manager = CollaborationManager()

        # Create users
        alice_profile = manager.user_manager.create_user("alice", "Alice")
        bob_profile = manager.user_manager.create_user("bob", "Bob")

        # Create user sessions
        alice_session = manager.user_manager.create_session(alice_profile, UserRole.OWNER)
        bob_session = manager.user_manager.create_session(bob_profile, UserRole.CONTRIBUTOR)

        # Create collaborative session
        collab_session = manager.create_session("alice", "Project Planning", mode=CollaborationMode.SEQUENTIAL)

        # Add participants
        manager.add_participant(collab_session.session_id, alice_session.session_id, UserRole.OWNER)
        manager.add_participant(collab_session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)

        # Alice takes turn and adds message
        manager.request_turn(collab_session.session_id, alice_session.session_id)
        manager.add_turn(collab_session.session_id, alice_session.session_id, "Let's plan the project!")

        # Bob takes turn and adds message
        manager.request_turn(collab_session.session_id, bob_session.session_id)
        manager.add_turn(collab_session.session_id, bob_session.session_id, "Great idea!")

        # Verify conversation
        history = manager.get_conversation_history(collab_session.session_id)
        assert len(history) == 2
        assert history[0]["user_id"] == "alice"
        assert history[1]["user_id"] == "bob"

    def test_moderated_collaboration(self):
        """Test moderated collaboration with approval workflow."""
        manager = CollaborationManager()
        controller = AccessController()

        # Create users
        alice_profile = manager.user_manager.create_user("alice", "Alice")
        bob_profile = manager.user_manager.create_user("bob", "Bob")

        # Create sessions
        alice_session = manager.user_manager.create_session(alice_profile, UserRole.ADMIN)
        bob_session = manager.user_manager.create_session(bob_profile, UserRole.CONTRIBUTOR)

        # Create moderated session
        collab_session = manager.create_session("alice", "Moderated Discussion", mode=CollaborationMode.MODERATED)

        # Add participants
        manager.add_participant(collab_session.session_id, alice_session.session_id, UserRole.ADMIN)
        manager.add_participant(collab_session.session_id, bob_session.session_id, UserRole.CONTRIBUTOR)

        # Bob submits contribution
        contrib_id = manager.add_turn(collab_session.session_id, bob_session.session_id, "My idea")

        # Verify pending
        assert len(collab_session.pending_contributions) == 1

        # Alice approves
        manager.approve_contribution(collab_session.session_id, contrib_id, alice_session.session_id)

        # Verify approved
        assert len(collab_session.conversation_turns) == 1
        assert len(collab_session.pending_contributions) == 0
